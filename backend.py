"""
AQI — FastAPI Backend
Run with: uvicorn backend:app --reload --port 8000
"""

import os
from pathlib import Path
import base64
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Optional

import asyncio

import httpx
import numpy as np
import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── env / config ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    pass

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value.strip() in {"", "YOUR_MONGODB_URI_HERE"}:
        hint = f"Set {name} in {_ENV_FILE}" if _ENV_FILE.exists() else f"Create {_ENV_FILE} with {name}"
        raise RuntimeError(hint)
    return value.strip()

MONGODB_URI  = _require_env("MONGODB_URI")
FEATURE_DB   = os.getenv("MONGODB_DB",                  "AQI_Project")
FEATURE_COL  = os.getenv("MONGODB_FEATURES_COLLECTION", "karachi_aqi_features")
MODEL_DB     = os.getenv("MODEL_DB",  "aqi_model_store")
MODEL_COL    = os.getenv("MODEL_COL", "AQI_72h_model")

KEY_HORIZONS = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72]
MAX_H = 72
BANDS = {
    "short":  list(range(1, 9)),
    "medium": [9, 12, 15, 18, 21, 24],
    "long":   [25, 30, 36, 42, 48, 54, 60, 66, 72],
}
FORECAST_WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "windspeed_10m", "wind_gusts_10m", "cloud_cover",
    "shortwave_radiation", "vapour_pressure_deficit",
    "precipitation", "dew_point_2m", "pressure_msl",
]
AQI_LEVELS = [
    (0,   50,  "Good",                    "#00e400"),
    (51,  100, "Moderate",                "#ffff00"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00"),
    (151, 200, "Unhealthy",               "#ff0000"),
    (201, 300, "Very Unhealthy",          "#8f3f97"),
    (301, 500, "Hazardous",               "#7e0023"),
]
LATITUDE  = 24.8607
LONGITUDE = 67.0011
PKT = pytz.timezone("Asia/Karachi")

# ── helpers ───────────────────────────────────────────────────────────────────

def aqi_color(val: float) -> str:
    for lo, hi, _, color in AQI_LEVELS:
        if lo <= val <= hi:
            return color
    return "#7e0023"

def aqi_label(val: float) -> str:
    for lo, hi, label, _ in AQI_LEVELS:
        if lo <= val <= hi:
            return label
    return "Hazardous"

def get_pkt_now() -> datetime:
    return datetime.now(PKT).replace(tzinfo=None)

def horizon_encoding(h: int) -> np.ndarray:
    band_flag = 0.0 if h <= 8 else (0.5 if h <= 24 else 1.0)
    return np.array([h / MAX_H, np.log1p(h) / np.log1p(MAX_H),
                     np.sqrt(h / MAX_H), band_flag], dtype=np.float32)

# ── MongoDB client (module-level singleton) ───────────────────────────────────
_mongo_client: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=15000,
        )
        _mongo_client.admin.command("ping")
    return _mongo_client

# ── model cache (loaded once at startup) ─────────────────────────────────────
_model_cache: dict = {}

# ── snapshot cache — rebuilt on every ingest trigger ─────────────────────────
# Holds the fully-computed current + forecast payload so /api/current and
# /api/forecast serve from memory instead of hitting MongoDB per request.
_snapshot: dict = {}          # keys: "current", "forecast", "updated_at"
_snapshot_lock = asyncio.Lock()

def load_models() -> dict:
    """
    Load models from MongoDB. Handles all pipeline versions (v5, v6, v7).

    Precedence: latest pipeline_version wins. For duplicate horizons,
    the document with the highest version number is used.
    Decompression: tries zlib first, falls back to raw pickle.
    """
    import zlib

    global _model_cache
    if _model_cache:
        return _model_cache

    client = get_mongo_client()
    col    = client[MODEL_DB][MODEL_COL]
    docs   = list(col.find({}, {"_id": 0}))
    if not docs:
        raise ValueError(f"No documents found in {MODEL_DB}.{MODEL_COL}")

    # ── Version priority map ──────────────────────────────────────────────
    # When multiple docs exist for the same horizon, keep the latest version.
    VERSION_ORDER = {"v7": 7, "v6": 6, "v5": 5, "v4": 4, "v3": 3}

    def version_rank(doc):
        return VERSION_ORDER.get(doc.get("pipeline_version", ""), 0)

    # Group docs by horizon, keep highest version per horizon
    best_docs = {}
    for doc in docs:
        key = doc.get("horizon")
        if key is None or "model_blob" not in doc:
            continue
        if key not in best_docs or version_rank(doc) > version_rank(best_docs[key]):
            best_docs[key] = doc

    def decode_blob(doc):
        """Decode model_blob: base64 -> try zlib decompress -> pickle.loads."""
        raw = base64.b64decode(doc["model_blob"])
        # Try zlib decompression first
        try:
            raw = zlib.decompress(raw)
        except Exception:
            pass  # Not compressed — use raw bytes directly
        return pickle.loads(raw)

    # Pull created_at from any horizon doc (they all share the same training run)
    created_at = None
    for doc in best_docs.values():
        if isinstance(doc.get("horizon"), int) and doc.get("created_at"):
            created_at = doc["created_at"]
            break

    models        = {}
    valid_indices = {}
    fc_indices_map= {}
    feature_cols  = []
    main_scaler   = None
    training_logs = {}
    model_name_used = "unknown"

    for key, doc in best_docs.items():
        try:
            if key == "_scaler":
                main_scaler  = decode_blob(doc)
                feature_cols = doc.get("feature_cols", [])
                continue

            # Skip non-integer horizon meta docs
            if not isinstance(key, int):
                continue

            payload = decode_blob(doc)

            # payload is either a dict {model, horizon, valid_indices, fc_indices}
            # or a raw model object (very old format)
            if isinstance(payload, dict):
                models[key]         = payload["model"]
                valid_indices[key]  = payload.get("valid_indices", [])
                fc_indices_map[key] = payload.get("fc_indices", [])
                model_name_used     = payload.get("model_name",
                                        doc.get("model_name", model_name_used))
            else:
                models[key]         = payload
                valid_indices[key]  = []
                fc_indices_map[key] = []

            training_logs[key] = doc.get("metrics", {})

        except Exception as e:
            print(f"  WARNING: skipping horizon={key} — {e}")
            continue

    if main_scaler is None or not models:
        raise ValueError(
            f"Models or scaler missing in MongoDB ({MODEL_DB}.{MODEL_COL}). "
            "Run Model_Retrain_pipeline.py to populate the collection."
        )

    print(f"Loaded {len(models)} horizon models ({model_name_used}) + scaler "
          f"| {len(feature_cols)} features")

    _model_cache = dict(
        models=models,
        valid_indices=valid_indices,
        fc_indices_map=fc_indices_map,
        main_scaler=main_scaler,
        feature_cols=feature_cols,
        training_logs=training_logs,
        model_name=model_name_used,
        created_at=created_at,
    )
    return _model_cache

async def fetch_weather_forecast(feature_cols, fc_indices_map, main_scaler) -> dict:
    sample_fci = next(
        (v for v in fc_indices_map.values() if isinstance(v, list) and len(v) > 0), []
    )
    if not sample_fci:
        return {h: np.zeros(0, dtype=np.float32) for h in KEY_HORIZONS}

    fc_cols  = [feature_cols[i] for i in sample_fci if i < len(feature_cols)]
    api_vars = [c for c in FORECAST_WEATHER_COLS if c in fc_cols]
    if not api_vars:
        return {h: np.zeros(len(sample_fci), dtype=np.float32) for h in KEY_HORIZONS}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": LATITUDE, "longitude": LONGITUDE,
                        "hourly": ",".join(api_vars),
                        "timezone": "UTC", "forecast_days": 4},
            )
            resp.raise_for_status()
        data  = resp.json()["hourly"]
        times = pd.to_datetime(data["time"], utc=True).tz_localize(None)
        now   = pd.Timestamp.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=None)
        result = {}
        for h in KEY_HORIZONS:
            fci    = fc_indices_map.get(h, sample_fci)
            fci    = fci if isinstance(fci, list) else sample_fci
            target = now + pd.Timedelta(hours=h)
            if target in times.values:
                idx = list(times.values).index(target)
                vec = np.zeros(len(feature_cols), dtype=np.float32)
                for col in api_vars:
                    if col in feature_cols:
                        val = data[col][idx]
                        vec[feature_cols.index(col)] = float(val) if val is not None else 0.0
                vec_scaled = main_scaler.transform([vec])[0]
                result[h] = vec_scaled[fci] if fci else np.zeros(0, dtype=np.float32)
            else:
                result[h] = np.zeros(len(fci), dtype=np.float32)
        return result
    except Exception:
        return {h: np.zeros(len(sample_fci), dtype=np.float32) for h in KEY_HORIZONS}

async def generate_forecast(cache: dict, df: pd.DataFrame) -> list:
    """
    Build a 72h AQI forecast using the loaded models.

    Sample layout per horizon h (must match Model_Retrain_pipeline.py):
      [ X_scaled[valid_indices]  |  live_weather[fc_indices]  |  horizon_encoding(h) ]
    """
    if df.empty:
        return []

    models         = cache["models"]
    valid_indices  = cache["valid_indices"]
    fc_indices_map = cache["fc_indices_map"]
    main_scaler    = cache["main_scaler"]
    feature_cols   = cache["feature_cols"]

    # Build the full scaled feature vector from the latest row
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        return []

    full_vec = np.zeros(len(feature_cols), dtype=np.float32)
    latest   = df[available_cols].iloc[-1]
    for col in available_cols:
        val = latest[col]
        full_vec[feature_cols.index(col)] = float(val) if pd.notna(val) else 0.0

    X_scaled = main_scaler.transform([full_vec])[0]

    # Fetch live weather forecast for future-weather injection
    sample_fci = next(
        (v for v in fc_indices_map.values() if isinstance(v, list) and v), []
    )
    live_fc = {}
    if sample_fci:
        live_fc = await fetch_weather_forecast(feature_cols, fc_indices_map, main_scaler)

    last_dt   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    forecasts = []

    for h in KEY_HORIZONS:
        if h not in models:
            continue
        vi  = valid_indices.get(h, list(range(len(feature_cols))))
        fci = fc_indices_map.get(h, [])

        try:
            base    = X_scaled[vi]
            fc_vec  = live_fc.get(h, np.zeros(len(fci), dtype=np.float32))
            enc     = horizon_encoding(h)
            row     = np.concatenate([base, fc_vec, enc]).reshape(1, -1)

            pred = max(0.0, float(models[h].predict(row)[0]))
            band = "short" if h <= 8 else ("medium" if h <= 24 else "long")
            forecasts.append({
                "hour":          h,
                "datetime":      (last_dt + timedelta(hours=h)).isoformat() + "Z",
                "predicted_aqi": round(pred, 1),
                "band":          band,
                "category":      aqi_label(pred),
                "color":         aqi_color(pred),
            })
        except Exception as exc:
            print(f"  Forecast error at t+{h}h: {exc}")
            continue

    return forecasts

# ── FastAPI app ───────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

# ── snapshot builder ──────────────────────────────────────────────────────────

async def _build_snapshot() -> dict:
    """
    Pull the latest feature data from MongoDB and run the 72h forecast.
    The result is cached in _snapshot so /api/forecast is served instantly.
    /api/current is NOT cached here — it always queries MongoDB live.
    Called at startup and by /api/ingest-trigger.
    """
    def _load_data():
        cache        = load_models()
        row, stale   = _fetch_latest_row()
        df, df_stale = _fetch_feature_df()
        return cache, row, stale, df, df_stale

    cache, row, stale, df, df_stale = await run_in_threadpool(_load_data)

    # ── forecast payload (the expensive ML part worth caching) ────────────
    forecast_payload: list = []
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        forecast_payload = await generate_forecast(cache, df)

    return {
        "forecast":   forecast_payload,
        "stale":      stale or df_stale,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models, warm up MongoDB, and build the initial snapshot at startup."""
    global _snapshot
    loop = asyncio.get_event_loop()
    try:
        print("Preloading MongoDB connection and ML models...")
        await loop.run_in_executor(None, get_mongo_client)
        await loop.run_in_executor(None, load_models)
        print("Building initial forecast snapshot...")
        _snapshot = await _build_snapshot()
        print(f"Snapshot ready — forecast pts: {len(_snapshot.get('forecast', []))}")
    except Exception as e:
        print(f"Startup preload failed (will retry on first request): {e}")
    yield

app = FastAPI(title="AQI SeekAI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── routes ────────────────────────────────────────────────────────────────────

def _fetch_latest_row(max_age_hours: Optional[int] = 48) -> tuple[Optional[dict], bool]:
    """Return (row, stale).
    stale=True if the newest record is older than 2 hours (pipeline should run hourly).
    Falls back to the absolute newest row if nothing within max_age_hours.
    """
    client = get_mongo_client()
    col = client[FEATURE_DB][FEATURE_COL]

    # Always grab the single newest document regardless of age
    docs = list(col.find({}, {"_id": 0}).sort("datetime", -1).limit(1))
    if not docs:
        return None, False

    row = docs[0]
    # Compute actual age and mark stale if older than 2 hours
    try:
        dt = pd.Timestamp(row["datetime"])
        if dt.tzinfo is not None:
            dt = dt.tz_convert("UTC").tz_localize(None)
        age_hours = (datetime.utcnow() - dt.to_pydatetime()).total_seconds() / 3600
        stale = age_hours > 2.0
    except Exception:
        stale = False

    return row, stale


def _fetch_feature_df(recent_days: int = 5, fallback_hours: int = 72) -> tuple[pd.DataFrame, bool]:
    """Load hourly feature rows for ML inference. Returns (df, stale)."""
    client = get_mongo_client()
    col = client[FEATURE_DB][FEATURE_COL]
    cutoff = datetime.utcnow() - timedelta(days=recent_days)
    docs = list(col.find({"datetime": {"$gte": cutoff}}, {"_id": 0}).sort("datetime", 1))
    if docs:
        return pd.DataFrame(docs), False

    cutoff_h = datetime.utcnow() - timedelta(hours=fallback_hours)
    docs = list(col.find({"datetime": {"$gte": cutoff_h}}, {"_id": 0}).sort("datetime", 1))
    if docs:
        return pd.DataFrame(docs), True

    docs = list(col.find({}, {"_id": 0}).sort("datetime", -1).limit(fallback_hours))
    if not docs:
        return pd.DataFrame(), False
    df = pd.DataFrame(docs).sort_values("datetime").reset_index(drop=True)
    return df, True


@app.get("/")
async def root():
    """Root endpoint — returns API info. Required by Hugging Face health checks."""
    return {
        "name": "AQI SeekAI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/api/current", "/api/forecast",
                      "/api/historical", "/api/eda", "/api/model-info"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "time_pkt": get_pkt_now().isoformat()}


@app.post("/api/ingest-trigger")
async def ingest_trigger():
    """
    Called by GitHub Actions at the end of the hourly pipeline.
    Fetches the latest data from MongoDB, runs the forecast, and stores
    the result in the in-memory snapshot so all subsequent frontend
    requests are served instantly from cache.
    """
    global _snapshot
    async with _snapshot_lock:
        try:
            snap = await _build_snapshot()
            _snapshot = snap
            return {
                "status":       "ok",
                "updated_at":   snap["updated_at"],
                "forecast_pts": len(snap["forecast"]),
            }
        except Exception as e:
            raise HTTPException(500, f"Snapshot build failed: {e}")


@app.post("/api/reload-models")
async def reload_models():
    """
    Force the backend to reload models from MongoDB.
    Call this after running Model_Retrain_pipeline.py so the new
    models are picked up without restarting the server.
    """
    global _model_cache
    _model_cache = {}
    try:
        cache = await run_in_threadpool(load_models)
        return {
            "status":      "reloaded",
            "model_name":  cache.get("model_name", "unknown"),
            "horizons":    len(cache["models"]),
            "features":    len(cache["feature_cols"]),
        }
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {e}")


@app.get("/api/current")
async def get_current():
    """Latest AQI reading + weather — always fetched live from MongoDB."""
    try:
        row, stale = await run_in_threadpool(_fetch_latest_row)
        if row is None:
            raise HTTPException(404, "No data in feature collection")
        aqi = float(row.get("us_aqi", 0))

        weather_keys = [
            "temperature_2m", "relative_humidity_2m", "windspeed_10m",
            "cloud_cover", "dew_point_2m", "surface_pressure",
            "shortwave_radiation", "wind_gusts_10m", "precipitation",
        ]

        # Compute data age in hours
        try:
            dt = pd.Timestamp(row.get("datetime"))
            if dt.tzinfo is not None:
                dt = dt.tz_convert("UTC").tz_localize(None)
            data_age_hours = round(
                (datetime.utcnow() - dt.to_pydatetime()).total_seconds() / 3600, 1
            )
        except Exception:
            data_age_hours = None

        return {
            "aqi":             aqi,
            "category":        aqi_label(aqi),
            "color":           aqi_color(aqi),
            "datetime":        str(row.get("datetime", "")).replace(" ", "T") + "Z",
            "pkt_now":         get_pkt_now().isoformat(),
            "stale":           stale,
            "data_age_hours":  data_age_hours,
            "weather":         {k: row.get(k) for k in weather_keys},
            "sub_aqi": {
                "pm2_5":            row.get("us_aqi_pm2_5"),
                "pm10":             row.get("us_aqi_pm10"),
                "nitrogen_dioxide": row.get("us_aqi_nitrogen_dioxide"),
                "ozone":            row.get("us_aqi_ozone"),
                "sulphur_dioxide":  row.get("us_aqi_sulphur_dioxide"),
                "carbon_monoxide":  row.get("us_aqi_carbon_monoxide"),
            },
            "pollutants": {
                "pm2_5":            row.get("pm2_5"),
                "pm10":             row.get("pm10"),
                "ozone":            row.get("ozone"),
                "nitrogen_dioxide": row.get("nitrogen_dioxide"),
                "sulphur_dioxide":  row.get("sulphur_dioxide"),
                "carbon_monoxide":  row.get("carbon_monoxide"),
            },
            "dominant_pollutant": row.get("dominant_pollutant"),
            "snapshot_at":     _snapshot.get("updated_at"),  # when forecast cache was last built
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/historical")
async def get_historical(days: int = Query(5, ge=1, le=30)):
    """Recent hourly AQI + weather rows."""
    try:
        def _query():
            client = get_mongo_client()
            col    = client[FEATURE_DB][FEATURE_COL]
            cutoff = datetime.utcnow() - timedelta(days=days)
            cursor = col.find({"datetime": {"$gte": cutoff}}, {"_id": 0}).sort("datetime", 1)
            return pd.DataFrame(list(cursor))
        df = await run_in_threadpool(_query)
        if df.empty:
            return {"rows": []}

        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        keep = ["datetime", "us_aqi", "temperature_2m", "relative_humidity_2m",
                "windspeed_10m", "pm2_5", "pm10", "ozone",
                "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]
        keep = [c for c in keep if c in df.columns]
        df   = df[keep].sort_values("datetime")

        rows = []
        for _, r in df.iterrows():
            d = r.to_dict()
            d["datetime"] = str(d["datetime"])
            aqi = d.get("us_aqi")
            if aqi is not None and not pd.isna(aqi):
                d["category"] = aqi_label(float(aqi))
                d["color"]    = aqi_color(float(aqi))
            rows.append(d)

        return {"rows": rows}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/forecast")
async def get_forecast():
    """72-hour AQI forecast. Served from in-memory cache."""
    # Serve from snapshot if available
    if _snapshot.get("forecast") is not None:
        return {
            "forecasts":   _snapshot["forecast"],
            "stale":       _snapshot.get("stale", False),
            "snapshot_at": _snapshot.get("updated_at"),
        }

    # Fallback: compute live (e.g. first request before any trigger)
    try:
        def _load():
            cache = load_models()
            df, stale = _fetch_feature_df()
            return cache, df, stale
        cache, df, stale = await run_in_threadpool(_load)
        if df.empty:
            raise HTTPException(404, "No feature data for inference")

        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        forecasts = await generate_forecast(cache, df)
        return {"forecasts": forecasts, "stale": stale}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/eda")
async def get_eda(days: int = Query(5, ge=1, le=30)):
    """EDA summary statistics."""
    try:
        def _query():
            client = get_mongo_client()
            col    = client[FEATURE_DB][FEATURE_COL]
            cutoff = datetime.utcnow() - timedelta(days=days)
            cursor = col.find({"datetime": {"$gte": cutoff}}, {"_id": 0}).sort("datetime", 1)
            return pd.DataFrame(list(cursor))
        df = await run_in_threadpool(_query)
        if df.empty:
            return {}

        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        aqi_data  = df["us_aqi"].dropna()

        # Category breakdown
        cat_counts = {}
        for val in aqi_data:
            lbl = aqi_label(float(val))
            cat_counts[lbl] = cat_counts.get(lbl, 0) + 1

        # Hourly averages
        df["hour"] = df["datetime"].dt.hour
        hourly = df.groupby("hour")["us_aqi"].mean().reset_index()
        hourly_avg = {int(r["hour"]): round(float(r["us_aqi"]), 1)
                      for _, r in hourly.iterrows() if not pd.isna(r["us_aqi"])}

        # Day × Hour heatmap
        df["dow"] = df["datetime"].dt.day_name()
        piv_rows = []
        for dow, grp in df.groupby("dow"):
            hmap = grp.groupby("hour")["us_aqi"].mean().to_dict()
            piv_rows.append({"day": dow, "hours": {int(h): round(float(v), 1)
                                                    for h, v in hmap.items() if not pd.isna(v)}})

        # Dominant pollutant
        dom_pol = {}
        if "dominant_pollutant" in df.columns:
            dom_pol = df["dominant_pollutant"].value_counts().to_dict()

        # Temperature vs AQI scatter (sample 200)
        scatter = []
        if "temperature_2m" in df.columns:
            sdf = df[["temperature_2m", "us_aqi"]].dropna().sample(min(200, len(df)))
            for _, r in sdf.iterrows():
                scatter.append({"temp": round(float(r["temperature_2m"]), 1),
                                 "aqi":  round(float(r["us_aqi"]), 1)})

        return {
            "summary": {
                "mean":    round(float(aqi_data.mean()), 1)   if len(aqi_data) else None,
                "median":  round(float(aqi_data.median()), 1) if len(aqi_data) else None,
                "max":     round(float(aqi_data.max()), 1)    if len(aqi_data) else None,
                "min":     round(float(aqi_data.min()), 1)    if len(aqi_data) else None,
                "count":   int(len(aqi_data)),
            },
            "category_breakdown": cat_counts,
            "hourly_avg":         hourly_avg,
            "heatmap":            piv_rows,
            "dominant_pollutant": dom_pol,
            "temp_aqi_scatter":   scatter,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/manual-predict")
async def manual_predict(date: str = Query(..., description="Date in YYYY-MM-DD format (PKT)")):
    """
    Predict AQI for every hour of a given date (00:00 → 23:00 PKT).
    Allowed dates: today and the next 2 days (3 days total) in PKT.

    Strategy:
      - Convert the requested date to UTC midnight.
      - For each target hour 0..23 on that date (PKT), compute how many
        hours ahead that is from the current UTC time → horizon h.
      - Use the closest available KEY_HORIZON model (or interpolate between
        the two nearest) to predict AQI.
      - Inject Open-Meteo hourly weather for the exact target UTC timestamp.
    """
    import math

    # ── Validate date ─────────────────────────────────────────────────────
    try:
        requested_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "date must be YYYY-MM-DD")

    pkt_now   = get_pkt_now()
    today_pkt = pkt_now.date()

    if requested_date < today_pkt:
        raise HTTPException(400, "Cannot predict for past dates")
    if requested_date > today_pkt + timedelta(days=2):
        raise HTTPException(400, "Predictions only available for today and the next 2 days")

    # ── Load models ───────────────────────────────────────────────────────
    try:
        cache = await run_in_threadpool(load_models)
    except Exception as e:
        raise HTTPException(500, f"Model load failed: {e}")

    models         = cache["models"]
    valid_indices  = cache["valid_indices"]
    fc_indices_map = cache["fc_indices_map"]
    main_scaler    = cache["main_scaler"]
    feature_cols   = cache["feature_cols"]

    # ── Load latest feature row for base vector ───────────────────────────
    def _load_df():
        return _fetch_feature_df()

    df, _ = await run_in_threadpool(_load_df)
    if df.empty:
        raise HTTPException(404, "No feature data available for inference")

    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    available_cols = [c for c in feature_cols if c in df.columns]
    full_vec = np.zeros(len(feature_cols), dtype=np.float32)
    latest   = df[available_cols].iloc[-1]
    for col in available_cols:
        val = latest[col]
        full_vec[feature_cols.index(col)] = float(val) if pd.notna(val) else 0.0
    X_scaled = main_scaler.transform([full_vec])[0]

    # ── Fetch Open-Meteo weather for the full requested date ──────────────
    sample_fci = next(
        (v for v in fc_indices_map.values() if isinstance(v, list) and v), []
    )
    fc_cols  = [feature_cols[i] for i in sample_fci if i < len(feature_cols)]
    api_vars = [c for c in FORECAST_WEATHER_COLS if c in fc_cols]

    weather_by_utc: dict = {}   # utc_hour_str -> scaled fc_vec per horizon
    if api_vars and sample_fci:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude":     LATITUDE,
                        "longitude":    LONGITUDE,
                        "hourly":       ",".join(api_vars),
                        "timezone":     "UTC",
                        "forecast_days": 4,
                    },
                )
                resp.raise_for_status()
            wdata  = resp.json()["hourly"]
            wtimes = pd.to_datetime(wdata["time"], utc=True).tz_localize(None)
            for i, ts in enumerate(wtimes):
                vec = np.zeros(len(feature_cols), dtype=np.float32)
                for col in api_vars:
                    if col in feature_cols:
                        val = wdata[col][i]
                        vec[feature_cols.index(col)] = float(val) if val is not None else 0.0
                vec_scaled = main_scaler.transform([vec])[0]
                weather_by_utc[ts] = vec_scaled
        except Exception as e:
            print(f"  Open-Meteo fetch failed for manual predict: {e}")

    # ── Build predictions for each hour of the requested date (PKT) ───────
    # PKT = UTC+5, so PKT hour H on date D = UTC (H-5) on same or previous day
    UTC_OFFSET = timedelta(hours=5)
    now_utc = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    results = []
    for pkt_hour in range(24):
        # Exact UTC timestamp for this PKT hour
        pkt_dt  = datetime(requested_date.year, requested_date.month,
                           requested_date.day, pkt_hour, 0, 0)
        utc_dt  = pkt_dt - UTC_OFFSET

        # Horizon = how many hours from now_utc to utc_dt
        h_float = (utc_dt - now_utc).total_seconds() / 3600.0

        # For past hours on today's date, use the t+1h model (closest available)
        # Never skip — always predict all 24 hours of the selected date
        h_effective = max(1.0, h_float)

        # Clamp to [1, 72] — model range
        h_clamped = max(1, min(72, round(h_effective)))

        # Find the two nearest KEY_HORIZONS for interpolation
        lower = max((k for k in KEY_HORIZONS if k <= h_clamped), default=KEY_HORIZONS[0])
        upper = min((k for k in KEY_HORIZONS if k >= h_clamped), default=KEY_HORIZONS[-1])

        def _predict_at(h_model: int) -> float:
            if h_model not in models:
                return 0.0
            vi  = valid_indices.get(h_model, list(range(len(feature_cols))))
            fci = fc_indices_map.get(h_model, [])
            base = X_scaled[vi]

            # Weather injection: use the exact UTC timestamp
            fc_vec = np.zeros(len(fci), dtype=np.float32)
            if fci and utc_dt in weather_by_utc:
                wvec = weather_by_utc[utc_dt]
                fc_vec = wvec[fci]

            enc = horizon_encoding(h_model)
            row = np.concatenate([base, fc_vec, enc]).reshape(1, -1)
            return max(0.0, float(models[h_model].predict(row)[0]))

        if lower == upper:
            pred = _predict_at(lower)
        else:
            pred_lo = _predict_at(lower)
            pred_hi = _predict_at(upper)
            # Linear interpolation
            t = (h_clamped - lower) / (upper - lower) if upper != lower else 0
            pred = pred_lo + t * (pred_hi - pred_lo)

        pred = round(pred, 1)
        # actual_horizon = real sequential offset from now (1, 2, 3 ... up to ~72)
        actual_horizon = max(1, round(h_effective))
        band = "short" if actual_horizon <= 8 else ("medium" if actual_horizon <= 24 else "long")

        results.append({
            "pkt_hour":      pkt_hour,
            "pkt_time":      pkt_dt.strftime("%Y-%m-%d %H:%M"),
            "utc_time":      utc_dt.isoformat() + "Z",
            "horizon_h":     actual_horizon,
            "predicted_aqi": pred,
            "band":          band,
            "category":      aqi_label(pred),
            "color":         aqi_color(pred),
        })

    if not results:
        raise HTTPException(404, "No predictions could be generated for this date")

    # ── Summary stats ─────────────────────────────────────────────────────
    aqis = [r["predicted_aqi"] for r in results]
    summary = {
        "date":     date,
        "mean_aqi": round(sum(aqis) / len(aqis), 1),
        "max_aqi":  max(aqis),
        "min_aqi":  min(aqis),
        "hours":    len(results),
        "dominant_category": max(
            set(r["category"] for r in results),
            key=lambda c: sum(1 for r in results if r["category"] == c)
        ),
    }

    return {"date": date, "summary": summary, "predictions": results}


@app.get("/api/model-info")
async def get_model_info():
    """Model architecture & per-horizon metrics."""
    try:
        cache = await run_in_threadpool(load_models)
        training_logs = cache["training_logs"]
        feature_cols  = cache["feature_cols"]
        model_name    = cache.get("model_name", "unknown")
        created_at    = cache.get("created_at")

        # Format created_at as ISO string with Z suffix for frontend
        created_at_str = None
        if created_at:
            if hasattr(created_at, "isoformat"):
                created_at_str = created_at.isoformat() + "Z"
            else:
                created_at_str = str(created_at)

        rows = []
        for h in KEY_HORIZONS:
            m    = training_logs.get(h, {})
            band = "Short" if h <= 8 else ("Medium" if h <= 24 else "Long")
            rows.append({
                "horizon": f"t+{h}h",
                "band":    band,
                "r2":      round(float(m["r2"]),   4) if isinstance(m.get("r2"),      (int, float)) else None,
                "rmse":    round(float(m["rmse"]),  2) if isinstance(m.get("rmse"),    (int, float)) else None,
                "mae":     round(float(m["mae"]),   2) if isinstance(m.get("mae"),     (int, float)) else None,
                "samples": int(m["samples"])            if isinstance(m.get("samples"), int)         else None,
            })

        algo_map = {
            "lightgbm":     "LightGBM (Gradient Boosted Trees)",
            "xgboost":      "XGBoost (Gradient Boosted Trees)",
            "random_forest":"Random Forest",
        }

        return {
            "metrics":       rows,
            "feature_count": len(feature_cols),
            "features":      feature_cols,
            "created_at":    created_at_str,
            "architecture": {
                "algorithm":          algo_map.get(model_name, model_name),
                "model_name":         model_name,
                "strategy":           "Per-horizon model (18 models for t+1h to t+72h)",
                "selection":          "Best of LightGBM / XGBoost / RandomForest by avg RMSE",
                "scaling":            "RobustScaler (fit on train only)",
                "lag_exclusion":      "Lags shorter than horizon excluded per model",
                "weather_injection":  "Open-Meteo 72h forecast injected at inference",
                "early_stopping":     "10% internal validation, 50 rounds patience",
                "pipeline_version":   "v7",
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))