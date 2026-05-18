"""
AQI SeekAI — FastAPI Backend
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
from fastapi.responses import FileResponse
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

def load_models() -> dict:
    global _model_cache
    if _model_cache:
        return _model_cache

    client = get_mongo_client()
    col = client[MODEL_DB][MODEL_COL]
    docs = list(col.find())
    if not docs:
        raise ValueError(f"No documents found in {MODEL_DB}.{MODEL_COL}")

    models, valid_indices, fc_indices_map = {}, {}, {}
    feature_cols, main_scaler, training_logs = [], None, {}

    is_new_format = any("horizon" in doc for doc in docs)

    for doc in docs:
        blob = base64.b64decode(doc["model_blob"])
        if is_new_format:
            key = doc.get("horizon")
            if key is None:
                continue
            if key == "_scaler":
                main_scaler  = pickle.loads(blob)
                feature_cols = doc.get("feature_cols", [])
            else:
                payload = pickle.loads(blob)
                models[key]         = payload["model"]
                valid_indices[key]  = payload.get("valid_indices", [])
                fc_indices_map[key] = payload.get("fc_indices", [])
                training_logs[key]  = doc.get("metrics", {})
        else:
            key = doc.get("band")
            if key is None:
                continue
            if key == "_scaler":
                main_scaler  = pickle.loads(blob)
                feature_cols = doc.get("feature_cols", [])
            else:
                obj = pickle.loads(blob)
                band_model  = obj.get("model") if isinstance(obj, dict) else obj
                band_scaler = obj.get("scaler") if isinstance(obj, dict) else None
                band_ranges = {"short": range(1,9), "medium": range(9,25), "long": range(25,73)}
                for h in band_ranges.get(key, []):
                    if h in KEY_HORIZONS:
                        models[h]         = band_model
                        valid_indices[h]  = []
                        fc_indices_map[h] = []
                        training_logs[h]  = doc.get("training_log", {}).get("metrics", {})
                if band_scaler is not None:
                    for h in band_ranges.get(key, []):
                        if h in KEY_HORIZONS:
                            fc_indices_map[h] = band_scaler

    if main_scaler is None or not models:
        raise ValueError("Models or scaler missing in MongoDB.")

    _model_cache = dict(
        models=models, valid_indices=valid_indices, fc_indices_map=fc_indices_map,
        main_scaler=main_scaler, feature_cols=feature_cols, training_logs=training_logs,
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
    if df.empty:
        return []
    models, valid_indices, fc_indices_map, main_scaler, feature_cols = (
        cache["models"], cache["valid_indices"], cache["fc_indices_map"],
        cache["main_scaler"], cache["feature_cols"],
    )
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        return []

    full_vec = np.zeros((1, len(feature_cols)), dtype=np.float32)
    latest_row = df[available_cols].iloc[-1]
    for col in available_cols:
        if col in feature_cols:
            val = latest_row[col]
            full_vec[0, feature_cols.index(col)] = float(val) if pd.notna(val) else 0.0
    X_scaled = main_scaler.transform(full_vec)[0]

    has_fc  = any(isinstance(v, list) and len(v) > 0 for v in fc_indices_map.values())
    live_fc = await fetch_weather_forecast(feature_cols, fc_indices_map, main_scaler) if has_fc else {}

    last_dt  = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    forecasts = []

    for h in KEY_HORIZONS:
        if h not in models:
            continue
        vi  = valid_indices.get(h, [])
        fci = fc_indices_map.get(h, [])
        try:
            if isinstance(fci, list):
                base   = X_scaled[vi] if vi else X_scaled
                fc_vec = live_fc.get(h, np.zeros(len(fci), dtype=np.float32))
                row    = np.concatenate([base, fc_vec, horizon_encoding(h)]) if len(fc_vec) > 0 \
                         else np.concatenate([base, horizon_encoding(h)])
            else:
                band_scaler = fci
                aqi_history = df["us_aqi"].dropna().values
                if len(aqi_history) == 0:
                    continue
                ar = np.array([
                    aqi_history[-1],
                    aqi_history[-6]  if len(aqi_history) >= 6  else aqi_history[-1],
                    aqi_history[-12] if len(aqi_history) >= 12 else aqi_history[-1],
                    np.mean(aqi_history[-24:]),
                    np.std(aqi_history[-24:]) if len(aqi_history) > 1 else 0.0,
                ])
                row  = band_scaler.transform([np.concatenate([X_scaled, ar, [h / MAX_H]])])[0].reshape(1, -1)
                pred = max(0, float(models[h].predict(row)[0]))
                band = "short" if h <= 8 else ("medium" if h <= 24 else "long")
                forecasts.append({
                    "hour": h,
                    "datetime": (last_dt + timedelta(hours=h)).isoformat(),
                    "predicted_aqi": round(pred, 1),
                    "band": band,
                    "category": aqi_label(pred),
                    "color": aqi_color(pred),
                })
                continue

            pred = max(0, float(models[h].predict([row])[0]))
            band = "short" if h <= 8 else ("medium" if h <= 24 else "long")
            forecasts.append({
                "hour": h,
                "datetime": (last_dt + timedelta(hours=h)).isoformat(),
                "predicted_aqi": round(pred, 1),
                "band": band,
                "category": aqi_label(pred),
                "color": aqi_color(pred),
            })
        except Exception:
            continue

    return forecasts

# ── FastAPI app ───────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models and warm up MongoDB connection at startup."""
    loop = asyncio.get_event_loop()
    try:
        print("Preloading MongoDB connection and ML models...")
        await loop.run_in_executor(None, get_mongo_client)
        await loop.run_in_executor(None, load_models)
        print("Models and DB ready.")
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
    """Return (row, stale). Falls back to the newest row if nothing within max_age_hours."""
    client = get_mongo_client()
    col = client[FEATURE_DB][FEATURE_COL]
    if max_age_hours is not None:
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        docs = list(
            col.find({"datetime": {"$gte": cutoff}}, {"_id": 0})
            .sort("datetime", -1)
            .limit(1)
        )
        if docs:
            return docs[0], False
    docs = list(col.find({}, {"_id": 0}).sort("datetime", -1).limit(1))
    if not docs:
        return None, False
    return docs[0], max_age_hours is not None


@app.get("/")
async def root():
    index = _ROOT / "index.html"
    if not index.is_file():
        raise HTTPException(404, "index.html not found")
    return FileResponse(index)


@app.get("/health")
async def health():
    return {"status": "ok", "time_pkt": get_pkt_now().isoformat()}


@app.get("/api/current")
async def get_current():
    """Latest AQI reading + weather snapshot."""
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
        weather = {k: row.get(k) for k in weather_keys}
        sub_aqi = {
            "pm2_5":           row.get("us_aqi_pm2_5"),
            "pm10":            row.get("us_aqi_pm10"),
            "nitrogen_dioxide":row.get("us_aqi_nitrogen_dioxide"),
            "ozone":           row.get("us_aqi_ozone"),
            "sulphur_dioxide": row.get("us_aqi_sulphur_dioxide"),
            "carbon_monoxide": row.get("us_aqi_carbon_monoxide"),
        }
        pollutants = {
            "pm2_5":            row.get("pm2_5"),
            "pm10":             row.get("pm10"),
            "ozone":            row.get("ozone"),
            "nitrogen_dioxide": row.get("nitrogen_dioxide"),
            "sulphur_dioxide":  row.get("sulphur_dioxide"),
            "carbon_monoxide":  row.get("carbon_monoxide"),
        }
        return {
            "aqi":             aqi,
            "category":        aqi_label(aqi),
            "color":           aqi_color(aqi),
            "datetime":        str(row.get("datetime", "")),
            "pkt_now":         get_pkt_now().isoformat(),
            "stale":           stale,
            "weather":         weather,
            "sub_aqi":         sub_aqi,
            "pollutants":      pollutants,
            "dominant_pollutant": row.get("dominant_pollutant"),
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
    """72-hour AQI forecast from loaded ML models."""
    try:
        def _load():
            cache  = load_models()
            client = get_mongo_client()
            col    = client[FEATURE_DB][FEATURE_COL]
            cutoff = datetime.utcnow() - timedelta(days=5)
            cursor = col.find({"datetime": {"$gte": cutoff}}, {"_id": 0}).sort("datetime", 1)
            return cache, pd.DataFrame(list(cursor))
        cache, df = await run_in_threadpool(_load)
        if df.empty:
            raise HTTPException(404, "No feature data for inference")

        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        forecasts = await generate_forecast(cache, df)
        return {"forecasts": forecasts}
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


@app.get("/api/model-info")
async def get_model_info():
    """Model architecture & per-horizon metrics."""
    try:
        cache = await run_in_threadpool(load_models)
        training_logs = cache["training_logs"]
        feature_cols  = cache["feature_cols"]

        rows = []
        for h in KEY_HORIZONS:
            m    = training_logs.get(h, {})
            band = "Short" if h <= 8 else ("Medium" if h <= 24 else "Long")
            rows.append({
                "horizon":  f"t+{h}h",
                "band":     band,
                "r2":       round(float(m["r2"]),   4) if isinstance(m.get("r2"),   (int, float)) else None,
                "rmse":     round(float(m["rmse"]),  2) if isinstance(m.get("rmse"), (int, float)) else None,
                "mae":      round(float(m["mae"]),   2) if isinstance(m.get("mae"),  (int, float)) else None,
                "samples":  int(m["samples"])           if isinstance(m.get("samples"), int)       else None,
            })

        return {
            "metrics":       rows,
            "feature_count": len(feature_cols),
            "features":      feature_cols,
            "architecture": {
                "algorithm":    "LightGBM (Gradient Boosted Trees)",
                "strategy":     "Per-horizon model (18 models for t+1h to t+72h)",
                "scaling":      "RobustScaler (fit on train only)",
                "lag_exclusion":"Lags shorter than horizon excluded per model",
                "weather_injection": "Open-Meteo 72h forecast injected at inference",
                "early_stopping": "10% internal validation, 50 rounds patience",
                "short_params": "1500 trees, lr=0.01, leaves=63",
                "medium_params":"800 trees, lr=0.03, leaves=47",
                "long_params":  "500 trees, lr=0.05, leaves=15",
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))