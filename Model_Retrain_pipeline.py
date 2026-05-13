# -------------------------------------------------------------
# Pearls AQI Predictor — Model Retrain Pipeline
# Runs every 12 hours via GitHub Actions
# Fetch features from MongoDB → Train 3-band models → Save to MongoDB
# Overwrites previous model, keeps training logs with it
# -------------------------------------------------------------

import os
import re
import sys
import time
import base64
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pymongo import MongoClient, ReadPreference
from pymongo.read_concern import ReadConcern
from pymongo.server_api import ServerApi

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env locally; in CI secrets are injected as env vars
except ImportError:
    pass  # python-dotenv not installed — env vars already set by CI

MONGODB_URI = os.getenv("MONGODB_URI")
FEATURE_DB  = os.getenv("MONGODB_DB")                  or "AQI_Project"
FEATURE_COL = os.getenv("MONGODB_FEATURES_COLLECTION") or "karachi_aqi_features"
MODEL_DB    = os.getenv("MODEL_DB")  or "aqi_model_store"
MODEL_COL   = os.getenv("MODEL_COL") or "AQI_72h_model"

RAW_POLLUTANTS = {
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "sulphur_dioxide", "carbon_monoxide",
}

MAX_H = 72

# Key horizons — one model per horizon (no band grouping)
KEY_HORIZONS = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72]

# Horizon-specific hyperparams — longer = simpler + more regularised
def get_params(h: int) -> dict:
    if h <= 3:
        return dict(n_estimators=1500, learning_rate=0.01, num_leaves=63,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                    reg_alpha=0.0, reg_lambda=1.0, max_depth=8)
    elif h <= 12:
        return dict(n_estimators=1200, learning_rate=0.02, num_leaves=63,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=25,
                    reg_alpha=0.05, reg_lambda=1.5, max_depth=7)
    elif h <= 24:
        return dict(n_estimators=800, learning_rate=0.03, num_leaves=47,
                    subsample=0.75, colsample_bytree=0.75, min_child_samples=35,
                    reg_alpha=0.2, reg_lambda=2.0, max_depth=6)
    elif h <= 48:
        return dict(n_estimators=600, learning_rate=0.04, num_leaves=31,
                    subsample=0.7, colsample_bytree=0.65, min_child_samples=50,
                    reg_alpha=0.5, reg_lambda=3.0, max_depth=5)
    else:
        return dict(n_estimators=500, learning_rate=0.05, num_leaves=15,
                    subsample=0.6, colsample_bytree=0.6, min_child_samples=80,
                    reg_alpha=1.0, reg_lambda=5.0, max_depth=4)

# Keep BANDS for summary reporting only
BANDS = {
    "short":  list(range(1, 9)),
    "medium": [9, 12, 15, 18, 21, 24],
    "long":   [25, 30, 36, 42, 48, 54, 60, 66, 72],
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: FETCH DATA FROM MONGODB
# ═══════════════════════════════════════════════════════════════

def fetch_features(max_retries=3):
    """
    Load feature-engineered data.
    Primary:  karachi_aqi_features_engineered.csv  (local, fast)
    Fallback: MongoDB AQI_Project.karachi_aqi_features
    Run Complete_Pipeline.py first to generate the CSV.
    """
    print("[1/5] LOADING FEATURE DATA")
    print("-" * 50)

    # ── Primary: CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "karachi_aqi_features_engineered.csv"
    )
    if os.path.exists(csv_path):
        print(f"  Reading from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        print(f"  Loaded {len(df):,} records | "
              f"{df['datetime'].min()} → {df['datetime'].max()}")
        print(f"  Columns: {len(df.columns)}")
        return df

    # ── Fallback: MongoDB ──────────────────────────────────────────────────
    print(f"  CSV not found — reading from MongoDB: {FEATURE_DB}.{FEATURE_COL}")
    print(f"  (Run Complete_Pipeline.py to generate the CSV for faster loading)")

    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"),
                         serverSelectionTimeoutMS=15_000)
    client.admin.command("ping")
    print("  Connected to MongoDB")

    db  = client[FEATURE_DB]
    col = db.get_collection(
        FEATURE_COL,
        read_concern=ReadConcern("majority"),
        read_preference=ReadPreference.PRIMARY,
    )

    df = pd.DataFrame()
    for attempt in range(1, max_retries + 1):
        total_docs = col.count_documents({})

        if total_docs == 0:
            print(f"  WARNING: Collection is empty on attempt {attempt}")
            if attempt < max_retries:
                time.sleep(30 * attempt)
                continue
            else:
                client.close()
                raise RuntimeError(
                    "No data found. Run Complete_Pipeline.py first to generate "
                    "karachi_aqi_features_engineered.csv"
                )

        print(f"  Fetching {total_docs:,} documents …")
        records = []
        for i, doc in enumerate(col.find({}, {"_id": 0}).batch_size(1000), 1):
            records.append(doc)
            if i % 2000 == 0:
                print(f"    {i:,} / {total_docs:,} …")
        df = pd.DataFrame(records)

        if df.empty or "datetime" not in df.columns:
            print(f"  WARNING: No valid documents on attempt {attempt}")
            if attempt < max_retries:
                time.sleep(30 * attempt)
                continue
            else:
                client.close()
                raise RuntimeError("Collection returned no valid documents.")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        latest_dt = df["datetime"].max()
        cutoff    = datetime.utcnow() - timedelta(hours=36)
        print(f"  {len(df):,} records | {df['datetime'].min()} → {latest_dt}")

        if latest_dt >= cutoff:
            print("  Data is fresh"); break
        else:
            print(f"  WARNING: Latest record older than {cutoff}")
            if attempt < max_retries:
                print(f"  Waiting {30 * attempt}s …"); time.sleep(30 * attempt)
            else:
                print("  Proceeding with available data")

    client.close()
    print(f"  Records : {len(df):,} | Columns : {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

# Weather columns available from forecast API at any future hour
FORECAST_WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "windspeed_10m", "wind_gusts_10m", "cloud_cover",
    "shortwave_radiation", "vapour_pressure_deficit",
    "precipitation", "dew_point_2m", "pressure_msl",
]


def get_valid_indices(feature_cols: list, h: int) -> list:
    """Return indices of features available at forecast horizon h.
    Excludes any lag_Xh column where X < h (not yet available)."""
    valid = []
    for i, col in enumerate(feature_cols):
        m = re.search(r"_lag_(\d+)h$", col)
        if m and int(m.group(1)) < h:
            continue   # this lag is in the future — exclude
        valid.append(i)
    return valid


def get_forecast_indices(feature_cols: list) -> list:
    """Indices of weather columns usable as future forecast features."""
    return [i for i, c in enumerate(feature_cols)
            if c in FORECAST_WEATHER_COLS]


def horizon_encoding(h: int) -> np.ndarray:
    """4-feature horizon encoding."""
    band_flag = 0.0 if h <= 8 else (0.5 if h <= 24 else 1.0)
    return np.array([
        h / MAX_H,
        np.log1p(h) / np.log1p(MAX_H),
        np.sqrt(h / MAX_H),
        band_flag,
    ], dtype=np.float32)


def prepare_data(df: pd.DataFrame) -> dict:
    """80/20 temporal split → feature select → scale (fit on train only)."""
    print("\n[2/5] DATA PREPARATION")
    print("-" * 50)

    df = df.dropna(subset=["us_aqi"]).sort_values("datetime").reset_index(drop=True)

    # Add Karachi pollution season flags if not present
    if "is_smog_season" not in df.columns:
        df["is_smog_season"]    = df["datetime"].dt.month.isin([10,11,12,1,2]).astype(int)
    if "is_monsoon_season" not in df.columns:
        df["is_monsoon_season"] = df["datetime"].dt.month.isin([6,7,8,9]).astype(int)

    exclude = RAW_POLLUTANTS | {
        "us_aqi", "datetime", "location", "latitude", "longitude", "dominant_pollutant",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].isnull().mean() < 0.10
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df["us_aqi"].values.astype(np.float32)

    split   = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fill nulls AFTER split — train and test independently
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().bfill().values.astype(np.float32)
    X_test  = pd.DataFrame(X_test,  columns=feature_cols).ffill().bfill().values.astype(np.float32)

    scaler  = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Pre-compute forecast weather indices (used to inject future weather)
    fc_indices = get_forecast_indices(feature_cols)

    print(f"  Features: {len(feature_cols)} (forecast weather cols: {len(fc_indices)})")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  AQI range: [{y.min():.0f}, {y.max():.0f}]")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler": scaler, "feature_cols": feature_cols,
        "fc_indices": fc_indices,   # indices of weather cols for future injection
    }


# ═══════════════════════════════════════════════════════════════
# STEP 3: TRAIN PER-HORIZON MODELS
# ═══════════════════════════════════════════════════════════════

def build_sample(x_now: np.ndarray, x_future: np.ndarray,
                 valid_indices: list, fc_indices: list, h: int) -> np.ndarray:
    """
    Build one training sample:
      - x_now[valid_indices]: current features with future lags excluded
      - x_future[fc_indices]: weather at target hour t+h (future forecast)
      - horizon_encoding(h): horizon metadata
    """
    base    = x_now[valid_indices]
    future  = x_future[fc_indices]   # actual weather at t+h from historical data
    return np.concatenate([base, future, horizon_encoding(h)])


def train_models(data: dict) -> tuple:
    """One LightGBM per key horizon with early stopping + future weather injection."""
    print("\n[3/5] MODEL TRAINING")
    print("-" * 50)

    X_train      = data["X_train"]
    y_train      = data["y_train"]
    X_test       = data["X_test"]
    y_test       = data["y_test"]
    feature_cols = data["feature_cols"]
    fc_indices   = data["fc_indices"]

    models          = {}
    horizon_metrics = {}

    print(f"\n  {'Horizon':>8} {'Feats':>6} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for h in KEY_HORIZONS:
        vi = get_valid_indices(feature_cols, h)

        # Build training samples — inject actual weather at t+h
        rows, targets = [], []
        for t in range(24, len(X_train) - h):
            rows.append(build_sample(X_train[t], X_train[t + h],
                                     vi, fc_indices, h))
            targets.append(y_train[t + h])

        rows    = np.array(rows,    dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        # 10% internal val for early stopping
        val_split   = int(len(rows) * 0.9)
        X_tr, X_val = rows[:val_split], rows[val_split:]
        y_tr, y_val = targets[:val_split], targets[val_split:]

        model = lgb.LGBMRegressor(**get_params(h), random_state=42,
                                   n_jobs=-1, verbose=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        models[h] = model

        # Test evaluation — inject actual weather at t+h from test set
        test_rows, test_targets = [], []
        for t in range(24, len(X_test) - h):
            test_rows.append(build_sample(X_test[t], X_test[t + h],
                                          vi, fc_indices, h))
            test_targets.append(y_test[t + h])

        test_rows    = np.array(test_rows,    dtype=np.float32)
        test_targets = np.array(test_targets, dtype=np.float32)
        preds        = model.predict(test_rows)

        rmse = float(np.sqrt(mean_squared_error(test_targets, preds)))
        mae  = float(mean_absolute_error(test_targets, preds))
        r2   = float(r2_score(test_targets, preds))

        horizon_metrics[h] = {"rmse": rmse, "mae": mae, "r2": r2,
                               "samples": len(rows),
                               "valid_indices": vi,
                               "fc_indices": fc_indices}
        print(f"  t+{h:<5d} {len(vi)+len(fc_indices):>6} "
              f"{rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}")

    return models, horizon_metrics


# ═══════════════════════════════════════════════════════════════
# STEP 4: SAVE MODELS TO MONGODB
# ═══════════════════════════════════════════════════════════════

def save_models(models: dict, horizon_metrics: dict, data: dict) -> None:
    print("\n[4/5] SAVING MODELS TO MONGODB")
    print("-" * 50)

    client           = MongoClient(MONGODB_URI, server_api=ServerApi("1"),
                                   serverSelectionTimeoutMS=15_000)
    model_collection = client[MODEL_DB][MODEL_COL]
    now              = datetime.utcnow()

    for h, model in models.items():
        hm   = horizon_metrics[h]
        blob = pickle.dumps({
            "model":         model,
            "horizon":       h,
            "valid_indices": hm["valid_indices"],
            "fc_indices":    hm["fc_indices"],
        })
        model_collection.update_one(
            {"horizon": h},
            {"$set": {
                "horizon":      h,
                "model_blob":   base64.b64encode(blob).decode(),
                "feature_cols": data["feature_cols"],
                "metrics":      {k: v for k, v in hm.items() if k != "valid_indices"},
                "created_at":   now,
                "pipeline_version": "v5",
            }},
            upsert=True,
        )

    scaler_blob = pickle.dumps(data["scaler"])
    model_collection.update_one(
        {"horizon": "_scaler"},
        {"$set": {
            "horizon":      "_scaler",
            "model_blob":   base64.b64encode(scaler_blob).decode(),
            "feature_cols": data["feature_cols"],
            "created_at":   now,
        }},
        upsert=True,
    )
    client.close()
    print(f"  Saved {len(models)} horizon models + scaler → {MODEL_DB}.{MODEL_COL}")


# ═══════════════════════════════════════════════════════════════
# STEP 5: FORECAST & EVALUATE
# ═══════════════════════════════════════════════════════════════

def fetch_live_forecast(feature_cols: list, fc_indices: list,
                        scaler) -> dict:
    """
    Fetch 72h weather forecast from Open-Meteo and return a dict:
    {h: scaled_fc_vector} for each horizon h in KEY_HORIZONS.
    Falls back to zeros if fetch fails.
    """
    import requests
    fc_cols = [feature_cols[i] for i in fc_indices]
    api_vars = [c for c in FORECAST_WEATHER_COLS if c in fc_cols]

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 24.8607, "longitude": 67.0011,
                "hourly": ",".join(api_vars),
                "timezone": "UTC", "forecast_days": 4,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()["hourly"]
        times = pd.to_datetime(data["time"], utc=True).tz_localize(None)

        result = {}
        now = pd.Timestamp.utcnow().replace(minute=0, second=0, microsecond=0,
                                            tzinfo=None)
        for h in KEY_HORIZONS:
            target = now + pd.Timedelta(hours=h)
            if target in times.values:
                idx = list(times.values).index(target)
                # Build a zero vector for all feature_cols, fill forecast cols
                vec = np.zeros(len(feature_cols), dtype=np.float32)
                for col in api_vars:
                    if col in feature_cols:
                        fi = feature_cols.index(col)
                        vec[fi] = data[col][idx] or 0.0
                # Scale using the fitted scaler (transform single row)
                vec_scaled = scaler.transform([vec])[0]
                result[h] = vec_scaled[fc_indices]
            else:
                result[h] = np.zeros(len(fc_indices), dtype=np.float32)
        print(f"  Live weather forecast fetched for {len(result)} horizons")
        return result
    except Exception as e:
        print(f"  WARNING: Weather forecast fetch failed ({e}) — using zeros")
        return {h: np.zeros(len(fc_indices), dtype=np.float32)
                for h in KEY_HORIZONS}


def run_predictions(models: dict, horizon_metrics: dict, data: dict) -> tuple:
    print("\n[5/5] 72h FORECAST & EVALUATION")
    print("-" * 50)

    X_test       = data["X_test"]
    y_test       = data["y_test"]
    feature_cols = data["feature_cols"]
    fc_indices   = data["fc_indices"]
    scaler       = data["scaler"]

    # Fetch live weather forecast for inference
    live_fc = fetch_live_forecast(feature_cols, fc_indices, scaler)

    # 72h forecast from last test point using live weather
    last_x   = X_test[-1]
    forecast = {}
    for h in KEY_HORIZONS:
        vi  = horizon_metrics[h]["valid_indices"]
        fci = horizon_metrics[h]["fc_indices"]
        row = np.concatenate([last_x[vi], live_fc[h], horizon_encoding(h)]).reshape(1, -1)
        forecast[h] = round(float(models[h].predict(row)[0]), 1)

    print("\n  72h Forecast (from last test point):")
    print(f"  {'Hour':>6} {'Predicted AQI':>14}")
    print(f"  {'-'*6} {'-'*14}")
    for h in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
        if h in forecast:
            print(f"  t+{h:<4d} {forecast[h]:>14.1f}")

    # Per-horizon evaluation using actual future weather from test set
    print("\n  Per-horizon accuracy (full test set):")
    print(f"  {'Horizon':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Samples':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    eval_metrics = {}
    for h in KEY_HORIZONS:
        vi  = horizon_metrics[h]["valid_indices"]
        fci = horizon_metrics[h]["fc_indices"]
        rows, targets = [], []
        for t in range(24, len(X_test) - h):
            rows.append(build_sample(X_test[t], X_test[t + h],
                                     vi, fci, h))
            targets.append(y_test[t + h])

        if len(rows) < 2:
            continue

        rows    = np.array(rows,    dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        preds   = models[h].predict(rows)

        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        mae  = float(mean_absolute_error(targets, preds))
        r2   = float(r2_score(targets, preds))
        eval_metrics[h] = {"rmse": rmse, "mae": mae, "r2": r2, "n": len(rows)}

    for h in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
        if h in eval_metrics:
            m = eval_metrics[h]
            print(f"  t+{h:<5d} {m['rmse']:>8.2f} {m['mae']:>8.2f} "
                  f"{m['r2']:>8.4f} {m['n']:>8d}")

    # Band summary
    print("\n  Band-level summary:")
    for band_name, horizons in BANDS.items():
        bh = [h for h in horizons if h in eval_metrics]
        if not bh: continue
        print(f"    {band_name:>8}: avg RMSE={np.mean([eval_metrics[h]['rmse'] for h in bh]):.2f}  "
              f"avg MAE={np.mean([eval_metrics[h]['mae'] for h in bh]):.2f}  "
              f"avg R²={np.mean([eval_metrics[h]['r2'] for h in bh]):.4f}")

    return forecast, eval_metrics


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_retrain():
    """Full retrain pipeline: Fetch → Prepare → Train → Save → Predict."""
    print("=" * 70)
    print(" Pearls AQI Predictor — Model Retrain Pipeline")
    print(f" Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 70)

    df                        = fetch_features()
    data                      = prepare_data(df)
    models, horizon_metrics   = train_models(data)
    save_models(models, horizon_metrics, data)
    forecast, eval_metrics    = run_predictions(models, horizon_metrics, data)

    print("\n" + "=" * 70)
    print(" RETRAIN COMPLETE")
    print("=" * 70)
    for band_name, horizons in BANDS.items():
        bh = [h for h in horizons if h in eval_metrics]
        if bh:
            avg_r2 = np.mean([eval_metrics[h]["r2"] for h in bh])
            print(f"  {band_name:>8}: avg R²={avg_r2:.4f}")
    fc_vals = list(forecast.values())
    print(f"\n  72h Forecast range: [{min(fc_vals):.1f}, {max(fc_vals):.1f}]")
    # print(f"  Best  R²: t+1h  = {eval_metrics.get(1,  {}).get('r2', 'N/A')}")
    # print(f"  Worst R²: t+72h = {eval_metrics.get(72, {}).get('r2', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    run_retrain()
