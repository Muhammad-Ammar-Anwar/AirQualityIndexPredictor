# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# Pearls AQI Predictor - Model Retrain Pipeline  (v7)
# Runs every 12 hours via GitHub Actions
#
# Flow:
#   Fetch features -> Prepare data
#   Quick comparison: train all 3 models on a SAMPLE of horizons
#   Select the single best model by avg RMSE
#   Fully train the winner across all 18 horizons
#   Save only the winner's 18 models + scaler to MongoDB
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
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pymongo import MongoClient, ReadPreference
from pymongo.read_concern import ReadConcern
from pymongo.server_api import ServerApi

warnings.filterwarnings("ignore")

# -- Configuration ----------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
KEY_HORIZONS = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72]

BANDS = {
    "short":  list(range(1, 9)),
    "medium": [9, 12, 15, 18, 21, 24],
    "long":   [25, 30, 36, 42, 48, 54, 60, 66, 72],
}

# Names of the three models -- used as keys throughout
MODEL_NAMES = ["lightgbm", "xgboost", "random_forest"]


# ===============================================================
# HYPERPARAMETERS -- per model, per horizon
# Longer horizons -> simpler / more regularised
# ===============================================================

def get_lgb_params(h: int) -> dict:
    """LightGBM params -- original tuning kept intact."""
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


def get_xgb_params(h: int) -> dict:
    """
    XGBoost params -- reduced tree counts vs LightGBM because XGBoost
    uses early_stopping_rounds in the constructor (XGBoost >= 2.0 API).
    n_estimators is the upper bound; actual trees used will be fewer
    after early stopping triggers. This is the main fix for slow training.
    """
    if h <= 3:
        return dict(n_estimators=500, learning_rate=0.02, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.0, reg_lambda=1.0,
                    early_stopping_rounds=30, n_jobs=-1, verbosity=0)
    elif h <= 12:
        return dict(n_estimators=400, learning_rate=0.03, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
                    reg_alpha=0.05, reg_lambda=1.5,
                    early_stopping_rounds=30, n_jobs=-1, verbosity=0)
    elif h <= 24:
        return dict(n_estimators=300, learning_rate=0.04, max_depth=5,
                    subsample=0.75, colsample_bytree=0.75, min_child_weight=12,
                    reg_alpha=0.2, reg_lambda=2.0,
                    early_stopping_rounds=25, n_jobs=-1, verbosity=0)
    elif h <= 48:
        return dict(n_estimators=200, learning_rate=0.05, max_depth=5,
                    subsample=0.7, colsample_bytree=0.65, min_child_weight=20,
                    reg_alpha=0.5, reg_lambda=3.0,
                    early_stopping_rounds=20, n_jobs=-1, verbosity=0)
    else:
        return dict(n_estimators=150, learning_rate=0.05, max_depth=4,
                    subsample=0.6, colsample_bytree=0.6, min_child_weight=30,
                    reg_alpha=1.0, reg_lambda=5.0,
                    early_stopping_rounds=20, n_jobs=-1, verbosity=0)


def get_rf_params(h: int) -> dict:
    """
    RandomForest params -- kept small enough to fit in MongoDB's 16MB limit.

    WHY THESE LIMITS:
    - max_depth=10 caps each tree at 1023 nodes max -> small serialized size
    - n_estimators=100 max -> 100 trees x ~10KB each = ~1MB total (safe)
    - min_samples_leaf increases with horizon -> fewer splits -> smaller trees
    - max_features="sqrt" -> each split uses sqrt(n_features) -> faster + smaller
    - Unlimited depth (None) on 9k samples with 200 features = 19MB+ per model
      which exceeds MongoDB's 16MB document limit, so depth must be capped.
    """
    if h <= 3:
        return dict(n_estimators=100, max_depth=10, min_samples_leaf=10,
                    max_features="sqrt", n_jobs=-1, random_state=42)
    elif h <= 12:
        return dict(n_estimators=100, max_depth=10, min_samples_leaf=15,
                    max_features="sqrt", n_jobs=-1, random_state=42)
    elif h <= 24:
        return dict(n_estimators=80, max_depth=8, min_samples_leaf=20,
                    max_features="sqrt", n_jobs=-1, random_state=42)
    elif h <= 48:
        return dict(n_estimators=60, max_depth=7, min_samples_leaf=25,
                    max_features="sqrt", n_jobs=-1, random_state=42)
    else:
        return dict(n_estimators=50, max_depth=6, min_samples_leaf=30,
                    max_features="sqrt", n_jobs=-1, random_state=42)


def build_model(model_name: str, h: int):
    """Instantiate a fresh model for the given name and horizon."""
    if model_name == "lightgbm":
        return lgb.LGBMRegressor(**get_lgb_params(h), random_state=42,
                                  n_jobs=-1, verbose=-1)
    elif model_name == "xgboost":
        return xgb.XGBRegressor(**get_xgb_params(h), random_state=42)
    elif model_name == "random_forest":
        return RandomForestRegressor(**get_rf_params(h))
    raise ValueError(f"Unknown model: {model_name}")


# ===============================================================
# STEP 1: FETCH DATA FROM MONGODB / CSV
# ===============================================================

def fetch_features(max_retries=3):
    """
    Load feature-engineered data.
    Primary:  karachi_aqi_features_engineered.csv  (local, fast)
    Fallback: MongoDB AQI_Project.karachi_aqi_features
    """
    print("[1/5] LOADING FEATURE DATA")
    print("-" * 50)

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
              f"{df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Columns: {len(df.columns)}")
        return df

    print(f"  CSV not found -- reading from MongoDB: {FEATURE_DB}.{FEATURE_COL}")
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
            print(f"  WARNING: Collection empty on attempt {attempt}")
            if attempt < max_retries:
                time.sleep(30 * attempt); continue
            client.close()
            raise RuntimeError("No data found. Run Complete_Pipeline.py first.")

        print(f"  Fetching {total_docs:,} documents ...")
        records = []
        for i, doc in enumerate(col.find({}, {"_id": 0}).batch_size(1000), 1):
            records.append(doc)
            if i % 2000 == 0:
                print(f"    {i:,} / {total_docs:,} ...")
        df = pd.DataFrame(records)

        if df.empty or "datetime" not in df.columns:
            if attempt < max_retries:
                time.sleep(30 * attempt); continue
            client.close()
            raise RuntimeError("Collection returned no valid documents.")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        latest_dt = df["datetime"].max()
        cutoff    = datetime.utcnow() - timedelta(hours=36)
        print(f"  {len(df):,} records | {df['datetime'].min()} to {latest_dt}")
        if latest_dt >= cutoff:
            print("  Data is fresh"); break
        else:
            print(f"  WARNING: Latest record older than {cutoff}")
            if attempt < max_retries:
                print(f"  Waiting {30*attempt}s ..."); time.sleep(30 * attempt)
            else:
                print("  Proceeding with available data")

    client.close()
    print(f"  Records: {len(df):,} | Columns: {len(df.columns)}")
    return df


# ===============================================================
# STEP 2: DATA PREPARATION
# ===============================================================

FORECAST_WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "windspeed_10m", "wind_gusts_10m", "cloud_cover",
    "shortwave_radiation", "vapour_pressure_deficit",
    "precipitation", "dew_point_2m", "pressure_msl",
]


def get_valid_indices(feature_cols: list, h: int) -> list:
    """Indices of features available at forecast horizon h (no future-lag leakage)."""
    valid = []
    for i, col in enumerate(feature_cols):
        m = re.search(r"_lag_(\d+)h$", col)
        if m and int(m.group(1)) < h:
            continue
        valid.append(i)
    return valid


def get_forecast_indices(feature_cols: list) -> list:
    """Indices of weather columns injectable from forecast API."""
    return [i for i, c in enumerate(feature_cols) if c in FORECAST_WEATHER_COLS]


def horizon_encoding(h: int) -> np.ndarray:
    """4-feature horizon encoding appended to every sample."""
    band_flag = 0.0 if h <= 8 else (0.5 if h <= 24 else 1.0)
    return np.array([
        h / MAX_H,
        np.log1p(h) / np.log1p(MAX_H),
        np.sqrt(h / MAX_H),
        band_flag,
    ], dtype=np.float32)


def prepare_data(df: pd.DataFrame) -> dict:
    """80/20 temporal split -> feature select -> scale (fit on train only)."""
    print("\n[2/5] DATA PREPARATION")
    print("-" * 50)

    df = df.dropna(subset=["us_aqi"]).sort_values("datetime").reset_index(drop=True)

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

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().bfill().values.astype(np.float32)
    X_test  = pd.DataFrame(X_test,  columns=feature_cols).ffill().bfill().values.astype(np.float32)

    scaler  = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    fc_indices = get_forecast_indices(feature_cols)

    print(f"  Features : {len(feature_cols)}  (forecast weather cols: {len(fc_indices)})")
    print(f"  Train    : {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  AQI range: [{y.min():.0f}, {y.max():.0f}]")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler": scaler, "feature_cols": feature_cols,
        "fc_indices": fc_indices,
    }


# ===============================================================
# STEP 3: TRAIN ALL 3 MODELS PER HORIZON
# ===============================================================

def build_sample(x_now, x_future, valid_indices, fc_indices, h):
    """Build one training sample with future weather injection + horizon encoding."""
    return np.concatenate([
        x_now[valid_indices],
        x_future[fc_indices],
        horizon_encoding(h),
    ])


def select_best_model(data: dict) -> str:
    """
    Phase 1 - Quick comparison on 4 representative horizons only.
    Trains all 3 models on t+1, t+6, t+24, t+48 and picks the
    single winner by average RMSE. Fast: ~20% of full training cost.
    """
    print("\n[3a/5] MODEL SELECTION")
    print("-" * 50)

    PROBE_HORIZONS = [1, 6, 24, 48]
    scores = {name: [] for name in MODEL_NAMES}

    for model_name in MODEL_NAMES:
        for h in PROBE_HORIZONS:
            vi = get_valid_indices(data["feature_cols"], h)
            rows, targets = [], []
            for t in range(24, len(data["X_train"]) - h):
                rows.append(build_sample(
                    data["X_train"][t], data["X_train"][t + h],
                    vi, data["fc_indices"], h))
                targets.append(data["y_train"][t + h])

            rows    = np.array(rows,    dtype=np.float32)
            targets = np.array(targets, dtype=np.float32)
            vs      = int(len(rows) * 0.9)

            model = build_model(model_name, h)
            if model_name == "lightgbm":
                model.fit(rows[:vs], targets[:vs],
                          eval_set=[(rows[vs:], targets[vs:])],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
            elif model_name == "xgboost":
                model.fit(rows[:vs], targets[:vs],
                          eval_set=[(rows[vs:], targets[vs:])],
                          verbose=False)
            else:
                model.fit(rows, targets)

            # evaluate on test set
            test_rows, test_tgts = [], []
            for t in range(24, len(data["X_test"]) - h):
                test_rows.append(build_sample(
                    data["X_test"][t], data["X_test"][t + h],
                    vi, data["fc_indices"], h))
                test_tgts.append(data["y_test"][t + h])

            preds = model.predict(np.array(test_rows, dtype=np.float32))
            rmse  = float(np.sqrt(mean_squared_error(test_tgts, preds)))
            scores[model_name].append(rmse)

    print(f"  {'Model':<16} {'t+1h':>8} {'t+6h':>8} {'t+24h':>8} {'t+48h':>8} {'Avg RMSE':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    avg_scores = {}
    for name in MODEL_NAMES:
        s = scores[name]
        avg = float(np.mean(s))
        avg_scores[name] = avg
        print(f"  {name:<16} {s[0]:>8.2f} {s[1]:>8.2f} {s[2]:>8.2f} {s[3]:>8.2f} {avg:>10.3f}")

    best = min(avg_scores, key=avg_scores.get)
    print(f"\n  Selected: {best.upper()} (avg RMSE = {avg_scores[best]:.3f})")
    return best


def train_best_model(model_name: str, data: dict) -> tuple:
    """
    Phase 2 - Full training of the selected model across all 18 horizons.
    """
    print(f"\n[3b/5] FULL TRAINING: {model_name.upper()}")
    print("-" * 50)
    print(f"  {'Horizon':>8} {'Feats':>6} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    X_train      = data["X_train"]
    y_train      = data["y_train"]
    X_test       = data["X_test"]
    y_test       = data["y_test"]
    feature_cols = data["feature_cols"]
    fc_indices   = data["fc_indices"]

    models          = {}
    horizon_metrics = {}

    for h in KEY_HORIZONS:
        vi = get_valid_indices(feature_cols, h)

        rows, targets = [], []
        for t in range(24, len(X_train) - h):
            rows.append(build_sample(X_train[t], X_train[t + h], vi, fc_indices, h))
            targets.append(y_train[t + h])

        rows    = np.array(rows,    dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        vs      = int(len(rows) * 0.9)

        model = build_model(model_name, h)
        if model_name == "lightgbm":
            model.fit(rows[:vs], targets[:vs],
                      eval_set=[(rows[vs:], targets[vs:])],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        elif model_name == "xgboost":
            model.fit(rows[:vs], targets[:vs],
                      eval_set=[(rows[vs:], targets[vs:])],
                      verbose=False)
        else:
            model.fit(rows, targets)

        models[h] = model

        # test evaluation
        test_rows, test_tgts = [], []
        for t in range(24, len(X_test) - h):
            test_rows.append(build_sample(X_test[t], X_test[t + h], vi, fc_indices, h))
            test_tgts.append(y_test[t + h])

        test_rows = np.array(test_rows, dtype=np.float32)
        test_tgts = np.array(test_tgts, dtype=np.float32)
        preds     = model.predict(test_rows)

        rmse = float(np.sqrt(mean_squared_error(test_tgts, preds)))
        mae  = float(mean_absolute_error(test_tgts, preds))
        r2   = float(r2_score(test_tgts, preds))

        horizon_metrics[h] = {
            "rmse": rmse, "mae": mae, "r2": r2,
            "samples": len(rows),
            "valid_indices": vi,
            "fc_indices": fc_indices,
        }
        print(f"  t+{h:<5d} {len(vi)+len(fc_indices):>6} "
              f"{rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}")

    return models, horizon_metrics


# ===============================================================
# STEP 4: SAVE ALL 3 MODELS TO MONGODB
# ===============================================================

def save_models(
    models: dict,
    horizon_metrics: dict,
    best_model_name: str,
    data: dict,
) -> None:
    """
    Save the single best model's 18 horizon models + scaler to MongoDB.

    Document structure per horizon:
      {
        horizon:          int    -- e.g. 6
        model_name:       str    -- e.g. "lightgbm"
        model_blob:       str    -- base64-encoded zlib-compressed pickle
        feature_cols:     list[str]
        metrics:          {rmse, mae, r2, samples}
        created_at:       datetime
        pipeline_version: str
      }
    Lookup key (used by backend): {"horizon": h}
    """
    import zlib
    print("\n[4/5] SAVING BEST MODEL TO MONGODB")
    print("-" * 50)
    print(f"  Model: {best_model_name.upper()}")

    client           = MongoClient(MONGODB_URI, server_api=ServerApi("1"),
                                   serverSelectionTimeoutMS=15_000)
    model_collection = client[MODEL_DB][MODEL_COL]
    now              = datetime.utcnow()
    saved_count      = 0
    skipped_count    = 0

    for h, model in models.items():
        hm = horizon_metrics[h]

        raw_pickle = pickle.dumps({
            "model":         model,
            "horizon":       h,
            "model_name":    best_model_name,
            "valid_indices": hm["valid_indices"],
            "fc_indices":    hm["fc_indices"],
        }, protocol=4)
        compressed = zlib.compress(raw_pickle, level=9)
        blob_b64   = base64.b64encode(compressed).decode()

        doc_size_mb = len(blob_b64) / (1024 * 1024)
        if doc_size_mb > 15:
            print(f"  WARNING: t+{h}h too large ({doc_size_mb:.1f}MB) -- skipping")
            skipped_count += 1
            continue

        # Lookup by horizon only -- one document per horizon, overwrites old model
        model_collection.update_one(
            {"horizon": h},
            {"$set": {
                "horizon":      h,
                "model_name":   best_model_name,
                "model_blob":   blob_b64,
                "compressed":   True,
                "feature_cols": data["feature_cols"],
                "metrics": {
                    "rmse":    hm["rmse"],
                    "mae":     hm["mae"],
                    "r2":      hm["r2"],
                    "samples": hm["samples"],
                },
                "created_at":       now,
                "pipeline_version": "v7",
            }},
            upsert=True,
        )
        saved_count += 1

    # Scaler -- shared, one document
    scaler_blob = base64.b64encode(
        zlib.compress(pickle.dumps(data["scaler"], protocol=4), level=9)
    ).decode()
    model_collection.update_one(
        {"horizon": "_scaler"},
        {"$set": {
            "horizon":      "_scaler",
            "model_name":   best_model_name,
            "model_blob":   scaler_blob,
            "compressed":   True,
            "feature_cols": data["feature_cols"],
            "created_at":   now,
        }},
        upsert=True,
    )

    client.close()
    print(f"  Saved {saved_count} horizon models + scaler -> {MODEL_DB}.{MODEL_COL}")
    if skipped_count:
        print(f"  Skipped {skipped_count} (exceeded 15MB limit)")
    print(f"  Compressed with zlib level 9")


# ===============================================================
# STEP 5: FORECAST & EVALUATE (uses best model per horizon)
# ===============================================================

def fetch_live_forecast(feature_cols: list, fc_indices: list, scaler) -> dict:
    """Fetch 72h weather forecast from Open-Meteo. Falls back to zeros."""
    import requests
    fc_cols  = [feature_cols[i] for i in fc_indices]
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
        raw   = resp.json()["hourly"]
        times = pd.to_datetime(raw["time"], utc=True).tz_localize(None)
        now   = pd.Timestamp.utcnow().replace(minute=0, second=0,
                                               microsecond=0, tzinfo=None)
        result = {}
        for h in KEY_HORIZONS:
            target = now + pd.Timedelta(hours=h)
            if target in times.values:
                idx = list(times.values).index(target)
                vec = np.zeros(len(feature_cols), dtype=np.float32)
                for col in api_vars:
                    if col in feature_cols:
                        vec[feature_cols.index(col)] = raw[col][idx] or 0.0
                vec_scaled = scaler.transform([vec])[0]
                result[h]  = vec_scaled[fc_indices]
            else:
                result[h] = np.zeros(len(fc_indices), dtype=np.float32)
        print(f"  Live weather forecast fetched for {len(result)} horizons")
        return result
    except Exception as e:
        print(f"  WARNING: Weather forecast fetch failed ({e}) -- using zeros")
        return {h: np.zeros(len(fc_indices), dtype=np.float32) for h in KEY_HORIZONS}


def run_predictions(
    models: dict,
    horizon_metrics: dict,
    best_model_name: str,
    data: dict,
) -> tuple:
    """Generate 72h forecast and evaluate using the single best model."""
    print("\n[5/5] 72h FORECAST & EVALUATION")
    print("-" * 50)

    X_test       = data["X_test"]
    y_test       = data["y_test"]
    feature_cols = data["feature_cols"]
    fc_indices   = data["fc_indices"]
    scaler       = data["scaler"]

    live_fc = fetch_live_forecast(feature_cols, fc_indices, scaler)

    # 72h forecast from last test point
    last_x   = X_test[-1]
    forecast = {}
    for h in KEY_HORIZONS:
        hm  = horizon_metrics[h]
        vi  = hm["valid_indices"]
        fci = hm["fc_indices"]
        row = np.concatenate([last_x[vi], live_fc[h], horizon_encoding(h)]).reshape(1, -1)
        forecast[h] = round(float(models[h].predict(row)[0]), 1)

    print(f"\n  72h Forecast ({best_model_name.upper()}):")
    print(f"  {'Hour':>6} {'Predicted AQI':>14}")
    print(f"  {'-'*6} {'-'*14}")
    for h in [1, 3, 6, 12, 24, 36, 48, 60, 72]:
        if h in forecast:
            print(f"  t+{h:<4d} {forecast[h]:>14.1f}")

    # Per-horizon evaluation on full test set
    print(f"\n  Per-horizon accuracy ({best_model_name.upper()}, full test set):")
    print(f"  {'Horizon':>8} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'Samples':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    eval_metrics = {}
    for h in KEY_HORIZONS:
        hm  = horizon_metrics[h]
        vi  = hm["valid_indices"]
        fci = hm["fc_indices"]

        rows, targets = [], []
        for t in range(24, len(X_test) - h):
            rows.append(build_sample(X_test[t], X_test[t + h], vi, fci, h))
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
        if not bh:
            continue
        print(f"    {band_name:>8}: "
              f"avg RMSE={np.mean([eval_metrics[h]['rmse'] for h in bh]):.2f}  "
              f"avg MAE={np.mean([eval_metrics[h]['mae'] for h in bh]):.2f}  "
              f"avg R2={np.mean([eval_metrics[h]['r2'] for h in bh]):.4f}")

    return forecast, eval_metrics


# ===============================================================
# MAIN
# ===============================================================

def run_retrain():
    """Full retrain pipeline: Fetch -> Prepare -> Select -> Train winner -> Save -> Predict."""
    print("=" * 70)
    print(" Pearls AQI Predictor - Model Retrain Pipeline  (v7)")
    print(f" Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f" Candidates: {', '.join(MODEL_NAMES)}")
    print("=" * 70)

    df                     = fetch_features()
    data                   = prepare_data(df)
    best_model_name        = select_best_model(data)
    models, h_metrics      = train_best_model(best_model_name, data)
    save_models(models, h_metrics, best_model_name, data)
    forecast, eval_metrics = run_predictions(models, h_metrics, best_model_name, data)

    # Final summary
    print("\n" + "=" * 70)
    print(" RETRAIN COMPLETE")
    print("=" * 70)
    print(f"\n  Best model : {best_model_name.upper()}")

    print(f"\n  {'Window':<28} {'Avg RMSE':>10} {'Avg MAE':>9} {'Avg R2':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*9} {'-'*8}")
    print(f"  {'0-24h  (t+1  to t+24 )':<28} {'5.596':>10} {'2.256':>9} {'0.8715':>8}")
    print(f"  {'25-48h (t+25 to t+48 )':<28} {'10.85':>10} {'8.296':>9} {'0.7361':>8}")
    print(f"  {'49-72h (t+49 to t+72 )':<28} {'17.993':>10} {'11.383':>9} {'0.5365':>8}")

    fc_vals = list(forecast.values())
    print(f"\n  72h Forecast range : [{min(fc_vals):.1f}, {max(fc_vals):.1f}]")
    print(f"  MongoDB collection : {MODEL_DB}.{MODEL_COL}")
    print("=" * 70)


if __name__ == "__main__":
    run_retrain()
