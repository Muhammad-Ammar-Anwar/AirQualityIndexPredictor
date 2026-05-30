# -----------------------------
# Pearls AQI Predictor — Hourly Pipeline (v2)
# Self-contained: Fetch last 3h → Check duplicates → Engineer → Upload new only
# Restructured: Uses API-provided US AQI + sub-indices, expanded weather vars
# Designed for CI/CD (GitHub Actions, cron, etc.)
# -----------------------------

import os
import sys
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ASCENDING, errors
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Configuration ──
LATITUDE = 24.8607
LONGITUDE = 67.0011
LOCATION = "Karachi"

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB") or "AQI_Project"
COLLECTION_NAME = (
    os.getenv("MONGODB_FEATURES_COLLECTION")
    or os.getenv("MONGODB_COLLECTION")
    or "karachi_aqi_features"
)

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FORECAST_URL_FALLBACK = "https://previous-runs-api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

LOOKBACK_HOURS = 3       # Fetch last 3 hours to avoid missing data
HISTORY_HOURS = 48       # MongoDB history for lag/rolling feature computation

# ── Weather variables (must match api_data_fetch.py / feature_engineering.py) ──
WEATHER_VARS = [
    # Core meteorological
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "surface_pressure",
    "pressure_msl",                # Sea-level pressure → atmospheric stability
    # Cloud cover (total + altitude bands)
    "cloud_cover",
    "cloud_cover_low",             # Low clouds <2km → fog, trapping
    "cloud_cover_mid",             # Mid clouds 2-6km
    "cloud_cover_high",            # High clouds >6km
    # Wind
    "windspeed_10m", "winddirection_10m",
    "wind_gusts_10m",              # Gust intensity → pollutant dispersion
    # Radiation & moisture
    "shortwave_radiation",         # Solar radiation → O3 photochemistry
    "vapour_pressure_deficit",     # VPD → particle hygroscopicity
]

# ── Air Quality variables ──
AQ_VARS = [
    # Raw pollutant concentrations (µg/m³)
    "pm10", "pm2_5", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    # Additional atmospheric variables
    "aerosol_optical_depth", "dust",
    "uv_index", "uv_index_clear_sky",
    "carbon_dioxide",
    # API-computed US AQI (proper EPA rolling averages)
    "us_aqi",
    # Individual pollutant AQI sub-indices
    "us_aqi_pm2_5", "us_aqi_pm10",
    "us_aqi_nitrogen_dioxide", "us_aqi_ozone",
    "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide",
]


# ═══════════════════════════════════════════════════════════════
# SELF-CONTAINED FEATURE ENGINEERING (mirrors feature_engineering.py)
# ═══════════════════════════════════════════════════════════════

# EPA breakpoint tables
PM25_BP = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
]
PM10_BP = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500)
]
O3_BP = [
    (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
    (86, 105, 151, 200), (106, 200, 201, 300)
]
NO2_BP = [
    (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
    (361, 649, 151, 200), (650, 1249, 201, 300)
]
SO2_BP = [
    (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
    (186, 304, 151, 200), (305, 604, 201, 300)
]
CO_BP = [
    (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300)
]


def _aqi_sub(C, bp):
    for Cl, Ch, Il, Ih in bp:
        if Cl <= C <= Ch:
            return ((Ih - Il) / (Ch - Cl)) * (C - Cl) + Il
    return None


def compute_aqi(pm25, pm10, o3=None, no2=None, so2=None, co=None):
    """Compute US EPA AQI from pollutant concentrations (µg/m³)."""
    subs = {}
    if pm25 is not None and not np.isnan(pm25) and pm25 >= 0:
        v = _aqi_sub(pm25, PM25_BP)
        if v: subs['PM2.5'] = v
    if pm10 is not None and not np.isnan(pm10) and pm10 >= 0:
        v = _aqi_sub(pm10, PM10_BP)
        if v: subs['PM10'] = v
    if o3 is not None and not np.isnan(o3) and o3 >= 0:
        v = _aqi_sub(o3 / 2.0, O3_BP)
        if v: subs['O3'] = v
    if no2 is not None and not np.isnan(no2) and no2 >= 0:
        v = _aqi_sub(no2 / 1.88, NO2_BP)
        if v: subs['NO2'] = v
    if so2 is not None and not np.isnan(so2) and so2 >= 0:
        v = _aqi_sub(so2 / 2.62, SO2_BP)
        if v: subs['SO2'] = v
    if co is not None and not np.isnan(co) and co >= 0:
        v = _aqi_sub(co / 1145.0, CO_BP)
        if v: subs['CO'] = v
    if not subs:
        return np.nan, 'N/A'
    dom = max(subs, key=subs.get)
    return round(subs[dom], 1), dom


# ═══════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════

def _parse_datetimes(series):
    """Parse datetime series and strip any timezone info to naive UTC."""
    dt = pd.to_datetime(series, utc=True)
    return dt.dt.tz_localize(None)


def _api_get(url: str, params: dict, max_retries: int = 5, fallback_url: str = None) -> dict:
    """GET with exponential backoff — handles transient 5xx errors.
    If fallback_url is provided, switches to it immediately on first 502/503/504.
    """
    import time as _t
    current_url = url
    fallback_used = False

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(current_url, params=params, timeout=30)
            resp.raise_for_status()
            if fallback_used:
                print(f"[FETCH] Fallback URL succeeded: {current_url}")
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (502, 503, 504):
                if not fallback_used and fallback_url:
                    fallback_used = True
                    current_url = fallback_url
                    print(f"[FETCH] {status} on primary — switching to fallback URL (attempt {attempt}/{max_retries})")
                    continue
                if attempt < max_retries:
                    wait = 60 * attempt
                    print(f"[FETCH] {status} error — retrying in {wait}s (attempt {attempt}/{max_retries})")
                    _t.sleep(wait)
                else:
                    raise
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait = 60 * attempt
                print(f"[FETCH] Request error ({e}) — retrying in {wait}s (attempt {attempt}/{max_retries})")
                _t.sleep(wait)
            else:
                raise
    raise RuntimeError("API request failed after all retries")


def fetch_recent_hours(lookback_hours=None):
    """Fetch recent weather + air quality data.

    Uses past_days=1 on both APIs to get confirmed historical data rather
    than relying solely on forecast data which can lag 1-6h behind real time.
    lookback_hours overrides LOOKBACK_HOURS when provided (used for backfill).
    """
    hours = lookback_hours if lookback_hours is not None else LOOKBACK_HOURS
    print(f"[FETCH] Fetching last {hours}h weather + air quality ...")

    # Use past_days=1 + forecast_days=1 to ensure current and recent hours
    # are always available regardless of forecast model update lag
    w_params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
        "past_days": 1,
        "forecast_days": 1,
    }
    print(f"[FETCH] Weather URL: {WEATHER_FORECAST_URL}")
    w_data = _api_get(WEATHER_FORECAST_URL, w_params, fallback_url=WEATHER_FORECAST_URL_FALLBACK)

    weather_df = pd.DataFrame({"datetime": w_data["hourly"]["time"]})
    for var in WEATHER_VARS:
        weather_df[var] = w_data["hourly"].get(var)
    weather_df["datetime"] = _parse_datetimes(weather_df["datetime"])

    aq_params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": ",".join(AQ_VARS),
        "timezone": "UTC",
        "past_days": 1,
        "forecast_days": 1,
    }
    print(f"[FETCH] AQ URL: {AIR_QUALITY_URL}")
    aq_data = _api_get(AIR_QUALITY_URL, aq_params)

    aq_df = pd.DataFrame({"datetime": aq_data["hourly"]["time"]})
    for var in AQ_VARS:
        aq_df[var] = aq_data["hourly"].get(var)
    aq_df["datetime"] = _parse_datetimes(aq_df["datetime"])

    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    cutoff = now - timedelta(hours=hours - 1)

    print(f"[FETCH] UTC now (truncated): {now}")
    print(f"[FETCH] Cutoff: {cutoff}")
    if not merged.empty:
        print(f"[FETCH] API data range: {merged['datetime'].min()} to {merged['datetime'].max()} ({len(merged)} rows)")
        print(f"[FETCH] Sample timestamps: {list(merged['datetime'].head(3))}")
    else:
        print("[FETCH] WARNING: merged DataFrame is empty (weather/AQ merge produced 0 rows)")
        return pd.DataFrame()

    # Primary filter: last LOOKBACK_HOURS up to now
    recent = merged[(merged["datetime"] >= cutoff) & (merged["datetime"] <= now)]
    recent = recent.sort_values("datetime").reset_index(drop=True)

    if recent.empty:
        # Fallback: widen to 12h in case of API lag
        print(f"[FETCH] Strict filter ({cutoff} to {now}) found 0 rows — trying wider 12h window ...")
        wider_cutoff = now - timedelta(hours=12)
        recent = merged[(merged["datetime"] >= wider_cutoff) & (merged["datetime"] <= now)]
        recent = recent.sort_values("datetime").reset_index(drop=True)
        if recent.empty:
            print(f"[FETCH] Wide filter also empty. Using nearest past hours from API data ...")
            past = merged[merged["datetime"] <= now].sort_values("datetime", ascending=False)
            if not past.empty:
                recent = past.head(LOOKBACK_HOURS).sort_values("datetime").reset_index(drop=True)
                print(f"[FETCH] Using {len(recent)} nearest past hours: {recent['datetime'].min()} to {recent['datetime'].max()}")
            else:
                print("[FETCH] No past data available at all in API response.")
                return pd.DataFrame()
        else:
            print(f"[FETCH] Wide filter got {len(recent)} rows")

    print(f"[FETCH] Got {len(recent)} hours: {recent['datetime'].min()} to {recent['datetime'].max()}")
    return recent


# ═══════════════════════════════════════════════════════════════
# MONGODB HELPERS
# ═══════════════════════════════════════════════════════════════

def get_mongo_client():
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    client.admin.command('ping')
    return client


def get_existing_datetimes(datetimes):
    """Check which datetimes already exist in MongoDB.

    Returns:
        set of datetime values already in the collection
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]

    # Normalise to naive UTC before querying — MongoDB may store with or without tz
    def _to_naive(dt):
        ts = pd.Timestamp(dt)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()

    dt_list = [_to_naive(dt) for dt in datetimes]

    # Query both naive and UTC-aware variants to be safe
    cursor = collection.find(
        {"datetime": {"$in": dt_list}},
        {"datetime": 1, "_id": 0}
    )
    existing = set()
    for doc in cursor:
        ts = pd.Timestamp(doc["datetime"])
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        existing.add(ts)
    client.close()
    return existing


def fetch_recent_history(hours=HISTORY_HOURS):
    """Fetch last N hours from MongoDB for computing lag/rolling features."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    cursor = collection.find(
        {"datetime": {"$gte": cutoff}}, {"_id": 0}
    ).sort("datetime", 1)

    records = list(cursor)
    client.close()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    print(f"[MONGO] Fetched {len(df)} recent records.")
    return df


def upload_single_record(record_dict):
    """Upload one engineered record with dedup."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    collection.create_index([("datetime", ASCENDING)], unique=True)

    try:
        collection.insert_one(record_dict)
        print("[MONGO] Inserted 1 new record.")
        ok = True
    except errors.DuplicateKeyError:
        print("[MONGO] Duplicate — skipped.")
        ok = False

    total = collection.count_documents({})
    print(f"[MONGO] Total documents: {total}")
    client.close()
    return ok


def prepare_record(row_dict):
    """Convert to MongoDB-safe types. Datetimes stored as naive UTC."""
    clean = {}
    for k, v in row_dict.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        elif isinstance(v, pd.Timestamp):
            # Always store as naive UTC — strip tz if present
            ts = v
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            clean[k] = ts.to_pydatetime()
        elif isinstance(v, datetime):
            # Strip tz from plain datetime too
            if v.tzinfo is not None:
                import pytz as _pytz
                v = v.astimezone(_pytz.utc).replace(tzinfo=None)
            clean[k] = v
        elif pd.api.types.is_scalar(v) and pd.isna(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR CURRENT HOUR
# ═══════════════════════════════════════════════════════════════

def engineer_current_hour(raw_row, history_df):
    """Engineer features for the current hour using recent MongoDB history.

    Mirrors feature_engineering.py (v2) but operates on a single row
    with context from recent history (for lag/rolling features).

    v2 changes:
    - Uses API-provided us_aqi (proper EPA rolling averages)
    - Adds sub-AQI index features (us_aqi_pm2_5, etc.)
    - New weather variables: wind_gusts, shortwave_radiation, VPD, pressure_msl
    - New atmospheric variables: carbon_dioxide, uv_index_clear_sky
    """
    print("[FEATURES] Engineering features for current hour ...")
    raw_row = raw_row.copy()

    # ── AQI — prefer API-provided value ──
    row = raw_row.iloc[0]
    extra_cols = {}  # accumulate all new columns here, concat once at end

    if 'us_aqi' in raw_row.columns and pd.notna(row.get('us_aqi')):
        print(f"[FEATURES] Using API-provided US AQI: {row['us_aqi']}")
        # Determine dominant pollutant from sub-indices
        sub_cols = ['us_aqi_pm2_5', 'us_aqi_pm10', 'us_aqi_nitrogen_dioxide',
                    'us_aqi_ozone', 'us_aqi_sulphur_dioxide', 'us_aqi_carbon_monoxide']
        available_subs = {c: row.get(c) for c in sub_cols
                         if c in raw_row.columns and pd.notna(row.get(c))}
        if available_subs:
            dominant_col = max(available_subs, key=available_subs.get)
            name_map = {
                'us_aqi_pm2_5': 'PM2.5', 'us_aqi_pm10': 'PM10',
                'us_aqi_nitrogen_dioxide': 'NO2', 'us_aqi_ozone': 'O3',
                'us_aqi_sulphur_dioxide': 'SO2', 'us_aqi_carbon_monoxide': 'CO'
            }
            extra_cols["dominant_pollutant"] = name_map.get(dominant_col, 'N/A')
        else:
            extra_cols["dominant_pollutant"] = 'N/A'
    else:
        # Fallback: compute AQI manually
        aqi_val, dominant = compute_aqi(
            pm25=row.get("pm2_5"), pm10=row.get("pm10"),
            o3=row.get("ozone"), no2=row.get("nitrogen_dioxide"),
            so2=row.get("sulphur_dioxide"), co=row.get("carbon_monoxide")
        )
        extra_cols["us_aqi"] = aqi_val
        extra_cols["dominant_pollutant"] = dominant
        print(f"[FEATURES] Computed AQI manually (fallback): {aqi_val}")

    # ── Wind decomposition ──
    if "windspeed_10m" in raw_row.columns and "winddirection_10m" in raw_row.columns:
        rad = np.deg2rad(raw_row["winddirection_10m"].values[0])
        ws = raw_row["windspeed_10m"].values[0]
        extra_cols["wind_speed"] = ws
        extra_cols["wind_u"] = -ws * np.sin(rad)
        extra_cols["wind_v"] = -ws * np.cos(rad)

    # ── Time features ──
    dt = raw_row["datetime"].iloc[0]
    hour = dt.hour
    dow = dt.weekday()
    month = dt.month
    doy = dt.timetuple().tm_yday
    extra_cols["hour_sin"]        = np.sin(2 * np.pi * hour / 24)
    extra_cols["hour_cos"]        = np.cos(2 * np.pi * hour / 24)
    extra_cols["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    extra_cols["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    extra_cols["month_sin"]       = np.sin(2 * np.pi * month / 12)
    extra_cols["month_cos"]       = np.cos(2 * np.pi * month / 12)
    extra_cols["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
    extra_cols["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)
    extra_cols["is_weekend"]      = 1.0 if dow >= 5 else 0.0

    # ── Interaction features (v2: added radiation×aerosol, vpd×temp) ──
    if "relative_humidity_2m" in raw_row.columns and "temperature_2m" in raw_row.columns:
        extra_cols["humidity_temp_interaction"] = (
            raw_row["relative_humidity_2m"].values[0] *
            raw_row["temperature_2m"].values[0]
        )
    if "temperature_2m" in raw_row.columns and "surface_pressure" in raw_row.columns:
        extra_cols["temp_pressure_interaction"] = (
            raw_row["temperature_2m"].values[0] *
            raw_row["surface_pressure"].values[0] / 1000.0
        )
    wind_speed_val = extra_cols.get("wind_speed", raw_row.get("wind_speed", [None])[0] if "wind_speed" in raw_row.columns else None)
    if wind_speed_val is not None and "relative_humidity_2m" in raw_row.columns:
        extra_cols["wind_humidity_interaction"] = (
            wind_speed_val *
            raw_row["relative_humidity_2m"].values[0]
        )
    if "cloud_cover" in raw_row.columns and "temperature_2m" in raw_row.columns:
        extra_cols["cloud_temp_interaction"] = (
            raw_row["cloud_cover"].values[0] *
            raw_row["temperature_2m"].values[0]
        )
    if "aerosol_optical_depth" in raw_row.columns and "relative_humidity_2m" in raw_row.columns:
        extra_cols["aerosol_humidity_interaction"] = (
            raw_row["aerosol_optical_depth"].values[0] *
            raw_row["relative_humidity_2m"].values[0]
        )
    # NEW v2 interactions
    if "shortwave_radiation" in raw_row.columns and "aerosol_optical_depth" in raw_row.columns:
        extra_cols["radiation_aerosol_interaction"] = (
            raw_row["shortwave_radiation"].values[0] *
            raw_row["aerosol_optical_depth"].values[0]
        )
    if "vapour_pressure_deficit" in raw_row.columns and "temperature_2m" in raw_row.columns:
        extra_cols["vpd_temp_interaction"] = (
            raw_row["vapour_pressure_deficit"].values[0] *
            raw_row["temperature_2m"].values[0]
        )

    # ── Lag / rolling / AQI-AR from history ──
    if not history_df.empty:
        # Build combined series for lag/rolling
        core_cols = ["datetime", "us_aqi"]
        # v2: expanded weather columns for rolling/lags
        weather_cols = [c for c in [
            "temperature_2m", "relative_humidity_2m", "surface_pressure",
            "wind_u", "wind_v", "cloud_cover", "dew_point_2m",
            "aerosol_optical_depth", "dust", "uv_index",
            # NEW in v2
            "wind_gusts_10m", "shortwave_radiation",
            "vapour_pressure_deficit", "pressure_msl",
            "uv_index_clear_sky", "carbon_dioxide",
        ] if c in history_df.columns and c in raw_row.columns]

        # Sub-AQI columns for rolling/lags
        sub_aqi_cols = [c for c in [
            "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide",
            "us_aqi_ozone", "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide",
        ] if c in history_df.columns and c in raw_row.columns]

        use_cols = [c for c in core_cols + weather_cols + sub_aqi_cols
                    if c in history_df.columns]
        hist = history_df[use_cols].copy()

        new_row_data = {}
        for c in use_cols:
            if c in raw_row.columns:
                new_row_data[c] = raw_row[c].values[0]
        new_df = pd.DataFrame([new_row_data])
        combined = pd.concat([hist, new_df], ignore_index=True)
        combined = combined.sort_values("datetime").reset_index(drop=True)
        last_idx = len(combined) - 1

        # Weather + atmospheric derivatives (rolling/lags)
        for col in weather_cols:
            for w in [6, 12, 24]:
                window = combined[col].iloc[max(0, last_idx - w + 1):last_idx + 1]
                extra_cols[f"{col}_rolling_mean_{w}h"] = float(window.mean())
            window_24 = combined[col].iloc[max(0, last_idx - 23):last_idx + 1]
            extra_cols[f"{col}_rolling_std_24h"] = float(window_24.std()) if len(window_24) > 1 else 0.0
            for lag in [12, 24]:
                idx = last_idx - lag
                if idx >= 0:
                    extra_cols[f"{col}_lag_{lag}h"] = float(combined[col].iloc[idx])
                else:
                    extra_cols[f"{col}_lag_{lag}h"] = np.nan

        # Sub-AQI index derivatives (NEW in v2)
        for col in sub_aqi_cols:
            for lag in [6, 12, 24]:
                idx = last_idx - lag
                if idx >= 0:
                    extra_cols[f"{col}_lag_{lag}h"] = float(combined[col].iloc[idx])
                else:
                    extra_cols[f"{col}_lag_{lag}h"] = np.nan
            for w in [12, 24]:
                window = combined[col].iloc[max(0, last_idx - w + 1):last_idx + 1]
                extra_cols[f"{col}_rolling_mean_{w}h"] = float(window.mean())

        # AQI autoregressive features
        aqi_col = "us_aqi"
        if aqi_col in combined.columns:
            for lag in [1, 3, 6, 12, 24]:
                idx = last_idx - lag
                extra_cols[f"us_aqi_lag_{lag}h"] = (
                    float(combined[aqi_col].iloc[idx]) if idx >= 0 else np.nan
                )
            for w in [6, 12, 24]:
                window = combined[aqi_col].iloc[max(0, last_idx - w + 1):last_idx + 1]
                extra_cols[f"us_aqi_rolling_mean_{w}h"] = float(window.mean())
            w6 = combined[aqi_col].iloc[max(0, last_idx - 5):last_idx + 1]
            w24 = combined[aqi_col].iloc[max(0, last_idx - 23):last_idx + 1]
            extra_cols["us_aqi_rolling_std_6h"] = float(w6.std()) if len(w6) > 1 else 0.0
            extra_cols["us_aqi_rolling_std_24h"] = float(w24.std()) if len(w24) > 1 else 0.0
            cur = combined[aqi_col].iloc[last_idx]
            lag1 = combined[aqi_col].iloc[last_idx - 1] if last_idx >= 1 else cur
            lag6 = combined[aqi_col].iloc[last_idx - 6] if last_idx >= 6 else cur
            extra_cols["us_aqi_delta_1h"] = float(cur - lag1)
            extra_cols["us_aqi_delta_6h"] = float((cur - lag6) / 6.0)
    else:
        print("[FEATURES] No history — lag/rolling will be NaN.")
        # Set all derived features to NaN
        weather_cols = [
            "temperature_2m", "relative_humidity_2m", "surface_pressure",
            "wind_u", "wind_v", "cloud_cover", "dew_point_2m",
            "aerosol_optical_depth", "dust", "uv_index",
            "wind_gusts_10m", "shortwave_radiation",
            "vapour_pressure_deficit", "pressure_msl",
            "uv_index_clear_sky", "carbon_dioxide",
        ]
        for col in weather_cols:
            for w in [6, 12, 24]:
                extra_cols[f"{col}_rolling_mean_{w}h"] = np.nan
            extra_cols[f"{col}_rolling_std_24h"] = np.nan
            for lag in [12, 24]:
                extra_cols[f"{col}_lag_{lag}h"] = np.nan

        # Sub-AQI NaN defaults
        sub_aqi_cols = [
            "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide",
            "us_aqi_ozone", "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide",
        ]
        for col in sub_aqi_cols:
            for lag in [6, 12, 24]:
                extra_cols[f"{col}_lag_{lag}h"] = np.nan
            for w in [12, 24]:
                extra_cols[f"{col}_rolling_mean_{w}h"] = np.nan

        # AQI AR NaN defaults
        for lag in [1, 3, 6, 12, 24]:
            extra_cols[f"us_aqi_lag_{lag}h"] = np.nan
        for w in [6, 12, 24]:
            extra_cols[f"us_aqi_rolling_mean_{w}h"] = np.nan
        extra_cols["us_aqi_rolling_std_6h"] = np.nan
        extra_cols["us_aqi_rolling_std_24h"] = np.nan
        extra_cols["us_aqi_delta_1h"] = np.nan
        extra_cols["us_aqi_delta_6h"] = np.nan

    # ── Concat all new columns at once to avoid DataFrame fragmentation ──
    if extra_cols:
        extra_df = pd.DataFrame([extra_cols], index=raw_row.index)
        raw_row = pd.concat([raw_row, extra_df], axis=1)

    print(f"[FEATURES] Engineered {len(raw_row.columns)} columns.")
    return raw_row


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_hourly_pipeline():
    """Full hourly pipeline:
    1. Auto-detect gap from MongoDB — extend lookback if data is missing
    2. Fetch missing hours from API
    3. Check MongoDB for duplicates — skip already uploaded hours
    4. Fetch 48h history from MongoDB for lag/rolling features
    5. Engineer features for each new hour
    6. Upload only new records to MongoDB
    """
    print("=" * 70)
    print(" Pearls AQI Predictor — Hourly Pipeline")
    print(f" Location: {LOCATION} ({LATITUDE}, {LONGITUDE})")
    print(f" Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Target DB: {DB_NAME}.{COLLECTION_NAME}")
    print("=" * 70)

    # ── Auto-detect gap: how many hours are missing from MongoDB? ──
    lookback = LOOKBACK_HOURS
    try:
        client = get_mongo_client()
        col = client[DB_NAME][COLLECTION_NAME]
        latest_doc = list(col.find({}, {"datetime": 1, "_id": 0})
                          .sort("datetime", -1).limit(1))
        client.close()
        if latest_doc:
            latest_dt = pd.Timestamp(latest_doc[0]["datetime"])
            if latest_dt.tzinfo is not None:
                latest_dt = latest_dt.tz_convert("UTC").tz_localize(None)
            now = pd.Timestamp.utcnow().replace(minute=0, second=0,
                                                microsecond=0, tzinfo=None)
            gap_hours = int((now - latest_dt).total_seconds() / 3600)
            if gap_hours > LOOKBACK_HOURS:
                lookback = min(gap_hours + 1, 48)  # cap at 48h
                print(f"[GAP] Last record: {latest_dt} — gap of {gap_hours}h detected.")
                print(f"[GAP] Extending lookback to {lookback}h to backfill.")
    except Exception as e:
        print(f"[GAP] Could not check gap: {e} — using default lookback {lookback}h")

    # ── Step 1: Fetch missing hours ──
    print(f"\n[1/4] FETCHING LAST {lookback} HOURS")
    print("-" * 50)

    raw_df = fetch_recent_hours(lookback_hours=lookback)

    if raw_df.empty:
        print("  No data fetched — skipping this run (will retry next hour).")
        print("=" * 70)
        return None

    print(f"  Fetched {len(raw_df)} rows")

    # ── Step 2: Check for duplicates in MongoDB ──
    print(f"\n[2/4] CHECKING FOR DUPLICATES IN MONGODB")
    print("-" * 50)

    fetched_datetimes = raw_df["datetime"].tolist()
    existing = get_existing_datetimes(fetched_datetimes)

    if existing:
        existing_strs = [str(dt) for dt in sorted(existing)]
        print(f"  Already in MongoDB: {', '.join(existing_strs)}")
        raw_df = raw_df[~raw_df["datetime"].isin(existing)].reset_index(drop=True)
    else:
        print("  No duplicates found")

    if raw_df.empty:
        print("\n  All hours already uploaded. Nothing to do.")
        print("=" * 70)
        return None

    new_datetimes = raw_df["datetime"].tolist()
    print(f"  New hours to process: {len(raw_df)}")
    for dt in new_datetimes:
        print(f"    - {dt}")

    # ── Step 3: Fetch history + engineer features ──
    print(f"\n[3/4] FEATURE ENGINEERING")
    print("-" * 50)

    print(f"  Fetching {HISTORY_HOURS}h history from MongoDB ...")
    history_df = fetch_recent_history(hours=HISTORY_HOURS)

    if not history_df.empty:
        print(f"  History: {len(history_df)} rows ({history_df['datetime'].min()} to {history_df['datetime'].max()})")

    # Process each new hour individually (each needs its own lag context)
    all_engineered = []
    for i, (_, row_data) in enumerate(raw_df.iterrows()):
        row_df = pd.DataFrame([row_data])
        print(f"\n  Processing hour {i+1}/{len(raw_df)}: {row_data['datetime']}")
        engineered = engineer_current_hour(row_df, history_df)
        all_engineered.append(engineered)

    # ── Step 4: Upload to MongoDB ──
    print(f"\n[4/4] UPLOADING TO MONGODB")
    print("-" * 50)

    inserted = 0
    skipped = 0
    for eng_row in all_engineered:
        record = prepare_record(eng_row.iloc[0].to_dict())
        ok = upload_single_record(record)
        if ok:
            inserted += 1
        else:
            skipped += 1

    # ── Summary ──
    print("\n" + "=" * 70)
    print(" HOURLY PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  Hours checked:      {len(fetched_datetimes)}")
    print(f"  Already in MongoDB: {len(existing)}")
    print(f"  New rows processed: {len(all_engineered)}")
    print(f"  Records uploaded:   {inserted}")
    print(f"  Duplicates skipped: {skipped}")
    aqi_vals = [e.iloc[0].get('us_aqi', 'N/A') for e in all_engineered]
    print(f"  AQI values:         {aqi_vals}")
    print("=" * 70)

    return all_engineered


if __name__ == "__main__":
    import time as _time
    import traceback as _tb

    MAX_RETRIES = 3
    RETRY_DELAY = 90  # seconds — longer wait for API recovery

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n>>> Attempt {attempt}/{MAX_RETRIES}")
            run_hourly_pipeline()
            break
        except Exception as exc:
            print(f"\n[ERROR] Attempt {attempt} failed: {exc}")
            _tb.print_exc()
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt  # exponential: 90s, 180s
                print(f"[RETRY] Waiting {wait}s before retry ...\n")
                _time.sleep(wait)
            else:
                print("\n" + "=" * 70)
                print(" HOURLY PIPELINE FAILED after all retries")
                print(f" Error: {exc}")
                # Check if it's a transient API error (5xx) — exit 0 so CI
                # doesn't alert on temporary Open-Meteo outages
                err_str = str(exc).lower()
                is_transient = any(code in err_str for code in
                                   ["502", "503", "504", "bad gateway",
                                    "service unavailable", "timeout"])
                print(f" Transient API error: {is_transient}")
                print("=" * 70)
                sys.exit(0 if is_transient else 1)

