# ─────────────────────────────────────────────────────────────────
# Pearls AQI Predictor — Feature Engineering Module (v2)
# Restructured for API-provided US AQI + sub-indices
#
# KEY CHANGES FROM v1:
#   1. Uses Open-Meteo API-provided us_aqi (proper EPA rolling averages:
#      PM=24h, O3/CO=8h, SO2/NO2=1h) instead of manual instantaneous calc
#   2. AQI sub-indices (us_aqi_pm2_5, us_aqi_pm10, etc.) used as features
#      — tells the model WHICH pollutant drives AQI
#   3. New weather variables: wind_gusts, shortwave_radiation, VPD,
#      pressure_msl, cloud cover bands (low/mid/high)
#   4. New atmospheric variables: carbon_dioxide, uv_index_clear_sky
#   5. Manual AQI calc retained as fallback only
#
# FEATURE CATEGORIES:
#   WEATHER:      temperature, humidity, pressure, wind_u/v, cloud, dew_point,
#                 wind_gusts, shortwave_radiation, VPD, pressure_msl
#                 + rolling means (6/12/24h), rolling std (24h), lags (12/24h)
#   ATMOSPHERIC:  aerosol_optical_depth, dust, uv_index, uv_index_clear_sky,
#                 carbon_dioxide + derivatives
#   SUB_AQI:      us_aqi_pm2_5, us_aqi_pm10, us_aqi_no2, us_aqi_o3,
#                 us_aqi_so2, us_aqi_co + lags (6/12/24h), rolling (12/24h)
#   TIME:         hour/dow/month/doy cyclical + is_weekend
#   INTERACTION:  humidity×temp, temp×pressure, wind×humidity, cloud×temp,
#                 aerosol×humidity, radiation×aerosol, vpd×temp
#   AQI_AR:       AQI lags (1/3/6/12/24h), rolling mean/std (6/12/24h), deltas
#   POLLUTANT:    Raw concentrations — used for AQI fallback only, NOT features
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# US EPA AQI CALCULATION (FALLBACK ONLY)
# Prefer API-provided us_aqi which uses proper EPA rolling averages
# ═══════════════════════════════════════════════════════════════

def _aqi_subindex(C, breakpoints):
    """Calculate AQI sub-index for a single pollutant concentration."""
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= C <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (C - Clow) + Ilow
    return None


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


def calculate_us_aqi(row):
    """Calculate US EPA AQI from instantaneous pollutant concentrations.

    IMPORTANT: This uses instantaneous values. The EPA standard requires
    rolling averages (PM: 24h, O3/CO: 8h, SO2/NO2: 1h). Prefer the
    API-provided us_aqi when available.

    Unit conversions (µg/m³ → ppb/ppm) applied per EPA standards.
    Returns (aqi_value, dominant_pollutant).
    """
    subindices = {}

    pm25 = row.get('pm2_5')
    pm10 = row.get('pm10')
    o3   = row.get('ozone')
    no2  = row.get('nitrogen_dioxide')
    so2  = row.get('sulphur_dioxide')
    co   = row.get('carbon_monoxide')

    if pd.notna(pm25) and pm25 >= 0:
        val = _aqi_subindex(pm25, PM25_BP)
        if val is not None: subindices['PM2.5'] = val

    if pd.notna(pm10) and pm10 >= 0:
        val = _aqi_subindex(pm10, PM10_BP)
        if val is not None: subindices['PM10'] = val

    if pd.notna(o3) and o3 >= 0:
        val = _aqi_subindex(o3 / 2.0, O3_BP)
        if val is not None: subindices['O3'] = val

    if pd.notna(no2) and no2 >= 0:
        val = _aqi_subindex(no2 / 1.88, NO2_BP)
        if val is not None: subindices['NO2'] = val

    if pd.notna(so2) and so2 >= 0:
        val = _aqi_subindex(so2 / 2.62, SO2_BP)
        if val is not None: subindices['SO2'] = val

    if pd.notna(co) and co >= 0:
        val = _aqi_subindex(co / 1145.0, CO_BP)
        if val is not None: subindices['CO'] = val

    if not subindices:
        return np.nan, 'N/A'

    dominant = max(subindices, key=subindices.get)
    return round(subindices[dominant], 1), dominant


# ═══════════════════════════════════════════════════════════════
# AQI TARGET SETUP
# ═══════════════════════════════════════════════════════════════

def add_aqi_target(df, verbose=True):
    """Set us_aqi target — prefer API-provided value, fallback to manual calc.

    The API-provided us_aqi from Open-Meteo uses proper EPA averaging periods:
      - PM2.5, PM10: 24-hour rolling average
      - O3, CO: 8-hour rolling average
      - NO2, SO2: 1-hour instantaneous
    This is MORE ACCURATE than our manual calculation which uses instantaneous
    values for all pollutants.
    """
    if 'us_aqi' in df.columns:
        # API-provided AQI exists — use it, fill gaps with manual calc
        api_valid = df['us_aqi'].notna().sum()
        api_missing = df['us_aqi'].isna().sum()

        if api_missing > 0:
            missing_mask = df['us_aqi'].isna()
            manual = df[missing_mask].apply(calculate_us_aqi, axis=1)
            df.loc[missing_mask, 'us_aqi'] = manual.apply(lambda x: x[0])
            if verbose:
                print(f"    ✓ US AQI: {api_valid} from API (proper EPA rolling avg) "
                      f"+ {api_missing} manual fallback")
        else:
            if verbose:
                print(f"    ✓ US AQI: {api_valid} from API (proper EPA rolling averages)")

        # Determine dominant pollutant from sub-indices
        sub_cols = [c for c in ['us_aqi_pm2_5', 'us_aqi_pm10',
                     'us_aqi_nitrogen_dioxide', 'us_aqi_ozone',
                     'us_aqi_sulphur_dioxide', 'us_aqi_carbon_monoxide']
                     if c in df.columns]
        if sub_cols:
            sub_df = df[sub_cols].copy()
            dominant = sub_df.idxmax(axis=1)
            name_map = {
                'us_aqi_pm2_5': 'PM2.5', 'us_aqi_pm10': 'PM10',
                'us_aqi_nitrogen_dioxide': 'NO2', 'us_aqi_ozone': 'O3',
                'us_aqi_sulphur_dioxide': 'SO2', 'us_aqi_carbon_monoxide': 'CO'
            }
            df['dominant_pollutant'] = dominant.map(name_map).fillna('N/A')
        else:
            aqi_results = df.apply(calculate_us_aqi, axis=1)
            df['dominant_pollutant'] = aqi_results.apply(lambda x: x[1])
    else:
        # No API AQI at all — compute manually
        if verbose:
            print("    ⚠ No API us_aqi found — computing manually (less accurate)")
        aqi_results = df.apply(calculate_us_aqi, axis=1)
        df['us_aqi'] = aqi_results.apply(lambda x: x[0])
        df['dominant_pollutant'] = aqi_results.apply(lambda x: x[1])

    if verbose:
        valid = df['us_aqi'].notna().sum()
        print(f"    ✓ Final AQI: {valid}/{len(df)} valid "
              f"(mean={df['us_aqi'].mean():.1f}, range="
              f"[{df['us_aqi'].min():.1f}, {df['us_aqi'].max():.1f}])")

    return df


# ═══════════════════════════════════════════════════════════════
# INDIVIDUAL FEATURE ENGINEERING STEPS
# ═══════════════════════════════════════════════════════════════

def add_wind_components(df):
    """Decompose wind into u/v components (better for ML than speed+direction)."""
    if 'windspeed_10m' in df.columns and 'winddirection_10m' in df.columns:
        rad = np.deg2rad(df['winddirection_10m'])
        df['wind_speed'] = df['windspeed_10m']
        df['wind_u'] = -df['windspeed_10m'] * np.sin(rad)
        df['wind_v'] = -df['windspeed_10m'] * np.cos(rad)
    return df


def add_time_features(df):
    """Cyclical time encoding — deterministic, drives diurnal AQI patterns."""
    dt = df['datetime']
    hour = dt.dt.hour
    dow  = dt.dt.dayofweek
    month = dt.dt.month
    doy  = dt.dt.dayofyear

    df['hour_sin']         = np.sin(2 * np.pi * hour / 24)
    df['hour_cos']         = np.cos(2 * np.pi * hour / 24)
    df['day_of_week_sin']  = np.sin(2 * np.pi * dow / 7)
    df['day_of_week_cos']  = np.cos(2 * np.pi * dow / 7)
    df['month_sin']        = np.sin(2 * np.pi * month / 12)
    df['month_cos']        = np.cos(2 * np.pi * month / 12)
    df['day_of_year_sin']  = np.sin(2 * np.pi * doy / 365)
    df['day_of_year_cos']  = np.cos(2 * np.pi * doy / 365)
    df['is_weekend']       = (dow >= 5).astype(float)
    return df


def add_weather_derivatives(df):
    """Rolling stats + lags for weather variables.

    These are SAFE at forecast time (weather forecast APIs provide them)
    and CHANGE during the forecast → prevent flat predictions.

    v2: Added wind_gusts_10m, shortwave_radiation, vapour_pressure_deficit,
        pressure_msl for better pollutant dispersion modeling.
    """
    weather_cols = [c for c in [
        'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
        'wind_u', 'wind_v', 'cloud_cover', 'dew_point_2m',
        # NEW in v2
        'wind_gusts_10m', 'shortwave_radiation',
        'vapour_pressure_deficit', 'pressure_msl',
    ] if c in df.columns]

    for col in weather_cols:
        # Rolling means
        for w in [6, 12, 24]:
            df[f'{col}_rolling_mean_{w}h'] = df[col].rolling(w, min_periods=1).mean()
        # Rolling std (24h) — weather volatility
        df[f'{col}_rolling_std_24h'] = df[col].rolling(24, min_periods=1).std()
        # Lags
        for lag in [12, 24]:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

    return df


def add_atmospheric_derivatives(df):
    """Rolling stats + lags for atmospheric variables.

    v2: Added uv_index_clear_sky, carbon_dioxide.
    """
    atm_cols = [c for c in [
        'aerosol_optical_depth', 'dust', 'uv_index',
        # NEW in v2
        'uv_index_clear_sky', 'carbon_dioxide',
    ] if c in df.columns]

    for col in atm_cols:
        for w in [6, 12, 24]:
            df[f'{col}_rolling_mean_{w}h'] = df[col].rolling(w, min_periods=1).mean()
        for lag in [12, 24]:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

    return df


def add_subaqi_features(df):
    """AQI sub-index features from API — which pollutant drives AQI.

    The sub-indices (us_aqi_pm2_5, us_aqi_pm10, etc.) are EPA-calculated
    AQI contributions with proper rolling averages. They tell the model:
    - Which pollutant is currently dominant
    - How each pollutant's AQI contribution is trending

    These are available at forecast launch time (current + historical)
    and provide crucial compositional information for prediction.
    """
    sub_aqi_cols = [c for c in [
        'us_aqi_pm2_5', 'us_aqi_pm10', 'us_aqi_nitrogen_dioxide',
        'us_aqi_ozone', 'us_aqi_sulphur_dioxide', 'us_aqi_carbon_monoxide'
    ] if c in df.columns]

    if not sub_aqi_cols:
        return df

    for col in sub_aqi_cols:
        # Lags — what was this pollutant's AQI contribution N hours ago?
        for lag in [6, 12, 24]:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        # Rolling mean — average contribution over window
        for w in [12, 24]:
            df[f'{col}_rolling_mean_{w}h'] = df[col].rolling(w, min_periods=1).mean()

    return df


def add_interactions(df):
    """Weather interaction features — capture non-linear pollution dynamics.

    v2: Added radiation×aerosol and VPD×temperature interactions.
    """
    if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
        df['humidity_temp_interaction'] = (
            df['relative_humidity_2m'] * df['temperature_2m']
        )

    if 'temperature_2m' in df.columns and 'surface_pressure' in df.columns:
        df['temp_pressure_interaction'] = (
            df['temperature_2m'] * df['surface_pressure'] / 1000.0
        )

    if 'wind_speed' in df.columns and 'relative_humidity_2m' in df.columns:
        df['wind_humidity_interaction'] = (
            df['wind_speed'] * df['relative_humidity_2m']
        )

    if 'cloud_cover' in df.columns and 'temperature_2m' in df.columns:
        df['cloud_temp_interaction'] = (
            df['cloud_cover'] * df['temperature_2m']
        )

    if 'aerosol_optical_depth' in df.columns and 'relative_humidity_2m' in df.columns:
        df['aerosol_humidity_interaction'] = (
            df['aerosol_optical_depth'] * df['relative_humidity_2m']
        )

    # NEW in v2
    if 'shortwave_radiation' in df.columns and 'aerosol_optical_depth' in df.columns:
        df['radiation_aerosol_interaction'] = (
            df['shortwave_radiation'] * df['aerosol_optical_depth']
        )

    if 'vapour_pressure_deficit' in df.columns and 'temperature_2m' in df.columns:
        df['vpd_temp_interaction'] = (
            df['vapour_pressure_deficit'] * df['temperature_2m']
        )

    return df


def add_aqi_autoregressive(df):
    """AQI autoregressive features — strongest short-term predictors.

    At forecast time t, we KNOW AQI(t), AQI(t-1), ..., AQI(t-24).
    These fixed features + horizon encoding tell the model how much
    to trust persistence vs. regression to the mean.
    """
    if 'us_aqi' not in df.columns:
        return df

    # Lags
    for lag in [1, 3, 6, 12, 24]:
        df[f'us_aqi_lag_{lag}h'] = df['us_aqi'].shift(lag)

    # Rolling stats
    for w in [6, 12, 24]:
        df[f'us_aqi_rolling_mean_{w}h'] = df['us_aqi'].rolling(w, min_periods=1).mean()
    df['us_aqi_rolling_std_6h']  = df['us_aqi'].rolling(6, min_periods=1).std()
    df['us_aqi_rolling_std_24h'] = df['us_aqi'].rolling(24, min_periods=1).std()

    # Trends
    df['us_aqi_delta_1h'] = df['us_aqi'] - df['us_aqi'].shift(1)
    df['us_aqi_delta_6h'] = (df['us_aqi'] - df['us_aqi'].shift(6)) / 6.0

    return df


# ═══════════════════════════════════════════════════════════════
# FEATURE SELECTION HELPER
# ═══════════════════════════════════════════════════════════════

# Raw pollutant columns — used for AQI reference, NOT model features
RAW_POLLUTANTS = {'pm2_5', 'pm10', 'ozone', 'nitrogen_dioxide',
                  'sulphur_dioxide', 'carbon_monoxide'}

# Columns to exclude from model input
# NOTE: us_aqi_pm2_5, us_aqi_pm10, etc. are NOT excluded — they are valid
# features (EPA-processed sub-indices, not raw concentrations)
EXCLUDE_FROM_MODEL = (
    RAW_POLLUTANTS |
    {'us_aqi', 'datetime', 'dominant_pollutant',
     'location', 'latitude', 'longitude',
     'windspeed_10m', 'winddirection_10m',
     'rain', 'snowfall', 'apparent_temperature'}
)


def get_model_feature_names(df):
    """Return list of features safe for AQI prediction (no leakage).

    Excludes:
      - Target: us_aqi
      - Metadata: datetime, dominant_pollutant, location, latitude, longitude
      - Raw pollutants: pm2_5, pm10, ozone, no2, so2, co
        (these DEFINE AQI — using them is circular)
      - Redundant: windspeed_10m/direction (u/v), rain≈precipitation,
        snowfall (0 in Karachi), apparent_temperature (≈temperature_2m)

    INCLUDES (valid features):
      - us_aqi_pm2_5, us_aqi_pm10, etc. (EPA sub-indices, not raw pollutants)
      - All weather, atmospheric, time, interaction features
      - AQI autoregressive features (lags, rolling, deltas)
    """
    return [c for c in df.columns
            if c not in EXCLUDE_FROM_MODEL
            and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]


def categorize_features(feature_names):
    """Categorize features for reporting/diagnostics."""
    categories = {
        'weather': [], 'atmospheric': [], 'sub_aqi': [],
        'time': [], 'interaction': [], 'aqi_ar': [], 'other': []
    }

    WEATHER_KEYS = ['temperature', 'humidity', 'pressure', 'wind',
                    'cloud', 'dew_point', 'shortwave', 'vapour',
                    'precipitation']
    ATM_KEYS = ['aerosol', 'dust', 'uv_index', 'carbon_dioxide']
    SUB_AQI_KEYS = ['us_aqi_pm', 'us_aqi_nitrogen', 'us_aqi_ozone',
                    'us_aqi_sulphur', 'us_aqi_carbon_monoxide']
    TIME_KEYS = ['hour_sin', 'hour_cos', 'day_of_week', 'month_sin',
                 'month_cos', 'day_of_year', 'is_weekend']

    for feat in feature_names:
        fl = feat.lower()
        if 'interaction' in fl:
            categories['interaction'].append(feat)
        elif any(k in fl for k in SUB_AQI_KEYS):
            categories['sub_aqi'].append(feat)
        elif 'us_aqi_' in fl:
            categories['aqi_ar'].append(feat)
        elif any(k in fl for k in ATM_KEYS):
            categories['atmospheric'].append(feat)
        elif any(k in fl for k in WEATHER_KEYS):
            categories['weather'].append(feat)
        elif any(k in fl for k in TIME_KEYS):
            categories['time'].append(feat)
        else:
            categories['other'].append(feat)

    return categories


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def engineer_features(df, verbose=True):
    """Full feature engineering pipeline (v2).

    Input:  Raw data with weather + air quality columns from API
            (including API-provided us_aqi and sub-indices)
    Output: Feature-engineered DataFrame with:
              - us_aqi (target — from API with proper EPA rolling averages)
              - dominant_pollutant (metadata — derived from sub-indices)
              - ~120+ model-ready features across 7 categories
              - Raw pollutant columns (kept for reference, not model features)

    Every feature is designed to be AVAILABLE at forecast time:
      ⏰ Time       → deterministic (known for any future hour)
      🌤️ Weather    → from weather forecast APIs
      🌫️ Atmospheric → from atmospheric forecast APIs
      📊 Sub-AQI    → EPA sub-indices (known at launch time)
      🔗 Interaction → computed from weather
      📈 AQI-AR     → known at forecast launch time
    """
    if verbose:
        print(f"  Starting feature engineering (v2) on {len(df)} rows...")

    # 1. Set AQI target — prefer API-provided (proper EPA rolling averages)
    df = add_aqi_target(df, verbose=verbose)

    # 2. Wind decomposition (u, v components)
    df = add_wind_components(df)
    if verbose:
        print(f"    ✓ Wind decomposed into u/v components")

    # 3. Time features (cyclical)
    df = add_time_features(df)
    if verbose:
        print(f"    ✓ Time features: 9 cyclical encodings")

    # 4. Weather derivatives (rolling/lags) — now includes gusts, radiation, VPD
    df = add_weather_derivatives(df)
    if verbose:
        n_wd = sum(1 for c in df.columns
                   if any(w in c for w in ['temperature', 'humidity', 'pressure',
                                           'wind_u', 'wind_v', 'cloud', 'dew_point',
                                           'wind_gusts', 'shortwave', 'vapour'])
                   and ('rolling' in c or 'lag' in c))
        print(f"    ✓ Weather derivatives: {n_wd} features")

    # 5. Atmospheric derivatives — now includes CO2, UV clear sky
    df = add_atmospheric_derivatives(df)
    if verbose:
        n_ad = sum(1 for c in df.columns
                   if any(a in c for a in ['aerosol', 'dust', 'uv_index',
                                           'carbon_dioxide'])
                   and ('rolling' in c or 'lag' in c))
        print(f"    ✓ Atmospheric derivatives: {n_ad} features")

    # 6. Sub-AQI index features (NEW in v2)
    df = add_subaqi_features(df)
    if verbose:
        n_sub = sum(1 for c in df.columns
                    if any(k in c for k in ['us_aqi_pm', 'us_aqi_nitrogen',
                                            'us_aqi_ozone', 'us_aqi_sulphur',
                                            'us_aqi_carbon_monoxide'])
                    and ('lag' in c or 'rolling' in c))
        n_sub_raw = sum(1 for c in df.columns
                        if c.startswith('us_aqi_') and 'lag' not in c
                        and 'rolling' not in c and 'delta' not in c
                        and 'std' not in c and 'mean' not in c)
        print(f"    ✓ Sub-AQI features: {n_sub_raw} raw + {n_sub} derivatives")

    # 7. Interaction features — now includes radiation×aerosol, VPD×temp
    df = add_interactions(df)
    if verbose:
        n_int = sum(1 for c in df.columns if 'interaction' in c)
        print(f"    ✓ Interactions: {n_int} features")

    # 8. AQI autoregressive features
    df = add_aqi_autoregressive(df)
    if verbose:
        n_ar = sum(1 for c in df.columns
                   if c.startswith('us_aqi_lag_') or c.startswith('us_aqi_rolling_')
                   or c.startswith('us_aqi_delta'))
        print(f"    ✓ AQI autoregressive: {n_ar} features")

    # 9. Fill NaN from lags/rolling (forward then backward fill)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # 10. Drop rows where AQI couldn't be calculated
    before = len(df)
    df = df.dropna(subset=['us_aqi'])
    if verbose and before != len(df):
        print(f"    ✓ Dropped {before - len(df)} rows with invalid AQI")

    # 11. Summary
    if verbose:
        model_feats = get_model_feature_names(df)
        cats = categorize_features(model_feats)
        print(f"\n    Final shape: {df.shape}")
        print(f"    Model features: {len(model_feats)}")
        for cat, feats in cats.items():
            if feats:
                print(f"      {cat:>14}: {len(feats)}")

    return df
