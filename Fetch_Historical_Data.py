# -----------------------------
# Pearls AQI Predictor — Data Fetch Module
# Fetches hourly weather + air quality data from Open-Meteo APIs
# Restructured: Uses API-provided US AQI + sub-indices for accuracy
# Weather vars expanded for better pollutant dispersion modeling
# -----------------------------

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011
LOCATION = "Karachi"

# Open-Meteo API endpoints
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ── Weather variables (Historical Archive & Forecast APIs) ──
# Each variable is available from both archive-api and forecast-api
WEATHER_HOURLY_VARS = [
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
# Raw pollutants + API-computed US AQI with proper EPA rolling averages
AIR_QUALITY_HOURLY_VARS = [
    # Raw pollutant concentrations (µg/m³) — used for reference
    "pm10", "pm2_5", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    # Additional atmospheric variables
    "aerosol_optical_depth", "dust",
    "uv_index", "uv_index_clear_sky",
    "carbon_dioxide",
    # API-computed US AQI (proper EPA rolling averages: PM=24h, O3/CO=8h, etc.)
    # Much more accurate than manual instantaneous calculation
    "us_aqi",
    # Individual pollutant AQI sub-indices — shows WHICH pollutant drives AQI
    "us_aqi_pm2_5", "us_aqi_pm10",
    "us_aqi_nitrogen_dioxide", "us_aqi_ozone",
    "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide",
]


def fetch_weather_data(latitude, longitude, start_date, end_date):
    """Fetch hourly weather data from Open-Meteo Archive API."""
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "hourly": WEATHER_HOURLY_VARS, "timezone": "auto"
    }
    resp = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({"datetime": data["hourly"]["time"]})
    for var in WEATHER_HOURLY_VARS:
        df[var] = data["hourly"].get(var)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_air_quality_data(latitude, longitude, start_date, end_date):
    """Fetch hourly air quality data from Open-Meteo Air Quality API."""
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "hourly": AIR_QUALITY_HOURLY_VARS, "timezone": "auto"
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({"datetime": data["hourly"]["time"]})
    for var in AIR_QUALITY_HOURLY_VARS:
        df[var] = data["hourly"].get(var)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_monthly_data(latitude, longitude, start_date, end_date):
    """Fetch one month of merged weather + air quality data."""
    weather_df = fetch_weather_data(latitude, longitude, start_date, end_date)
    aq_df = fetch_air_quality_data(latitude, longitude, start_date, end_date)
    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")
    merged["location"] = LOCATION
    merged["latitude"] = latitude
    merged["longitude"] = longitude
    return merged


def fetch_range_data(latitude, longitude, start_date, end_date=None):
    """
    Fetch data in ~30-day chunks from start_date to end_date (or today).
    Includes rate-limiting (1s sleep between API calls).
    """
    if end_date is None:
        end_date = datetime.utcnow()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    all_data = []
    current = start_date

    while current < end_date:
        chunk_end = current + timedelta(days=30)
        if chunk_end > end_date:
            chunk_end = end_date

        s_str = current.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")
        print(f"  Fetching {s_str} to {e_str} ...")

        try:
            df = fetch_monthly_data(latitude, longitude, s_str, e_str)
            all_data.append(df)
            print(f"    -> {len(df)} records")
        except requests.exceptions.RequestException as e:
            print(f"    -> ERROR: {e}")

        time.sleep(1)
        current = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def fetch_and_save(output_path="karachi_raw_data.csv",
                   start_date="2025-04-14", end_date=None):
    """Main fetch entry point — fetches all data and saves to CSV.
    
    Default: Fetches last 10 months to one day before current date.
    """
    # Set end_date to one day before current date if not specified
    if end_date is None:
        end_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Fetching weather + air quality data for {LOCATION} ...")
    print(f"  Coordinates: ({LATITUDE}, {LONGITUDE})")
    print(f"  Range: {start_date} to {end_date}\n")

    df = fetch_range_data(LATITUDE, LONGITUDE, start_date, end_date)

    if df.empty:
        print("No data fetched.")
        return df

    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}\n")

    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")
    return df


# -------------------------------
if __name__ == "__main__":
    fetch_and_save()
