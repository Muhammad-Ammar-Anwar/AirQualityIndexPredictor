import requests
import pandas as pd
from datetime import datetime
import os
import hopsworks
from dotenv import load_dotenv
import pytz


# === Load environment variables ===
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


import requests
import pandas as pd
from datetime import datetime
import pytz

def fetch_current_hour_data(latitude, longitude):
    """Fetch live (current hour) weather and air quality data using forecast API."""
    from datetime import datetime
    import requests
    import pandas as pd

    # --- WEATHER DATA ---
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "surface_pressure", "cloud_cover", "windspeed_10m", "winddirection_10m"
        ],
        "timezone": "auto",
        "forecast_days": 1
    }

    w_resp = requests.get(weather_url, params=weather_params)
    w_resp.raise_for_status()
    w_data = w_resp.json()

    weather_df = pd.DataFrame({
        "datetime": w_data["hourly"]["time"],
        "temperature_2m": w_data["hourly"]["temperature_2m"],
        "relative_humidity_2m": w_data["hourly"]["relative_humidity_2m"],
        "dew_point_2m": w_data["hourly"]["dew_point_2m"],
        "apparent_temperature": w_data["hourly"]["apparent_temperature"],
        "precipitation": w_data["hourly"]["precipitation"],
        "rain": w_data["hourly"]["rain"],
        "snowfall": w_data["hourly"]["snowfall"],
        "surface_pressure": w_data["hourly"]["surface_pressure"],
        "cloud_cover": w_data["hourly"]["cloud_cover"],
        "windspeed_10m": w_data["hourly"]["windspeed_10m"],
        "winddirection_10m": w_data["hourly"]["winddirection_10m"]
    })
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    # --- AIR QUALITY DATA ---
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "aerosol_optical_depth",
            "dust", "uv_index"
        ],
        "timezone": "auto",
        "forecast_days": 1
    }

    aq_resp = requests.get(aq_url, params=aq_params)
    aq_resp.raise_for_status()
    aq_data = aq_resp.json()

    aq_df = pd.DataFrame({
        "datetime": aq_data["hourly"]["time"],
        "pm10": aq_data["hourly"]["pm10"],
        "pm2_5": aq_data["hourly"]["pm2_5"],
        "carbon_monoxide": aq_data["hourly"]["carbon_monoxide"],
        "nitrogen_dioxide": aq_data["hourly"]["nitrogen_dioxide"],
        "sulphur_dioxide": aq_data["hourly"]["sulphur_dioxide"],
        "ozone": aq_data["hourly"]["ozone"],
        "aerosol_optical_depth": aq_data["hourly"]["aerosol_optical_depth"],
        "dust": aq_data["hourly"]["dust"],
        "uv_index": aq_data["hourly"]["uv_index"]
    })
    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    # --- MERGE BOTH ---
    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")

    # Filter to most recent hour (≤ now)
    now = datetime.now()
    current_hour = merged[merged["datetime"] <= now].sort_values("datetime").tail(1)

    return current_hour


def append_to_csv(df_now, csv_path):
    """Append new data to CSV if not already present."""
    if not df_now.empty:
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])

            if df_now.iloc[0]["datetime"] not in existing_df["datetime"].values:
                updated_df = pd.concat([existing_df, df_now], ignore_index=True)
                updated_df = updated_df.sort_values("datetime")
                updated_df.to_csv(csv_path, index=False)
                print(f"📈 Data appended to {csv_path}")
            else:
                print("ℹ️ This hour's data already exists in the CSV — no duplicate added.")
        else:
            df_now.to_csv(csv_path, index=False)
            print(f"📁 Created new file {csv_path}")


def push_to_hopsworks(df_now):
    """Push the current hour data to Hopsworks Feature Store."""
    import hopsworks
    import pandas as pd
    import os
    from dotenv import load_dotenv

    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

    print("\n🚀 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    feature_group = fs.get_feature_group(name="weather_data_2", version=1)

    # --- SCHEMA FIXES ---
    df_now = df_now.copy()

    # 1️⃣ Datetime: convert to UTC timestamp
    df_now["datetime"] = pd.to_datetime(df_now["datetime"], errors="coerce", utc=True)
    df_now["datetime"] = df_now["datetime"].astype("datetime64[ns, UTC]")

    # 2️⃣ Convert all other numeric columns to float64 (double)
    float_cols = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "precipitation", "rain", "snowfall",
        "surface_pressure", "cloud_cover", "windspeed_10m", "winddirection_10m",
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "uv_index"
    ]

    for col in float_cols:
        if col in df_now.columns:
            df_now[col] = pd.to_numeric(df_now[col], errors="coerce").astype("float64")

    # --- Debugging Info ---
    print("\n🧾 DataFrame dtypes before upload:")
    print(df_now.dtypes)
    print("\n🔍 Preview:")
    print(df_now.head(1))
    

    from pandas.api.types import is_datetime64_any_dtype, is_float_dtype

    assert is_datetime64_any_dtype(df_now["datetime"]), "❌ datetime not timestamp"
    for c in float_cols:
        if c in df_now.columns:
            assert is_float_dtype(df_now[c]), f"❌ {c} not float64"
    print("\n✅ Schema verified. Uploading to Hopsworks...")
    feature_group.insert(df_now)
    print("🎯 Successfully inserted into Feature Group!")


if __name__ == "__main__":
    latitude = 24.8607  # Karachi
    longitude = 67.0011
    csv_path = "karachi_weather_pollutants_2024_2025.csv"

    print("🌤 Fetching real-time (latest hour) weather and air quality data for Karachi...")
    df_now = fetch_current_hour_data(latitude, longitude)

    if not df_now.empty:
        print("\n✅ Current Hour Data:\n")
        print(df_now.to_string(index=False))

        # Append to CSV
        append_to_csv(df_now, csv_path)

        # Push to Hopsworks
        push_to_hopsworks(df_now)

    else:
        print("⚠️ No data available yet for the current hour.")

    print("✅ Done!")
