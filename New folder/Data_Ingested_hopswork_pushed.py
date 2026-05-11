import os
from datetime import datetime

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv


# =========================================================
# Load Environment Variables
# =========================================================
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


# =========================================================
# Fetch Current Hour Weather + AQ Data
# =========================================================
def fetch_current_hour_data(latitude, longitude):

    # ---------------- WEATHER API ----------------
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "surface_pressure",
            "cloud_cover",
            "windspeed_10m",
            "winddirection_10m"
        ],
        "timezone": "auto",
        "forecast_days": 1
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    weather_df = pd.DataFrame({
        "datetime": weather_data["hourly"]["time"],
        "temperature_2m": weather_data["hourly"]["temperature_2m"],
        "relative_humidity_2m": weather_data["hourly"]["relative_humidity_2m"],
        "dew_point_2m": weather_data["hourly"]["dew_point_2m"],
        "apparent_temperature": weather_data["hourly"]["apparent_temperature"],
        "precipitation": weather_data["hourly"]["precipitation"],
        "rain": weather_data["hourly"]["rain"],
        "snowfall": weather_data["hourly"]["snowfall"],
        "surface_pressure": weather_data["hourly"]["surface_pressure"],
        "cloud_cover": weather_data["hourly"]["cloud_cover"],
        "windspeed_10m": weather_data["hourly"]["windspeed_10m"],
        "winddirection_10m": weather_data["hourly"]["winddirection_10m"]
    })

    weather_df["datetime"] = pd.to_datetime(
        weather_df["datetime"]
    )

    # ---------------- AIR QUALITY API ----------------
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "aerosol_optical_depth",
            "dust",
            "uv_index"
        ],
        "timezone": "auto",
        "forecast_days": 1
    }

    aq_response = requests.get(
        aq_url,
        params=aq_params
    )

    aq_response.raise_for_status()

    aq_data = aq_response.json()

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

    aq_df["datetime"] = pd.to_datetime(
        aq_df["datetime"]
    )

    # ---------------- MERGE ----------------
    merged_df = pd.merge(
        weather_df,
        aq_df,
        on="datetime",
        how="inner"
    )

    # Get latest available hour
    now = datetime.now()

    current_hour_df = (
        merged_df[merged_df["datetime"] <= now]
        .sort_values("datetime")
        .tail(1)
    )

    return current_hour_df


# =========================================================
# Append to CSV
# =========================================================
def append_to_csv(df_now, csv_path):

    if df_now.empty:
        return

    if os.path.exists(csv_path):

        existing_df = pd.read_csv(csv_path)

        existing_df["datetime"] = pd.to_datetime(
            existing_df["datetime"]
        )

        if (
            df_now.iloc[0]["datetime"]
            not in existing_df["datetime"].values
        ):

            updated_df = pd.concat(
                [existing_df, df_now],
                ignore_index=True
            )

            updated_df = updated_df.sort_values(
                "datetime"
            )

            updated_df.to_csv(
                csv_path,
                index=False
            )

            print(f"📈 Data appended to {csv_path}")

        else:
            print(
                "ℹ️ This hour data already exists in CSV."
            )

    else:
        df_now.to_csv(csv_path, index=False)

        print(f"📁 Created new file {csv_path}")


# =========================================================
# Push Data to Hopsworks
# =========================================================
def push_to_hopsworks(df_now):

    print("\n🚀 Connecting to Hopsworks...")

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project="AQI_Predictor_Internship",
        api_key_value=HOPSWORKS_API_KEY
    )

    feature_store = project.get_feature_store()

    # Create or get feature group
    feature_group = (
        feature_store.get_or_create_feature_group(
            name="weather_data_2",
            version=1,
            description="Karachi weather and air quality data",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=True
        )
    )

    # Make dataframe copy
    df_now = df_now.copy()

    # ---------------- FIX DATETIME ----------------
    df_now["datetime"] = pd.to_datetime(
        df_now["datetime"],
        errors="coerce"
    )

    # Remove timezone
    df_now["datetime"] = (
        df_now["datetime"]
        .dt.tz_localize(None)
    )

    # ---------------- FIX NUMERIC TYPES ----------------
    float_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloud_cover",
        "windspeed_10m",
        "winddirection_10m",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "aerosol_optical_depth",
        "dust",
        "uv_index"
    ]

    for col in float_columns:

        if col in df_now.columns:

            df_now[col] = pd.to_numeric(
                df_now[col],
                errors="coerce"
            ).astype("float64")

    # ---------------- DEBUG INFO ----------------
    print("\n🧾 DataFrame dtypes before upload:\n")

    print(df_now.dtypes)

    print("\n🔍 Preview:\n")

    print(df_now.head(1))

    # ---------------- INSERT ----------------
    print("\n✅ Uploading to Hopsworks...")

    feature_group.insert(df_now)

    print(
        "\n🎯 Successfully inserted into Hopsworks!"
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    latitude = 24.8607
    longitude = 67.0011

    csv_path = (
        "karachi_weather_pollutants_2024_2025.csv"
    )

    print(
        "🌍 Starting weather and AQ ingestion..."
    )

    print(
        "\n🌤 Fetching real-time weather and AQ data for Karachi..."
    )

    df_now = fetch_current_hour_data(
        latitude,
        longitude
    )

    if not df_now.empty:

        print("\n✅ Current Hour Data:\n")

        print(df_now.to_string(index=False))

        # Save locally
        append_to_csv(
            df_now,
            csv_path
        )

        # Push to Hopsworks
        push_to_hopsworks(df_now)

    else:
        print(
            "⚠️ No current hour data available."
        )

    print("\n✅ Done!")