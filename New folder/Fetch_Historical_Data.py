import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_monthly_data(latitude, longitude, start_date, end_date):
    """Fetch one month of weather and pollutant data from Open-Meteo archive APIs."""

    # --- WEATHER (ARCHIVE API) ---
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "surface_pressure", "cloud_cover", "windspeed_10m", "winddirection_10m"
        ],
        "timezone": "auto"
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

    # --- AIR QUALITY (ARCHIVE API) ---
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
            "aerosol_optical_depth", "dust", "uv_index"
        ],
        "timezone": "auto"
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

    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")
    return merged


def fetch_yearly_data(latitude, longitude, start_year, end_year):
    """Fetch data month-by-month for a full year to avoid API limits."""
    all_data = []

    start = datetime(start_year, 10, 21)
    end = datetime(end_year, 10, 21)

    current = start
    while current < end:
        month_end = (current + timedelta(days=30))
        if month_end > end:
            month_end = end

        print(f"Fetching data from {current.date()} to {month_end.date()}...")
        df = fetch_monthly_data(latitude, longitude, current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
        all_data.append(df)
        current = month_end + timedelta(days=1)

    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    latitude = 24.8607
    longitude = 67.0011

    print("🌤 Fetching 1 year of data for Karachi...")
    df = fetch_yearly_data(latitude, longitude, 2024, 2025)

    print(f"\n✅ Total records fetched: {len(df)}")
    print(df.head())

    df.to_csv("karachi_weather_pollutants_2024_2025.csv", index=False)
    print("📁 Saved as karachi_weather_pollutants_2024_2025.csv")
