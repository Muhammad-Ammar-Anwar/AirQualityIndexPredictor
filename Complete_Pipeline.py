# -----------------------------
# Pearls AQI Predictor — Complete Pipeline
#
# Flow:
#   1. Fetch historical data from Open-Meteo API (2022-01-01 → yesterday)
#   2. Also read any existing data from MongoDB weather_data
#   3. Merge both sources, deduplicate on datetime
#   4. Engineer 120+ features
#   5. Save to karachi_aqi_features_engineered.csv
#   6. Upload to MongoDB: AQI_Project.karachi_aqi_features
#
# This ensures the CSV always has the full history needed for model training.
# Run this once to backfill, then Model_Retrain_pipeline.py reads the CSV.
# -----------------------------

import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from feature_engineering import engineer_features
from mongodb_upload import upload_dataframe, clear_collection
from Fetch_Historical_Data import fetch_range_data, LATITUDE, LONGITUDE
from datetime import datetime, timedelta

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
MONGODB_URI     = os.getenv("MONGODB_URI")
MONGODB_DB      = os.getenv("MONGODB_DB", "AQI_Project")
RAW_COLLECTION  = os.getenv("MONGODB_COLLECTION", "weather_data")
FEAT_COLLECTION = os.getenv("MONGODB_FEATURES_COLLECTION", "karachi_aqi_features")
CSV_PATH        = "karachi_aqi_features_engineered.csv"
HISTORY_START   = "2022-01-01"   # 4 years — needed for long-horizon accuracy


def fetch_from_api(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical weather + AQ data from Open-Meteo API."""
    print(f"  Fetching from Open-Meteo API: {start_date} → {end_date}")
    df = fetch_range_data(LATITUDE, LONGITUDE, start_date, end_date)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        print(f"  API: {len(df):,} rows fetched")
    return df


def fetch_from_mongodb() -> pd.DataFrame:
    """Read raw records from MongoDB weather_data collection."""
    try:
        print(f"  Reading from MongoDB: {MONGODB_DB}.{RAW_COLLECTION}")
        client = MongoClient(MONGODB_URI, server_api=ServerApi("1"),
                             serverSelectionTimeoutMS=8_000,
                             socketTimeoutMS=30_000)
        client.admin.command("ping")
        col   = client[MONGODB_DB][RAW_COLLECTION]
        total = col.count_documents({})

        if total == 0:
            print("  MongoDB weather_data is empty — skipping")
            client.close()
            return pd.DataFrame()

        print(f"  Fetching {total:,} documents …")
        records = []
        for i, doc in enumerate(col.find({}, {"_id": 0}).batch_size(500), 1):
            records.append(doc)
            if i % 1000 == 0:
                print(f"    {i:,} / {total:,} …")

        client.close()
        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        print(f"  MongoDB: {len(df):,} rows loaded")
        return df

    except Exception as e:
        print(f"  WARNING: MongoDB fetch failed or timed out ({e})")
        print(f"  Continuing with API data only …")
        return pd.DataFrame()


def run_pipeline(start_date: str = HISTORY_START,
                 end_date: str = None,
                 clear_first: bool = False,
                 skip_api: bool = False) -> pd.DataFrame:
    """
    Full pipeline: fetch → merge → engineer → save CSV → upload to MongoDB.

    Parameters:
        start_date:  Start of historical fetch from API (default: 2022-01-01)
        end_date:    End date (default: yesterday)
        clear_first: Wipe karachi_aqi_features before uploading
        skip_api:    Skip API fetch, use only MongoDB weather_data
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 70)
    print(" Pearls AQI Predictor — Complete Pipeline")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Step 1: Collect raw data ───────────────────────────────────────────
    print(f"\n[1/3] COLLECTING RAW DATA")
    print("-" * 50)

    frames = []

    # API historical data (primary — full history from 2022)
    if not skip_api:
        df_api = fetch_from_api(start_date, end_date)
        if not df_api.empty:
            frames.append(df_api)

    # MongoDB weather_data (secondary — recent hourly upserts)
    df_mongo = fetch_from_mongodb()
    if not df_mongo.empty:
        frames.append(df_mongo)

    if not frames:
        print("No data available from any source. Aborting.")
        return None

    # Merge and deduplicate on datetime
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    print(f"\n  ✓ Combined: {before:,} rows → {len(df):,} after dedup")
    print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Columns: {len(df.columns)}")

    # ── Step 2: Feature engineering ───────────────────────────────────────
    print(f"\n[2/3] FEATURE ENGINEERING")
    print("-" * 50)

    df = engineer_features(df, verbose=True)
    print(f"\n  ✓ Engineered {df.shape[1]} columns for {len(df):,} rows")

    # Save CSV — this is what Model_Retrain_pipeline.py reads
    df.to_csv(CSV_PATH, index=False)
    print(f"  ✓ Saved to {CSV_PATH}")

    # ── Step 3: Upload to MongoDB ──────────────────────────────────────────
    print(f"\n[3/3] UPLOADING TO MONGODB ({MONGODB_DB}.{FEAT_COLLECTION})")
    print("-" * 50)

    if clear_first:
        print("  Clearing existing collection …")
        clear_collection(collection_name=FEAT_COLLECTION)

    inserted, skipped, total = upload_dataframe(
        df, collection_name=FEAT_COLLECTION
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  Date range:         {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Rows processed:     {len(df):,}")
    print(f"  Features:           {df.shape[1]}")
    print(f"  AQI range:          [{df['us_aqi'].min():.1f}, {df['us_aqi'].max():.1f}]")
    print(f"  CSV saved to:       {CSV_PATH}")
    print(f"  MongoDB inserted:   {inserted:,}")
    print(f"  MongoDB skipped:    {skipped:,}")
    print(f"  MongoDB total:      {total:,}")
    print("=" * 70)
    print(f"\n  Next step: python Model_Retrain_pipeline.py")

    return df


if __name__ == "__main__":
    # skip_api=False  → fetch from Open-Meteo API (2022 → yesterday) — full history
    # skip_api=True   → use only MongoDB weather_data (faster, recent data only)
    # Set clear_first=True to wipe karachi_aqi_features and re-upload from scratch
    run_pipeline(start_date="2025-01-01", clear_first=True, skip_api=False)
