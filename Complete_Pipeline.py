# -----------------------------
# Pearls AQI Predictor — Batch Pipeline (v2)
# Orchestrates: Fetch → Feature Engineer → Upload to MongoDB
# v2: Uses API-provided US AQI + sub-indices, expanded weather variables
# Run this to populate the feature store from scratch
# -----------------------------

from Fetch_Historical_Data import fetch_range_data, LATITUDE, LONGITUDE
from feature_engineering import engineer_features
from mongodb_upload import upload_dataframe, clear_collection, get_collection_info
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def run_pipeline(start_date=None, end_date=None,
                 clear_first=False, mongodb_uri=None):
    """Full batch pipeline: Fetch → Engineer → Upload.

    Parameters:
        start_date:   Start date for data fetch (YYYY-MM-DD, default: 10 months ago)
        end_date:     End date (None = one day before current date)
        clear_first:  If True, delete all existing data before uploading
        mongodb_uri:  Override MongoDB URI (uses default if None)
    """
    # Set end_date to one day before current date if not specified
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Default start_date: 10 months before current date
    if start_date is None:
        start_date = (datetime.now() - relativedelta(months=10)).strftime('%Y-%m-%d')
    
    print("=" * 70)
    print(" Pearls AQI Predictor — Batch Pipeline")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Step 1: Fetch data ──
    print(f"\n[1/3] FETCHING DATA ({start_date} to {end_date})")
    print("-" * 50)

    df = fetch_range_data(LATITUDE, LONGITUDE, start_date, end_date)

    if df.empty:
        print("No data fetched. Aborting.")
        return None

    print(f"\n  ✓ Fetched {len(df):,} rows")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Columns: {len(df.columns)}")

    # ── Step 2: Feature engineering ──
    print(f"\n[2/3] FEATURE ENGINEERING")
    print("-" * 50)

    df = engineer_features(df, verbose=True)

    print(f"\n  ✓ Engineered {df.shape[1]} columns for {len(df):,} rows")

    # Save CSV backup
    csv_path = "karachi_aqi_features_engineered.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved backup to {csv_path}")

    # ── Step 3: Upload to MongoDB ──
    print(f"\n[3/3] UPLOADING TO MONGODB")
    print("-" * 50)

    if clear_first:
        print("  Clearing existing collection...")
        clear_collection(uri=mongodb_uri)

    kwargs = {}
    if mongodb_uri:
        kwargs['uri'] = mongodb_uri

    inserted, skipped, total = upload_dataframe(df, **kwargs)

    # ── Summary ──
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  Records fetched:    {len(df):,}")
    print(f"  Records uploaded:   {inserted:,}")
    print(f"  Duplicates skipped: {skipped:,}")
    print(f"  Total in MongoDB:   {total:,}")
    print(f"  Features:           {len(df.columns)}")
    print(f"  AQI range:          [{df['us_aqi'].min():.1f}, {df['us_aqi'].max():.1f}]")
    print("=" * 70)

    return df


if __name__ == "__main__":
    run_pipeline()
