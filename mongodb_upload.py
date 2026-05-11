# -----------------------------
# Pearls AQI Predictor — MongoDB Upload Module
# Uploads feature-engineered data to MongoDB Atlas
# Handles deduplication via datetime unique index
# -----------------------------

import numpy as np
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ASCENDING, errors


# Default MongoDB settings
DEFAULT_URI = ('MONGODB_URI')
DEFAULT_DB = "aqi_feature_store"
DEFAULT_COLLECTION = "karachi_aqi_features"


def get_client(uri=None):
    """Create and verify MongoDB connection."""
    uri = uri or DEFAULT_URI
    client = MongoClient(uri, server_api=ServerApi('1'),
                         serverSelectionTimeoutMS=10000)
    client.admin.command('ping')
    return client


def prepare_record(row_dict):
    """Convert a row dict to MongoDB-safe types (no numpy/NaN issues)."""
    clean = {}
    for k, v in row_dict.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        elif isinstance(v, pd.Timestamp):
            clean[k] = v.to_pydatetime()
        elif pd.api.types.is_scalar(v) and pd.isna(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean


def upload_dataframe(df, uri=None, db_name=None, collection_name=None,
                     batch_size=500):
    """Upload a DataFrame to MongoDB with deduplication on datetime.

    Parameters:
        df:              Feature-engineered DataFrame with 'datetime' column
        uri:             MongoDB connection URI
        db_name:         Database name
        collection_name: Collection name
        batch_size:      Number of documents per insert batch

    Returns:
        (inserted_count, skipped_count, total_in_collection)
    """
    uri = uri or DEFAULT_URI
    db_name = db_name or DEFAULT_DB
    collection_name = collection_name or DEFAULT_COLLECTION

    print(f"\n[MongoDB] Connecting to {db_name}.{collection_name} ...")
    client = get_client(uri)
    db = client[db_name]
    collection = db[collection_name]

    # Ensure unique index on datetime
    collection.create_index([("datetime", ASCENDING)], unique=True)

    # Prepare records
    records = []
    for _, row in df.iterrows():
        record = prepare_record(row.to_dict())
        records.append(record)

    # Batch insert with dedup
    inserted = 0
    skipped = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)
        except errors.BulkWriteError as bwe:
            # Some docs may be duplicates
            n_inserted = bwe.details.get('nInserted', 0)
            n_errors = len(bwe.details.get('writeErrors', []))
            inserted += n_inserted
            skipped += n_errors
        except Exception as e:
            print(f"  ERROR in batch {i//batch_size}: {e}")
            skipped += len(batch)

        if (i + batch_size) % 2000 == 0 or i + batch_size >= len(records):
            print(f"  Progress: {min(i + batch_size, len(records))}/{len(records)}")

    total = collection.count_documents({})
    client.close()

    print(f"\n[MongoDB] Upload complete:")
    print(f"  Inserted: {inserted}")
    print(f"  Skipped (duplicates): {skipped}")
    print(f"  Total in collection: {total}")

    return inserted, skipped, total


def clear_collection(uri=None, db_name=None, collection_name=None):
    """Delete all documents from a collection (for fresh re-upload)."""
    uri = uri or DEFAULT_URI
    db_name = db_name or DEFAULT_DB
    collection_name = collection_name or DEFAULT_COLLECTION

    client = get_client(uri)
    db = client[db_name]
    collection = db[collection_name]

    count = collection.count_documents({})
    result = collection.delete_many({})
    client.close()

    print(f"[MongoDB] Cleared {result.deleted_count} documents from "
          f"{db_name}.{collection_name}")
    return result.deleted_count


def get_collection_info(uri=None, db_name=None, collection_name=None):
    """Get basic info about the MongoDB collection."""
    uri = uri or DEFAULT_URI
    db_name = db_name or DEFAULT_DB
    collection_name = collection_name or DEFAULT_COLLECTION

    client = get_client(uri)
    db = client[db_name]
    collection = db[collection_name]

    total = collection.count_documents({})

    # Get date range
    oldest = collection.find_one(sort=[("datetime", 1)])
    newest = collection.find_one(sort=[("datetime", -1)])

    # Get column count from a sample
    sample = collection.find_one()
    n_cols = len(sample.keys()) if sample else 0

    client.close()

    info = {
        'total_documents': total,
        'n_columns': n_cols,
        'oldest': oldest['datetime'] if oldest else None,
        'newest': newest['datetime'] if newest else None,
    }

    print(f"\n[MongoDB] Collection: {db_name}.{collection_name}")
    print(f"  Documents: {total}")
    print(f"  Columns: {n_cols}")
    if oldest:
        print(f"  Date range: {oldest['datetime']} to {newest['datetime']}")

    return info

