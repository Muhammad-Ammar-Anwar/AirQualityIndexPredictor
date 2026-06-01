# AQI SeekAI — Karachi Air Quality Intelligence

**End-to-end serverless ML pipeline for real-time AQI monitoring, 72-hour forecasting & manual date prediction — Karachi, Pakistan**

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Vercel-black?logo=vercel)](https://air-quality-index-predictor.vercel.app/)
[![Backend API](https://img.shields.io/badge/Backend_API-Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](https://ammaranwar-aqi-backend.hf.space)
[![Hourly Pipeline](https://img.shields.io/badge/Hourly_Pipeline-active-brightgreen?logo=githubactions)](https://github.com/actions)
[![Retrain](https://img.shields.io/badge/Model_Retrain-every_12h-orange?logo=githubactions)](https://github.com/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![MongoDB](https://img.shields.io/badge/Feature_Store-MongoDB_Atlas-47A248?logo=mongodb&logoColor=white)](https://mongodb.com/atlas)

---

## What It Does

- Collects **35 weather & air quality variables** every hour from Open-Meteo APIs
- Engineers **35+ features** — rolling stats, lags, cyclical time encodings, interaction terms, AQI autoregressive features — stored in MongoDB Atlas
- Trains and compares **3 ML model families** (XGBoost, LightGBM, Random Forest) across 18 forecast horizons every 12 hours — best model selected automatically by RMSE
- Serves a **6-page dark-theme dashboard** (static HTML on Vercel + FastAPI on Hugging Face) with:
  - Real-time AQI gauge with live weather conditions and pollutant breakdown
  - 72-hour ML forecast with short / medium / long band visualization
  - **Manual date prediction** — pick any date (today + next 2 days) and get full 24-hour AQI predictions
  - Historical data explorer with pollutant trends
  - EDA analytics — heatmaps, scatter plots, category breakdowns, dominant pollutant
  - Model performance metrics and architecture details for all 3 models

Everything runs on free-tier services — **$0/month**.

---

## Live Architecture

```
Open-Meteo APIs
├── Weather Forecast API   (18 variables)
└── Air Quality API        (17 variables)
         │
         ▼  GitHub Actions — every hour at :10 UTC
hourly_pipeline.py
  Fetch 3h → Dedup → Engineer 186+ features → Upload to MongoDB
         │
         ▼
MongoDB Atlas
├── AQI_Project.karachi_aqi_features    ← hourly feature store
└── aqi_model_store.AQI_72h_model       ← trained model blobs (zlib + base64)
         │                   ▲
         │                   │  GitHub Actions — every 12h at :30 UTC
         ▼                   │
    backend.py          Model_Retrain_pipeline.py
  (FastAPI on             Phase 1: train all 3 models on probe horizons
  Hugging Face)           Phase 2: fully train winner across 18 horizons
         │                Saves best model + scaler to MongoDB
         ▼
    index.html  (Vercel)
  6-page dashboard — auto-refreshes every 60s
```

---

## Feature Pipeline

### Data Collection

**Weather variables :**
`temperature_2m` · `relative_humidity_2m` · `dew_point_2m` · `apparent_temperature` · `precipitation` · `rain` · `snowfall` · `surface_pressure` · `pressure_msl` · `cloud_cover` · `cloud_cover_low` · `cloud_cover_mid` · `cloud_cover_high` · `windspeed_10m` · `winddirection_10m` · `wind_gusts_10m` · `shortwave_radiation` · `vapour_pressure_deficit`

**Air quality variables :**
`pm10` · `pm2_5` · `carbon_monoxide` · `nitrogen_dioxide` · `sulphur_dioxide` · `ozone` · `aerosol_optical_depth` · `dust` · `uv_index` · `uv_index_clear_sky` · `carbon_dioxide` · `us_aqi` · `us_aqi_pm2_5` · `us_aqi_pm10` · `us_aqi_nitrogen_dioxide` · `us_aqi_ozone` · `us_aqi_sulphur_dioxide` · `us_aqi_carbon_monoxide`

The `us_aqi` target uses Open-Meteo's EPA-compliant rolling averages (PM: 24h, O₃/CO: 8h, NO₂/SO₂: 1h).

### Feature Engineering — 186+ Features

| Category | Count | Description |
|:---------|:------|:------------|
| Weather derivatives | ~88 | Rolling means (6/12/24h), rolling std (24h), lags (12/24h) for 11 weather vars |
| Atmospheric derivatives | ~20 | Rolling/lags for aerosol, dust, UV index, CO₂ |
| Sub-AQI features | ~30 | EPA sub-index lags (6/12/24h) + rolling means (12/24h) for 6 pollutants |
| Time features | 9 | Cyclical sin/cos for hour, day-of-week, month, day-of-year + `is_weekend` |
| Interaction features | 7 | humidity×temp, temp×pressure, wind×humidity, cloud×temp, aerosol×humidity, radiation×aerosol, VPD×temp |
| AQI autoregressive | 13 | Lags (1/3/6/12/24h), rolling mean/std (6/12/24h), delta trends (1h, 6h) |
| Seasonal flags | 2 | `is_smog_season` (Oct–Feb), `is_monsoon_season` (Jun–Sep) |

### Hourly Pipeline (`hourly_pipeline.py`)

Runs **every hour at :10 UTC** via GitHub Actions:

1. Auto-detects data gaps — extends lookback if pipeline was delayed (up to 48h backfill)
2. Fetches last 3h from Open-Meteo with fallback URL and exponential backoff retry
3. Deduplicates against MongoDB — skips already-uploaded hours
4. Fetches 48h history from MongoDB for lag/rolling feature computation
5. Engineers all 186+ features for each new hour
6. Uploads only new records with a unique datetime index
7. Triggers backend snapshot rebuild via `POST /api/ingest-trigger`

---

## Training Pipeline

### Strategy

`Model_Retrain_pipeline.py` (v7) runs **every 12h at :30 UTC** in two phases:

**Phase 1 — Model selection** (probe horizons: t+1, t+6, t+24, t+48):
All 3 models trained on a sample → winner picked by avg RMSE. Cost: ~20% of full training.

**Phase 2 — Full training** of the winning model across all 18 horizons:
`t+1, t+2, t+3, t+6, t+9, t+12, t+15, t+18, t+21, t+24, t+30, t+36, t+42, t+48, t+54, t+60, t+66, t+72`

**Training design:**
- 80/20 temporal split — no shuffling, preserves time order
- `RobustScaler` fitted on train only, applied to test
- Lag exclusion per horizon — lags shorter than `h` excluded (no data leakage)
- Horizon encoding appended to every sample: `[h/72, log(h)/log(72), √(h/72), band_flag]`
- Live weather injection — Open-Meteo 72h forecast vectors injected at inference time
- Serialization: `pickle` → `zlib` (level 9) → `base64` → MongoDB (one doc per horizon)

---

## The Three Models

### 1. ⚡ XGBoost — **Selected (Best)**

Gradient boosted trees with level-wise growth. `early_stopping_rounds` used to cap actual tree count below `n_estimators`.

| Horizon range | n_estimators | lr | max_depth | α / λ |
|:--------------|:-------------|:---|:----------|:------|
| t+1–3h | 500 | 0.02 | 7 | 0.0 / 1.0 |
| t+4–12h | 400 | 0.03 | 6 | 0.05 / 1.5 |
| t+13–24h | 300 | 0.04 | 5 | 0.2 / 2.0 |
| t+25–48h | 200 | 0.05 | 5 | 0.5 / 3.0 |
| t+49–72h | 150 | 0.05 | 4 | 1.0 / 5.0 |

**Performance:**

| Metric | Value |
|:-------|:------|
| Avg R² (all horizons) | **0.8480** |
| Avg RMSE | **13.49** |
| Avg MAE | **9.55** |
| Short R² (t+1–24h) | **0.8715** |
| Medium R² (t+25–48h) | **0.7361** |
| Long R² (t+49–72h) | **0.5365** |

Window breakdown:

| Window | Horizons | Avg RMSE | Avg MAE | Avg R² | Quality |
|:-------|:---------|:---------|:--------|:-------|:--------|
| Short | t+1 → t+24h | 5.596 | 2.256 | 0.8715 | ✅ Excellent |
| Medium | t+25 → t+48h | 10.85 | 9.996 | 0.7361 | 🟡 Good |
| Long | t+49 → t+72h | 17.993 | 12.383 | 0.5365 | 🟠 Moderate |


---

### 2. 💡 LightGBM

Leaf-wise gradient boosting — faster training than XGBoost, competitive accuracy. Uses `early_stopping` callback with 50-round patience.

| Horizon range | n_estimators | lr | num_leaves | α / λ |
|:--------------|:-------------|:---|:-----------|:------|
| t+1–3h | 1500 | 0.01 | 63 | 0.0 / 1.0 |
| t+4–12h | 1200 | 0.02 | 63 | 0.05 / 1.5 |
| t+13–24h | 800 | 0.03 | 47 | 0.2 / 2.0 |
| t+25–48h | 600 | 0.04 | 31 | 0.5 / 3.0 |
| t+49–72h | 500 | 0.05 | 15 | 1.0 / 5.0 |

**Performance:**

| Metric | Value |
|:-------|:------|
| Avg R² (all horizons) | 0.8210 |
| Avg RMSE | 14.31 |
| Avg MAE | 10.18 |
| Short R² (t+1–24h) | 0.8590 |
| Medium R² (t+25–48h) | 0.7142 |
| Long R² (t+49–72h) | 0.4898 |

Fast training, competitive accuracy. Slightly behind XGBoost on RMSE across all horizons.

---

### 3. 🌲 Random Forest

Bagged decision trees. Depth-capped to stay within MongoDB's 16MB document limit per horizon model.

| Horizon range | n_estimators | max_depth | min_samples_leaf |
|:--------------|:-------------|:----------|:-----------------|
| t+1–3h | 100 | 10 | 10 |
| t+4–12h | 100 | 10 | 15 |
| t+13–24h | 80 | 8 | 20 |
| t+25–48h | 60 | 7 | 25 |
| t+49–72h | 50 | 6 | 30 |

**Performance:**

| Metric | Value |
|:-------|:------|
| Avg R² (all horizons) | 0.7820 |
| Avg RMSE | 15.82 |
| Avg MAE | 11.24 |
| Short R² (t+1–24h) | 0.8421 |
| Medium R² (t+25–48h) | 0.6934 |
| Long R² (t+49–72h) | 0.4105 |

Good short-term performance. Higher variance on long-horizon predictions. Weakest of the three on RMSE.

---

### Model Comparison Summary

| Model | Avg R² | Avg RMSE | Avg MAE | Short R² | Medium R² | Long R² |
|:------|:-------|:---------|:--------|:---------|:----------|:--------|
| **XGBoost** ✅ | **0.8480** | **13.49** | **9.55** | **0.8715** | **0.7361** | **0.5365** |
| LightGBM | 0.8210 | 14.31 | 10.18 | 0.8590 | 0.7142 | 0.4898 |
| Random Forest | 0.7820 | 15.82 | 11.24 | 0.8421 | 0.6934 | 0.4105 |

---

## Backend API

`backend.py` — FastAPI on **Hugging Face Spaces** (free tier, auto-sleeps after inactivity):

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/health` | GET | Health check + PKT timestamp |
| `/api/current` | GET | Latest AQI + weather + sub-AQI (live from MongoDB) |
| `/api/forecast` | GET | 72h forecast from in-memory snapshot |
| `/api/historical?days=N` | GET | Historical AQI + pollutant rows (1–30 days) |
| `/api/eda?days=N` | GET | EDA stats, heatmap, scatter, category breakdown |
| `/api/model-info` | GET | Per-horizon metrics, architecture, feature list |
| `/api/manual-predict?date=YYYY-MM-DD` | GET | Full 24-hour prediction for a specific date |
| `/api/ingest-trigger` | POST | Rebuild forecast snapshot (called by hourly pipeline) |
| `/api/reload-models` | POST | Hot-reload models from MongoDB without restart |

**Snapshot caching:** At startup and after each hourly trigger, the backend pre-computes the full 72h forecast and caches it in memory — `/api/forecast` is served instantly with no per-request ML inference.

---

## Dashboard

Single-file static dashboard on **[Vercel](https://air-quality-index-predictor.vercel.app/)** — 6 pages, auto-refreshes every 60s, fully responsive:

| Page | Description |
|:-----|:------------|
| 🏠 **Dashboard** | Live AQI gauge, 72h forecast preview, milestone chips (t+1/6/12/24/48/72h), weather panel, pollutant sub-AQI bars |
| 📈 **72h Forecast** | Full forecast chart (3 bands), band summary cards, detailed table with category badges |
| 🗓️ **Manual Predict** | Date calendar (today + 2 days), 24-hour prediction table + full-day chart + hourly chips |
| 📊 **Historical Data** | AQI time series (up to 30 days), hourly averages bar chart, pollutant trends |
| 🔬 **Analytics** | Category distribution pie, temperature vs AQI scatter, day×hour heatmap, dominant pollutant frequency |
| 🤖 **Model Info** | All 3 model comparison, band performance, window performance table, per-horizon metrics, feature tags |

---

## Continuous Automation

| Workflow | Schedule | Steps |
|:---------|:---------|:------|
| `hourly_api_fetch.yml` | Every hour at :10 UTC | Fetch → Engineer → Upload → Trigger backend snapshot |
| `72h_retrain.yml` | Every 12h at :30 UTC | Historical pipeline → Train all 3 models → Select best → Save to MongoDB |

---

## Setup

### Prerequisites

- Python 3.12+
- MongoDB Atlas account (free M0 tier)
- GitHub account (for Actions CI/CD)

### Local Development

```bash
git clone https://github.com/SeekAI-786/AQI_Predictor.git
cd AQI_Predictor

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Create a `.env` file:

```env
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/
MONGODB_DB=AQI_Project
MONGODB_FEATURES_COLLECTION=karachi_aqi_features
MODEL_DB=aqi_model_store
MODEL_COL=AQI_72h_model
```

Run the backend:

```bash
uvicorn backend:app --reload --port 8000
```

Open `index.html` in your browser (or serve with any static file server).

### GitHub Actions Secrets

| Secret | Value |
|:-------|:------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB` | `AQI_Project` |
| `MONGODB_COLLECTION` | `weather_data` |
| `MONGODB_FEATURES_COLLECTION` | `karachi_aqi_features` |
| `MODEL_DB` | `aqi_model_store` |
| `MODEL_COL` | `AQI_72h_model` |

### Hugging Face Spaces (Backend)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload `backend.py` and `requirements.txt`
3. Add all env vars in Space Settings → Variables and Secrets
4. Update the `BACKEND` URL in `hourly_api_fetch.yml` to your Space URL

---

## Project Structure

```
├── index.html                    # 6-page dashboard (Vercel, ~2500 lines)
├── backend.py                    # FastAPI backend — all API routes + ML inference
├── hourly_pipeline.py            # Hourly data ingestion + feature engineering
├── Model_Retrain_pipeline.py     # 3-model training pipeline (v7)
├── Complete_Pipeline.py          # Historical data fetch + feature engineering
├── feature_engineering.py        # Feature engineering module (186+ features, v2)
├── Fetch_Historical_Data.py      # Historical data fetching utilities
├── mongodb_upload.py             # MongoDB upload helpers                     # Streamlit dashboard (legacy)
├── requirements.txt              # Python dependencies
├── .env                          # Local secrets (gitignored)
├── .github/
│   └── workflows/
│       ├── hourly_api_fetch.yml  # Cron: every hour at :10 UTC
│       └── 72h_retrain.yml       # Cron: every 12h at :30 UTC
└── Research/               # Jupyter notebooks — Hopsworks experiments
    └── Mongodb/                  # Jupyter notebooks — MongoDB experiments
```

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| **ML Models** | XGBoost, LightGBM, scikit-learn (RandomForest, RobustScaler) |
| **Backend** | FastAPI, Uvicorn, httpx, Pydantic |
| **Feature Store** | MongoDB Atlas (free M0 tier) |
| **Data Source** | Open-Meteo Weather API + Air Quality API (free, no key required) |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js 4.4 |
| **CI/CD** | GitHub Actions (hourly + 12h cron) |
| **Hosting** | Vercel (frontend), Hugging Face Spaces (backend) |
| **Language** | Python 3.12 |

---

## Location

📍 **Karachi, Pakistan** — 24.8607°N, 67.0011°E  
🕐 All times displayed in **PKT (UTC+5)**  
🌐 Data: [Open-Meteo](https://open-meteo.com) — free, no API key required

---

## AQI Scale Reference

| AQI Range | Category | Health Implication |
|:----------|:---------|:-------------------|
| 0–50 | 🟢 Good | Air quality is satisfactory |
| 51–100 | 🟡 Moderate | Acceptable; sensitive people may be affected |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | Sensitive groups should reduce outdoor exertion |
| 151–200 | 🔴 Unhealthy | Everyone should reduce prolonged outdoor exertion |
| 201–300 | 🟣 Very Unhealthy | Everyone should avoid prolonged outdoor exertion |
| 301–500 | ⛔ Hazardous | Everyone should avoid all outdoor exertion |
