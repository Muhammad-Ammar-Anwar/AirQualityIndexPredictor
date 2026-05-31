# AQI SeekAI — Karachi Air Quality Intelligence

**Real-time Air Quality Monitoring, 72-Hour ML Forecast & Manual Date Prediction for Karachi, Pakistan**

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-air--quality--index--predictor.vercel.app-blue?logo=vercel)](https://air-quality-index-predictor.vercel.app/)
[![Hourly Pipeline](https://img.shields.io/badge/Hourly_Pipeline-active-brightgreen?logo=githubactions)](https://github.com/actions)
[![Retrain Pipeline](https://img.shields.io/badge/Retrain-every_12h-orange?logo=githubactions)](https://github.com/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb)](https://mongodb.com/atlas)

---

## What It Does

- Collects **35 weather & air quality variables** every hour from Open-Meteo APIs
- Engineers **186+ features** (rolling stats, lags, cyclical encodings, interactions, AQI autoregressive) and stores them in MongoDB Atlas
- Trains **per-horizon XGBoost / LightGBM / Random Forest models** (18 models, t+1h to t+72h) every 12 hours — best model selected automatically by RMSE
- Serves a **6-page dark-theme dashboard** (HTML + FastAPI backend) with:
  - Real-time AQI gauge with live weather conditions
  - 72-hour ML forecast with band visualization
  - **Manual date prediction** — pick any date (today + next 2 days) and get full 24-hour AQI predictions
  - Historical data explorer with pollutant trends
  - EDA analytics with heatmaps, scatter plots, category breakdowns
  - Model performance metrics and architecture details

Everything runs on free-tier services — **$0/month**.

---

## Live Architecture

```
Open-Meteo APIs (Weather + Air Quality)
        │
        ▼
hourly_pipeline.py          ← GitHub Actions cron: every hour at :10 UTC
        │  Fetch last 3h → Dedup → Engineer 186+ features → Upload to MongoDB
        │
        ▼
MongoDB Atlas
├── AQI_Project.karachi_aqi_features     ← Hourly feature store
└── aqi_model_store.AQI_72h_model        ← Trained model blobs (base64 + zlib)
        │                    ▲
        │                    │
        ▼                    │
   backend.py           Model_Retrain_pipeline.py
  (FastAPI on           ← GitHub Actions cron: every 12h at :30 UTC
Hugging Face Space)       Trains XGBoost / LightGBM / RF
        │                 Selects best by avg RMSE across probe horizons
        ▼                 Saves 18 per-horizon models + scaler to MongoDB
   index.html
  (Static Dashboard)
```

---

## Model Performance

**Selected model: XGBoost** — chosen by avg RMSE across probe horizons (t+1, t+6, t+24, t+48).

### Model Comparison

| Model | Avg R² | Avg RMSE | Avg MAE | Short R² | Medium R² | Long R² |
|:------|:-------|:---------|:--------|:---------|:----------|:--------|
| **XGBoost** ✓ | **0.8480** | **13.49** | **9.55** | **0.8715** | **0.7361** | **0.5365** |
| LightGBM | 0.8210 | 14.31 | 10.18 | 0.8590 | 0.7142 | 0.4898 |
| Random Forest | 0.7820 | 15.82 | 11.24 | 0.8421 | 0.6934 | 0.4105 |

### XGBoost Window Performance

| Window | Horizons | Avg RMSE | Avg MAE | Avg R² | Quality |
|:-------|:---------|:---------|:--------|:-------|:--------|
| Short | t+1 → t+24h | 5.596 | 2.256 | **0.8715** | ✅ Excellent |
| Medium | t+25 → t+48h | 10.85 | 9.996 | **0.7361** | 🟡 Good |
| Long | t+49 → t+72h | 17.993 | 12.383 | **0.5365** | 🟠 Moderate |

### LightLGM Band-Level Detail

| Band | Horizons | Avg RMSE | Avg MAE | Avg R² |
|:-----|:---------|:---------|:--------|:-------|
| Short | t+1 → t+24h | 5.52 | 2.86 | 0.9138 |
| Medium | t+25 → t+48h | 12.31 | 8.52 | 0.6171 |
| Long | t+49 → t+72h | 18.44 | 13.69 | 0.1347 |

---

## Implementation
### 1. Data Collection

Weather and pollutant data are collected from two Open-Meteo endpoints:

**Weather variables (18):** `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `apparent_temperature`, `precipitation`, `rain`, `snowfall`, `surface_pressure`, `pressure_msl`, `cloud_cover` (total + low/mid/high bands), `windspeed_10m`, `winddirection_10m`, `wind_gusts_10m`, `shortwave_radiation`, `vapour_pressure_deficit`

**Air quality variables (17):** `pm10`, `pm2_5`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `aerosol_optical_depth`, `dust`, `uv_index`, `uv_index_clear_sky`, `carbon_dioxide`

The `us_aqi` value is sourced directly from the Open-Meteo API which applies proper EPA rolling averages (PM: 24h, O₃/CO: 8h, NO₂/SO₂: 1h) — more accurate than instantaneous manual calculation.

### 2. Hourly Data Ingestion & Feature Engineering

`hourly_pipeline.py` runs **every hour at :10 UTC** via GitHub Actions:

1. **Auto-detects gaps** — checks MongoDB for the latest record and extends lookback if the pipeline was delayed
2. **Fetches** last 3h from Open-Meteo (with fallback URL and exponential backoff retry)
3. **Deduplicates** against MongoDB — skips already-uploaded hours
4. **Fetches 48h history** from MongoDB for lag/rolling feature computation
5. **Engineers 186+ features** per hour:
   - **Weather derivatives** — rolling means (6/12/24h), rolling std (24h), lags (12/24h) for 11 weather variables
   - **Atmospheric derivatives** — rolling/lags for aerosol, dust, UV, CO₂
   - **Sub-AQI features** — EPA sub-index lags (6/12/24h) and rolling means (12/24h) for 6 pollutants
   - **Time features** — cyclical sin/cos encoding for hour, day-of-week, month, day-of-year + `is_weekend`
   - **Interaction features** — humidity×temp, temp×pressure, wind×humidity, cloud×temp, aerosol×humidity, radiation×aerosol, VPD×temp
   - **AQI autoregressive** — lags (1/3/6/12/24h), rolling mean/std (6/12/24h), delta trends (1h, 6h)
6. **Uploads** only new records to MongoDB with deduplication index
7. **Triggers backend snapshot rebuild** via `POST /api/ingest-trigger` on Hugging Face Space

### 3. Model Training

`Model_Retrain_pipeline.py` runs **every 12 hours at :30 UTC** via GitHub Actions:

- Pulls all feature data from MongoDB
- Trains **3 model families**: XGBoost, LightGBM, Random Forest
- Uses **per-horizon strategy**: 18 separate models for t+1h, t+2h, t+3h, t+6h, t+9h, t+12h, t+15h, t+18h, t+21h, t+24h, t+30h, t+36h, t+42h, t+48h, t+54h, t+60h, t+66h, t+72h
- Each model excludes lags shorter than its horizon (no data leakage)
- **Horizon encoding** appended to each sample: `[h/72, log(h)/log(72), √(h/72), band_flag]`
- **Live weather injection**: Open-Meteo 72h forecast weather vectors injected at inference time
- Selects best model family by avg RMSE across probe horizons (t+1, t+6, t+24, t+48)
- Serializes models with `pickle` + `zlib` compression + `base64` encoding → stored in MongoDB
- Supports pipeline versions v5/v6/v7 with automatic version priority on load

### 4. Backend API

`backend.py` — FastAPI application deployed on **Hugging Face Spaces** (free tier):

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `GET /health` | GET | Health check |
| `GET /api/current` | GET | Latest AQI + weather from MongoDB (live) |
| `GET /api/forecast` | GET | 72h forecast served from in-memory snapshot |
| `GET /api/historical?days=N` | GET | Historical AQI + pollutant rows |
| `GET /api/eda?days=N` | GET | EDA summary stats, heatmap, scatter, category breakdown |
| `GET /api/model-info` | GET | Per-horizon metrics, architecture, feature list |
| `GET /api/manual-predict?date=YYYY-MM-DD` | GET | Full 24-hour prediction for a specific date |
| `POST /api/ingest-trigger` | POST | Rebuild forecast snapshot (called by hourly pipeline) |
| `POST /api/reload-models` | POST | Hot-reload models from MongoDB without restart |

**Snapshot caching**: At startup and after each hourly pipeline trigger, the backend pre-computes the full 72h forecast and caches it in memory. All `/api/forecast` requests are served instantly from cache — no ML inference per request.

### 5. Manual Date Prediction

The `/api/manual-predict` endpoint predicts AQI for all 24 hours (00:00–23:00 PKT) of a selected date:

- Allowed range: **today and next 2 days** (PKT timezone)
- For each PKT hour, computes the UTC horizon from current time
- Finds the two nearest KEY_HORIZON models and **linearly interpolates** between them
- Injects Open-Meteo weather forecast for the exact UTC timestamp of each hour
- Past hours of today use the t+1h model (closest available)
- Returns per-hour predictions with AQI value, band, category, and health advice

### 6. Dashboard (index.html)

Single-file static dashboard deployed on **[Vercel](https://air-quality-index-predictor.vercel.app/)** with 6 pages:

| Page | Description |
|:-----|:------------|
| 🏠 Dashboard | Live AQI gauge, 72h forecast preview, milestone chips, weather panel, pollutant bars |
| 📈 72h Forecast | Full forecast chart, band summary cards, detailed table |
| 🗓️ Manual Predict | Date calendar (today + 2 days), 24-hour prediction table + charts |
| 📊 Historical Data | AQI time series, hourly averages, pollutant trends |
| 🔬 Analytics | Category pie, temp vs AQI scatter, day×hour heatmap, dominant pollutant |
| 🤖 Model Info | Model comparison, band performance, per-horizon metrics, feature tags |

Auto-refreshes every 60 seconds. Fully responsive (mobile sidebar, touch-friendly).

### 7. Continuous Automation

| Workflow | Schedule | Action |
|:---------|:---------|:-------|
| `hourly_api_fetch.yml` | Every hour at :10 UTC | Fetch → Engineer → Upload → Trigger backend |
| `72h_retrain.yml` | Every 12h at :30 UTC | Full pipeline + retrain all models |

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
MONGODB_URI
MONGODB_DB
MONGODB_FEATURES_COLLECTION
MODEL_DB
MODEL_COL
```

Run the backend:

```bash
uvicorn backend:app --reload --port 8000
```

Then open `index.html` in your browser (or serve it with any static file server).

### GitHub Actions Secrets

Add these repository secrets under **Settings → Secrets → Actions**:

| Secret | Description |
|:-------|:------------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB` | Feature store DB name (`AQI_Project`) |
| `MONGODB_COLLECTION` | Raw collection name (`weather_data`) |
| `MONGODB_FEATURES_COLLECTION` | Feature collection (`karachi_aqi_features`) |
| `MODEL_DB` | Model store DB name (`aqi_model_store`) |
| `MODEL_COL` | Model collection (`AQI_72h_model`) |

### Hugging Face Spaces Deployment (Backend)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) (Docker or Python SDK)
2. Upload `backend.py` and `requirements.txt`
3. Add `MONGODB_URI` and other env vars in Space Settings → Variables and Secrets
4. Update the `BACKEND` URL in `hourly_api_fetch.yml` to your Space URL

---

## Project Structure

```
├── index.html                    # 6-page dark dashboard (single file, ~2400 lines)
├── backend.py                    # FastAPI backend — all API routes + ML inference
├── hourly_pipeline.py            # Hourly data ingestion + feature engineering
├── Model_Retrain_pipeline.py     # Per-horizon model training (XGBoost/LightGBM/RF)
├── Complete_Pipeline.py          # Full historical data fetch + feature engineering
├── feature_engineering.py        # Feature engineering module (186+ features, v2)
├── Fetch_Historical_Data.py      # Historical data fetching utilities
├── mongodb_upload.py             # MongoDB upload helpers
├── requirements.txt              # Python dependencies
├── .env                          # Local secrets (gitignored)
├── .github/
│   └── workflows/
│       ├── hourly_api_fetch.yml  # Cron: every hour at :10 UTC
│       ├── 72h_retrain.yml       # Cron: every 12h at :30 UTC
└── mongo/                        # MongoDB binaries (local dev)
```

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| **ML Models** | XGBoost, LightGBM, scikit-learn (RandomForest, RobustScaler) |
| **Backend** | FastAPI, Uvicorn, httpx, Pydantic |
| **Database** | MongoDB Atlas (free M0 tier) |
| **Data Source** | Open-Meteo Weather API + Air Quality API |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js 4.4 |
| **CI/CD** | GitHub Actions (hourly + 12h cron) |
| **Hosting** | Hugging Face Spaces (backend), static file (frontend) |
| **Language** | Python 3.12 |

---

## Location

📍 **Karachi, Pakistan** — 24.8607°N, 67.0011°E  
🕐 All times displayed in **PKT (UTC+5)**  
🌐 Data source: [Open-Meteo](https://open-meteo.com) (free, no API key required)

---

## AQI Scale Reference

| AQI Range | Category | Color |
|:----------|:---------|:------|
| 0–50 | Good | 🟢 Green |
| 51–100 | Moderate | 🟡 Yellow |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 Orange |
| 151–200 | Unhealthy | 🔴 Red |
| 201–300 | Very Unhealthy | 🟣 Purple |
| 301–500 | Hazardous | ⛔ Maroon |
