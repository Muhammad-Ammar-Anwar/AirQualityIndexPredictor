import os
import base64
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pytz

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AQI — Karachi",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption p {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600 !important;
        color: #475569 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Headers */
    .main h1 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main h2, .main h3 {
        color: #1e293b;
        font-weight: 600;
        margin-top: 1.5rem;
    }

    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #334155;
        border-radius: 8px;
    }

    /* Sidebar radio */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: background 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.08);
    }

    .stAlert { border-radius: 10px; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _get_mongo_uri():
    # 1. Streamlit secrets (production)
    try:
        return st.secrets["MONGODB_URI"]
    except Exception:
        pass
    # 2. Environment variable (local .env)
    uri = os.getenv("MONGODB_URI")
    if uri:
        return uri
    # 3. Hardcoded fallback — update this to your actual URI
    return "Invalid URI"

MONGODB_URI = _get_mongo_uri()
FEATURE_DB  = os.getenv("MONGODB_DB", "AQI_Project")
FEATURE_COL = os.getenv("MONGODB_FEATURES_COLLECTION", "karachi_aqi_features")
MODEL_DB    = os.getenv("MODEL_DB", "aqi_model_store")
MODEL_COL   = os.getenv("MODEL_COL", "AQI_72h_model")

KEY_HORIZONS = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72]

FORECAST_WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "windspeed_10m", "wind_gusts_10m", "cloud_cover",
    "shortwave_radiation", "vapour_pressure_deficit",
    "precipitation", "dew_point_2m", "pressure_msl",
]

MAX_H = 72
BANDS = {
    "short":  list(range(1, 9)),
    "medium": [9, 12, 15, 18, 21, 24],
    "long":   [25, 30, 36, 42, 48, 54, 60, 66, 72],
}

RAW_POLLUTANTS = {
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "sulphur_dioxide", "carbon_monoxide",
}

LATITUDE = 24.8607
LONGITUDE = 67.0011

AQI_LEVELS = [
    (0,   50,  "Good",                     "#00e400"),
    (51,  100, "Moderate",                  "#ffff00"),
    (101, 150, "Unhealthy for Sensitive",   "#ff7e00"),
    (151, 200, "Unhealthy",                 "#ff0000"),
    (201, 300, "Very Unhealthy",            "#8f3f97"),
    (301, 500, "Hazardous",                 "#7e0023"),
]

BRAND = {
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "short": "#3b82f6",
    "medium": "#f59e0b",
    "long": "#8b5cf6",
}


PKT = pytz.timezone("Asia/Karachi")

def get_pkt_now():
    return datetime.now(PKT).replace(tzinfo=None)


def aqi_color(val):
    for lo, hi, _, color in AQI_LEVELS:
        if lo <= val <= hi:
            return color
    return "#7e0023"


def aqi_label(val):
    for lo, hi, label, _ in AQI_LEVELS:
        if lo <= val <= hi:
            return label
    return "Hazardous"


def aqi_emoji(val):
    if val <= 50:    return "🟢"
    if val <= 100:   return "🟡"
    if val <= 150:   return "🟠"
    if val <= 200:   return "🔴"
    if val <= 300:   return "🟣"
    return "⛔"



@st.cache_resource(ttl=600, show_spinner="Connecting to MongoDB …")
def get_mongo_client():
    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"),
                         serverSelectionTimeoutMS=15000)
    client.admin.command("ping")
    return client


@st.cache_data(ttl=120, show_spinner="Loading models \u2026")
def load_models():
    """Load models from MongoDB — supports both old band format and new per-horizon format."""
    client = get_mongo_client()
    col = client[MODEL_DB][MODEL_COL]

    models        = {}
    valid_indices = {}
    fc_indices_map= {}
    feature_cols  = []
    main_scaler   = None
    training_logs = {}

    docs = list(col.find())
    if not docs:
        raise ValueError(f"No documents found in {MODEL_DB}.{MODEL_COL}")

    # Detect format: new format uses "horizon" key, old uses "band"
    is_new_format = any("horizon" in doc for doc in docs)

    for doc in docs:
        blob = base64.b64decode(doc["model_blob"])

        if is_new_format:
            key = doc.get("horizon")
            if key is None:
                continue
            if key == "_scaler":
                main_scaler  = pickle.loads(blob)
                feature_cols = doc.get("feature_cols", [])
            else:
                payload = pickle.loads(blob)
                models[key]         = payload["model"]
                valid_indices[key]  = payload.get("valid_indices", [])
                fc_indices_map[key] = payload.get("fc_indices", [])
                training_logs[key]  = doc.get("metrics", {})
        else:
            # Old band format — convert to per-horizon
            key = doc.get("band")
            if key is None:
                continue
            if key == "_scaler":
                main_scaler  = pickle.loads(blob)
                feature_cols = doc.get("feature_cols", [])
            else:
                obj = pickle.loads(blob)
                band_model  = obj.get("model") if isinstance(obj, dict) else obj
                band_scaler = obj.get("scaler") if isinstance(obj, dict) else None
                band_ranges = {"short": range(1, 9), "medium": range(9, 25), "long": range(25, 73)}
                for h in band_ranges.get(key, []):
                    if h in KEY_HORIZONS:
                        models[h]         = band_model
                        valid_indices[h]  = []   # no lag exclusion in old format
                        fc_indices_map[h] = []
                        training_logs[h]  = doc.get("training_log", {}).get("metrics", {})
                # Store band scaler for old-format inference
                if band_scaler is not None:
                    for h in band_ranges.get(key, []):
                        if h in KEY_HORIZONS:
                            fc_indices_map[h] = band_scaler  # repurpose to carry scaler

    if main_scaler is None:
        raise ValueError("Main feature scaler not found in MongoDB. "
                         "Run Model_Retrain_pipeline.py to train and save models.")
    if not models:
        raise ValueError("No models found in MongoDB.")

    return models, valid_indices, fc_indices_map, main_scaler, feature_cols, training_logs


@st.cache_data(ttl=120, show_spinner="Loading data …")
def load_features(days: int = 5):
    client = get_mongo_client()
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Primary: karachi_aqi_features (has us_aqi + all engineered features)
    col = client[FEATURE_DB][FEATURE_COL]
    count = col.count_documents({})
    if count > 0:
        cursor = col.find({"datetime": {"$gte": cutoff}}, {"_id": 0}).sort("datetime", 1)
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
            if df["datetime"].dt.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_localize(None)
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

    # Fallback: weather_data (raw hourly upserts)
    raw_col = os.getenv("MONGODB_COLLECTION", "weather_data")
    col = client[FEATURE_DB][raw_col]
    cursor = col.find({"datetime": {"$gte": cutoff}}, {"_id": 0}).sort("datetime", 1)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df



def fetch_weather_forecast_for_app(feature_cols, fc_indices_map, main_scaler):
    """Fetch live 72h weather forecast from Open-Meteo for inference."""
    import requests

    # Find valid fc_indices (list type with items)
    sample_fci = []
    for v in fc_indices_map.values():
        if isinstance(v, list) and len(v) > 0:
            sample_fci = v
            break

    if not sample_fci:
        return {h: np.zeros(0, dtype=np.float32) for h in KEY_HORIZONS}

    fc_cols  = [feature_cols[i] for i in sample_fci if i < len(feature_cols)]
    api_vars = [c for c in FORECAST_WEATHER_COLS if c in fc_cols]
    if not api_vars:
        return {h: np.zeros(len(sample_fci), dtype=np.float32) for h in KEY_HORIZONS}

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": LATITUDE, "longitude": LONGITUDE,
                    "hourly": ",".join(api_vars),
                    "timezone": "UTC", "forecast_days": 4},
            timeout=15,
        )
        resp.raise_for_status()
        data  = resp.json()["hourly"]
        times = pd.to_datetime(data["time"], utc=True).tz_localize(None)
        now   = pd.Timestamp.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=None)
        result = {}
        for h in KEY_HORIZONS:
            fci    = fc_indices_map.get(h, sample_fci)
            fci    = fci if isinstance(fci, list) else sample_fci
            target = now + pd.Timedelta(hours=h)
            if target in times.values:
                idx = list(times.values).index(target)
                vec = np.zeros(len(feature_cols), dtype=np.float32)
                for col in api_vars:
                    if col in feature_cols:
                        val = data[col][idx]
                        vec[feature_cols.index(col)] = float(val) if val is not None else 0.0
                vec_scaled = main_scaler.transform([vec])[0]
                result[h] = vec_scaled[fci] if fci else np.zeros(0, dtype=np.float32)
            else:
                result[h] = np.zeros(len(fci), dtype=np.float32)
        return result
    except Exception as e:
        return {h: np.zeros(len(sample_fci), dtype=np.float32) for h in KEY_HORIZONS}


def horizon_encoding_app(h: int) -> np.ndarray:
    band_flag = 0.0 if h <= 8 else (0.5 if h <= 24 else 1.0)
    return np.array([h / MAX_H, np.log1p(h) / np.log1p(MAX_H),
                     np.sqrt(h / MAX_H), band_flag], dtype=np.float32)


def generate_forecast(models, valid_indices, fc_indices_map,
                      main_scaler, feature_cols, df):
    """Generate 72h forecast using per-horizon models with weather injection."""
    if df.empty or not models or main_scaler is None:
        return pd.DataFrame()

    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame()

    # Build scaled feature vector from latest observation
    full_vec = np.zeros((1, len(feature_cols)), dtype=np.float32)
    latest_row = df[available_cols].iloc[-1]
    for col in available_cols:
        if col in feature_cols:
            val = latest_row[col]
            full_vec[0, feature_cols.index(col)] = float(val) if pd.notna(val) else 0.0
    X_scaled = main_scaler.transform(full_vec)[0]

    # Fetch live weather forecast (only if fc_indices exist)
    has_fc = any(isinstance(v, list) and len(v) > 0 for v in fc_indices_map.values())
    live_fc = fetch_weather_forecast_for_app(feature_cols, fc_indices_map, main_scaler) if has_fc else {}

    last_dt   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    forecasts = []

    for h in KEY_HORIZONS:
        if h not in models:
            continue

        vi  = valid_indices.get(h, [])
        fci = fc_indices_map.get(h, [])

        try:
            if isinstance(fci, list):
                # New format
                base   = X_scaled[vi] if vi else X_scaled
                fc_vec = live_fc.get(h, np.zeros(len(fci), dtype=np.float32))
                if len(fc_vec) > 0:
                    row = np.concatenate([base, fc_vec, horizon_encoding_app(h)])
                else:
                    row = np.concatenate([base, horizon_encoding_app(h)])
            else:
                # Old format — fci is a band scaler
                band_scaler = fci
                aqi_history = df["us_aqi"].dropna().values
                if len(aqi_history) == 0:
                    continue
                ar = np.array([
                    aqi_history[-1],
                    aqi_history[-6]  if len(aqi_history) >= 6  else aqi_history[-1],
                    aqi_history[-12] if len(aqi_history) >= 12 else aqi_history[-1],
                    np.mean(aqi_history[-24:]),
                    np.std(aqi_history[-24:]) if len(aqi_history) > 1 else 0.0,
                ])
                row = band_scaler.transform(
                    [np.concatenate([X_scaled, ar, [h / MAX_H]])]
                )[0].reshape(1, -1)
                pred = max(0, float(models[h].predict(row)[0]))
                band = "short" if h <= 8 else ("medium" if h <= 24 else "long")
                forecasts.append({"hour": h,
                                   "datetime": last_dt + timedelta(hours=h),
                                   "predicted_aqi": round(pred, 1),
                                   "band": band})
                continue

            pred = max(0, float(models[h].predict([row])[0]))
            band = "short" if h <= 8 else ("medium" if h <= 24 else "long")
            forecasts.append({"hour": h,
                               "datetime": last_dt + timedelta(hours=h),
                               "predicted_aqi": round(pred, 1),
                               "band": band})
        except Exception:
            continue

    return pd.DataFrame(forecasts)



_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,0.5)",
    font=dict(family="Inter, sans-serif"),
    template="plotly_white",
)


def render_aqi_gauge(value, label="Current AQI"):
    color = aqi_color(value)
    category = aqi_label(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 56, "color": color}},
        title={
            "text": (f"<b>{label}</b><br>"
                     f"<span style='font-size:1em;color:{color}'>"
                     f"{category}</span>"),
            "font": {"size": 16},
        },
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],    "color": "rgba(0,228,0,0.15)"},
                {"range": [50, 100],  "color": "rgba(255,255,0,0.12)"},
                {"range": [100, 150], "color": "rgba(255,126,0,0.12)"},
                {"range": [150, 200], "color": "rgba(255,0,0,0.12)"},
                {"range": [200, 300], "color": "rgba(143,63,151,0.12)"},
                {"range": [300, 500], "color": "rgba(126,0,35,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#0f172a", "width": 3},
                "thickness": 0.8, "value": value,
            },
        },
    ))
    fig.update_layout(height=310, margin=dict(t=90, b=20, l=40, r=40), **_CHART_LAYOUT)
    return fig


def render_forecast_chart(fc_df):
    fig = go.Figure()
    for lo, hi, label, color in AQI_LEVELS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.07,
                      layer="below", line_width=0,
                      annotation_text=label if lo < 300 else "",
                      annotation_position="top left",
                      annotation_font_size=9, annotation_font_color=color,
                      annotation_opacity=0.7)

    for band in ("short", "medium", "long"):
        bdf = fc_df[fc_df["band"] == band]
        if bdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=bdf["datetime"], y=bdf["predicted_aqi"],
            mode="lines+markers",
            name=f"{band.capitalize()} (t+{BANDS[band][0]}–{BANDS[band][-1]}h)",
            line=dict(color=BRAND[band], width=2.5, shape="spline"),
            marker=dict(size=5, line=dict(width=1, color="white")),
            hovertemplate="<b>t+%{customdata}h</b><br>AQI: %{y:.1f}<extra></extra>",
            customdata=bdf["hour"],
        ))

    fig.update_layout(
        title=dict(text="72-Hour AQI Forecast", font=dict(size=18)),
        xaxis_title="Time", yaxis_title="Predicted AQI",
        yaxis=dict(range=[0, max(fc_df["predicted_aqi"].max() * 1.15, 100)]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450, **_CHART_LAYOUT,
    )
    return fig


def render_historical_chart(df, column="us_aqi"):
    fig = go.Figure()
    for lo, hi, _, color in AQI_LEVELS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.06,
                      layer="below", line_width=0)
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[column], mode="lines",
        name="Observed AQI",
        line=dict(color=BRAND["primary"], width=1.8),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="AQI: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Historical AQI — Last {len(df)} hours", font=dict(size=18)),
        xaxis_title="Date / Time", yaxis_title="US AQI",
        hovermode="x unified", height=420, **_CHART_LAYOUT,
    )
    return fig


def render_pollutant_breakdown(df):
    sub_cols = {
        "us_aqi_pm2_5": "PM2.5",
        "us_aqi_pm10": "PM10",
        "us_aqi_nitrogen_dioxide": "NO\u2082",
        "us_aqi_ozone": "O\u2083",
        "us_aqi_sulphur_dioxide": "SO\u2082",
        "us_aqi_carbon_monoxide": "CO",
    }
    available = {k: v for k, v in sub_cols.items() if k in df.columns}
    if not available:
        return None

    latest = df.iloc[-1]
    names, values, colors = [], [], []
    for k, v in available.items():
        val = latest.get(k, 0)
        val = val if pd.notna(val) else 0
        names.append(v)
        values.append(val)
        colors.append(aqi_color(val))

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=14, color="#334155"),
        width=0.5,
    ))
    fig.update_layout(
        title=dict(text="Pollutant Sub-AQI Breakdown", font=dict(size=16)),
        yaxis_title="Sub-AQI Index",
        yaxis=dict(range=[0, max(values) * 1.3 if values else 100]),
        height=370, bargap=0.35, **_CHART_LAYOUT,
    )
    return fig


def render_weather_panel(df):
    if df.empty:
        return
    latest = df.iloc[-1]

    def fmt(val, suffix, d=1):
        return f"{val:.{d}f}{suffix}" if pd.notna(val) else "N/A"

    def card(emoji, label, val):
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#f8fafc,#f1f5f9);
                    border:1px solid #e2e8f0;border-radius:12px;
                    padding:16px 8px;text-align:center;height:120px;
                    display:flex;flex-direction:column;justify-content:center;
                    align-items:center;'>
            <div style='font-size:1.5rem;'>{emoji}</div>
            <div style='font-size:0.7rem;font-weight:600;color:#64748b;
                        text-transform:uppercase;letter-spacing:0.04em;
                        margin:4px 0 2px 0;'>{label}</div>
            <div style='font-size:1.1rem;font-weight:700;color:#0f172a;
                        white-space:nowrap;'>{val}</div>
        </div>""", unsafe_allow_html=True)

    row1 = st.columns(4)
    with row1[0]: card("\U0001f321\ufe0f", "Temp",     fmt(latest.get('temperature_2m'), "\u00b0C"))
    with row1[1]: card("\U0001f4a7",       "Humidity",  fmt(latest.get('relative_humidity_2m'), "%", 0))
    with row1[2]: card("\U0001f4a8",       "Wind",      fmt(latest.get('windspeed_10m'), " km/h"))
    with row1[3]: card("\u2601\ufe0f",     "Clouds",    fmt(latest.get('cloud_cover'), "%", 0))

    row2 = st.columns(4)
    with row2[0]: card("\U0001f321\ufe0f", "Dew Pt",    fmt(latest.get('dew_point_2m'), "\u00b0C"))
    with row2[1]: card("\U0001f4ca",       "Pressure",  fmt(latest.get('surface_pressure'), " hPa", 0))
    with row2[2]: card("\u2600\ufe0f",     "Radiation",  fmt(latest.get('shortwave_radiation'), " W/m\u00b2", 0))
    with row2[3]: card("\U0001f32c\ufe0f", "Gusts",     fmt(latest.get('wind_gusts_10m'), " km/h"))



def main():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
            <div style='font-size:2.2rem; font-weight:700; letter-spacing:-0.03em;
                        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        font-family: Inter, sans-serif;'>
                AQI Prediction
            </div>
            <div style='font-size:0.8rem; color:#94a3b8; margin-top:2px;
                        font-family:Inter,sans-serif;'>
                Karachi Air Quality Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        page = st.radio("Navigate", [
            "\U0001f3e0 Dashboard",
            "\U0001f4c8 72h Forecast",
            "\U0001f4ca Historical Data",
            "\U0001f52c EDA & Analytics",
            "\U0001f916 Model Info",
        ], index=0, label_visibility="collapsed")

        st.divider()
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.7;
                    font-family:Inter,sans-serif;'>
            <b style='color:#cbd5e1'>\U0001f4cd Location</b><br>Karachi, Pakistan<br>
            <b style='color:#cbd5e1'>\U0001f310 Coordinates</b><br>{LATITUDE}\u00b0N, {LONGITUDE}\u00b0E
        </div>
        """, unsafe_allow_html=True)



    try:
        models, valid_indices, fc_indices_map, main_scaler, feature_cols, training_logs = load_models()
        model_loaded = True
    except Exception as e:
        st.error(f"\u26a0\ufe0f Could not load models: {e}")
        models, valid_indices, fc_indices_map, main_scaler, feature_cols, training_logs = {}, {}, {}, None, [], {}
        model_loaded = False

    hist_df = load_features(days=5)

    current_aqi, current_dt = None, None
    if not hist_df.empty and "us_aqi" in hist_df.columns:
        valid = hist_df.dropna(subset=["us_aqi"])
        if not valid.empty:
            latest = valid.iloc[-1]
            current_aqi = float(latest["us_aqi"])
            current_dt = get_pkt_now()

    with st.sidebar:
        now_display = get_pkt_now()
        refresh_str = now_display.strftime('%Y-%m-%d %H:%M') + " PKT"
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.7;
                    font-family:Inter,sans-serif;'>
            <b style='color:#cbd5e1'>\U0001f550 Last Refresh</b><br>{refresh_str}
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        if st.button("\U0001f504 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    fc_df = pd.DataFrame()
    if model_loaded and not hist_df.empty:
        fc_df = generate_forecast(models, valid_indices, fc_indices_map,
                                  main_scaler, feature_cols, hist_df)

    if page == "\U0001f3e0 Dashboard":
        st.markdown("""
        <div style='margin-bottom:1.5rem;'>
            <h1 style='margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.02em;
                       font-family:Inter,sans-serif; color:#0f172a;'>
                Karachi Air Quality Dashboard
            </h1>
            <p style='margin:4px 0 0 0; font-size:0.9rem; color:#64748b;
                      font-family:Inter,sans-serif;'>
                Real-time AQI monitoring & 72-hour ML forecast powered by SeekAI
            </p>
        </div>
        """, unsafe_allow_html=True)

        if current_aqi is not None:
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                st.plotly_chart(render_aqi_gauge(current_aqi), use_container_width=True)
                ts = now_display.strftime('%b %d %Y, %H:%M') + " PKT"
                st.markdown(f"""
                <div style='text-align:center; margin-top:-15px;'>
                    <span style='font-size:0.78rem; color:#94a3b8;
                                font-family:Inter,sans-serif;'>
                        Latest from MongoDB \u00b7 {ts}
                    </span>
                </div>""", unsafe_allow_html=True)
            with col2:
                if not fc_df.empty:
                    st.plotly_chart(render_forecast_chart(fc_df), use_container_width=True)
                else:
                    st.info("Forecast not available \u2014 model or data missing.")

            if not fc_df.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### \u23f1\ufe0f Forecast Milestones")
                milestones = [1, 6, 12, 24, 48, 72]
                mcols = st.columns(len(milestones))
                for c, h in zip(mcols, milestones):
                    row = fc_df[fc_df["hour"] == h]
                    if not row.empty:
                        val = row.iloc[0]["predicted_aqi"]
                        c.metric(f"t+{h}h", f"{val:.0f}",
                                 delta=f"{val - current_aqi:+.0f}" if current_aqi else None,
                                 delta_color="inverse")

            st.divider()

            if not hist_df.empty:
                st.markdown("##### \U0001f324\ufe0f Current Weather")
                render_weather_panel(hist_df)

            st.divider()

            if not hist_df.empty:
                fig_poll = render_pollutant_breakdown(hist_df)
                if fig_poll:
                    st.plotly_chart(fig_poll, use_container_width=True)
        else:
            st.warning("No AQI data available. Ensure the hourly pipeline is running.")

    elif page == "\U0001f4c8 72h Forecast":
        st.markdown("<h1 style='font-family:Inter,sans-serif;font-weight:700;"
                    "letter-spacing:-0.02em;'>72-Hour AQI Forecast</h1>",
                    unsafe_allow_html=True)

        if fc_df.empty:
            st.warning("Forecast unavailable \u2014 check model and data availability.")
        else:
            st.plotly_chart(render_forecast_chart(fc_df), use_container_width=True)

            st.markdown("##### \U0001f4ca Band Summary")
            bcols = st.columns(3, gap="large")
            for i, (band, hrs) in enumerate(BANDS.items()):
                bdf = fc_df[fc_df["band"] == band]
                if not bdf.empty:
                    bcols[i].markdown(
                        f"<div style='font-size:0.85rem;font-weight:600;"
                        f"color:{BRAND[band]};margin-bottom:8px;'>"
                        f"\u25cf {band.capitalize()} Band "
                        f"(t+{hrs[0]}\u2013{hrs[-1]}h)</div>",
                        unsafe_allow_html=True)
                    bcols[i].metric("Avg AQI", f"{bdf['predicted_aqi'].mean():.1f}")
                    bcols[i].metric("Peak AQI", f"{bdf['predicted_aqi'].max():.1f}")
                    bcols[i].metric("Min AQI", f"{bdf['predicted_aqi'].min():.1f}")

            st.divider()
            st.markdown("##### \U0001f4cb Detailed Forecast")
            disp = fc_df.copy()
            disp["Category"] = disp["predicted_aqi"].apply(aqi_label)
            disp = disp.rename(columns={"hour": "Hour Ahead", "datetime": "Forecast Time",
                                        "predicted_aqi": "Predicted AQI", "band": "Model Band"})
            st.dataframe(
                disp[["Hour Ahead", "Forecast Time", "Predicted AQI", "Model Band", "Category"]],
                use_container_width=True, hide_index=True)

            st.markdown("##### \U0001f4c8 Forecast Distribution")
            fig_dist = px.histogram(fc_df, x="predicted_aqi", nbins=20, color="band",
                                    color_discrete_map=BRAND)
            fig_dist.update_layout(height=350, xaxis_title="Predicted AQI",
                                   yaxis_title="Count", **_CHART_LAYOUT)
            st.plotly_chart(fig_dist, use_container_width=True)

    elif page == "\U0001f4ca Historical Data":
        st.markdown("<h1 style='font-family:Inter,sans-serif;font-weight:700;"
                    "letter-spacing:-0.02em;'>Historical AQI Data</h1>",
                    unsafe_allow_html=True)

        if hist_df.empty:
            st.warning("No historical data found in MongoDB.")
        else:
            min_dt, max_dt = hist_df["datetime"].min().date(), hist_df["datetime"].max().date()
            c1, c2 = st.columns(2)
            start_date = c1.date_input("From", value=min_dt, min_value=min_dt, max_value=max_dt)
            end_date = c2.date_input("To", value=max_dt, min_value=min_dt, max_value=max_dt)

            mask = (hist_df["datetime"].dt.date >= start_date) & (hist_df["datetime"].dt.date <= end_date)
            filtered = hist_df[mask]

            if filtered.empty:
                st.info("No data in selected range.")
            else:
                st.plotly_chart(render_historical_chart(filtered), use_container_width=True)

                st.markdown("##### \U0001f4ca Summary Statistics")
                aqi_data = filtered["us_aqi"].dropna()
                sc = st.columns(5)
                sc[0].metric("Mean AQI", f"{aqi_data.mean():.1f}")
                sc[1].metric("Median AQI", f"{aqi_data.median():.1f}")
                sc[2].metric("Peak AQI", f"{aqi_data.max():.0f}")
                sc[3].metric("Min AQI", f"{aqi_data.min():.0f}")
                sc[4].metric("Data Points", f"{len(aqi_data):,}")

                st.divider()

                st.markdown("##### \U0001f550 Average AQI by Hour of Day")
                hourly = filtered.copy()
                hourly["hour"] = hourly["datetime"].dt.hour
                hourly_avg = hourly.groupby("hour")["us_aqi"].mean().reset_index()
                fig_h = px.bar(hourly_avg, x="hour", y="us_aqi",
                               labels={"hour": "Hour of Day", "us_aqi": "Average AQI"},
                               color="us_aqi",
                               color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"])
                fig_h.update_layout(height=360, **_CHART_LAYOUT)
                fig_h.update_traces(marker_cornerradius=5)
                st.plotly_chart(fig_h, use_container_width=True)

                st.markdown("##### \U0001f9ea Pollutant Trends")
                poll_map = {"pm2_5": "PM2.5 (\u00b5g/m\u00b3)", "pm10": "PM10 (\u00b5g/m\u00b3)",
                            "ozone": "Ozone (\u00b5g/m\u00b3)", "nitrogen_dioxide": "NO\u2082 (\u00b5g/m\u00b3)",
                            "sulphur_dioxide": "SO\u2082 (\u00b5g/m\u00b3)",
                            "carbon_monoxide": "CO (\u00b5g/m\u00b3)"}
                avail_p = [p for p in poll_map if p in filtered.columns]
                if avail_p:
                    sel_p = st.multiselect("Select pollutants", avail_p,
                                           default=avail_p[:3],
                                           format_func=lambda x: poll_map.get(x, x))
                    if sel_p:
                        fig_p = go.Figure()
                        palette = [BRAND["primary"], BRAND["danger"], BRAND["success"],
                                   BRAND["warning"], BRAND["secondary"], BRAND["accent"]]
                        for i, p in enumerate(sel_p):
                            fig_p.add_trace(go.Scatter(
                                x=filtered["datetime"], y=filtered[p],
                                name=poll_map[p],
                                line=dict(color=palette[i % len(palette)], width=1.8)))
                        fig_p.update_layout(xaxis_title="Date / Time",
                                            yaxis_title="Concentration",
                                            height=400, hovermode="x unified",
                                            **_CHART_LAYOUT)
                        st.plotly_chart(fig_p, use_container_width=True)

                with st.expander("\U0001f4cb View Raw Data"):
                    show = ["datetime", "us_aqi"] + avail_p
                    show = [c for c in show if c in filtered.columns]
                    st.dataframe(filtered[show].tail(120), use_container_width=True,
                                 hide_index=True)

    elif page == "\U0001f52c EDA & Analytics":
        st.markdown("<h1 style='font-family:Inter,sans-serif;font-weight:700;"
                    "letter-spacing:-0.02em;'>Exploratory Data Analysis</h1>",
                    unsafe_allow_html=True)

        if hist_df.empty:
            st.warning("No data available for analysis.")
        else:
            st.markdown("##### \U0001f4ca AQI Distribution")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = px.histogram(hist_df.dropna(subset=["us_aqi"]), x="us_aqi",
                                   nbins=40, title="AQI Value Distribution",
                                   labels={"us_aqi": "US AQI"},
                                   color_discrete_sequence=[BRAND["primary"]])
                fig.update_layout(height=360, **_CHART_LAYOUT)
                fig.update_traces(marker_cornerradius=4)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(hist_df.dropna(subset=["us_aqi"]), y="us_aqi",
                             title="AQI Box Plot", labels={"us_aqi": "US AQI"},
                             color_discrete_sequence=[BRAND["secondary"]])
                fig.update_layout(height=360, **_CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("##### \U0001f3af AQI Category Breakdown")
            cats = hist_df.dropna(subset=["us_aqi"]).copy()
            cats["category"] = cats["us_aqi"].apply(aqi_label)
            cc = cats["category"].value_counts().reset_index()
            cc.columns = ["Category", "Count"]
            cmap = {label: color for _, _, label, color in AQI_LEVELS}
            fig = px.pie(cc, values="Count", names="Category", color="Category",
                         color_discrete_map=cmap, hole=0.4,
                         title="Time Spent in Each AQI Category")
            fig.update_layout(height=420, **_CHART_LAYOUT)
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              textfont_size=12)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("##### \U0001f517 Feature Correlations")
            corr_cols = ["us_aqi", "temperature_2m", "relative_humidity_2m",
                         "surface_pressure", "cloud_cover", "windspeed_10m",
                         "pm2_5", "pm10", "ozone", "nitrogen_dioxide"]
            ac = [c for c in corr_cols if c in hist_df.columns]
            if len(ac) > 2:
                cd = hist_df[ac].dropna()
                if len(cd) > 10:
                    fig = px.imshow(cd.corr(), text_auto=".2f",
                                   color_continuous_scale="RdBu_r",
                                   title="Correlation Matrix")
                    fig.update_layout(height=500, **_CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("##### \U0001f5d3\ufe0f AQI Heatmap \u2014 Day \u00d7 Hour")
            hdf = hist_df.dropna(subset=["us_aqi"]).copy()
            hdf["hour"] = hdf["datetime"].dt.hour
            hdf["dow"] = hdf["datetime"].dt.day_name()
            piv = hdf.pivot_table(values="us_aqi", index="dow", columns="hour", aggfunc="mean")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            piv = piv.reindex([d for d in dow_order if d in piv.index])
            if not piv.empty:
                fig = px.imshow(piv, color_continuous_scale="YlOrRd",
                                labels={"x": "Hour", "y": "Day", "color": "AQI"},
                                title="Average AQI by Day & Hour")
                fig.update_layout(height=400, **_CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            dc1, dc2 = st.columns(2, gap="large")
            with dc1:
                if "dominant_pollutant" in hist_df.columns:
                    st.markdown("##### \U0001f3ed Dominant Pollutant")
                    dmc = hist_df["dominant_pollutant"].value_counts().reset_index()
                    dmc.columns = ["Pollutant", "Hours"]
                    fig = px.bar(dmc, x="Pollutant", y="Hours", color="Pollutant",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_layout(height=370, showlegend=False, **_CHART_LAYOUT)
                    fig.update_traces(marker_cornerradius=5)
                    st.plotly_chart(fig, use_container_width=True)
            with dc2:
                if "temperature_2m" in hist_df.columns:
                    st.markdown("##### \U0001f321\ufe0f Temperature vs AQI")
                    sdf = hist_df.dropna(subset=["us_aqi", "temperature_2m"])
                    fig = px.scatter(sdf, x="temperature_2m", y="us_aqi",
                                    color="us_aqi", opacity=0.45,
                                    color_continuous_scale=["#10b981", "#f59e0b",
                                                            "#ef4444", "#7e0023"],
                                    labels={"temperature_2m": "Temp (\u00b0C)",
                                            "us_aqi": "US AQI"})
                    fig.update_layout(height=370, **_CHART_LAYOUT)
                    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
                    st.plotly_chart(fig, use_container_width=True)

    elif page == "\U0001f916 Model Info":
        st.markdown("<h1 style='font-family:Inter,sans-serif;font-weight:700;"
                    "letter-spacing:-0.02em;'>Model Performance & Details</h1>",
                    unsafe_allow_html=True)

        if not model_loaded:
            st.warning("Models not loaded from MongoDB.")
        else:
            st.markdown("##### \U0001f4ca Per-Horizon Model Metrics")
            rows = []
            for h in KEY_HORIZONS:
                m = training_logs.get(h, {})
                band = "Short" if h <= 8 else ("Medium" if h <= 24 else "Long")
                rows.append({
                    "Horizon": f"t+{h}h",
                    "Band": band,
                    "R\u00b2": f"{m['r2']:.4f}" if isinstance(m.get('r2'), (int, float)) else "N/A",
                    "RMSE":   f"{m['rmse']:.2f}" if isinstance(m.get('rmse'), (int, float)) else "N/A",
                    "MAE":    f"{m['mae']:.2f}" if isinstance(m.get('mae'), (int, float)) else "N/A",
                    "Samples": f"{m['samples']:,}" if isinstance(m.get('samples'), int) else "N/A",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("##### \U0001f4c8 R² by Horizon")
            r2_data = [(h, training_logs.get(h, {}).get("r2", 0)) for h in KEY_HORIZONS
                       if isinstance(training_logs.get(h, {}).get("r2"), (int, float))]
            if r2_data:
                fig = go.Figure(go.Bar(
                    x=[f"t+{h}h" for h, _ in r2_data],
                    y=[r2 for _, r2 in r2_data],
                    marker=dict(
                        color=[BRAND["short"] if h <= 8 else
                               (BRAND["medium"] if h <= 24 else BRAND["long"])
                               for h, _ in r2_data],
                        cornerradius=5,
                    ),
                    text=[f"{r2:.3f}" for _, r2 in r2_data],
                    textposition="outside",
                ))
                fig.update_layout(yaxis_title="R²", height=380,
                                  yaxis=dict(range=[min(0, min(r for _, r in r2_data)) - 0.05, 1.05]),
                                  **_CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("##### \U0001f9ec Model Features")
            if feature_cols:
                st.info(f"The model uses **{len(feature_cols)}** engineered features.")
                with st.expander("View all features"):
                    n = len(feature_cols)
                    t = n // 3 + 1
                    f1, f2, f3 = st.columns(3)
                    for f in feature_cols[:t]:    f1.code(f, language=None)
                    for f in feature_cols[t:2*t]: f2.code(f, language=None)
                    for f in feature_cols[2*t:]:  f3.code(f, language=None)

            st.divider()
            st.markdown("##### \U0001f3d7\ufe0f Architecture")
            st.markdown("""
            | Parameter | Value |
            |:----------|:------|
            | **Algorithm** | LightGBM (Gradient Boosted Trees) |
            | **Strategy** | Per-horizon model (18 models for t+1h to t+72h) |
            | **Feature Scaling** | RobustScaler (fit on train only) |
            | **Lag Exclusion** | Lags shorter than horizon excluded per model |
            | **Future Weather** | Open-Meteo 72h forecast injected at inference |
            | **Early Stopping** | 10% internal validation, 50 rounds patience |
            | **Short (t+1–8h)** | 1500 trees, lr=0.01, leaves=63 |
            | **Medium (t+9–24h)** | 800 trees, lr=0.03, leaves=47 |
            | **Long (t+25–72h)** | 500 trees, lr=0.05, leaves=15 |
            """)


if __name__ == "__main__":
    main()
