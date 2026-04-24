"""
dashboard.py — Streamlit Interactive Dashboard
================================================
GenAI-Powered Flight Delay Analytics System

Tabs:
  📊 Overview          — KPI cards + dataset summary
  📈 Delay Analysis    — Heatmaps, bar charts, trends
  🤖 Prediction        — Real-time delay prediction form
  🧠 AI Explanation    — GenAI natural-language explanation
  ⚠️  Anomalies        — Isolation Forest anomaly report
  💬 AI Chatbot        — Query flight insights in plain English

Run: streamlit run dashboard.py
"""

import os
import sys
import json
import requests
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.ingestion import load_data
from data.feature_engineering import build_features, build_inference_row, ALL_FEATURES
from models.trainer import load_model
from models.anomaly_detector import detect_anomalies, get_anomaly_summary
from utils.genai_explainer import explain_prediction, answer_flight_query
from utils.visualizations import (
    delay_distribution, airline_performance_bar, monthly_trend,
    delay_cause_heatmap, airport_delay_map, prediction_gauge,
    shap_bar, anomaly_scatter, carrier_month_heatmap,
)

FLASK_URL = "http://127.0.0.1:5000"

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ FlightIQ — GenAI Delay Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp { background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%); }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(99,102,241,0.4);
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='30' cy='30' r='1' fill='rgba(255,255,255,0.06)'/%3E%3C/svg%3E");
  }
  .hero h1 { color: white; font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
  .hero p  { color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-top: 0.5rem; }

  /* KPI cards */
  .kpi-card {
    background: rgba(30,41,59,0.9);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    backdrop-filter: blur(10px);
  }
  .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(99,102,241,0.35); }
  .kpi-value { font-size: 2rem; font-weight: 800; color: #A5B4FC; }
  .kpi-label { font-size: 0.78rem; color: #94A3B8; font-weight: 500; margin-top: 0.2rem; letter-spacing: 0.5px; text-transform: uppercase; }
  .kpi-delta { font-size: 0.85rem; margin-top: 0.4rem; }

  /* Section headers */
  .section-title {
    font-size: 1.25rem; font-weight: 700; color: #E2E8F0;
    border-left: 4px solid #6366F1; padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
  }

  /* Prediction cards */
  .pred-card-high {
    background: linear-gradient(135deg, rgba(127,29,29,0.5), rgba(185,28,28,0.3));
    border: 1px solid #EF4444;
    border-radius: 16px; padding: 1.5rem; text-align: center;
  }
  .pred-card-low {
    background: linear-gradient(135deg, rgba(6,78,59,0.5), rgba(5,150,105,0.3));
    border: 1px solid #10B981;
    border-radius: 16px; padding: 1.5rem; text-align: center;
  }
  .pred-label { font-size: 1.6rem; font-weight: 800; margin-top: 0.5rem; }

  /* Explanation box */
  .explanation-box {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(99,102,241,0.4);
    border-left: 4px solid #6366F1;
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #CBD5E1;
  }

  /* Chat bubbles */
  .chat-user {
    background: rgba(79,70,229,0.25);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 15%;
    color: #E0E7FF;
  }
  .chat-bot {
    background: rgba(30,41,59,0.9);
    border: 1px solid rgba(100,116,139,0.3);
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 15% 0.5rem 0;
    color: #CBD5E1;
    line-height: 1.6;
  }

  /* Anomaly badge */
  .anomaly-badge {
    background: rgba(127,29,29,0.4);
    border: 1px solid #EF4444;
    color: #FCA5A5;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 12px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    background: rgba(30,41,59,0.5);
    color: #94A3B8;
    border: 1px solid rgba(100,116,139,0.2);
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    border-color: transparent !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95);
    border-right: 1px solid rgba(99,102,241,0.2);
  }

  /* Streamlit buttons */
  .stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 1.5rem; font-weight: 600;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ── Cached Data Loading ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_anomalies(_df):
    return detect_anomalies(_df)


@st.cache_data(show_spinner=False)
def compute_dataset_summary(_df):
    agg = _df.groupby("carrier_name")["delay_rate"].mean()
    month_agg = _df.groupby("month")["arr_flights"].sum()
    month_labels = {1:"January",2:"February",3:"March",4:"April",5:"May",
                    6:"June",7:"July",8:"August",9:"September",10:"October",
                    11:"November",12:"December"}
    return {
        "total_flights":           int(_df["arr_flights"].sum()),
        "avg_delay_rate":          float(_df["delay_rate"].mean()),
        "n_carriers":              int(_df["carrier"].nunique()),
        "n_airports":              int(_df["airport"].nunique()),
        "worst_carrier":           agg.idxmax(),
        "worst_carrier_delay_rate": float(agg.max()),
        "best_carrier":            agg.idxmin(),
        "best_carrier_delay_rate": float(agg.min()),
        "avg_weather_delay_pct":   float(_df["weather_delay_pct"].mean()) if "weather_delay_pct" in _df.columns else 0,
        "avg_cancel_rate":         float(_df["cancel_rate"].mean()) if "cancel_rate" in _df.columns else 0,
        "busiest_month":           month_labels.get(int(month_agg.idxmax()), "Unknown"),
    }


# ── Load Everything ────────────────────────────────────────────────────────
with st.spinner("🔄 Loading data and model …"):
    df      = get_data()
    bundle  = get_model()
    anom_df = get_anomalies(df)
    summary = compute_dataset_summary(df)

MODEL       = bundle["model"]
SCALER      = bundle["scaler"]
CARRIER_ENC = bundle["carrier_enc"]
AIRPORT_ENC = bundle["airport_enc"]
THRESHOLD   = bundle["threshold"]

CARRIERS = sorted(df["carrier"].unique().tolist())
AIRPORTS = sorted(df["airport"].unique().tolist())
CARRIER_MAP = dict(zip(df["carrier"], df["carrier_name"]))
AIRPORT_MAP = dict(zip(df["airport"], df["airport_name"]))

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ FlightIQ")
    st.markdown("**GenAI Flight Delay Analytics**")
    st.markdown("---")

    st.markdown("### 📌 Dataset Info")
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown(f"**Carriers:** {df['carrier'].nunique()}")
    st.markdown(f"**Airports:** {df['airport'].nunique()}")
    st.markdown(f"**Period:** {df['year'].min()}–{df['year'].max()}")
    st.markdown(f"**Delay threshold:** {THRESHOLD:.1%}")
    st.markdown("---")

    # Load model meta
    meta_path = os.path.join(BASE_DIR, "models", "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        metrics = meta.get("metrics", {})
        st.markdown("### 🎯 Model Performance")
        for k, v in metrics.items():
            colour = "#10B981" if v >= 0.80 else "#F59E0B" if v >= 0.70 else "#EF4444"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin:4px 0">'
                f'<span style="color:#94A3B8">{k.replace("_"," ").title()}</span>'
                f'<span style="color:{colour};font-weight:700">{v:.4f}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    gemini_key = st.text_input("🔑 Gemini API Key (optional)", type="password",
                               help="Set for live GenAI explanations; leave blank for smart mock")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        st.success("✅ Gemini API key set!")


# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>✈️ FlightIQ — GenAI Delay Analytics</h1>
  <p>XGBoost · SHAP · Isolation Forest · Generative AI Explanations · BTS Dataset</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "📈 Delay Analysis",
    "🤖 Prediction",
    "🧠 AI Explanation",
    "⚠️ Anomalies",
    "💬 AI Chatbot",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{summary['total_flights']:,.0f}</div>
          <div class="kpi-label">Total Arriving Flights</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{summary['avg_delay_rate']:.1%}</div>
          <div class="kpi-label">Avg Delay Rate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{summary['n_carriers']}</div>
          <div class="kpi-label">Airlines Tracked</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{summary['n_airports']}</div>
          <div class="kpi-label">Airports Covered</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        worst_rate = summary['worst_carrier_delay_rate']
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#F87171;font-size:1.4rem">{summary['worst_carrier']}</div>
          <div class="kpi-label">Most Delayed Airline</div>
          <div class="kpi-delta" style="color:#F87171">● {worst_rate:.1%} avg delay rate</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        best_rate = summary['best_carrier_delay_rate']
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#34D399;font-size:1.4rem">{summary['best_carrier']}</div>
          <div class="kpi-label">Best On-Time Airline</div>
          <div class="kpi-delta" style="color:#34D399">● {best_rate:.1%} avg delay rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Delay Rate Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(delay_distribution(df), use_container_width=True)

    st.markdown('<div class="section-title">Raw Dataset (first 200 rows)</div>', unsafe_allow_html=True)
    preview_cols = ["year","month","carrier_name","airport_name",
                    "arr_flights","arr_del15","delay_rate","cancel_rate"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    st.dataframe(
        df[preview_cols].head(200).style.format({
            "delay_rate": "{:.1%}",
            "cancel_rate": "{:.2%}",
            "arr_flights": "{:,.0f}",
            "arr_del15": "{:,.0f}",
        }),
        use_container_width=True, height=350,
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DELAY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Airline Performance</div>', unsafe_allow_html=True)
    st.plotly_chart(airline_performance_bar(df), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Monthly Trend</div>', unsafe_allow_html=True)
        st.plotly_chart(monthly_trend(df), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">Carrier × Month Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(carrier_month_heatmap(df), use_container_width=True)

    st.markdown('<div class="section-title">Delay Cause Breakdown by Airline</div>', unsafe_allow_html=True)
    st.plotly_chart(delay_cause_heatmap(df), use_container_width=True)

    st.markdown('<div class="section-title">Top 30 Most Delayed Airports</div>', unsafe_allow_html=True)
    st.plotly_chart(airport_delay_map(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">🤖 Real-Time Delay Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        "Fill in the flight details below. The XGBoost model will predict whether "
        "this carrier-airport-month combination has a **HIGH** or **LOW** delay risk.",
    )

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            sel_carrier = st.selectbox("✈️ Airline (Carrier Code)", CARRIERS,
                                       format_func=lambda c: f"{c} — {CARRIER_MAP.get(c, c)}")
            sel_month   = st.slider("📅 Month", 1, 12, 7)
            arr_flights = st.number_input("🛬 Arriving Flights", min_value=10,
                                          max_value=50000, value=500, step=50)

        with col2:
            sel_airport = st.selectbox("🏢 Airport Code", AIRPORTS,
                                       format_func=lambda a: f"{a} — {AIRPORT_MAP.get(a, a)}")
            cancel_rate = st.slider("❌ Cancellation Rate", 0.0, 0.15, 0.02, 0.001,
                                    format="%.3f")

        with col3:
            st.markdown("**Delay Cause Mix** *(must sum to ~1)*")
            carrier_pct    = st.slider("Carrier Delays %",       0.0, 1.0, 0.35, 0.01)
            weather_pct    = st.slider("Weather Delays %",       0.0, 1.0, 0.10, 0.01)
            nas_pct        = st.slider("NAS / ATC Delays %",     0.0, 1.0, 0.25, 0.01)
            security_pct   = st.slider("Security Delays %",      0.0, 0.1, 0.01, 0.005)
            late_ac_pct    = st.slider("Late Aircraft Delays %", 0.0, 1.0, 0.29, 0.01)

        submitted = st.form_submit_button("🚀 Predict Delay Risk", use_container_width=True)

    if submitted:
        with st.spinner("Running prediction …"):
            payload = {
                "carrier":                 sel_carrier,
                "airport":                 sel_airport,
                "carrier_name":            CARRIER_MAP.get(sel_carrier, sel_carrier),
                "airport_name":            AIRPORT_MAP.get(sel_airport, sel_airport),
                "month":                   sel_month,
                "arr_flights":             arr_flights,
                "carrier_delay_pct":       carrier_pct,
                "weather_delay_pct":       weather_pct,
                "nas_delay_pct":           nas_pct,
                "security_delay_pct":      security_pct,
                "late_aircraft_delay_pct": late_ac_pct,
                "cancel_rate":             cancel_rate,
            }

            # Try Flask API first, fall back to direct model call
            pred_result = None
            try:
                resp = requests.post(f"{FLASK_URL}/predict", json=payload, timeout=5)
                if resp.status_code == 200:
                    pred_result = resp.json()
            except Exception:
                pass

            if pred_result is None:
                # Direct model call (standalone mode)
                from data.feature_engineering import build_inference_row
                X_row = build_inference_row(
                    carrier=sel_carrier, airport=sel_airport, month=sel_month,
                    arr_flights=arr_flights,
                    carrier_delay_pct=carrier_pct, weather_delay_pct=weather_pct,
                    nas_delay_pct=nas_pct, security_delay_pct=security_pct,
                    late_aircraft_delay_pct=late_ac_pct, cancel_rate=cancel_rate,
                    scaler=SCALER, carrier_enc=CARRIER_ENC,
                    airport_enc=AIRPORT_ENC, threshold=THRESHOLD,
                )
                prediction  = int(MODEL.predict(X_row)[0])
                probability = float(MODEL.predict_proba(X_row)[0][1])
                shap_values = {}
                try:
                    import shap
                    exp_sv = shap.TreeExplainer(MODEL).shap_values(X_row)
                    shap_values = dict(zip(ALL_FEATURES, exp_sv[0].tolist()))
                except Exception:
                    pass
                pred_result = {
                    "prediction": prediction, "probability": probability,
                    "shap_values": shap_values, "threshold": THRESHOLD,
                    "label": "HIGH DELAY RISK" if prediction == 1 else "LOW DELAY RISK",
                }

            st.session_state["pred_result"] = pred_result
            st.session_state["pred_payload"] = payload

        # ── Display results ────────────────────────────────────────────────
        prediction  = pred_result["prediction"]
        probability = pred_result["probability"]
        label       = pred_result["label"]
        shap_values = pred_result.get("shap_values", {})

        col_g, col_d = st.columns([1, 1])
        with col_g:
            st.plotly_chart(prediction_gauge(probability, prediction),
                            use_container_width=True)
        with col_d:
            card_cls = "pred-card-high" if prediction == 1 else "pred-card-low"
            colour   = "#F87171" if prediction == 1 else "#34D399"
            icon     = "⚠️" if prediction == 1 else "✅"
            st.markdown(f"""
            <div class="{card_cls}" style="margin-top:1rem">
              <div style="font-size:3rem">{icon}</div>
              <div class="pred-label" style="color:{colour}">{label}</div>
              <div style="color:#94A3B8;margin-top:0.5rem">
                Confidence: <b style="color:{colour}">{probability:.1%}</b>
              </div>
              <div style="color:#64748B;font-size:0.8rem;margin-top:0.5rem">
                Carrier: {CARRIER_MAP.get(sel_carrier, sel_carrier)}<br>
                Airport: {AIRPORT_MAP.get(sel_airport, sel_airport)}<br>
                Month: {sel_month} | Flights: {arr_flights:,}
              </div>
            </div>
            """, unsafe_allow_html=True)

        if shap_values:
            st.markdown('<div class="section-title">Feature Impact (SHAP)</div>', unsafe_allow_html=True)
            from utils.visualizations import FEATURE_LABELS
            st.plotly_chart(shap_bar(shap_values, FEATURE_LABELS), use_container_width=True)

        st.info("👉 Go to the **🧠 AI Explanation** tab for a natural-language explanation of this prediction.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — AI EXPLANATION
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🧠 GenAI-Powered Explanation</div>', unsafe_allow_html=True)

    if "pred_result" not in st.session_state:
        st.info("⬅️ Go to the **🤖 Prediction** tab, fill in the form, and click **Predict** first.")
    else:
        pred_result = st.session_state["pred_result"]
        payload     = st.session_state["pred_payload"]

        prediction  = pred_result["prediction"]
        probability = pred_result["probability"]
        shap_values = pred_result.get("shap_values", {})

        with st.spinner("Generating AI explanation …"):
            explanation = None
            try:
                resp = requests.post(
                    f"{FLASK_URL}/explain",
                    json={**payload, "prediction": prediction,
                          "probability": probability, "shap_values": shap_values},
                    timeout=15,
                )
                if resp.status_code == 200:
                    explanation = resp.json().get("explanation")
            except Exception:
                pass

            if not explanation:
                explanation = explain_prediction(payload, prediction, probability, shap_values)

        api_status = "🟢 Gemini API" if os.getenv("GEMINI_API_KEY") else "🟡 Smart Mock (set Gemini key for live AI)"
        st.caption(f"Explanation source: {api_status}")

        st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)

        # Feature breakdown table
        st.markdown('<div class="section-title">Input Feature Summary</div>', unsafe_allow_html=True)
        from utils.visualizations import FEATURE_LABELS
        feature_display = {
            "Carrier":              payload.get("carrier_name"),
            "Airport":              payload.get("airport_name"),
            "Month":                payload.get("month"),
            "Arriving Flights":     f"{payload.get('arr_flights', 0):,}",
            "Carrier Delay %":      f"{payload.get('carrier_delay_pct', 0):.1%}",
            "Weather Delay %":      f"{payload.get('weather_delay_pct', 0):.1%}",
            "NAS/ATC Delay %":      f"{payload.get('nas_delay_pct', 0):.1%}",
            "Security Delay %":     f"{payload.get('security_delay_pct', 0):.1%}",
            "Late Aircraft %":      f"{payload.get('late_aircraft_delay_pct', 0):.1%}",
            "Cancellation Rate":    f"{payload.get('cancel_rate', 0):.2%}",
            "Prediction":           "HIGH DELAY RISK ⚠️" if prediction == 1 else "LOW DELAY RISK ✅",
            "Confidence":           f"{probability:.1%}",
        }
        st.table(pd.DataFrame(
            list(feature_display.items()), columns=["Feature", "Value"]
        ).set_index("Feature"))


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — ANOMALIES
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">⚠️ Anomaly Detection — Isolation Forest</div>', unsafe_allow_html=True)

    anomaly_summary = get_anomaly_summary(anom_df)
    total_anom = len(anomaly_summary)

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#F87171">{total_anom}</div>
          <div class="kpi-label">Anomalous Records</div>
        </div>""", unsafe_allow_html=True)
    with col_a2:
        pct = total_anom / len(df) * 100
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#FB923C">{pct:.1f}%</div>
          <div class="kpi-label">Of Total Records</div>
        </div>""", unsafe_allow_html=True)
    with col_a3:
        top_carrier = (
            anomaly_summary["carrier_name"].value_counts().index[0]
            if len(anomaly_summary) > 0 and "carrier_name" in anomaly_summary.columns
            else "N/A"
        )
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#FBBF24;font-size:1.2rem">{top_carrier}</div>
          <div class="kpi-label">Most Anomalous Carrier</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(anomaly_scatter(anom_df), use_container_width=True)

    st.markdown('<div class="section-title">Anomalous Records</div>', unsafe_allow_html=True)
    if len(anomaly_summary) > 0:
        fmt_cols = {}
        if "delay_rate" in anomaly_summary.columns:
            fmt_cols["delay_rate"] = "{:.1%}"
        if "cancel_rate" in anomaly_summary.columns:
            fmt_cols["cancel_rate"] = "{:.2%}"
        if "anomaly_score" in anomaly_summary.columns:
            fmt_cols["anomaly_score"] = "{:.4f}"
        st.dataframe(
            anomaly_summary.style.format(fmt_cols),
            use_container_width=True,
            height=400,
        )
    else:
        st.success("No anomalies detected.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">💬 AI Flight Insights Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        "Ask any question about the flight dataset in plain English. "
        "Powered by GenAI (Gemini) with a smart fallback.\n\n"
        "**Try:** *Which airline has the worst delays?* · *What causes most delays?* · "
        "*Which month is busiest?* · *Tell me about cancellations.*"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        css_cls = "chat-user" if role == "user" else "chat-bot"
        icon    = "🧑" if role == "user" else "🤖"
        st.markdown(f'<div class="{css_cls}">{icon} {msg}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", placeholder="Which airline delays the most?")
        send       = st.form_submit_button("Send ✈️", use_container_width=True)

    if send and user_input.strip():
        with st.spinner("Thinking …"):
            answer = answer_flight_query(user_input.strip(), summary)
        st.session_state.chat_history.append(("user", user_input.strip()))
        st.session_state.chat_history.append(("bot",  answer))
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
