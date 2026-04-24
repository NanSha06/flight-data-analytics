# GenAI-Powered Flight Delay Analytics System

## Overview

Build a full-stack, explainable AI system for flight delay analytics using the BTS `Airline_Delay_Cause.csv` dataset (1,880 rows, 21 columns — aggregate monthly delay causes by carrier and airport for 2025). The system predicts whether a carrier-airport combo will have a **high delay rate** and explains predictions using a GenAI layer (Google Gemini API with graceful mock fallback).

> [!NOTE]
> The dataset is **aggregate-level**, not individual flight-level. We derive a binary classification target: `high_delay = 1` if the delay rate (`arr_del15 / arr_flights`) exceeds the median threshold. This is a realistic and appropriate use of this specific BTS dataset.

---

## Proposed Project Structure

```
flight_data_analytics/
├── data/
│   ├── __init__.py
│   ├── ingestion.py          # CSV loading, cleaning, type casting
│   └── feature_engineering.py # Feature creation, encoding, scaling
│
├── models/
│   ├── __init__.py
│   ├── trainer.py            # XGBoost training, eval, SHAP
│   └── anomaly_detector.py   # IsolationForest anomaly detection
│
├── utils/
│   ├── __init__.py
│   ├── genai_explainer.py    # Gemini API + mock fallback explanations
│   └── visualizations.py    # Reusable plot helpers (plotly/seaborn)
│
├── app.py                    # Flask REST API (/predict, /explain, /anomalies)
├── dashboard.py              # Streamlit interactive dashboard
├── requirements.txt
└── Airline_Delay_Cause.csv   # Source dataset (already present)
```

---

## Proposed Changes

### Data Layer

#### [NEW] data/__init__.py
Empty package init.

#### [NEW] data/ingestion.py
- Load `Airline_Delay_Cause.csv` with `pandas`
- Drop the single null row
- Compute derived columns: `delay_rate`, `carrier_delay_pct`, `weather_delay_pct`, `nas_delay_pct`, `late_aircraft_delay_pct`
- Return a clean DataFrame

#### [NEW] data/feature_engineering.py
- Binary target: `high_delay = 1` if `delay_rate > median(delay_rate)`
- Features:
  - `month` (1–12, cyclic sin/cos encoding)
  - `arr_flights` (log-scaled)
  - `carrier_delay_pct`, `weather_delay_pct`, `nas_delay_pct`, `late_aircraft_delay_pct`, `security_delay_pct`
  - `distance_bucket` (based on arr_flights quintiles)
  - `carrier` (Label Encoded)
  - `airport` (Label Encoded)
- Scale numeric features with `StandardScaler`
- Return `X_train, X_test, y_train, y_test`, encoders, scaler

---

### Models Layer

#### [NEW] models/__init__.py
Empty package init.

#### [NEW] models/trainer.py
- Train `XGBoostClassifier` (with `scale_pos_weight` for class balance)
- 80/20 train-test split, `random_state=42`
- Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
- Compute SHAP values for feature importance
- Save model to `models/xgb_model.pkl`

#### [NEW] models/anomaly_detector.py
- Use `IsolationForest` to flag unusual carrier-airport-month combinations
- Returns DataFrame with `anomaly_score` and `is_anomaly` flag

---

### Utils Layer

#### [NEW] utils/__init__.py
Empty package init.

#### [NEW] utils/genai_explainer.py
- **Primary**: Calls Google Gemini API (`gemini-pro`) with a structured prompt built from flight features + prediction
- **Fallback (mock)**: Rule-based natural language generation using feature values when API key is missing/fails
- Returns a clean English explanation string

#### [NEW] utils/visualizations.py
- Plotly-based reusable charts: delay distribution histogram, airline performance bar, monthly trends line, SHAP heatmap, anomaly scatter

---

### Backend (Flask)

#### [NEW] app.py
Three endpoints:
- `POST /predict` — Accepts JSON `{carrier, airport, month, arr_flights, ...}`, returns `{prediction, probability, shap_values}`
- `POST /explain` — Accepts same JSON + prediction, returns `{explanation}` (GenAI or mock)
- `GET /anomalies` — Returns list of detected anomalies from the dataset

---

### Frontend (Streamlit)

#### [NEW] dashboard.py
5-tab interactive dashboard:
1. **📊 Overview** — KPI cards (total flights, avg delay rate, top carrier, most delayed airport)
2. **📈 Delay Analysis** — Heatmap (carrier × month), bar chart (airline performance), monthly trend line
3. **🤖 Prediction** — Form to input flight details → calls Flask `/predict` → shows result with gauge chart
4. **🧠 AI Explanation** — Shows GenAI-generated explanation for the prediction
5. **⚠️ Anomalies** — Table + scatter of anomalous routes detected by IsolationForest

---

## Verification Plan

### Automated
```
# Install dependencies
pip install -r requirements.txt

# Run data pipeline unit check
python -c "from data.ingestion import load_data; from data.feature_engineering import build_features; df = load_data(); print(build_features(df)[0].shape)"

# Train model
python models/trainer.py

# Start Flask API
python app.py  # runs on :5000

# Start Streamlit
streamlit run dashboard.py  # runs on :8501
```

### Manual
- Verify all 5 Streamlit tabs render without errors
- Verify `/predict` returns `{prediction, probability}` JSON
- Verify `/explain` returns a non-empty natural language string
- Verify anomalies table shows flagged rows
