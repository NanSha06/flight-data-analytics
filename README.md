# GenAI-Powered Flight Delay Analytics System

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (saves to models/xgb_model.pkl)
python models/trainer.py

# 3. (Optional) Set Gemini key for live AI explanations
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4a. Run ONLY the Streamlit dashboard (standalone mode — no Flask needed)
streamlit run dashboard.py

# 4b. OR run Flask API + Streamlit together
#   Terminal 1:
python app.py
#   Terminal 2:
streamlit run dashboard.py
```

## Project Structure

```
flight_data_analytics/
├── data/
│   ├── ingestion.py          # CSV loading, cleaning
│   └── feature_engineering.py # Features, encoding, scaling
├── models/
│   ├── trainer.py            # XGBoost + SHAP + metrics
│   ├── anomaly_detector.py   # Isolation Forest
│   ├── xgb_model.pkl         # Saved model (after training)
│   └── model_meta.json       # Metrics + SHAP importances
├── utils/
│   ├── genai_explainer.py    # Gemini API + mock fallback
│   └── visualizations.py    # Plotly chart factories
├── app.py                    # Flask REST API
├── dashboard.py              # Streamlit dashboard (6 tabs)
├── requirements.txt
├── .env.example              # API key template
└── Airline_Delay_Cause.csv   # BTS source data
```

## API Endpoints (Flask — port 5000)

| Method | Endpoint     | Description                          |
|--------|--------------|--------------------------------------|
| GET    | /health      | Health check                         |
| GET    | /metadata    | Model metrics + feature importances  |
| POST   | /predict     | Delay risk prediction + SHAP         |
| POST   | /explain     | GenAI natural-language explanation   |
| GET    | /anomalies   | Isolation Forest anomaly records     |

### Example /predict Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "carrier": "DL",
    "airport": "ATL",
    "month": 7,
    "arr_flights": 12500,
    "carrier_delay_pct": 0.35,
    "weather_delay_pct": 0.10,
    "nas_delay_pct": 0.28,
    "security_delay_pct": 0.01,
    "late_aircraft_delay_pct": 0.26,
    "cancel_rate": 0.015
  }'
```

### Example Response

```json
{
  "label": "HIGH DELAY RISK",
  "prediction": 1,
  "probability": 0.7832,
  "shap_values": { "carrier_delay_pct": 0.12, "nas_delay_pct": 0.08, ... },
  "threshold": 0.2341
}
```

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| 📊 Overview | KPI cards, delay distribution, dataset preview |
| 📈 Delay Analysis | Airline performance, monthly trends, heatmaps |
| 🤖 Prediction | Real-time prediction form with gauge + SHAP bars |
| 🧠 AI Explanation | GenAI natural-language reasoning |
| ⚠️ Anomalies | Isolation Forest results, scatter plot |
| 💬 AI Chatbot | Plain-English Q&A about the dataset |

## GenAI Explanation

- **With Gemini key**: Uses `gemini-1.5-flash` for rich, contextual explanations
- **Without key**: Rule-based smart mock using SHAP + feature values (fully offline)
