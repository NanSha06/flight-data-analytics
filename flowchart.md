# GenAI-Powered Flight Delay Analytics System — Architecture Flowchart

![System Architecture Flowchart](C:\Users\hp\.gemini\antigravity\brain\2dce2a33-dfb5-4ea2-ae40-f5f2a6168c1f\flight_analytics_flowchart_1777005038884.png)

---

## Layer-by-Layer Breakdown

### 🟦 Layer 1 — Data Ingestion & Cleaning
| File | Responsibility |
|------|---------------|
| `Airline_Delay_Cause.csv` | BTS source data: monthly carrier × airport delay stats |
| `data/ingestion.py` | Loads CSV, drops nulls, computes `delay_rate`, `cancel_rate`, and 5 delay-cause percentage columns |

### 🟦 Layer 2 — Feature Engineering
| File | Responsibility |
|------|---------------|
| `data/feature_engineering.py` | `log1p` flight volume, cyclic `month_sin/cos`, label-encodes carrier & airport, quintile distance bucket, StandardScaler → returns feature matrix **X** + binary target **y** (`high_delay`) |

---

### 🟣 Layer 3 — ML Models (trained in parallel on same cleaned data)

| Model | File | Output |
|-------|------|--------|
| **XGBoost Classifier** | `models/trainer.py` | Predicts `HIGH DELAY / ON-TIME` + calibrated probability + SHAP values → saved as `xgb_model.pkl` + `model_meta.json` |
| **Isolation Forest** | `models/anomaly_detector.py` | Detects statistically unusual carrier-airport-month records → `anomaly_score` + `is_anomaly` flag + human-readable label |

---

### 🟢 Layer 4 — GenAI Explanation Layer
| File | Responsibility |
|------|---------------|
| `utils/genai_explainer.py` | **Priority 1**: `gemini-1.5-flash` via `GEMINI_API_KEY` for rich contextual explanations. **Fallback**: Rule-based mock explainer using SHAP values + feature values (fully offline). Also handles chatbot Q&A (`answer_flight_query`). |

---

### 🟠 Layer 5 — Serving Layer

| Component | File | Details |
|-----------|------|---------|
| **Flask REST API** | `app.py` | Port 5000. Endpoints: `GET /health`, `GET /metadata`, `POST /predict`, `POST /explain`, `GET /anomalies` |
| **Streamlit Dashboard** | `dashboard.py` | 6 tabs: 📊 Overview · 📈 Delay Analysis · 🤖 Prediction · 🧠 AI Explanation · ⚠️ Anomalies · 💬 AI Chatbot |

---

## End-to-End Data Flow

```mermaid
flowchart TD
    A[("🗄️ Airline_Delay_Cause.csv\nBTS Source Data")] --> B

    B["📥 data/ingestion.py\nLoad · Clean · Compute Rates"] --> C

    C["⚙️ data/feature_engineering.py\nLog Transform · Encode · Scale\n→ Feature Matrix X + Target y"] --> D & E

    D["🤖 models/trainer.py\nXGBoost Classifier\n300 trees · SHAP\n→ xgb_model.pkl + model_meta.json"]
    E["🔍 models/anomaly_detector.py\nIsolation Forest\n200 trees · 5% contamination\n→ anomaly_score + is_anomaly"]

    D --> F
    E --> F

    F["🧠 utils/genai_explainer.py\nGemini-1.5-Flash API\nOR Rule-Based Mock\n→ Natural Language Explanations"]

    F --> G & H

    G["🌐 app.py — Flask API :5000\n/health /metadata\n/predict /explain /anomalies"]
    H["📊 dashboard.py — Streamlit\n📊 Overview · 📈 Analysis\n🤖 Prediction · 🧠 AI · ⚠️ Anomalies · 💬 Chat"]

    G --> I
    H --> I

    I[("👤 End User / Operations Manager")]

    style A fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style B fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style C fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style D fill:#2d1b69,stroke:#8b5cf6,color:#fff
    style E fill:#2d1b69,stroke:#8b5cf6,color:#fff
    style F fill:#064e3b,stroke:#10b981,color:#fff
    style G fill:#451a03,stroke:#f59e0b,color:#fff
    style H fill:#451a03,stroke:#f59e0b,color:#fff
    style I fill:#374151,stroke:#9ca3af,color:#fff
```

> [!NOTE]
> The Streamlit dashboard (`dashboard.py`) can run in **standalone mode** — it loads `xgb_model.pkl` directly and does not require the Flask API to be running. The Flask API is only needed if you want external REST access to the prediction endpoints.

> [!TIP]
> Set `GEMINI_API_KEY` in your `.env` file to unlock live AI explanations via Gemini. Without it, the offline rule-based explainer activates automatically with no code changes needed.
