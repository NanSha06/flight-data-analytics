"""
app.py — Flask REST API
========================
Endpoints:
  POST /predict    — predict high-delay risk for a carrier-airport-month combo
  POST /explain    — generate GenAI natural-language explanation
  GET  /anomalies  — return anomalous records from the dataset
  GET  /health     — health check
  GET  /metadata   — model meta (metrics, feature names, SHAP importances)

Start: python app.py
Runs on: http://127.0.0.1:5000
"""

import os
import sys
import json
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Path setup (works regardless of cwd) ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.ingestion import load_data
from data.feature_engineering import build_features, build_inference_row, ALL_FEATURES
from models.trainer import load_model
from models.anomaly_detector import detect_anomalies, get_anomaly_summary
from utils.genai_explainer import explain_prediction

# ── App Initialisation ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow Streamlit to call Flask

# ── Load model once at startup ─────────────────────────────────────────────
print("[app] Loading model …")
BUNDLE = load_model()
MODEL       = BUNDLE["model"]
SCALER      = BUNDLE["scaler"]
CARRIER_ENC = BUNDLE["carrier_enc"]
AIRPORT_ENC = BUNDLE["airport_enc"]
THRESHOLD   = BUNDLE["threshold"]

# ── Pre-compute anomalies once ─────────────────────────────────────────────
print("[app] Computing anomalies …")
RAW_DF       = load_data()
ANOMALY_DF   = detect_anomalies(RAW_DF)
ANOMALY_LIST = get_anomaly_summary(ANOMALY_DF).to_dict(orient="records")

# ── Model metadata ─────────────────────────────────────────────────────────
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.json")
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        MODEL_META = json.load(f)
else:
    MODEL_META = {}

print("[app] Ready. Endpoints: /predict  /explain  /anomalies  /health  /metadata")


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_features(data: dict) -> dict:
    """Extract and validate required fields from request JSON."""
    required = ["carrier", "airport", "month", "arr_flights"]
    missing  = [r for r in required if r not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return {
        "carrier":                  str(data["carrier"]),
        "airport":                  str(data["airport"]),
        "month":                    int(data["month"]),
        "arr_flights":              float(data.get("arr_flights", 100)),
        "carrier_delay_pct":        float(data.get("carrier_delay_pct", 0.35)),
        "weather_delay_pct":        float(data.get("weather_delay_pct", 0.10)),
        "nas_delay_pct":            float(data.get("nas_delay_pct", 0.25)),
        "security_delay_pct":       float(data.get("security_delay_pct", 0.01)),
        "late_aircraft_delay_pct":  float(data.get("late_aircraft_delay_pct", 0.29)),
        "cancel_rate":              float(data.get("cancel_rate", 0.02)),
        # Human-readable extras (not model inputs, used for explanations)
        "carrier_name":  str(data.get("carrier_name", data["carrier"])),
        "airport_name":  str(data.get("airport_name", data["airport"])),
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/metadata", methods=["GET"])
def metadata():
    """Return model metadata, metrics, and feature importances."""
    return jsonify(MODEL_META)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict flight delay risk.

    Request JSON
    ------------
    {
      "carrier":   "DL",
      "airport":   "ATL",
      "month":     7,
      "arr_flights": 12500,
      "carrier_delay_pct":       0.35,
      "weather_delay_pct":       0.10,
      "nas_delay_pct":           0.28,
      "security_delay_pct":      0.01,
      "late_aircraft_delay_pct": 0.26,
      "cancel_rate":             0.015
    }

    Response JSON
    -------------
    {
      "prediction":   1,
      "label":        "HIGH DELAY RISK",
      "probability":  0.78,
      "shap_values":  { "carrier_delay_pct": 0.12, ... },
      "threshold":    0.2341
    }
    """
    try:
        data   = request.get_json(force=True)
        feats  = _parse_features(data)

        # Build model input
        X_row = build_inference_row(
            carrier=feats["carrier"],
            airport=feats["airport"],
            month=feats["month"],
            arr_flights=feats["arr_flights"],
            carrier_delay_pct=feats["carrier_delay_pct"],
            weather_delay_pct=feats["weather_delay_pct"],
            nas_delay_pct=feats["nas_delay_pct"],
            security_delay_pct=feats["security_delay_pct"],
            late_aircraft_delay_pct=feats["late_aircraft_delay_pct"],
            cancel_rate=feats["cancel_rate"],
            scaler=SCALER,
            carrier_enc=CARRIER_ENC,
            airport_enc=AIRPORT_ENC,
            threshold=THRESHOLD,
        )

        prediction  = int(MODEL.predict(X_row)[0])
        probability = float(MODEL.predict_proba(X_row)[0][1])

        # SHAP for this instance
        shap_values = {}
        try:
            import shap
            explainer     = shap.TreeExplainer(MODEL)
            sv            = explainer.shap_values(X_row)
            shap_values   = dict(zip(ALL_FEATURES, sv[0].tolist()))
        except Exception:
            pass

        return jsonify({
            "prediction":  prediction,
            "label":       "HIGH DELAY RISK" if prediction == 1 else "LOW DELAY RISK",
            "probability": round(probability, 4),
            "shap_values": shap_values,
            "threshold":   round(THRESHOLD, 4),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """
    Generate a GenAI natural-language explanation for a prediction.

    Request JSON
    ------------
    Same as /predict, PLUS optionally:
      "prediction":  1          (if omitted, /predict is called internally)
      "probability": 0.78
      "shap_values": { ... }

    Response JSON
    -------------
    { "explanation": "The flight is likely delayed due to …" }
    """
    try:
        data = request.get_json(force=True)

        # If prediction not provided, run inline prediction
        if "prediction" not in data:
            with app.test_client() as c:
                pred_resp = c.post(
                    "/predict",
                    json=data,
                    content_type="application/json",
                )
                pred_data = pred_resp.get_json()
            prediction  = pred_data.get("prediction", 0)
            probability = pred_data.get("probability", 0.5)
            shap_values = pred_data.get("shap_values", {})
        else:
            prediction  = int(data["prediction"])
            probability = float(data.get("probability", 0.5))
            shap_values = data.get("shap_values", {})

        feats       = _parse_features(data)
        explanation = explain_prediction(feats, prediction, probability, shap_values)

        return jsonify({"explanation": explanation})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/anomalies", methods=["GET"])
def anomalies():
    """
    Return detected anomalous carrier-airport-month records.

    Query params:
      limit   int  (default 50)

    Response JSON
    -------------
    { "count": 94, "anomalies": [ {...}, ... ] }
    """
    try:
        limit = int(request.args.get("limit", 50))
        return jsonify({
            "count":     len(ANOMALY_LIST),
            "anomalies": ANOMALY_LIST[:limit],
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
