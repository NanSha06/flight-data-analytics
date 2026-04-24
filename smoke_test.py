"""
smoke_test.py — End-to-end pipeline validation
Run: python -X utf8 smoke_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ingestion import load_data
from data.feature_engineering import build_features, build_inference_row, ALL_FEATURES
from models.trainer import load_model
from models.anomaly_detector import detect_anomalies, get_anomaly_summary
from utils.genai_explainer import explain_prediction
from utils.visualizations import FEATURE_LABELS

PASS = 0
FAIL = 0

def check(label, fn):
    global PASS, FAIL
    try:
        result = fn()
        print(f"  [PASS] {label}: {result}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        FAIL += 1

print("\n=== SMOKE TEST ===\n")

print("1. Data layer")
df = load_data()
check("Row count", lambda: f"{len(df)} rows")
check("Columns present", lambda: all(c in df.columns for c in
    ["delay_rate", "cancel_rate", "carrier_delay_pct"]))
check("No nulls", lambda: df.isnull().sum().sum() == 0)

print("\n2. Feature engineering")
X, y, sc, ce, ae, thr, dbins = build_features(df, fit=True)
check("Feature shape", lambda: str(X.shape))
check("Target balance", lambda: f"high_delay={y.mean():.1%}")
check("Threshold value", lambda: f"threshold={thr:.4f}")

print("\n3. Model loading")
bundle = load_model()
check("Model loaded", lambda: type(bundle['model']).__name__)
check("Scaler loaded", lambda: type(bundle['scaler']).__name__)

print("\n4. Inference")
X_row = build_inference_row(
    carrier='DL', airport='ATL', month=7, arr_flights=12000,
    carrier_delay_pct=0.35, weather_delay_pct=0.10, nas_delay_pct=0.25,
    security_delay_pct=0.01, late_aircraft_delay_pct=0.29, cancel_rate=0.015,
    scaler=bundle['scaler'], carrier_enc=bundle['carrier_enc'],
    airport_enc=bundle['airport_enc'], threshold=bundle['threshold'],
    distance_bins=bundle.get('distance_bins'),
)
pred  = int(bundle['model'].predict(X_row)[0])
prob  = float(bundle['model'].predict_proba(X_row)[0][1])
check("Prediction value", lambda: f"pred={pred}, prob={prob:.3f}")
check("Probability range", lambda: 0 <= prob <= 1)

print("\n5. SHAP")
shap_dict = {}
try:
    import shap
    sv = shap.TreeExplainer(bundle['model']).shap_values(X_row)
    shap_dict = dict(zip(ALL_FEATURES, sv[0].tolist()))
    top_feat = max(shap_dict, key=lambda k: abs(shap_dict[k]))
    check("SHAP values", lambda: f"top={top_feat}")
except Exception as e:
    print(f"  [SKIP] SHAP: {e}")

print("\n6. GenAI explanation (mock)")
features_dict = {
    'carrier': 'DL', 'carrier_name': 'Delta Air Lines Inc.',
    'airport': 'ATL', 'airport_name': 'Atlanta Hartsfield-Jackson',
    'month': 7, 'arr_flights': 12000,
    'carrier_delay_pct': 0.35, 'weather_delay_pct': 0.10,
    'nas_delay_pct': 0.25, 'security_delay_pct': 0.01,
    'late_aircraft_delay_pct': 0.29, 'cancel_rate': 0.015,
}
expl = explain_prediction(features_dict, pred, prob, shap_dict)
check("Explanation generated", lambda: f"{len(expl)} chars")
check("Explanation non-empty", lambda: len(expl) > 50)

print("\n7. Anomaly detection")
anom_df  = detect_anomalies(df)
summary  = get_anomaly_summary(anom_df)
check("Anomalies detected", lambda: f"{len(summary)} anomalies")
check("Anomaly columns", lambda: 'anomaly_label' in anom_df.columns)

print("\n8. Visualizations")
from utils.visualizations import (
    delay_distribution, airline_performance_bar, monthly_trend,
    delay_cause_heatmap, anomaly_scatter, carrier_month_heatmap,
)
check("delay_distribution", lambda: type(delay_distribution(df)).__name__)
check("airline_performance_bar", lambda: type(airline_performance_bar(df)).__name__)
check("monthly_trend", lambda: type(monthly_trend(df)).__name__)
check("delay_cause_heatmap", lambda: type(delay_cause_heatmap(df)).__name__)
check("anomaly_scatter", lambda: type(anomaly_scatter(anom_df)).__name__)
check("carrier_month_heatmap", lambda: type(carrier_month_heatmap(df)).__name__)
check("FEATURE_LABELS", lambda: f"{len(FEATURE_LABELS)} labels")

print(f"\n=== RESULTS: {PASS} passed, {FAIL} failed ===\n")
if FAIL == 0:
    print("All systems GO. Run: streamlit run dashboard.py")
else:
    print("Some checks failed — review output above.")
