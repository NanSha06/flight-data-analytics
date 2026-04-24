"""
models/trainer.py
-----------------
Trains an XGBoost binary classifier to predict high-delay carrier-airport
combinations from BTS monthly aggregate data.

Usage (standalone training)
---------------------------
    python models/trainer.py
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)

from data.ingestion import load_data
from data.feature_engineering import build_features, train_test_split_data, ALL_FEATURES

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")


def train(save: bool = True) -> dict:
    """
    Full training pipeline.

    Returns
    -------
    dict with keys: model, scaler, carrier_enc, airport_enc, threshold,
                    metrics, feature_names, shap_values, X_test, y_test
    """
    # ── Data ──────────────────────────────────────────────────────────────
    df = load_data()
    X, y, scaler, carrier_enc, airport_enc, threshold = build_features(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print(f"[trainer] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Class balance ──────────────────────────────────────────────────────
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # ── Model ──────────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    print("\n[trainer] -- Evaluation Metrics ---------------------------")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("\n[trainer] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))

    # ── SHAP feature importances ───────────────────────────────────────────
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = dict(zip(
            ALL_FEATURES,
            np.abs(shap_values).mean(axis=0).tolist()
        ))
    except Exception as e:
        print(f"[trainer] SHAP unavailable: {e}")
        shap_values   = None
        mean_abs_shap = {}

    # ── Persist artifacts ──────────────────────────────────────────────────
    if save:
        bundle = {
            "model":       model,
            "scaler":      scaler,
            "carrier_enc": carrier_enc,
            "airport_enc": airport_enc,
            "threshold":   threshold,
            "feature_names": ALL_FEATURES,
            "shap_mean_abs": mean_abs_shap,
        }
        joblib.dump(bundle, MODEL_PATH)
        print(f"\n[trainer] Model saved → {MODEL_PATH}")

        meta = {
            "metrics": metrics,
            "threshold": threshold,
            "feature_names": ALL_FEATURES,
            "shap_mean_abs": mean_abs_shap,
            "n_train": len(X_train),
            "n_test":  len(X_test),
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[trainer] Metadata saved → {META_PATH}")

    return {
        "model":        model,
        "scaler":       scaler,
        "carrier_enc":  carrier_enc,
        "airport_enc":  airport_enc,
        "threshold":    threshold,
        "metrics":      metrics,
        "feature_names": ALL_FEATURES,
        "shap_values":  shap_values,
        "X_test":       X_test,
        "y_test":       y_test,
        "mean_abs_shap": mean_abs_shap,
    }


def load_model() -> dict:
    """Load persisted model bundle. Train first if not present."""
    if not os.path.exists(MODEL_PATH):
        print("[trainer] No saved model found — training now …")
        return train(save=True)
    bundle = joblib.load(MODEL_PATH)
    print(f"[trainer] Model loaded from {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    results = train(save=True)
    print(f"\n✅  Training complete. F1-score: {results['metrics']['f1_score']:.4f}")
