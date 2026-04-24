"""
models/anomaly_detector.py
--------------------------
Uses Isolation Forest to detect statistically unusual delay patterns
in the BTS dataset (e.g., sudden spikes in a carrier-airport-month combo).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ANOMALY_FEATURES = [
    "delay_rate",
    "carrier_delay_pct",
    "weather_delay_pct",
    "nas_delay_pct",
    "late_aircraft_delay_pct",
    "cancel_rate",
]


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Identify anomalous carrier-airport-month records using Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ingestion.load_data() (must contain
        delay_rate, carrier_delay_pct, etc.).
    contamination : float
        Expected fraction of anomalies (default 5%).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Original df augmented with:
        - 'anomaly_score'  : raw IF decision score (lower = more anomalous)
        - 'is_anomaly'     : bool flag
        - 'anomaly_label'  : human-readable label
    """
    result = df.copy()

    # Only use columns that exist in the dataframe
    features = [f for f in ANOMALY_FEATURES if f in result.columns]
    X = result[features].fillna(0.0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    result["anomaly_score"] = iso.decision_function(X_scaled).round(4)
    result["is_anomaly"]    = iso.predict(X_scaled) == -1

    result["anomaly_label"] = result.apply(
        lambda r: _label_anomaly(r, df[features].mean()),
        axis=1,
    )

    n_anom = result["is_anomaly"].sum()
    print(f"[anomaly] Detected {n_anom} anomalies ({n_anom/len(result):.1%} of records)")
    return result


def _label_anomaly(row: pd.Series, means: pd.Series) -> str:
    """Generate a short human-readable anomaly reason."""
    if not row.get("is_anomaly", False):
        return "Normal"

    reasons = []
    if "delay_rate" in means.index and row.get("delay_rate", 0) > means["delay_rate"] * 1.5:
        reasons.append("very high delay rate")
    if "weather_delay_pct" in means.index and row.get("weather_delay_pct", 0) > means["weather_delay_pct"] * 2:
        reasons.append("extreme weather delays")
    if "carrier_delay_pct" in means.index and row.get("carrier_delay_pct", 0) > means["carrier_delay_pct"] * 2:
        reasons.append("carrier-driven delays")
    if "cancel_rate" in means.index and row.get("cancel_rate", 0) > means["cancel_rate"] * 2:
        reasons.append("high cancellation rate")

    if reasons:
        return "Anomaly: " + "; ".join(reasons)
    return "Anomaly: unusual delay pattern"


def get_anomaly_summary(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted summary of all anomalous records."""
    cols = ["carrier_name", "airport_name", "month", "delay_rate",
            "cancel_rate", "anomaly_score", "anomaly_label"]
    cols = [c for c in cols if c in anomaly_df.columns]
    return (
        anomaly_df[anomaly_df["is_anomaly"]][cols]
        .sort_values("anomaly_score")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from data.ingestion import load_data
    df = load_data()
    result = detect_anomalies(df)
    summary = get_anomaly_summary(result)
    print(summary.to_string())
