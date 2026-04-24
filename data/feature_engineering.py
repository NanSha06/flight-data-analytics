"""
data/feature_engineering.py
----------------------------
Transforms cleaned BTS data into model-ready features.

Features produced
-----------------
Numeric (scaled):
  arr_flights_log      – log1p of arriving flights (volume proxy)
  carrier_delay_pct    – share of delays caused by carrier
  weather_delay_pct    – share of delays caused by weather
  nas_delay_pct        – share of delays caused by NAS
  security_delay_pct   – share of delays caused by security
  late_aircraft_delay_pct – share of delays caused by late aircraft
  cancel_rate          – cancellation rate

Cyclic:
  month_sin / month_cos – cyclic encoding of month (1-12)

Categorical (encoded):
  carrier_enc          – label-encoded carrier code
  airport_enc          – label-encoded airport code
  distance_bucket      – quintile bucket of arr_flights (0-4)

Target:
  high_delay (binary)  – 1 if delay_rate > median(delay_rate)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Features that are numeric and need standard scaling
NUMERIC_FEATURES = [
    "arr_flights_log",
    "carrier_delay_pct",
    "weather_delay_pct",
    "nas_delay_pct",
    "security_delay_pct",
    "late_aircraft_delay_pct",
    "cancel_rate",
]

# All features fed to the model (order matters for SHAP)
ALL_FEATURES = NUMERIC_FEATURES + [
    "month_sin",
    "month_cos",
    "carrier_enc",
    "airport_enc",
    "distance_bucket",
]


def build_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    carrier_enc: LabelEncoder | None = None,
    airport_enc: LabelEncoder | None = None,
    threshold: float | None = None,
    distance_bins: list | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler, LabelEncoder, LabelEncoder, float, list]:
    """
    Engineer features and create the binary target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ingestion.load_data().
    scaler : StandardScaler, optional
        Pre-fitted scaler (used during inference).
    carrier_enc : LabelEncoder, optional
        Pre-fitted carrier label encoder (used during inference).
    airport_enc : LabelEncoder, optional
        Pre-fitted airport label encoder (used during inference).
    threshold : float, optional
        Pre-computed delay-rate median (used during inference).
    distance_bins : list, optional
        Pre-computed quantile bin edges for arr_flights (used during inference).
    fit : bool
        If True, fit encoders/scaler on the data. Set False for inference.

    Returns
    -------
    X, y, scaler, carrier_enc, airport_enc, threshold, distance_bins
    """
    feat = df.copy()

    # ── Log-scale flight volume ────────────────────────────────────────────
    feat["arr_flights_log"] = np.log1p(feat["arr_flights"])

    # ── Cyclic month encoding ──────────────────────────────────────────────
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # -- Distance bucket (quintile of arr_flights) -------------------------
    if fit:
        # Compute bin edges from training data; clip labels to [0, 4]
        _, bin_edges = pd.qcut(
            feat["arr_flights"], q=5, labels=False, duplicates="drop", retbins=True
        )
        distance_bins = bin_edges.tolist()
        feat["distance_bucket"] = pd.qcut(
            feat["arr_flights"], q=5, labels=False, duplicates="drop"
        ).fillna(0).astype(int)
    else:
        # For inference, assign bucket using stored bin edges
        bins = np.array(distance_bins)
        feat["distance_bucket"] = np.clip(
            np.searchsorted(bins[1:-1], feat["arr_flights"].values), 0, 4
        ).astype(int)

    # ── Label encode carrier and airport ──────────────────────────────────
    if fit:
        carrier_enc = LabelEncoder()
        airport_enc = LabelEncoder()
        feat["carrier_enc"] = carrier_enc.fit_transform(feat["carrier"])
        feat["airport_enc"] = airport_enc.fit_transform(feat["airport"])
    else:
        # Handle unseen labels gracefully during inference
        def safe_transform(enc: LabelEncoder, values):
            known = set(enc.classes_)
            mapped = [v if v in known else enc.classes_[0] for v in values]
            return enc.transform(mapped)

        feat["carrier_enc"] = safe_transform(carrier_enc, feat["carrier"])
        feat["airport_enc"] = safe_transform(airport_enc, feat["airport"])

    # ── Binary target ──────────────────────────────────────────────────────
    if fit:
        threshold = float(feat["delay_rate"].median())

    feat["high_delay"] = (feat["delay_rate"] > threshold).astype(int)

    # ── Scale numeric features ─────────────────────────────────────────────
    if fit:
        scaler = StandardScaler()
        feat[NUMERIC_FEATURES] = scaler.fit_transform(feat[NUMERIC_FEATURES])
    else:
        feat[NUMERIC_FEATURES] = scaler.transform(feat[NUMERIC_FEATURES])

    X = feat[ALL_FEATURES].copy()
    y = feat["high_delay"].copy()

    if fit:
        print(f"[features] X shape={X.shape} | "
              f"high_delay rate={y.mean():.1%} | threshold={threshold:.4f}")

    return X, y, scaler, carrier_enc, airport_enc, threshold, distance_bins


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train-test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_inference_row(
    carrier: str,
    airport: str,
    month: int,
    arr_flights: float,
    carrier_delay_pct: float,
    weather_delay_pct: float,
    nas_delay_pct: float,
    security_delay_pct: float,
    late_aircraft_delay_pct: float,
    cancel_rate: float,
    scaler: StandardScaler,
    carrier_enc: LabelEncoder,
    airport_enc: LabelEncoder,
    threshold: float,
    distance_bins: list | None = None,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for real-time inference.
    Compatible with the trained model's expected feature schema.
    """
    row = {
        "carrier": [carrier],
        "airport": [airport],
        "month": [month],
        "arr_flights": [arr_flights],
        "carrier_delay_pct": [carrier_delay_pct],
        "weather_delay_pct": [weather_delay_pct],
        "nas_delay_pct": [nas_delay_pct],
        "security_delay_pct": [security_delay_pct],
        "late_aircraft_delay_pct": [late_aircraft_delay_pct],
        "cancel_rate": [cancel_rate],
        "delay_rate": [0.0],  # dummy — not used in inference
    }
    df_row = pd.DataFrame(row)

    # Reuse build_features in inference mode (fit=False)
    X, _, _, _, _, _, _ = build_features(
        df_row,
        scaler=scaler,
        carrier_enc=carrier_enc,
        airport_enc=airport_enc,
        threshold=threshold,
        distance_bins=distance_bins,
        fit=False,
    )
    return X

