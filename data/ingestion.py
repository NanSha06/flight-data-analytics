"""
data/ingestion.py
-----------------
Loads and cleans the BTS Airline_Delay_Cause.csv dataset.
Handles nulls, computes delay-cause breakdowns, and returns a
production-ready DataFrame for downstream feature engineering.
"""

import os
import pandas as pd
import numpy as np


# ── Column Definitions ─────────────────────────────────────────────────────
RAW_COLS = [
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct",
    "security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted",
    "arr_delay", "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay",
]

# Columns dropped because they are totals recomputed as percentages
DROP_COLS = ["arr_delay", "carrier_delay", "weather_delay",
             "nas_delay", "security_delay", "late_aircraft_delay"]


def load_data(filepath: str | None = None) -> pd.DataFrame:
    """
    Load and clean the BTS Airline Delay dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to Airline_Delay_Cause.csv
        in the same directory as this script (project root).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with derived delay-cause percentage columns.
    """
    if filepath is None:
        # Resolve relative to project root regardless of cwd
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base, "Airline_Delay_Cause.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place Airline_Delay_Cause.csv in the project root."
        )

    # ── Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── Drop rows with any nulls (only 1 such row in this dataset) ─────────
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Type coercion ──────────────────────────────────────────────────────
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["arr_flights"] = df["arr_flights"].astype(float)
    df["arr_del15"] = df["arr_del15"].astype(float)

    # ── Derived: delay rate (fraction of arriving flights delayed ≥15 min) ─
    df["delay_rate"] = df["arr_del15"] / df["arr_flights"].replace(0, np.nan)
    df["delay_rate"] = df["delay_rate"].fillna(0.0).clip(0, 1)

    # ── Derived: cancellation & diversion rates ────────────────────────────
    df["cancel_rate"] = df["arr_cancelled"] / df["arr_flights"].replace(0, np.nan)
    df["cancel_rate"] = df["cancel_rate"].fillna(0.0)

    # ── Derived: delay-cause percentages (share of total delayed count) ────
    total_delay_ct = (
        df["carrier_ct"] + df["weather_ct"] + df["nas_ct"]
        + df["security_ct"] + df["late_aircraft_ct"]
    ).replace(0, np.nan)

    df["carrier_delay_pct"]       = df["carrier_ct"]       / total_delay_ct
    df["weather_delay_pct"]       = df["weather_ct"]       / total_delay_ct
    df["nas_delay_pct"]           = df["nas_ct"]           / total_delay_ct
    df["security_delay_pct"]      = df["security_ct"]      / total_delay_ct
    df["late_aircraft_delay_pct"] = df["late_aircraft_ct"] / total_delay_ct

    for col in ["carrier_delay_pct", "weather_delay_pct", "nas_delay_pct",
                "security_delay_pct", "late_aircraft_delay_pct"]:
        df[col] = df[col].fillna(0.0)

    # ── Drop raw minute-total delay columns (replaced by percentages) ──────
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    print(f"[ingestion] Loaded {len(df):,} records | "
          f"{df['carrier'].nunique()} carriers | "
          f"{df['airport'].nunique()} airports")

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.describe())
