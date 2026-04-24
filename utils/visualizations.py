"""
utils/visualizations.py
------------------------
Reusable Plotly and Seaborn chart factories for the Streamlit dashboard
and Flask API.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Palette ────────────────────────────────────────────────────────────────
PRIMARY   = "#6366F1"   # indigo
SECONDARY = "#F59E0B"   # amber
DANGER    = "#EF4444"   # red
SUCCESS   = "#10B981"   # emerald
BG        = "#0F172A"   # slate-900
SURFACE   = "#1E293B"   # slate-800
TEXT      = "#F1F5F9"   # slate-100

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Inter, sans-serif", size=13),
    margin=dict(l=40, r=40, t=50, b=40),
)

# Re-export for convenience (dashboard imports FEATURE_LABELS from here)
FEATURE_LABELS = {
    "arr_flights_log":          "Flight Volume (log)",
    "carrier_delay_pct":        "Carrier-caused Delay Share",
    "weather_delay_pct":        "Weather-caused Delay Share",
    "nas_delay_pct":            "NAS/ATC-caused Delay Share",
    "security_delay_pct":       "Security-caused Delay Share",
    "late_aircraft_delay_pct":  "Late Aircraft Delay Share",
    "cancel_rate":              "Cancellation Rate",
    "month_sin":                "Month (sin)",
    "month_cos":                "Month (cos)",
    "carrier_enc":              "Airline Carrier",
    "airport_enc":              "Airport",
    "distance_bucket":          "Route Distance Bucket",
}



def delay_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of delay rates across all carrier-airport combos."""
    fig = px.histogram(
        df, x="delay_rate",
        nbins=40,
        color_discrete_sequence=[PRIMARY],
        labels={"delay_rate": "Delay Rate (fraction of flights delayed ≥15 min)"},
        title="Flight Delay Rate Distribution",
    )
    median_val = df["delay_rate"].median()
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color=SECONDARY,
        annotation_text=f"Median: {median_val:.1%}",
        annotation_font_color=SECONDARY,
    )
    fig.update_layout(**LAYOUT_BASE)
    fig.update_traces(marker_line_color=BG, marker_line_width=0.5)
    return fig


def airline_performance_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of avg delay rate per carrier."""
    agg = (
        df.groupby("carrier_name")["delay_rate"]
        .mean()
        .reset_index()
        .sort_values("delay_rate", ascending=True)
    )
    agg["color"] = agg["delay_rate"].apply(
        lambda x: DANGER if x > agg["delay_rate"].median() else SUCCESS
    )

    fig = go.Figure(go.Bar(
        x=agg["delay_rate"],
        y=agg["carrier_name"],
        orientation="h",
        marker_color=agg["color"],
        text=agg["delay_rate"].apply(lambda x: f"{x:.1%}"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Airline On-Time Performance (Avg Delay Rate)",
        xaxis_title="Average Delay Rate",
        yaxis_title="",
        **LAYOUT_BASE,
    )
    return fig


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of avg delay rate by month."""
    agg = df.groupby("month")["delay_rate"].agg(["mean", "std"]).reset_index()
    month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    agg["month_name"] = agg["month"].map(month_labels)

    fig = go.Figure()
    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(agg["month_name"]) + list(agg["month_name"])[::-1],
        y=list(agg["mean"] + agg["std"]) + list((agg["mean"] - agg["std"]).clip(0))[::-1],
        fill="toself",
        fillcolor=f"rgba(99,102,241,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=agg["month_name"],
        y=agg["mean"],
        mode="lines+markers",
        name="Avg Delay Rate",
        line=dict(color=PRIMARY, width=3),
        marker=dict(size=8, color=SECONDARY),
    ))
    fig.update_layout(
        title="Monthly Delay Rate Trend",
        xaxis_title="Month",
        yaxis_title="Average Delay Rate",
        **LAYOUT_BASE,
    )
    return fig


def delay_cause_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: carrier × delay cause (average delay-cause percentages)."""
    cause_cols = {
        "carrier_delay_pct":        "Carrier",
        "weather_delay_pct":        "Weather",
        "nas_delay_pct":            "NAS/ATC",
        "security_delay_pct":       "Security",
        "late_aircraft_delay_pct":  "Late Aircraft",
    }
    available = {k: v for k, v in cause_cols.items() if k in df.columns}
    if not available:
        return go.Figure()

    agg = df.groupby("carrier_name")[list(available.keys())].mean()
    agg.columns = list(available.values())

    fig = px.imshow(
        agg.T,
        color_continuous_scale="RdYlGn_r",
        title="Delay Cause Breakdown by Airline (Heatmap)",
        labels={"color": "Avg Share of Delays"},
        aspect="auto",
    )
    fig.update_layout(**LAYOUT_BASE)
    fig.update_coloraxes(colorbar_title="Share")
    return fig


def airport_delay_map(df: pd.DataFrame) -> go.Figure:
    """Top-30 airports by delay rate — bubble bar chart."""
    agg = (
        df.groupby("airport_name")
        .agg(delay_rate=("delay_rate", "mean"), arr_flights=("arr_flights", "sum"))
        .reset_index()
        .nlargest(30, "delay_rate")
    )
    # Shorten long airport names
    agg["airport_short"] = agg["airport_name"].apply(
        lambda x: x[:35] + "…" if len(x) > 35 else x
    )

    agg["label"] = agg["arr_flights"].apply(lambda x: f"{x:,.0f} flights")

    fig = px.bar(
        agg.sort_values("delay_rate"),
        x="delay_rate",
        y="airport_short",
        orientation="h",
        color="delay_rate",
        color_continuous_scale="RdYlGn_r",
        text="label",
        hover_data={"arr_flights": ":,.0f", "delay_rate": ":.1%"},
        title="Top 30 Most Delayed Airports",
        labels={"delay_rate": "Avg Delay Rate", "airport_short": ""},
    )
    fig.update_layout(**LAYOUT_BASE)
    return fig


def prediction_gauge(probability: float, prediction: int) -> go.Figure:
    """Gauge chart showing delay probability."""
    color = DANGER if prediction == 1 else SUCCESS
    label = "HIGH DELAY RISK" if prediction == 1 else "LOW DELAY RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": f"<b>{label}</b>", "font": {"size": 16, "color": TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": SURFACE,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#14532d"},
                {"range": [40, 60], "color": "#713f12"},
                {"range": [60, 100],"color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": SECONDARY, "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(paper_bgcolor=BG, font=dict(color=TEXT), height=300,
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig


def shap_bar(shap_dict: dict, feature_labels: dict | None = None) -> go.Figure:
    """Horizontal bar chart of SHAP values for a prediction."""
    if not shap_dict:
        return go.Figure()

    labels = feature_labels or {}
    items  = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    names  = [labels.get(k, k) for k, _ in items]
    vals   = [v for _, v in items]
    colors = [DANGER if v > 0 else SUCCESS for v in vals]

    fig = go.Figure(go.Bar(
        x=vals[::-1],
        y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in vals[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Impact (SHAP Values) — Red = ↑ Delay Risk",
        xaxis_title="SHAP Value",
        **LAYOUT_BASE,
        height=350,
    )
    return fig


def anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter of delay_rate vs cancel_rate coloured by anomaly status."""
    if "is_anomaly" not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df,
        x="delay_rate",
        y="cancel_rate",
        color="is_anomaly",
        color_discrete_map={True: DANGER, False: PRIMARY},
        hover_data=["carrier_name", "airport_name", "month", "anomaly_label"]
            if "anomaly_label" in df.columns else ["carrier_name", "airport_name", "month"],
        size="arr_flights",
        size_max=18,
        title="Anomaly Detection — Delay Rate vs Cancellation Rate",
        labels={
            "delay_rate": "Delay Rate",
            "cancel_rate": "Cancellation Rate",
            "is_anomaly": "Anomaly",
        },
    )
    fig.update_layout(**LAYOUT_BASE)
    return fig


def carrier_month_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of avg delay rate: carrier × month."""
    pivot = (
        df.groupby(["carrier_name", "month"])["delay_rate"]
        .mean()
        .unstack(fill_value=0)
    )
    month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot.columns = [month_labels.get(c, str(c)) for c in pivot.columns]

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        title="Delay Rate Heatmap — Carrier × Month",
        labels={"color": "Avg Delay Rate"},
        aspect="auto",
    )
    fig.update_layout(**LAYOUT_BASE)
    return fig
