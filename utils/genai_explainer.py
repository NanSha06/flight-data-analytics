"""
utils/genai_explainer.py
-------------------------
Converts XGBoost predictions + SHAP feature importances into natural-language
explanations.

Strategy (in priority order):
  1. Google Gemini API  — if GEMINI_API_KEY env var is set
  2. Rule-based mock    — rich, deterministic explanation from feature values

Set GEMINI_API_KEY in a .env file or as an environment variable to enable
live GenAI explanations.
"""

import os
import json
from typing import Optional

# Load .env file if present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Feature display names ──────────────────────────────────────────────────
FEATURE_LABELS = {
    "arr_flights_log":          "Flight Volume (log)",
    "carrier_delay_pct":        "Carrier-caused Delay Share",
    "weather_delay_pct":        "Weather-caused Delay Share",
    "nas_delay_pct":            "NAS/ATC-caused Delay Share",
    "security_delay_pct":       "Security-caused Delay Share",
    "late_aircraft_delay_pct":  "Late Aircraft Delay Share",
    "cancel_rate":              "Cancellation Rate",
    "month_sin":                "Month (seasonal component)",
    "month_cos":                "Month (seasonal component)",
    "carrier_enc":              "Airline Carrier",
    "airport_enc":              "Airport",
    "distance_bucket":          "Route Distance Bucket",
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ── Public API ─────────────────────────────────────────────────────────────

def explain_prediction(
    features: dict,
    prediction: int,
    probability: float,
    shap_dict: Optional[dict] = None,
) -> str:
    """
    Generate a natural-language explanation for a flight delay prediction.

    Parameters
    ----------
    features : dict
        Raw input features (carrier, airport, month, arr_flights,
        carrier_delay_pct, weather_delay_pct, nas_delay_pct,
        security_delay_pct, late_aircraft_delay_pct, cancel_rate).
    prediction : int
        Model output — 1 = High Delay, 0 = On-Time.
    probability : float
        Model confidence (0–1) for the predicted class.
    shap_dict : dict, optional
        {feature_name: shap_value} for the predicted instance.

    Returns
    -------
    str
        Human-readable explanation paragraph.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            return _gemini_explain(features, prediction, probability, shap_dict, api_key)
        except Exception as e:
            print(f"[genai] Gemini API error — falling back to mock: {e}")

    return _mock_explain(features, prediction, probability, shap_dict)


# ── Gemini Integration ─────────────────────────────────────────────────────

def _gemini_explain(
    features: dict,
    prediction: int,
    probability: float,
    shap_dict: Optional[dict],
    api_key: str,
) -> str:
    """Call Google Gemini API to generate an explanation."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    outcome = "HIGH DELAY RISK" if prediction == 1 else "ON-TIME / LOW DELAY RISK"
    shap_summary = _format_shap_summary(shap_dict) if shap_dict else "Not available."

    prompt = f"""
You are an expert aviation analyst. The following data describes a carrier-airport-month
combination from the US Bureau of Transportation Statistics (BTS) dataset.

=== Prediction ===
Outcome: {outcome}
Model confidence: {probability:.1%}

=== Input Features ===
Carrier: {features.get('carrier', 'Unknown')}
Airport: {features.get('airport', 'Unknown')}
Month: {MONTH_NAMES.get(int(features.get('month', 1)), 'Unknown')}
Arriving flights: {features.get('arr_flights', 0):,.0f}
Cancellation rate: {features.get('cancel_rate', 0):.1%}
Carrier delay share: {features.get('carrier_delay_pct', 0):.1%}
Weather delay share: {features.get('weather_delay_pct', 0):.1%}
NAS/ATC delay share: {features.get('nas_delay_pct', 0):.1%}
Late aircraft delay share: {features.get('late_aircraft_delay_pct', 0):.1%}

=== Top Contributing Factors (SHAP) ===
{shap_summary}

Write a concise, clear, 3–4 sentence explanation for a non-technical airline operations
manager explaining WHY this prediction was made and WHAT the key drivers are.
Be specific about the dominant delay causes. End with one actionable recommendation.
Do NOT use bullet points. Write in plain prose.
""".strip()

    response = model.generate_content(prompt)
    return response.text.strip()


# ── Rule-Based Mock Explainer ──────────────────────────────────────────────

def _mock_explain(
    features: dict,
    prediction: int,
    probability: float,
    shap_dict: Optional[dict],
) -> str:
    """
    Rich rule-based explanation that mimics GenAI output.
    Produces different text based on feature values and SHAP importances.
    """
    carrier    = features.get("carrier_name", features.get("carrier", "This carrier"))
    airport    = features.get("airport_name", features.get("airport", "this airport"))
    month_name = MONTH_NAMES.get(int(features.get("month", 1)), "this month")
    flights    = int(features.get("arr_flights", 0))
    conf       = probability

    # Identify dominant delay cause
    causes = {
        "carrier operations":  features.get("carrier_delay_pct", 0),
        "weather conditions":  features.get("weather_delay_pct", 0),
        "NAS / air traffic control": features.get("nas_delay_pct", 0),
        "late-arriving aircraft": features.get("late_aircraft_delay_pct", 0),
    }
    top_cause = max(causes, key=causes.get)
    top_pct   = causes[top_cause]

    cancel_rate = features.get("cancel_rate", 0)

    # ── High-Delay Explanation ─────────────────────────────────────────────
    if prediction == 1:
        intro = (
            f"Based on our XGBoost model trained on BTS data, the {carrier} "
            f"operation at {airport} during {month_name} is predicted to experience "
            f"a **HIGH DELAY RISK** (confidence: {conf:.1%}). "
        )

        cause_sent = (
            f"The single largest driver of this prediction is **{top_cause}**, "
            f"which accounts for approximately {top_pct:.0%} of all delayed flights "
            f"at this carrier-airport pair. "
        )

        volume_sent = ""
        if flights > 500:
            volume_sent = (
                f"The high traffic volume of {flights:,} arriving flights amplifies "
                f"congestion effects, making recovery from initial delays slower. "
            )

        cancel_sent = ""
        if cancel_rate > 0.02:
            cancel_sent = (
                f"A cancellation rate of {cancel_rate:.1%} further signals operational "
                f"stress at this location. "
            )

        shap_sent = ""
        if shap_dict:
            top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
            shap_names = [FEATURE_LABELS.get(k, k) for k, _ in top_shap]
            shap_sent = (
                f"SHAP analysis confirms that **{shap_names[0]}** and "
                f"**{shap_names[1]}** had the greatest positive impact on this "
                f"high-risk classification. "
            )

        recommendation = (
            f"**Recommendation:** Focus on reducing {top_cause.lower()} through "
            f"proactive scheduling buffers, closer coordination with ATC, and "
            f"ensuring adequate ground crew availability during peak {month_name} traffic."
        )

        return intro + cause_sent + volume_sent + cancel_sent + shap_sent + recommendation

    # ── On-Time / Low-Delay Explanation ───────────────────────────────────
    else:
        intro = (
            f"The {carrier} operation at {airport} during {month_name} is predicted "
            f"to be **ON-TIME / LOW DELAY** (confidence: {conf:.1%}). "
        )

        cause_sent = (
            f"Delay-cause analysis shows that {top_cause} "
            f"({top_pct:.0%} of delays) is the primary concern, but overall delay rates "
            f"remain below the BTS dataset median threshold. "
        )

        positive_sent = (
            f"With {flights:,} arriving flights and a well-controlled operational "
            f"footprint, this carrier-airport combination demonstrates strong "
            f"on-time performance for this period. "
        )

        shap_sent = ""
        if shap_dict:
            top_shap = sorted(shap_dict.items(), key=lambda x: x[1])[:2]
            shap_names = [FEATURE_LABELS.get(k, k) for k, _ in top_shap]
            shap_sent = (
                f"SHAP analysis indicates **{shap_names[0]}** contributed most "
                f"strongly to the on-time prediction. "
            )

        recommendation = (
            f"**Recommendation:** Maintain current scheduling discipline and monitor "
            f"{top_cause.lower()} trends — particularly heading into historically "
            f"high-traffic months."
        )

        return intro + cause_sent + positive_sent + shap_sent + recommendation


# ── Helper ─────────────────────────────────────────────────────────────────

def _format_shap_summary(shap_dict: dict, top_n: int = 5) -> str:
    """Format top-N SHAP values as a readable list for prompt injection."""
    ranked = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    lines = []
    for feat, val in ranked:
        direction = "↑ increases" if val > 0 else "↓ decreases"
        label = FEATURE_LABELS.get(feat, feat)
        lines.append(f"  • {label}: {direction} delay risk (SHAP={val:+.3f})")
    return "\n".join(lines)


# ── Chatbot-style query handler ────────────────────────────────────────────

def answer_flight_query(query: str, df_summary: dict) -> str:
    """
    Answer free-text questions about the flight dataset.
    Uses Gemini if available, else a keyword-based mock.

    Parameters
    ----------
    query : str
        User's natural language question.
    df_summary : dict
        Pre-computed dataset summary statistics.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            return _gemini_chat(query, df_summary, api_key)
        except Exception as e:
            print(f"[genai] Chat API error: {e}")

    return _mock_chat(query, df_summary)


def _gemini_chat(query: str, summary: dict, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    context = json.dumps(summary, indent=2)
    prompt = (
        f"You are a flight analytics expert. Answer the following question based "
        f"on this BTS dataset summary:\n\n{context}\n\nQuestion: {query}\n\n"
        f"Provide a concise, data-driven answer in 2-3 sentences."
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def _mock_chat(query: str, summary: dict) -> str:
    """Simple keyword-based fallback for chat queries."""
    q = query.lower()

    if any(w in q for w in ["worst", "most delayed", "highest delay"]):
        carrier = summary.get("worst_carrier", "Unknown")
        rate    = summary.get("worst_carrier_delay_rate", 0)
        return (
            f"Based on the BTS dataset, **{carrier}** has the highest average "
            f"delay rate at {rate:.1%}. This is driven primarily by carrier-side "
            f"operational issues and late-arriving aircraft cascades."
        )
    elif any(w in q for w in ["best", "on-time", "lowest delay"]):
        carrier = summary.get("best_carrier", "Unknown")
        rate    = summary.get("best_carrier_delay_rate", 0)
        return (
            f"**{carrier}** shows the best on-time performance with an average "
            f"delay rate of only {rate:.1%}. Strong scheduling discipline and "
            f"route optimization contribute to this performance."
        )
    elif any(w in q for w in ["weather", "storm"]):
        pct = summary.get("avg_weather_delay_pct", 0)
        return (
            f"Weather-related delays account for approximately {pct:.1%} of all "
            f"delayed flights in the dataset. Impact is highest in winter months "
            f"(December-February) at northern hub airports."
        )
    elif any(w in q for w in ["cancel", "cancellation"]):
        rate = summary.get("avg_cancel_rate", 0)
        return (
            f"The average cancellation rate across all carriers and airports is "
            f"{rate:.1%}. Cancellations are most correlated with extreme weather "
            f"events and high-volume hubs during peak travel periods."
        )
    elif any(w in q for w in ["month", "season", "busiest"]):
        month = summary.get("busiest_month", "Unknown")
        return (
            f"The busiest month by flight volume is **{month}**. Delay rates tend "
            f"to peak during summer (June–August) due to thunderstorm activity and "
            f"in December due to holiday travel demand surges."
        )
    else:
        total   = summary.get("total_flights", 0)
        avg_dr  = summary.get("avg_delay_rate", 0)
        n_carr  = summary.get("n_carriers", 0)
        return (
            f"The BTS dataset covers {total:,} arriving flights across {n_carr} "
            f"major US carriers with an average delay rate of {avg_dr:.1%}. "
            f"Ask me about the best/worst carriers, weather impact, cancellation "
            f"trends, or seasonal patterns for more specific insights."
        )
