"""
WHO-guideline-based recommendation generator for fatty liver risk assessment.
"""

# Feature name sets for rule matching
_PHYSICAL_FEATURES = {"bmi", "waist_cm", "waist_circ"}
_LAB_FEATURES      = {"alt_u_l", "ast_u_l", "triglycerides", "hdl_cholesterol",
                      "fasting_glucose", "insulin_uiu_ml", "ggt_u_l"}

_RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

_CLINICAL_NEXT_STEPS = {
    0: "Annual health check-up recommended. Monitor BMI and waist circumference regularly.",
    1: "Schedule a GP consultation within 4 weeks. Request liver function tests and lipid panel.",
    2: "Urgent specialist referral required. FibroScan or liver ultrasound recommended within 2 weeks.",
}


def generate_recommendations(
    risk_level: int,
    top_factors: list[str],
    weekly_alcohol: int,
    eats_fatty_food: bool = False,
    eats_sugary_food: bool = False,
    daily_activity_mins: int = 0,
) -> dict:
    """
    Generate WHO-aligned lifestyle and clinical recommendations.

    Args:
        risk_level:          0 = Low, 1 = Moderate, 2 = High
        top_factors:         Top SHAP feature names driving the prediction
        weekly_alcohol:      Patient-reported weekly alcohol units
        eats_fatty_food:     Self-reported regular fatty food consumption
        eats_sugary_food:    Self-reported regular sugary food/drink consumption
        daily_activity_mins: Self-reported daily activity in minutes

    Returns:
        dict with keys: risk_label, remedies (list), next_steps (str)
    """
    factors = {f.lower() for f in top_factors}
    remedies: list[str] = []

    # ── WHO Physical Activity Guideline ───────────────────────────────────────
    # Source: WHO Global Action Plan on Physical Activity 2018–2030
    if factors & _PHYSICAL_FEATURES:
        remedies.append(
            "Physical Activity (WHO): Aim for 150–300 minutes/week of moderate-intensity "
            "aerobic exercise (e.g. brisk walking, cycling). Elevated BMI or waist "
            "circumference is a key driver of your risk."
        )

    # ── WHO Dietary Guideline ─────────────────────────────────────────────────
    # Source: WHO Guideline on Sugars Intake for Adults and Children (2015)
    if eats_sugary_food or (factors & (_LAB_FEATURES | _PHYSICAL_FEATURES)):
        remedies.append(
            "Diet — Sugar (WHO): Reduce free sugar intake to <10% of total daily energy. "
            "Swap sugary drinks for water and limit sweets and processed snacks."
        )
    if eats_fatty_food:
        remedies.append(
            "Diet — Fat (WHO): Limit saturated and trans fats. Replace fried and fatty foods "
            "with grilled, steamed, or baked options. Prioritise vegetables, legumes, and whole grains."
        )

    # ── WHO Alcohol Guideline ─────────────────────────────────────────────────
    # Source: WHO Global Status Report on Alcohol and Health (2018)
    if weekly_alcohol > 14:
        remedies.append(
            "Alcohol (WHO): Your reported intake exceeds 14 units/week. "
            "WHO recommends reducing alcohol consumption as no level is entirely safe. "
            "Target ≤14 units/week with at least 2 alcohol-free days."
        )

    # ── Risk-level baseline advice (always included) ──────────────────────────
    baseline = {
        0: "Maintain a healthy weight and stay physically active to preserve your low-risk status.",
        1: "Moderate lifestyle changes now can prevent progression to advanced liver disease.",
        2: "Immediate lifestyle intervention combined with medical management is strongly advised.",
    }
    remedies.append(baseline[risk_level])

    return {
        "risk_label":  _RISK_LABELS[risk_level],
        "remedies":    remedies,
        "next_steps":  _CLINICAL_NEXT_STEPS[risk_level],
    }
