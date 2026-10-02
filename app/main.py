"""
FastAPI application — Fatty Liver Risk Assessment API
Run: uvicorn app.main:app --reload
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AssessmentResponse, PatientInput
from src.recommend import generate_recommendations

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sanjeevani — Fatty Liver Risk Assessment API",
    version="1.0.0",
    description="AI-assisted fatty liver risk scoring aligned with WHO public health guidelines.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────
_state: dict = {}

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@app.on_event("startup")
async def load_models() -> None:
    try:
        _state["imputer"]       = joblib.load(MODEL_DIR / "imputer.pkl")
        _state["scaler"]        = joblib.load(MODEL_DIR / "scaler.pkl")
        _state["model"]         = joblib.load(MODEL_DIR / "fatty_liver_model.pkl")
        _state["feature_names"] = joblib.load(MODEL_DIR / "feature_names.pkl")
        _state["explainer"]     = shap.TreeExplainer(_state["model"])
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model artifact missing: {e}. Run `python src/train.py` first."
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
# Human-readable labels for SHAP feature names returned to the user
_FEATURE_LABELS: dict[str, str] = {
    "bmi":             "Body Weight (BMI)",
    "waist_cm":        "Waist Size",
    "triglycerides":   "Blood Fat Level (Triglycerides)",
    "is_diabetic":     "Diabetes",
    "alt_u_l":         "Liver Enzyme ALT",
    "ast_u_l":         "Liver Enzyme AST",
    "ggt_u_l":         "Liver Enzyme GGT",
    "fasting_glucose": "Blood Sugar Level",
    "hdl_cholesterol": "Good Cholesterol (HDL)",
    "insulin_uiu_ml":  "Insulin Level",
    "systolic_bp":     "Blood Pressure",
    "is_smoker":       "Smoking",
    "age":             "Age",
}

_BMI_CATEGORY = [
    (18.5, "Underweight"),
    (25.0, "Normal weight"),
    (30.0, "Overweight"),
    (float("inf"), "Obese"),
]


def _bmi_category(bmi: float) -> str:
    for threshold, label in _BMI_CATEGORY:
        if bmi < threshold:
            return label
    return "Obese"


def _build_feature_vector(p: PatientInput, feature_names: list[str]) -> np.ndarray:
    """
    Map self-reported PatientInput → model feature vector.
    Fields not collected from the user are set to np.nan and handled by the imputer.
    """
    # Estimate waist from BMI using a population regression proxy
    # (waist_cm ≈ 2.15 × BMI + 27.8, derived from NHANES data)
    estimated_waist = round(2.15 * p.bmi + 27.8, 1)

    mapping: dict[str, float | None] = {
        "age":             p.age,
        "bmi":             p.bmi,                       # derived from height + weight
        "waist_cm":        estimated_waist,              # estimated from BMI
        "systolic_bp":     None,                         # imputed
        "is_diabetic":     float(p.has_diabetes),
        "is_smoker":       float(p.is_smoker),
        "alt_u_l":         None,                         # imputed
        "ast_u_l":         None,                         # imputed
        "triglycerides":   None,                         # imputed
        "hdl_cholesterol": None,                         # imputed
        "fasting_glucose": None,                         # imputed
        "insulin_uiu_ml":  None,                         # imputed
        "ggt_u_l":         None,                         # imputed
    }
    return np.array(
        [mapping.get(f) if mapping.get(f) is not None else np.nan for f in feature_names],
        dtype=float,
    ).reshape(1, -1)


def _derive_risk_level(proba: float) -> int:
    """Map model probability to 3-tier risk level."""
    if proba < 0.40:
        return 0   # Low
    if proba < 0.70:
        return 1   # Moderate
    return 2       # High


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def health_check():
    return {"status": "ok", "service": "Sanjeevani Fatty Liver Risk API", "version": "1.0.0"}


@app.post("/api/v1/assess", response_model=AssessmentResponse, tags=["Assessment"])
def assess(patient: PatientInput) -> AssessmentResponse:
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    feature_names: list[str] = _state["feature_names"]

    # 1. Build raw feature vector
    X_raw = _build_feature_vector(patient, feature_names)

    # 2. Impute → Scale
    X_imp = _state["imputer"].transform(X_raw)
    X_sc  = _state["scaler"].transform(X_imp)

    # 3. Predict
    proba      = float(_state["model"].predict_proba(X_sc)[0, 1])
    risk_level = _derive_risk_level(proba)

    # 4. SHAP — top 3 contributing features (raw names for recommend.py, labels for response)
    shap_vals   = _state["explainer"].shap_values(X_sc)[0]
    top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_factors_raw    = [feature_names[i] for i in top_indices]
    top_factors_labels = [_FEATURE_LABELS.get(f, f) for f in top_factors_raw]

    # 5. Recommendations
    recs = generate_recommendations(
        risk_level=risk_level,
        top_factors=top_factors_raw,
        weekly_alcohol=patient.weekly_alcohol_units,
        eats_fatty_food=patient.eats_fatty_food,
        eats_sugary_food=patient.eats_sugary_food,
        daily_activity_mins=patient.daily_activity_mins,
    )

    return AssessmentResponse(
        risk_level=risk_level,
        risk_score_percent=round(proba * 100, 1),
        bmi=patient.bmi,
        bmi_category=_bmi_category(patient.bmi),
        top_contributing_factors=top_factors_labels,
        remedies=recs["remedies"],
        next_steps=recs["next_steps"],
    )
