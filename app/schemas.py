"""
Pydantic v2 schemas for the fatty liver risk assessment API.
All required fields are self-reportable — no medical tests needed.
"""

from pydantic import BaseModel, Field, model_validator


class PatientInput(BaseModel):
    # ── Demographics (self-reportable) ────────────────────────────────────────
    age:            float = Field(..., ge=18, le=100, description="Age in years")
    gender:         str   = Field(..., pattern="^(male|female|other)$")
    height_cm:      float = Field(..., ge=100, le=250, description="Height in centimetres")
    weight_kg:      float = Field(..., ge=30,  le=300, description="Weight in kilograms")

    # ── Lifestyle (self-reportable) ───────────────────────────────────────────
    is_smoker:            bool = Field(..., description="Do you currently smoke?")
    weekly_alcohol_units: int  = Field(..., ge=0, le=200,  description="Alcoholic drinks per week (1 drink = 1 unit)")
    daily_activity_mins:  int  = Field(..., ge=0, le=1440, description="Minutes of physical activity per day (walking, sports, gym, etc.)")
    eats_fatty_food:      bool = Field(..., description="Do you regularly eat fried or fatty food?")
    eats_sugary_food:     bool = Field(..., description="Do you regularly consume sugary drinks or sweets?")

    # ── Known conditions (self-reportable — doctor told you) ──────────────────
    has_diabetes:   bool = Field(..., description="Have you been told you have diabetes or pre-diabetes?")
    family_history: bool = Field(..., description="Does a close family member have liver disease, diabetes, or obesity?")

    # ── Derived field (computed, not supplied by user) ────────────────────────
    bmi: float = Field(0.0, exclude=True)   # populated by validator below

    @model_validator(mode="after")
    def compute_bmi(self) -> "PatientInput":
        self.bmi = round(self.weight_kg / (self.height_cm / 100) ** 2, 1)
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "age": 42, "gender": "male",
            "height_cm": 172, "weight_kg": 87,
            "is_smoker": False, "weekly_alcohol_units": 10,
            "daily_activity_mins": 20, "eats_fatty_food": True,
            "eats_sugary_food": True, "has_diabetes": False,
            "family_history": True,
        }
    }}


class AssessmentResponse(BaseModel):
    risk_level:               int        # 0 | 1 | 2
    risk_score_percent:       float      # model probability × 100, rounded to 1 dp
    bmi:                      float      # derived BMI shown back to user
    bmi_category:             str        # Underweight / Normal / Overweight / Obese
    top_contributing_factors: list[str]  # top 3 SHAP feature names (human-readable)
    remedies:                 list[str]  # WHO-guideline lifestyle recommendations
    next_steps:               str        # clinical action string
