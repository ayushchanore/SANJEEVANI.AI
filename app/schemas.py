"""
Pydantic v2 schemas for the fatty liver risk assessment API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    # ── Demographics ──────────────────────────────────────────────────────────
    age:                   float = Field(..., ge=18,  le=100,  description="Age in years")
    gender:                str   = Field(..., pattern="^(male|female|other)$")
    bmi:                   float = Field(..., ge=10,  le=70,   description="Body Mass Index (kg/m²)")
    waist_circ:            float = Field(..., ge=40,  le=200,  description="Waist circumference (cm)")
    has_diabetes:          bool  = Field(..., description="Type 2 diabetes diagnosis")
    has_hypertension:      bool  = Field(..., description="Hypertension diagnosis")
    weekly_alcohol_units:  int   = Field(..., ge=0,   le=200,  description="Alcohol units consumed per week")
    daily_activity_mins:   float = Field(..., ge=0,   le=1440, description="Daily physical activity (minutes)")

    # ── Optional lab biomarkers ───────────────────────────────────────────────
    triglycerides: Optional[float] = Field(None, ge=0, description="Triglycerides (mg/dL)")
    ast:           Optional[float] = Field(None, ge=0, description="AST liver enzyme (U/L)")
    alt:           Optional[float] = Field(None, ge=0, description="ALT liver enzyme (U/L)")
    ggt:           Optional[float] = Field(None, ge=0, description="GGT liver enzyme (U/L)")

    model_config = {"json_schema_extra": {
        "example": {
            "age": 42, "gender": "male", "bmi": 29.5, "waist_circ": 95.0,
            "has_diabetes": False, "has_hypertension": True,
            "weekly_alcohol_units": 10, "daily_activity_mins": 30,
            "triglycerides": 180.0, "ast": 38.0, "alt": 45.0, "ggt": None,
        }
    }}


class AssessmentResponse(BaseModel):
    risk_level:                int         # 0 | 1 | 2
    risk_score_percent:        float       # model probability × 100, rounded to 1 dp
    top_contributing_factors:  list[str]   # top 3 SHAP feature names
    lab_data_provided:         bool        # True if any optional lab field was supplied
    remedies:                  list[str]   # WHO-guideline lifestyle recommendations
    next_steps:                str         # clinical action string
