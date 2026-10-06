"""
FastAPI application — Fatty Liver Risk Assessment API
Run: uvicorn app.main:app --reload
"""

import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import shap
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.auth import create_access_token, get_current_user, hash_password, verify_password
from app.config import APP_TITLE, APP_VERSION, CORS_ORIGINS, LOG_LEVEL
from app.database import Base, engine, get_db
from app.models import User
from app.schemas import (
    AssessmentResponse, LoginRequest, PatientInput,
    RegisterRequest, TokenResponse,
)
from src.recommend import generate_recommendations

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sanjeevani")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Model state ───────────────────────────────────────────────────────────────
_state: dict = {}
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@app.on_event("startup")
async def startup() -> None:
    # Create DB tables
    Base.metadata.create_all(bind=engine)
    log.info("Database ready.")

    # Load ML artifacts
    log.info("Loading model artifacts...")
    try:
        _state["imputer"]       = joblib.load(MODEL_DIR / "imputer.pkl")
        _state["scaler"]        = joblib.load(MODEL_DIR / "scaler.pkl")
        _state["model"]         = joblib.load(MODEL_DIR / "fatty_liver_model.pkl")
        _state["feature_names"] = joblib.load(MODEL_DIR / "feature_names.pkl")
        _state["explainer"]     = shap.TreeExplainer(_state["model"])
        log.info("All artifacts loaded successfully.")
    except FileNotFoundError as e:
        log.error(f"Missing artifact: {e}")
        raise RuntimeError(f"Model artifact missing: {e}. Run `python src/train.py` first.")


# ── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    log.info(f"{request.method} {request.url.path} -> {response.status_code} ({ms:.1f}ms)")
    return response


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    (18.5, "Underweight"), (25.0, "Normal weight"),
    (30.0, "Overweight"),  (float("inf"), "Obese"),
]


def _bmi_category(bmi: float) -> str:
    for threshold, label in _BMI_CATEGORY:
        if bmi < threshold:
            return label
    return "Obese"


def _build_feature_vector(p: PatientInput, feature_names: list[str]) -> np.ndarray:
    estimated_waist = round(2.15 * p.bmi + 27.8, 1)
    mapping: dict[str, float | None] = {
        "age": p.age, "bmi": p.bmi, "waist_cm": estimated_waist,
        "systolic_bp": None, "is_diabetic": float(p.has_diabetes),
        "is_smoker": float(p.is_smoker), "alt_u_l": None, "ast_u_l": None,
        "triglycerides": None, "hdl_cholesterol": None,
        "fasting_glucose": None, "insulin_uiu_ml": None, "ggt_u_l": None,
    }
    return np.array(
        [mapping.get(f) if mapping.get(f) is not None else np.nan for f in feature_names],
        dtype=float,
    ).reshape(1, -1)


def _derive_risk_level(proba: float) -> int:
    if proba < 0.40: return 0
    if proba < 0.70: return 1
    return 2


# ── Frontend routes ───────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def landing():    return FileResponse(STATIC_DIR / "landing.html")

@app.get("/login", include_in_schema=False)
def login_page(): return FileResponse(STATIC_DIR / "login.html")

@app.get("/register", include_in_schema=False)
def register_page(): return FileResponse(STATIC_DIR / "register.html")

@app.get("/assess", include_in_schema=False)
def assess_page(): return FileResponse(STATIC_DIR / "index.html")


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/register", status_code=status.HTTP_201_CREATED, tags=["Auth"])
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=400, detail="Username already taken.")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")
    user = User(
        username=body.username, email=body.email,
        full_name=body.full_name, hashed_pw=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    log.info(f"New user registered: {body.username}")
    return {"message": "Account created successfully."}


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_access_token({"sub": user.username})
    log.info(f"User logged in: {user.username}")
    return TokenResponse(
        access_token=token, username=user.username, full_name=user.full_name
    )


# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Status"])
def health_check():
    return {"status": "ok", "service": APP_TITLE, "version": APP_VERSION}


# ── Assessment (protected) ────────────────────────────────────────────────────
@app.post("/api/v1/assess", response_model=AssessmentResponse, tags=["Assessment"])
def assess(
    patient: PatientInput,
    current_user: User = Depends(get_current_user),
) -> AssessmentResponse:
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    feature_names: list[str] = _state["feature_names"]
    X_raw = _build_feature_vector(patient, feature_names)
    X_imp = _state["imputer"].transform(X_raw)
    X_sc  = _state["scaler"].transform(X_imp)

    proba      = float(_state["model"].predict_proba(X_sc)[0, 1])
    risk_level = _derive_risk_level(proba)

    shap_vals          = _state["explainer"].shap_values(X_sc)[0]
    top_indices        = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_factors_raw    = [feature_names[i] for i in top_indices]
    top_factors_labels = [_FEATURE_LABELS.get(f, f) for f in top_factors_raw]

    recs = generate_recommendations(
        risk_level=risk_level, top_factors=top_factors_raw,
        weekly_alcohol=patient.weekly_alcohol_units,
        eats_fatty_food=patient.eats_fatty_food,
        eats_sugary_food=patient.eats_sugary_food,
        daily_activity_mins=patient.daily_activity_mins,
    )

    log.info(f"Assessment by {current_user.username}: risk={risk_level}, bmi={patient.bmi}")

    return AssessmentResponse(
        risk_level=risk_level,
        risk_score_percent=round(proba * 100, 1),
        bmi=patient.bmi,
        bmi_category=_bmi_category(patient.bmi),
        top_contributing_factors=top_factors_labels,
        remedies=recs["remedies"],
        next_steps=recs["next_steps"],
    )
