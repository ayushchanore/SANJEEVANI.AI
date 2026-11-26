# 🌿 Sanjeevani — AI Fatty Liver Risk Assessment

> AI-assisted Non-Alcoholic Fatty Liver Disease (NAFLD) risk scoring based on WHO public health guidelines.  
> Powered by real NHANES clinical data. No blood tests required — all inputs are self-reportable.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Dataset — NHANES 2017-2018](#dataset--nhanes-2017-2018)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [API Reference](#api-reference)
- [Authentication](#authentication)
- [WHO Guidelines Integration](#who-guidelines-integration)
- [Frontend Pages](#frontend-pages)
- [Environment & Configuration](#environment--configuration)
- [Known Limitations](#known-limitations)

---

## Overview

Sanjeevani is a full-stack AI web application that assesses a user's risk of Non-Alcoholic Fatty Liver Disease (NAFLD) using only self-reportable lifestyle and demographic inputs — no lab tests, no clinic visits required.

The system is trained on **real clinical data from the CDC's NHANES 2017-2018 survey** (1,549 records), uses an XGBoost classifier with SHAP explainability to surface the top contributing risk factors, and generates WHO-guideline-aligned recommendations to guide users toward actionable lifestyle changes.

---

## Features

- **Real Clinical Data** — Trained on NHANES 2017-2018 data from the CDC, not synthetic records
- **AI Risk Scoring** — XGBoost model with 85% accuracy and 0.93 ROC-AUC
- **SHAP Explainability** — Top 3 personalised risk factors per assessment
- **WHO-Aligned Recommendations** — Diet, activity, alcohol, and weight advice
- **No Blood Tests** — 11 self-reportable inputs only
- **JWT Authentication** — Secure register/login with bcrypt password hashing
- **Responsive UI** — Works on desktop, tablet, and mobile
- **Instant Results** — Risk level, BMI category, remedies, and clinical next steps
- **Dataset Download** — Download the processed NHANES dataset directly from the app

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost + SHAP (TreeExplainer) |
| API | FastAPI + Pydantic v2 |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Database | SQLite + SQLAlchemy ORM |
| Frontend | Vanilla HTML / CSS / JS |
| Server | Uvicorn (ASGI) |
| Data Source | NHANES 2017-2018 (CDC) via pyreadstat |

---

## Project Structure

```
SANJEEVANI/
├── app/
│   ├── main.py            # FastAPI app — routes, middleware, ML inference
│   ├── schemas.py         # Pydantic v2 request/response models
│   ├── auth.py            # JWT creation, verification, password hashing
│   ├── models.py          # SQLAlchemy User model
│   ├── database.py        # SQLite engine + session factory
│   ├── config.py          # Environment config (title, version, CORS, log level)
│   └── static/
│       ├── landing.html   # Public landing page
│       ├── login.html     # Login page
│       ├── register.html  # Registration page
│       └── index.html     # Assessment UI (protected)
├── src/
│   ├── train.py           # Model training pipeline
│   └── recommend.py       # WHO recommendation engine
├── data/
│   ├── fetch_nhanes.py      # Downloads & preprocesses real NHANES data
│   ├── generate_dataset.py  # Legacy synthetic dataset generator
│   └── fatty_liver_data.csv # Processed NHANES dataset (1,549 records)
├── models/                  # Saved .pkl artifacts (git-ignored)
│   ├── fatty_liver_model.pkl
│   ├── feature_names.pkl
│   ├── imputer.pkl
│   └── scaler.pkl
├── sanjeevani.db            # SQLite database (auto-created on startup)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip
- Internet connection (for NHANES download)

### 1. Clone the repository

```powershell
git clone <repo-url>
cd SANJEEVANI
```

### 2. Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** `bcrypt==4.0.1` is pinned explicitly for compatibility with `passlib 1.7.4`. Do not upgrade bcrypt without testing.

### 4. Fetch real NHANES data

```powershell
python data/fetch_nhanes.py
```

This downloads 10 XPT files from CDC servers, merges them, derives the NAFLD label using the Fatty Liver Index (FLI), and saves `data/fatty_liver_data.csv` with 1,549 real clinical records.

### 5. Train the model

```powershell
python src/train.py
```

This trains the XGBoost classifier and saves 4 artifacts to `models/`:
- `fatty_liver_model.pkl` — trained XGBClassifier
- `imputer.pkl` — median SimpleImputer (handles missing values)
- `scaler.pkl` — StandardScaler
- `feature_names.pkl` — ordered feature list

---

## Running the App

```powershell
.venv\Scripts\uvicorn.exe app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000` | Landing page |
| `http://localhost:8000/login` | Login page |
| `http://localhost:8000/register` | Registration page |
| `http://localhost:8000/assess` | Assessment UI (requires login) |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/download/dataset` | Download NHANES dataset CSV |

---

## How It Works

```
User registers / logs in
        ↓
JWT token stored in localStorage
        ↓
User fills in 11 self-reportable inputs
        ↓
Frontend sends POST /api/v1/assess with Bearer token
        ↓
FastAPI validates token → builds feature vector
        ↓
Imputer fills missing lab values (median strategy)
        ↓
StandardScaler normalises features
        ↓
XGBoost predicts NAFLD probability
        ↓
SHAP TreeExplainer identifies top 3 risk drivers
        ↓
WHO recommendation engine generates personalised advice
        ↓
Response: risk level, score, BMI, factors, remedies, next steps
```

---

## Dataset — NHANES 2017-2018

### Source

The dataset is sourced from the **National Health and Nutrition Examination Survey (NHANES) 2017-2018 cycle**, conducted by the **CDC (Centers for Disease Control and Prevention), USA**.

NHANES is a nationally representative survey that combines standardised interviews, physical examinations, and laboratory tests on thousands of Americans. All data is publicly available at [cdc.gov/nchs/nhanes](https://www.cdc.gov/nchs/nhanes/).

### Files Downloaded

| NHANES File | Contents | Rows |
|---|---|---|
| DEMO_J.xpt | Age, demographics | 9,254 |
| BMX_J.xpt | BMI, waist circumference | 8,704 |
| BPX_J.xpt | Systolic blood pressure | 8,704 |
| DIQ_J.xpt | Diabetes diagnosis | 8,897 |
| SMQ_J.xpt | Smoking status | 6,724 |
| BIOPRO_J.xpt | ALT, AST, GGT liver enzymes | 6,401 |
| TRIGLY_J.xpt | Triglycerides | 3,036 |
| HDL_J.xpt | HDL cholesterol | 7,435 |
| GLU_J.xpt | Fasting glucose | 3,036 |
| INS_J.xpt | Insulin | 3,036 |

### Feature Mapping

| Pipeline Feature | NHANES Variable | Description |
|---|---|---|
| age | RIDAGEYR | Age in years |
| systolic_bp | BPXSY1 | Systolic blood pressure |
| is_diabetic | DIQ010 | Diabetes / pre-diabetes diagnosis |
| is_smoker | SMQ020 | Ever smoked 100+ cigarettes |
| alt_u_l | LBXSATSI | Alanine aminotransferase (liver enzyme) |
| ast_u_l | LBXSASSI | Aspartate aminotransferase (liver enzyme) |
| hdl_cholesterol | LBDHDD | HDL cholesterol |
| fasting_glucose | LBXGLU | Fasting plasma glucose |
| insulin_uiu_ml | LBXIN | Fasting insulin |

### NAFLD Label — Fatty Liver Index (FLI)

Since NHANES does not include a direct NAFLD diagnosis, the binary label is derived using the **Fatty Liver Index (FLI)** — a clinically validated non-invasive surrogate for ultrasound-confirmed NAFLD (Bedogni et al., 2006):

```
FLI = exp(0.953×ln(TG) + 0.139×BMI + 0.718×ln(GGT) + 0.053×waist − 15.745)
      ─────────────────────────────────────────────────────────────────────── × 100
      1 + exp(0.953×ln(TG) + 0.139×BMI + 0.718×ln(GGT) + 0.053×waist − 15.745)
```

| FLI Score | Label |
|---|---|
| < 30 | fatty_liver = 0 (NAFLD unlikely) |
| 30 – 59 | Excluded (indeterminate) |
| ≥ 60 | fatty_liver = 1 (NAFLD likely) |

> The FLI input features (BMI, waist circumference, triglycerides, GGT) are **excluded from model training** to prevent label leakage. The model learns from the remaining independent clinical features only.

### Final Dataset Statistics

| Metric | Value |
|---|---|
| Total records | 1,549 |
| No NAFLD (0) | 659 (42.5%) |
| NAFLD (1) | 890 (57.5%) |
| Age range | 18–70 years |
| Missing values | systolic_bp (~10%), ast_u_l (<1%), insulin (<1%) |

---

## Machine Learning Pipeline

### Training (`src/train.py`)

| Step | Detail |
|---|---|
| Split | 80/20 train/test, stratified |
| Imputation | Median strategy for missing values |
| Scaling | StandardScaler |
| Model | XGBClassifier — 300 estimators, max_depth=4, lr=0.05 |
| Class balance | `scale_pos_weight` computed from training set |
| Explainability | SHAP TreeExplainer |

### Model Performance

| Metric | Value |
|---|---|
| Accuracy | 85% |
| ROC-AUC | 0.93 |
| Precision (NAFLD) | 0.89 |
| Recall (NAFLD) | 0.84 |
| F1-Score (NAFLD) | 0.86 |

### Top SHAP Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | Insulin Level | 1.919 |
| 2 | Liver Enzyme ALT | 0.942 |
| 3 | HDL Cholesterol | 0.750 |
| 4 | Age | 0.649 |
| 5 | Liver Enzyme AST | 0.354 |
| 6 | Blood Pressure | 0.339 |
| 7 | Smoking | 0.284 |
| 8 | Fasting Glucose | 0.180 |
| 9 | Diabetes | 0.087 |

### Inference

At inference time, only self-reportable features are available. Lab values (ALT, AST, HDL, glucose, insulin) are set to `NaN` and filled by the imputer using training-set medians. BMI is computed from height and weight inputs.

### Risk Levels

| Level | Probability | Label | Clinical Action |
|---|---|---|---|
| 0 | < 40% | Low Risk | Annual check-up |
| 1 | 40–70% | Moderate Risk | GP consultation within 4 weeks |
| 2 | > 70% | High Risk | Urgent specialist referral |

---

## API Reference

### `GET /health`

```json
{ "status": "ok", "service": "Sanjeevani", "version": "1.0.0" }
```

---

### `GET /download/dataset`

Downloads `fatty_liver_data.csv` — the processed NHANES 2017-2018 dataset used to train the model.

---

### `POST /auth/register`

**Request**
```json
{
  "full_name": "Rahul Sharma",
  "email": "rahul@example.com",
  "username": "rahul123",
  "password": "securepass"
}
```

**Response** `201 Created`
```json
{ "message": "Account created successfully." }
```

**Validation**
- `username` — 3–30 characters
- `password` — minimum 6 characters
- `email` — valid email format
- Duplicate username or email returns `400`

---

### `POST /auth/login`

**Request**
```json
{ "username": "rahul123", "password": "securepass" }
```

**Response** `200 OK`
```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "username": "rahul123",
  "full_name": "Rahul Sharma"
}
```

---

### `POST /api/v1/assess` _(requires Bearer token)_

**Headers**
```
Authorization: Bearer <access_token>
```

**Request**
```json
{
  "age": 42,
  "gender": "male",
  "height_cm": 172,
  "weight_kg": 87,
  "is_smoker": false,
  "weekly_alcohol_units": 10,
  "daily_activity_mins": 20,
  "eats_fatty_food": true,
  "eats_sugary_food": true,
  "has_diabetes": false,
  "family_history": true
}
```

**Response** `200 OK`
```json
{
  "risk_level": 1,
  "risk_score_percent": 62.4,
  "bmi": 29.4,
  "bmi_category": "Overweight",
  "top_contributing_factors": ["Insulin Level", "Liver Enzyme ALT", "Good Cholesterol (HDL)"],
  "remedies": [
    "Physical Activity (WHO): Aim for 150–300 minutes/week...",
    "Diet — Sugar (WHO): Reduce free sugar intake to <10%...",
    "Moderate lifestyle changes now can prevent progression..."
  ],
  "next_steps": "Schedule a GP consultation within 4 weeks. Request liver function tests and lipid panel."
}
```

**Input Validation**

| Field | Type | Constraints |
|---|---|---|
| age | float | 18–100 |
| gender | string | `male` / `female` / `other` |
| height_cm | float | 100–250 |
| weight_kg | float | 30–300 |
| weekly_alcohol_units | int | 0–200 |
| daily_activity_mins | int | 0–1440 |

---

## Authentication

- Passwords are hashed with **bcrypt** via `passlib` (`bcrypt==4.0.1` pinned for compatibility)
- Tokens are signed **HS256 JWTs** with a 24-hour expiry
- The `/api/v1/assess` endpoint requires a valid `Authorization: Bearer <token>` header
- The frontend stores the token in `localStorage` and redirects to `/login` on 401
- Auto-login after registration — no separate login step needed
- The `SECRET_KEY` in `auth.py` must be replaced with a strong random value in production

---

## WHO Guidelines Integration

Recommendations are generated by `src/recommend.py` based on three WHO sources:

| Guideline | Source |
|---|---|
| Physical Activity | WHO Global Action Plan on Physical Activity 2018–2030 |
| Diet & Sugar | WHO Guideline on Sugars Intake for Adults and Children (2015) |
| Alcohol | WHO Global Status Report on Alcohol and Health (2018) |

### Recommendation Rules

| Trigger | Advice Generated |
|---|---|
| BMI or waist in top SHAP factors | 150–300 mins/week aerobic activity |
| `eats_sugary_food = true` | Reduce free sugars to <10% daily energy |
| `eats_fatty_food = true` | Replace saturated fats with unsaturated alternatives |
| `weekly_alcohol_units > 14` | Reduce alcohol — no safe level per WHO |
| Risk level 0 | Maintain healthy lifestyle |
| Risk level 1 | Moderate lifestyle changes advised |
| Risk level 2 | Immediate intervention + medical management |

---

## Frontend Pages

| Page | Route | Auth Required | Description |
|---|---|---|---|
| Landing | `/` | No | Hero, features, WHO guidelines, early detection info |
| Login | `/login` | No | Username + password login |
| Register | `/register` | No | Account creation with auto-login |
| Assessment | `/assess` | Yes (JWT) | 11-input form + results |

### Responsiveness Breakpoints

| Breakpoint | Changes |
|---|---|
| ≤ 700px | Nav links hidden, hero font scales down, grids collapse to 1 column |
| ≤ 560px | Form grid collapses, card padding reduced, header scales down |
| ≤ 480px | Auth card padding tightened |

---

## Environment & Configuration

`app/config.py` controls:

| Variable | Default | Description |
|---|---|---|
| `APP_TITLE` | `Sanjeevani` | API title shown in docs |
| `APP_VERSION` | `1.0.0` | API version |
| `LOG_LEVEL` | `INFO` | Uvicorn log level |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

### Production Checklist

- Replace `SECRET_KEY` in `auth.py` with a strong random string stored as an environment variable
- Restrict `CORS_ORIGINS` to your domain
- Replace SQLite with PostgreSQL
- Run behind Nginx with TLS/HTTPS
- Set `LOG_LEVEL` to `WARNING` or `ERROR`

---

## Known Limitations

- **FLI-derived labels** — NAFLD labels are derived from the Fatty Liver Index formula, not direct ultrasound or biopsy diagnosis. FLI is clinically validated but not a gold standard.
- **Indeterminate zone excluded** — Records with FLI 30–59 are excluded to ensure label quality, reducing dataset size to 1,549 records.
- **Estimated lab values at inference** — Missing biomarkers (ALT, AST, HDL, glucose, insulin) are imputed from training medians at inference time, which reduces individual precision.
- **No HTTPS** — Run behind a reverse proxy (e.g. Nginx) with TLS in production.
- **SQLite** — Suitable for development. Replace with PostgreSQL for production deployments.
- **Secret key** — The JWT secret in `auth.py` must be rotated and stored as an environment variable before going live.
- **US population bias** — NHANES is a US national survey. Risk patterns may differ for other populations.

---

## References

- Bedogni G, et al. (2006). *The Fatty Liver Index: a simple and accurate predictor of hepatic steatosis in the general population.* BMC Gastroenterology.
- WHO Global Action Plan on Physical Activity 2018–2030.
- WHO Guideline on Sugars Intake for Adults and Children (2015).
- WHO Global Status Report on Alcohol and Health (2018).
- CDC NHANES 2017-2018: [https://www.cdc.gov/nchs/nhanes/](https://www.cdc.gov/nchs/nhanes/)

---

> ⚠️ **Disclaimer:** Sanjeevani is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
