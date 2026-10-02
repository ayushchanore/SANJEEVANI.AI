# Sanjeevani — AI Fatty Liver Risk Assessment

AI-assisted fatty liver (NAFLD) risk scoring based on WHO public health guidelines.
No blood tests required — all inputs are self-reportable.

## Stack
- **ML**: XGBoost + SHAP (TreeExplainer)
- **API**: FastAPI + Pydantic v2
- **Frontend**: Vanilla HTML/CSS/JS (served by FastAPI)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data/generate_dataset.py
python src/train.py
```

## Run

```powershell
.venv\Scripts\uvicorn.exe app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000` | Frontend UI |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/health` | Health check |

## API

### `POST /api/v1/assess`

**Request**
```json
{
  "age": 42, "gender": "male",
  "height_cm": 172, "weight_kg": 87,
  "is_smoker": false, "weekly_alcohol_units": 10,
  "daily_activity_mins": 20, "eats_fatty_food": true,
  "eats_sugary_food": true, "has_diabetes": false,
  "family_history": true
}
```

**Response**
```json
{
  "risk_level": 1,
  "risk_score_percent": 62.4,
  "bmi": 29.4,
  "bmi_category": "Overweight",
  "top_contributing_factors": ["Body Weight (BMI)", "Diabetes", "Blood Fat Level"],
  "remedies": ["..."],
  "next_steps": "Schedule a GP consultation within 4 weeks."
}
```

## Project Structure

```
SANJEEVANI/
├── app/
│   ├── main.py          # FastAPI app
│   ├── schemas.py       # Pydantic models
│   ├── config.py        # Environment config
│   └── static/
│       └── index.html   # Frontend UI
├── src/
│   ├── train.py         # Model training pipeline
│   └── recommend.py     # WHO recommendation engine
├── data/
│   └── generate_dataset.py
├── models/              # Saved .pkl artifacts (git-ignored)
└── requirements.txt
```
