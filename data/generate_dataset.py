"""
Generate synthetic fatty liver dataset → data/fatty_liver_data.csv
Run: python data/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 1000

# ── Physical indicators (always present) ─────────────────────────────────────
age         = RNG.integers(18, 70, N).astype(float)
bmi         = RNG.normal(27.5, 5.5, N).clip(16, 50)
waist_cm    = RNG.normal(88, 14, N).clip(55, 140)
systolic_bp = RNG.normal(125, 18, N).clip(85, 200)
is_diabetic = RNG.binomial(1, 0.25, N).astype(float)
is_smoker   = RNG.binomial(1, 0.30, N).astype(float)

# ── Lab biomarkers (optional — ~25% NaN each) ────────────────────────────────
def with_nan(arr, rate=0.25):
    arr = arr.astype(float)
    arr[RNG.random(N) < rate] = np.nan
    return arr

alt_u_l         = with_nan(RNG.normal(35, 20, N).clip(5, 150))
ast_u_l         = with_nan(RNG.normal(30, 15, N).clip(5, 120))
triglycerides   = with_nan(RNG.normal(160, 70, N).clip(50, 500))
hdl_cholesterol = with_nan(RNG.normal(50, 12, N).clip(20, 100))
fasting_glucose = with_nan(RNG.normal(100, 25, N).clip(60, 300))
insulin_uiu_ml  = with_nan(RNG.normal(12, 6, N).clip(2, 60))
ggt_u_l         = with_nan(RNG.normal(40, 25, N).clip(5, 200))

# ── Label: risk score → binary (threshold ≈ 40th percentile) ─────────────────
def safe(arr, fallback):
    """Replace NaN with population mean for scoring only."""
    filled = np.where(np.isnan(arr), fallback, arr)
    return filled

risk = (
    0.30 * (bmi / 30)
    + 0.20 * (safe(triglycerides, 160) / 200)
    + 0.15 * is_diabetic
    + 0.10 * (safe(alt_u_l, 35) / 50)
    + 0.10 * (safe(fasting_glucose, 100) / 120)
    + 0.08 * (waist_cm / 100)
    + 0.07 * (safe(ast_u_l, 30) / 40)
    + RNG.normal(0, 0.05, N)          # noise
)
label = (risk > np.percentile(risk, 40)).astype(int)  # ~60% positive, realistic

df = pd.DataFrame({
    "age": age,
    "bmi": bmi.round(1),
    "waist_cm": waist_cm.round(1),
    "systolic_bp": systolic_bp.round(1),
    "is_diabetic": is_diabetic,
    "is_smoker": is_smoker,
    "alt_u_l": alt_u_l.round(1),
    "ast_u_l": ast_u_l.round(1),
    "triglycerides": triglycerides.round(1),
    "hdl_cholesterol": hdl_cholesterol.round(1),
    "fasting_glucose": fasting_glucose.round(1),
    "insulin_uiu_ml": insulin_uiu_ml.round(2),
    "ggt_u_l": ggt_u_l.round(1),
    "fatty_liver": label,
})

out = Path(__file__).parent / "fatty_liver_data.csv"
df.to_csv(out, index=False)

print(f"Saved {len(df)} records → {out}")
print(f"Class balance — 0: {(label==0).sum()}  1: {(label==1).sum()}")
print(f"NaN counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
