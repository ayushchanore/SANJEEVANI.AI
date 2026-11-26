"""
Fetch real NHANES data and produce data/fatty_liver_data.csv
in the exact same format as the synthetic dataset.

NAFLD label is derived using the Fatty Liver Index (FLI):
  FLI >= 60  → fatty_liver = 1  (NAFLD likely)
  FLI <  30  → fatty_liver = 0  (NAFLD unlikely)
  30 <= FLI < 60 → excluded (indeterminate zone)

FLI formula (Bedogni et al., 2006):
  FLI = (e^(0.953*ln(TG) + 0.139*BMI + 0.718*ln(GGT) + 0.053*waist - 15.745))
        / (1 + e^(...)) * 100

Run: python data/fetch_nhanes.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import requests
import pyreadstat
from pathlib import Path

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "fatty_liver_data.csv"

# ── NHANES 2017-2018 cycle XPT URLs ──────────────────────────────────────────
URLS = {
    # Demographics
    "DEMO": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
    # Body measures
    "BMX":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt",
    # Blood pressure
    "BPX":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.xpt",
    # Diabetes
    "DIQ":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.xpt",
    # Smoking
    "SMQ":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.xpt",
    # Liver enzymes (ALT, AST, GGT)
    "BIOPRO": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.xpt",
    # Triglycerides
    "TRIGLY": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TRIGLY_J.xpt",
    # HDL cholesterol
    "HDL":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HDL_J.xpt",
    # Fasting glucose + insulin
    "GLU":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.xpt",
    "INS":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/INS_J.xpt",
}


def fetch_xpt(name: str, url: str) -> pd.DataFrame:
    print(f"  Downloading {name}...", end=" ", flush=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df, _ = pyreadstat.read_xport(io.BytesIO(r.content))
    print(f"{len(df)} rows")
    return df


def fli(bmi, waist, tg, ggt) -> np.ndarray:
    """Fatty Liver Index (Bedogni 2006). Returns 0–100."""
    ln_tg  = np.log(np.where(tg  > 0, tg,  np.nan))
    ln_ggt = np.log(np.where(ggt > 0, ggt, np.nan))
    exp = np.exp(0.953 * ln_tg + 0.139 * bmi + 0.718 * ln_ggt + 0.053 * waist - 15.745)
    return (exp / (1 + exp)) * 100


print("Downloading NHANES 2017-2018 files...")
frames = {k: fetch_xpt(k, v) for k, v in URLS.items()}

# ── Merge on SEQN (respondent ID) ────────────────────────────────────────────
df = frames["DEMO"][["SEQN", "RIDAGEYR"]].copy()
for key, cols in [
    ("BMX",    ["SEQN", "BMXBMI", "BMXWAIST"]),
    ("BPX",    ["SEQN", "BPXSY1"]),
    ("DIQ",    ["SEQN", "DIQ010"]),
    ("SMQ",    ["SEQN", "SMQ020"]),
    ("BIOPRO", ["SEQN", "LBXSATSI", "LBXSASSI", "LBXSGTSI"]),
    ("TRIGLY", ["SEQN", "LBXTR"]),
    ("HDL",    ["SEQN", "LBDHDD"]),
    ("GLU",    ["SEQN", "LBXGLU"]),
    ("INS",    ["SEQN", "LBXIN"]),
]:
    available = [c for c in cols if c in frames[key].columns]
    df = df.merge(frames[key][available], on="SEQN", how="left")

print(f"\nMerged dataset: {len(df)} rows, {df.shape[1]} columns")

# ── Rename to match pipeline feature names ────────────────────────────────────
df = df.rename(columns={
    "RIDAGEYR": "age",
    "BMXBMI":   "bmi",
    "BMXWAIST": "waist_cm",
    "BPXSY1":   "systolic_bp",
    "LBXSATSI": "alt_u_l",
    "LBXSASSI": "ast_u_l",
    "LBXSGTSI": "ggt_u_l",
    "LBXTR":    "triglycerides",
    "LBDHDD":   "hdl_cholesterol",
    "LBXGLU":   "fasting_glucose",
    "LBXIN":    "insulin_uiu_ml",
})

# ── Encode binary fields ──────────────────────────────────────────────────────
# DIQ010: 1=Yes, 2=No, 3=Borderline → treat 1 & 3 as diabetic
df["is_diabetic"] = df["DIQ010"].map({1: 1, 3: 1, 2: 0}).fillna(0).astype(float)

# SMQ020: 1=Yes, 2=No
df["is_smoker"] = df["SMQ020"].map({1: 1, 2: 0}).fillna(0).astype(float)

# ── Filter: adults 18–70 only ─────────────────────────────────────────────────
df = df[(df["age"] >= 18) & (df["age"] <= 70)].copy()

# ── Derive NAFLD label via FLI ────────────────────────────────────────────────
df["_fli"] = fli(df["bmi"], df["waist_cm"], df["triglycerides"], df["ggt_u_l"])

# Keep only definitive cases (FLI < 30 = no NAFLD, FLI >= 60 = NAFLD)
df = df[df["_fli"].notna() & ((df["_fli"] < 30) | (df["_fli"] >= 60))].copy()
df["fatty_liver"] = (df["_fli"] >= 60).astype(int)

# Drop FLI input columns to prevent label leakage
# (waist_cm, triglycerides, ggt_u_l, bmi were used to compute the label)
df = df.drop(columns=["waist_cm", "triglycerides", "ggt_u_l", "bmi"])

# ── Select final columns in pipeline order ────────────────────────────────────
FINAL_COLS = [
    "age", "systolic_bp",
    "is_diabetic", "is_smoker",
    "alt_u_l", "ast_u_l", "hdl_cholesterol",
    "fasting_glucose", "insulin_uiu_ml",
    "fatty_liver",
]
df = df[FINAL_COLS].reset_index(drop=True)

# ── Round numeric columns ─────────────────────────────────────────────────────
for col in ["systolic_bp", "alt_u_l", "ast_u_l", "hdl_cholesterol", "fasting_glucose"]:
    df[col] = df[col].round(1)
df["insulin_uiu_ml"] = df["insulin_uiu_ml"].round(2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nFinal dataset: {len(df)} records")
print(f"Class balance — 0 (No NAFLD): {(df['fatty_liver']==0).sum()}  "
      f"1 (NAFLD): {(df['fatty_liver']==1).sum()}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

df.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
print("Now retrain the model: python src/train.py")
