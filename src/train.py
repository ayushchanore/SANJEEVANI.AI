"""
Train fatty liver XGBClassifier pipeline and save model artifacts.
Run: python src/train.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fatty_liver_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["fatty_liver"])
y  = df["fatty_liver"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Impute → Scale ────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc  = scaler.transform(X_test_imp)

# ── Train ─────────────────────────────────────────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train_sc, y_train,
    eval_set=[(X_test_sc, y_test)],
    verbose=False,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]
print(classification_report(y_test, y_pred, target_names=["No NAFLD", "NAFLD"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── SHAP ──────────────────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_sc)

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_names
).sort_values(ascending=False)

print("\nTop SHAP feature importances:")
print(mean_abs_shap.to_string())

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(imputer,       MODEL_DIR / "imputer.pkl")
joblib.dump(scaler,        MODEL_DIR / "scaler.pkl")
joblib.dump(model,         MODEL_DIR / "fatty_liver_model.pkl")
joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

print(f"\nArtifacts saved to {MODEL_DIR}/")
