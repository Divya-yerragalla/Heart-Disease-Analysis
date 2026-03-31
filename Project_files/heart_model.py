import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
from imblearn.over_sampling import SMOTE

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("Heart_new2.csv")

# =========================
# CLEAN STRINGS
# =========================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.strip()

# =========================
# BASIC REPLACEMENTS
# =========================
df.replace({
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
}, inplace=True)

# =========================
# AGE MAPPING
# =========================
age_map = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4,
    "40-44": 5, "45-49": 6, "50-54": 7, "55-59": 8,
    "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12,
    "80 or older": 13
}
df["AgeCategory"] = df["AgeCategory"].map(age_map)

# =========================
# DIABETIC FIX
# =========================
df["Diabetic"] = df["Diabetic"].replace({
    "Yes": 1,
    "No": 0,
    "No, borderline diabetes": 0,
    "Yes (during pregnancy)": 1
})

# =========================
# GENHEALTH ENCODING
# =========================
# FIX 1: Use GenHealth — one of the strongest predictors
genhealth_map = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very good": 4,
    "Excellent": 5
}
df["GenHealth"] = df["GenHealth"].map(genhealth_map)

# =========================
# FIX TARGET
# =========================
df["HeartDisease"] = df["HeartDisease"].astype(str).str.strip()
df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
df = df[df["HeartDisease"].isin([0, 1])]
df["HeartDisease"] = df["HeartDisease"].astype(int)

# =========================
# DROP NULLS
# =========================
df = df.dropna()

# =========================
# FEATURE ENGINEERING
# =========================

# FIX 2: Consistent RiskScore logic (no_activity = 1, active = 0)
df["RiskScore"] = (
    df["Smoking"] +
    df["AlcoholDrinking"] +
    df["Stroke"] +
    df["Diabetic"] +
    df["PhysicalActivity"].apply(lambda x: 0 if x == 1 else 1) +  # inactive = risk
    df["KidneyDisease"] +
    df["DiffWalking"]  # FIX: DiffWalking is a strong predictor
)

# BMI Category
df["BMI_Category"] = pd.cut(df["BMI"],
                             bins=[0, 18.5, 25, 30, 100],
                             labels=[0, 1, 2, 3]).astype(int)

# Sleep Risk
df["SleepRisk"] = df["SleepTime"].apply(lambda x: 1 if x < 6 else 0)

# FIX 3: Age risk flag — older age groups are much higher risk
df["HighAgeRisk"] = df["AgeCategory"].apply(lambda x: 1 if x >= 8 else 0)  # 55+

# =========================
# CLASS DISTRIBUTION
# =========================
print("Class Distribution:\n", df["HeartDisease"].value_counts())

# =========================
# FIX 4: ADD MISSING STRONG FEATURES
# =========================
X = df[[
    "BMI",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "Sex",
    "AgeCategory",
    "Diabetic",
    "PhysicalActivity",
    "SleepTime",
    "Asthma",
    "KidneyDisease",
    "DiffWalking",       # NEW — strong predictor
    "GenHealth",         # NEW — strongest predictor
    "PhysicalHealth",    # NEW — days of poor physical health
    "MentalHealth",      # NEW — days of poor mental health
    "SkinCancer",        # NEW — comorbidity
    # Engineered features
    "RiskScore",
    "BMI_Category",
    "SleepRisk",
    "HighAgeRisk",       # NEW
]]

y = df["HeartDisease"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# FIX 5: USE SMOTE ONLY — remove class_weight double-compensation
# SMOTE alone handles imbalance cleanly
# =========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nClass Distribution AFTER SMOTE:\n", pd.Series(y_train_res).value_counts())

# =========================
# MODEL — no class_weight since SMOTE already balanced
# =========================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    class_weight=None,   # FIX: Don't double-compensate with SMOTE
    random_state=42
)

model.fit(X_train_res, y_train_res)

# =========================
# FIX 6: EVALUATE AT THE SAME THRESHOLD USED IN APP
# Use 0.3 threshold — tune for recall on minority class
# =========================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Feature Importances:\n", feat_imp.head(10))

# =========================
# SAVE
# =========================
joblib.dump(model, "heart_model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nModel saved successfully!")