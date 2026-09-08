#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABELED_DATA_PATH = Path("Data/processed/Training_Data_clean.csv")
RAW_DATA_PATH = Path("Data/processed/Training_Data_clean.csv")
QUALITY_PRIORITY = ["verified", "high_confidence", "confirmed"]
MODEL_PATH = Path("model/layoff_model.pkl")
META_PATH = Path("model/model_meta.pkl")
MIN_DEPLOYMENT_THRESHOLD = 0.30

TARGET_COL = "layoff_target"
FEATURES = [
    "Age",
    "DistanceFromHome",
    "monthly_income",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "JobSatisfaction",
    "BusinessTravel",
    "department",
    "EducationField",
    "Gender",
    "MaritalStatus",
    "job_role",
    "overtime",
    "industry_proxy",
]


def choose_threshold(y_true, y_prob, min_recall: float = 0.25) -> dict:
    """Pick threshold for balanced precision/recall using F1 with recall floor."""
    best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
        "strategy": "best_f1_with_recall_floor",
    }
    fallback = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
        "strategy": "best_f1_fallback",
    }

    for i in range(10, 91):
        thr = i / 100.0
        pred = (y_prob >= thr).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))

        if recall >= min_recall and f1 > best["f1"]:
            best = {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "strategy": "best_f1_with_recall_floor",
            }
        if f1 > fallback["f1"]:
            fallback = {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "strategy": "best_f1_fallback",
            }

    if best["f1"] < 0:
        return fallback
    return best


def main() -> None:
    data_path = LABELED_DATA_PATH if LABELED_DATA_PATH.exists() else RAW_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {LABELED_DATA_PATH} or {RAW_DATA_PATH}"
        )

    print(f"Using dataset: {data_path}")
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COL}")

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    # Prefer high-quality labels when available.
    if "label_quality" in df.columns:
        quality_series = df["label_quality"].astype(str).str.lower()
        mask = quality_series.isin(QUALITY_PRIORITY)
        if int(mask.sum()) >= 100:
            df = df.loc[mask].copy()
            print(f"Using high-quality labels only: {len(df)} rows")
        else:
            print(
                "High-quality labels not sufficient; using full dataset with current label quality."
            )

    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in FEATURES if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    threshold_info = choose_threshold(y_test, probs, min_recall=0.25)
    tuned_threshold = float(threshold_info["threshold"])
    deployed_threshold = max(tuned_threshold, MIN_DEPLOYMENT_THRESHOLD)
    tuned_preds = (probs >= deployed_threshold).astype(int)

    print(classification_report(y_test, preds, digits=4))
    print("ROC-AUC:", round(float(roc_auc_score(y_test, probs)), 4))
    print(
        "Tuned threshold:",
        deployed_threshold,
        "| precision:",
        round(float(precision_score(y_test, tuned_preds, zero_division=0)), 4),
        "| recall:",
        round(float(recall_score(y_test, tuned_preds, zero_division=0)), 4),
        "| f1:",
        round(float(f1_score(y_test, tuned_preds, zero_division=0)), 4),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(
        {
            "features": FEATURES,
            "target": TARGET_COL,
            "num_features": num_cols,
            "cat_features": cat_cols,
            "class_balance": y.value_counts().to_dict(),
            "best_model": "RandomForest",
            "selection_metric": "single_model_training",
            "decision_threshold": deployed_threshold,
            "tuned_threshold_raw": tuned_threshold,
            "min_deployment_threshold": MIN_DEPLOYMENT_THRESHOLD,
            "threshold_strategy": threshold_info["strategy"],
            "threshold_val_precision": round(threshold_info["precision"], 4),
            "threshold_val_recall": round(threshold_info["recall"], 4),
            "threshold_val_f1": round(threshold_info["f1"], 4),
            "quality_priority": QUALITY_PRIORITY,
            "training_rows_used": int(len(df)),
            "label_quality_distribution": (
                df["label_quality"].value_counts().to_dict()
                if "label_quality" in df.columns
                else {}
            ),
        },
        META_PATH,
    )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
