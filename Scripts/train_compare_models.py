
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("Data/processed/Training_Data.csv")
MODEL_PATH = Path("model/layoff_model.pkl")
META_PATH = Path("model/model_meta.pkl")
RESULTS_PATH = Path("model/model_comparison.csv")

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


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
    }


def find_best_threshold(y_true, y_prob) -> tuple[float, float, float, float]:
    best_t = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    # Search a practical threshold grid to improve minority-class detection.
    for i in range(10, 91):
        t = i / 100.0
        y_pred = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_precision = float(precision_score(y_true, y_pred, zero_division=0))
            best_recall = float(recall_score(y_true, y_pred, zero_division=0))
    return best_t, best_f1, best_precision, best_recall


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target column missing in training data: {TARGET_COL}")

    missing_features = [c for c in FEATURES if c not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in training data: {missing_features}")

    X = train_df[FEATURES].copy()
    y = train_df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in FEATURES if c not in num_cols]
    preprocessor = build_preprocessor(num_cols, cat_cols)

    rows: list[dict[str, float | str]] = []
    best_name = ""
    best_acc = -1.0
    best_val_probs = None
    best_val_y = None

    for name, model in build_models().items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)[:, 1]

        acc = float(accuracy_score(y_val, preds))
        auc = float(roc_auc_score(y_val, probs))
        f1 = float(f1_score(y_val, preds, zero_division=0))
        recall = float(recall_score(y_val, preds, zero_division=0))
        precision = float(precision_score(y_val, preds, zero_division=0))


        rows.append(
            {
                "model": name,
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4),
                "f1_class1": round(f1, 4),
                "recall_class1": round(recall, 4),
                "precision_class1": round(precision, 4),

            }
        )

        print(
            f"{name:18s} | accuracy={acc:.4f} | roc_auc={auc:.4f} | "
            f"f1_class1={f1:.4f} | recall_class1={recall:.4f} | precision_class1={precision:.4f} | "
        )

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_val_probs = probs
            best_val_y = y_val

    results_df = pd.DataFrame(rows).sort_values(["accuracy", "roc_auc"], ascending=False)

    if best_val_probs is None or best_val_y is None:
        raise RuntimeError("Could not compute validation probabilities for threshold tuning.")

    best_threshold, t_f1, t_precision, t_recall = find_best_threshold(best_val_y, best_val_probs)
    print(
        f"\nTuned threshold for {best_name}: {best_threshold:.2f} "
        f"(val f1_class1={t_f1:.4f}, precision={t_precision:.4f}, recall={t_recall:.4f})"
    )

    # Refit best model on full training dataset for deployment.
    best_model = build_models()[best_name]
    final_preprocessor = build_preprocessor(num_cols, cat_cols)
    final_pipe = Pipeline(steps=[("preprocess", final_preprocessor), ("model", best_model)])
    final_pipe.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    joblib.dump(final_pipe, MODEL_PATH)
    joblib.dump(
        {
            "features": FEATURES,
            "target": TARGET_COL,
            "num_features": num_cols,
            "cat_features": cat_cols,
            "class_balance": y.value_counts().to_dict(),
            "best_model": best_name,
            "selection_metric": "accuracy",
            "validation_split": "20% from Training_Data.csv",
            "decision_threshold": best_threshold,
            "threshold_metric": "best_f1_class1_on_validation",
            "threshold_val_f1_class1": round(t_f1, 4),
            "threshold_val_precision_class1": round(t_precision, 4),
            "threshold_val_recall_class1": round(t_recall, 4),
        },
        META_PATH,
    )

    print("\nModel ranking (best by accuracy, validation split from training data):")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison: {RESULTS_PATH}")
    print(f"Saved best model: {MODEL_PATH} ({best_name})")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
