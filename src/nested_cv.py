
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "va1_features_raw.csv"

FEATURES = [
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
]

# Load the same cohort used for the original baseline.
df = pd.read_csv(INPUT_FILE)
df = df[df["cohort_group"].isin(["ELDERLY_HC", "PD"])].copy()

if "sample_rate" in df.columns:
    df = df[df["sample_rate"] == 16000].copy()

if "possible_task_code" in df.columns:
    df = df[df["possible_task_code"] == "VA1"].copy()

if "include_primary_recording" in df.columns:
    df = df[
        df["include_primary_recording"].astype(str).str.lower()
        .isin(["true", "1"])
    ].copy()

if df["participant_folder"].duplicated().any():
    raise ValueError("More than one recording per participant.")

df = df.dropna(subset=["cohort_group"] + FEATURES).reset_index(drop=True)

X = df[FEATURES]
y = df["cohort_group"].map({"ELDERLY_HC": 0, "PD": 1}).to_numpy()

if len(df) != 29 or np.bincount(y).tolist() != [17, 12]:
    raise ValueError(
        f"Expected 29 participants (17 HC, 12 PD); got {len(df)} "
        f"and class counts {np.bincount(y).tolist()}. "
        "Stop and reconcile the cohort."
    )

# Preprocessing is fitted separately within every training fold.
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )),
])

# Predeclare a small search space; don't expand it after seeing results.
parameter_grid = [
    {
        "model": [
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )
        ],
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
    {
        "model": [
            SVC(
                kernel="rbf",
                class_weight="balanced",
            )
        ],
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", 0.1, 0.01],
    },
]

outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

fold_rows = []
prediction_rows = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # The inner CV only sees the current outer training participants.
    inner_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=100 + fold
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    scores = best_model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    fold_rows.append({
        "fold": fold,
        "selected_model": type(best_model.named_steps["model"]).__name__,
        "inner_best_balanced_accuracy": search.best_score_,
        "outer_balanced_accuracy": balanced_accuracy_score(
            y_test, y_pred
        ),
        "outer_roc_auc": roc_auc_score(y_test, scores),
        "outer_sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "outer_specificity": tn / (tn + fp) if tn + fp else np.nan,
        "best_parameters": str(search.best_params_),
    })

    for i, true, pred, score in zip(
        test_idx, y_test, y_pred, scores
    ):
        prediction_rows.append({
            "fold": fold,
            "participant_folder": df.iloc[i]["participant_folder"],
            "cohort_group": df.iloc[i]["cohort_group"],
            "true_label": int(true),
            "predicted_label": int(pred),
            "decision_score": float(score),
            "selected_model": type(
                best_model.named_steps["model"]
            ).__name__,
        })

fold_df = pd.DataFrame(fold_rows)
pred_df = pd.DataFrame(prediction_rows)

true = pred_df["true_label"].to_numpy()
pred = pred_df["predicted_label"].to_numpy()

tn, fp, fn, tp = confusion_matrix(
    true, pred, labels=[0, 1]
).ravel()

print("\n=== NESTED CROSS-VALIDATION: OUTER FOLDS ===")
print(fold_df.to_string(index=False))

print("\n=== POOLED OUT-OF-FOLD RESULTS ===")
print("Participants:", len(pred_df))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(true, pred), 3),
)
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(true, pred, labels=[0, 1]))
print("Sensitivity:", round(tp / (tp + fn), 3))
print("Specificity:", round(tn / (tn + fp), 3))
print(
    "Pooled ROC-AUC:",
    round(roc_auc_score(true, pred_df["decision_score"]), 3),
)

print("\nSelected model counts:")
print(fold_df["selected_model"].value_counts())

fold_path = ROOT / "data" / "nested_cv_fold_results.csv"
pred_path = ROOT / "data" / "nested_cv_oof_predictions.csv"

fold_df.to_csv(fold_path, index=False)
pred_df.to_csv(pred_path, index=False)

print(f"\nSaved fold results: {fold_path}")
print(f"Saved OOF predictions: {pred_path}")
