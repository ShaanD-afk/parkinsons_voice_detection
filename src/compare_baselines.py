
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
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

df = pd.read_csv(INPUT_FILE)

df = df[
    df["cohort_group"].isin(["ELDERLY_HC", "PD"])
].copy()

# Match the original experiment's six-feature cohort.
if "sample_rate" in df.columns:
    df = df[df["sample_rate"] == 16000].copy()

if "possible_task_code" in df.columns:
    df = df[df["possible_task_code"] == "VA1"].copy()

# Retain only the original eligible participants if this flag exists.
if "include_primary_recording" in df.columns:
    df = df[
        df["include_primary_recording"].astype(str).str.lower()
        .isin(["true", "1"])
    ].copy()

if df["participant_folder"].duplicated().any():
    raise ValueError("Expected one recording per participant.")

df = df.dropna(subset=["cohort_group"] + FEATURES).reset_index(drop=True)

X = df[FEATURES]
y = df["cohort_group"].map({"ELDERLY_HC": 0, "PD": 1}).to_numpy()

if len(df) != 29 or np.bincount(y).tolist() != [17, 12]:
    raise ValueError(
        f"Unexpected cohort: {len(df)} participants, "
        f"class counts {np.bincount(y).tolist()}. "
        "Stop and reconcile with the original cohort before comparing."
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),

    "Logistic regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    ),

    "RBF SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            probability=False,
            random_state=42,
        ),
    ),
}

results = []
prediction_rows = []

for model_name, model in models.items():
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # For ROC-AUC, use continuous scores rather than hard labels.
        if model_name == "RBF SVM":
            scores = model.decision_function(X_test)
        elif model_name == "Dummy (most frequent)":
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.predict_proba(X_test)[:, 1]

        fold_result = {
            "model": model_name,
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, scores),
        }
        fold_scores.append(fold_result)

        for i, true, pred, score in zip(
            test_idx, y_test, y_pred, scores
        ):
            prediction_rows.append({
                "model": model_name,
                "fold": fold,
                "participant_folder": df.iloc[i]["participant_folder"],
                "cohort_group": df.iloc[i]["cohort_group"],
                "true_label": int(true),
                "predicted_label": int(pred),
                "out_of_fold_score": float(score),
            })

    results.extend(fold_scores)

fold_df = pd.DataFrame(results)
pred_df = pd.DataFrame(prediction_rows)

print("\n=== FIVE-FOLD RESULTS ===")
for model_name, group in fold_df.groupby("model", sort=False):
    print(f"\n{model_name}")
    for metric in ["balanced_accuracy", "roc_auc"]:
        values = group[metric].to_numpy()
        print(
            f"  {metric}: folds={np.round(values, 3)}, "
            f"mean={values.mean():.3f}, SD={values.std(ddof=1):.3f}"
        )

print("\n=== AGGREGATED OUT-OF-FOLD RESULTS ===")
for model_name, group in pred_df.groupby("model", sort=False):
    true = group["true_label"].to_numpy()
    pred = group["predicted_label"].to_numpy()

    print(f"\n{model_name}")
    print(
        "  Balanced accuracy:",
        round(balanced_accuracy_score(true, pred), 3),
    )
    print("  Confusion matrix:")
    print(confusion_matrix(true, pred, labels=[0, 1]))

out_results = ROOT / "data" / "baseline_model_comparison_folds.csv"
out_predictions = ROOT / "data" / "baseline_model_comparison_predictions.csv"

fold_df.to_csv(out_results, index=False)
pred_df.to_csv(out_predictions, index=False)

print(f"\nSaved fold metrics: {out_results}")
print(f"Saved predictions: {out_predictions}")
