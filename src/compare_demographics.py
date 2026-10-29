
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FEATURES_FILE = DATA / "va1_features_raw.csv"
MANIFEST_FILE = DATA / "primary_cohort_manifest.csv"

ACOUSTIC_FEATURES = [
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
]

TARGET = "label"
ID = "relative_path"
PARTICIPANT_ID = "participant_folder"

N_SPLITS = 5
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Models
# --------------------------------------------------

def make_acoustic_model():
    """Logistic regression using acoustic features."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def make_demographic_model():
    """Logistic regression using age and sex only."""

    preprocessor = ColumnTransformer([
        ("age", StandardScaler(), ["metadata_age"]),
        (
            "sex",
            OneHotEncoder(handle_unknown="ignore"),
            ["metadata_sex"],
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# --------------------------------------------------
# 3. Load and validate data
# --------------------------------------------------

def load_data():
    features = pd.read_csv(FEATURES_FILE)
    manifest = pd.read_csv(MANIFEST_FILE)

    required_feature_columns = {
        ID,
        PARTICIPANT_ID,
        TARGET,
        *ACOUSTIC_FEATURES,
    }

    missing_features = required_feature_columns - set(features.columns)

    if missing_features:
        raise ValueError(
            f"Missing columns in feature file: {sorted(missing_features)}"
        )

    required_metadata_columns = {
        ID,
        "metadata_age",
        "metadata_sex",
    }

    missing_metadata = required_metadata_columns - set(manifest.columns)

    if missing_metadata:
        raise ValueError(
            f"Missing columns in manifest: {sorted(missing_metadata)}"
        )

    if manifest[ID].duplicated().any():
        raise ValueError(
            "The manifest has duplicate relative_path values."
        )

    metadata = manifest[
        [ID, "metadata_age", "metadata_sex"]
    ].copy()

    # Join metadata to the feature table using the recording path.
    df = features.merge(
        metadata,
        on=ID,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = df[df["_merge"] != "both"]

    if not unmatched.empty:
        raise ValueError(
            "Some recordings could not be matched to metadata:\n"
            + unmatched[ID].to_string(index=False)
        )

    df = df.drop(columns="_merge")

    # This experiment requires one recording per participant.
    if df[PARTICIPANT_ID].duplicated().any():
        duplicates = df.loc[
            df[PARTICIPANT_ID].duplicated(keep=False),
            PARTICIPANT_ID,
        ].tolist()

        raise ValueError(
            "Multiple recordings per participant detected. "
            "Resolve these before running participant-level CV. "
            f"Duplicates: {duplicates}"
        )

    # Normalize labels explicitly.
    labels = df[TARGET].astype(str).str.strip().str.upper()

        # Support either numeric labels (0/1) or group labels (HC/PD).
    if set(labels.unique()).issubset({"0", "1"}):
        y = labels.astype(int).to_numpy()

    elif set(labels.unique()).issubset({"HC", "PD"}):
        y = labels.map({"HC": 0, "PD": 1}).to_numpy(dtype=int)

    else:
        raise ValueError(
            f"Unexpected labels: {labels.value_counts().to_dict()}"
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"Expected both classes; encoded counts: "
            f"{np.bincount(y, minlength=2)}"
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Target encoding produced only one class. "
            f"Encoded counts: {np.bincount(y, minlength=2)}"
        )

    if df["metadata_age"].isna().any():
        raise ValueError("Missing age values found in the metadata.")

    if df["metadata_sex"].isna().any():
        raise ValueError("Missing sex values found in the metadata.")

    print("\nData validation")
    print("----------------")
    print("Total participants:", len(df))
    print("Original labels:", labels.value_counts().to_dict())
    print("Encoded counts [HC=0, PD=1]:",
          np.bincount(y, minlength=2))
    print("Unique participants:",
          df[PARTICIPANT_ID].nunique())

    return df, y


# --------------------------------------------------
# 4. Cross-validation and model evaluation
# --------------------------------------------------

def main():
    df, y = load_data()

    X_demo = df[
        ["metadata_age", "metadata_sex"]
    ].copy()

    X_acoustic = df[
        ACOUSTIC_FEATURES
    ].copy()

    models = {
        "Dummy": DummyClassifier(
            strategy="most_frequent"
        ),
        "Age + sex": make_demographic_model(),
        "Acoustic logistic regression": make_acoustic_model(),
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Materialize the splits once so every model uses identical folds.
    splits = list(cv.split(X_acoustic, y))

    print("\nCross-validation fold counts")
    print("----------------------------")

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_counts = np.bincount(
            y[train_idx], minlength=2
        )
        test_counts = np.bincount(
            y[test_idx], minlength=2
        )

        print(
            f"Fold {fold}: "
            f"train [HC, PD]={train_counts.tolist()}, "
            f"test [HC, PD]={test_counts.tolist()}"
        )

        if len(np.unique(y[train_idx])) != 2:
            raise ValueError(
                f"Fold {fold} training data contains only one class."
            )

        if len(np.unique(y[test_idx])) != 2:
            raise ValueError(
                f"Fold {fold} test data contains only one class."
            )

    oof_predictions = {}
    oof_scores = {}
    fold_results = []

    for model_name, model in models.items():

        if model_name == "Age + sex":
            X = X_demo
        else:
            X = X_acoustic

        predictions = np.full(len(df), -1, dtype=int)
        scores = np.full(len(df), np.nan, dtype=float)

        print(f"\nEvaluating: {model_name}")

        for fold, (train_idx, test_idx) in enumerate(
            splits, start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)

            pred = model.predict(X_test).astype(int)
            predictions[test_idx] = pred

            # Extract the probability corresponding to PD (class 1).
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                classes = list(model.classes_)

                if 1 in classes:
                    pd_column = classes.index(1)
                    scores[test_idx] = proba[:, pd_column]
                else:
                    raise ValueError(
                        f"{model_name}, fold {fold}: "
                        "PD class is missing from fitted model."
                    )
            else:
                scores[test_idx] = pred.astype(float)

            fold_ba = balanced_accuracy_score(y_test, pred)
            fold_auc = roc_auc_score(
                y_test, scores[test_idx]
            )

            fold_results.append({
                "model": model_name,
                "fold": fold,
                "balanced_accuracy": fold_ba,
                "roc_auc": fold_auc,
            })

            print(
                f"  Fold {fold}: "
                f"balanced accuracy={fold_ba:.3f}, "
                f"ROC-AUC={fold_auc:.3f}"
            )

        if np.any(predictions == -1) or np.isnan(scores).any():
            raise ValueError(
                f"Incomplete out-of-fold predictions for {model_name}."
            )

        oof_predictions[model_name] = predictions
        oof_scores[model_name] = scores

    # --------------------------------------------------
    # 5. Summarize out-of-fold predictions
    # --------------------------------------------------

    summary = []

    for model_name in models:
        pred = oof_predictions[model_name]
        score = oof_scores[model_name]

        tn, fp, fn, tp = confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        ).ravel()

        summary.append({
            "model": model_name,
            "participants": len(df),
            "healthy_controls": int((y == 0).sum()),
            "parkinsons": int((y == 1).sum()),
            "pooled_oof_balanced_accuracy":
                balanced_accuracy_score(y, pred),
            "pooled_oof_roc_auc":
                roc_auc_score(y, score),
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })

    summary_df = pd.DataFrame(summary)
    folds_df = pd.DataFrame(fold_results)

    # Save comparison results.
    summary_path = DATA / "demographic_baseline_comparison.csv"
    folds_path = DATA / "demographic_baseline_folds.csv"
    predictions_path = DATA / "demographic_baseline_oof_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    prediction_df = df[
        [ID, PARTICIPANT_ID, TARGET, "metadata_age", "metadata_sex"]
    ].copy()

    prediction_df["true_label_encoded"] = y

    for model_name in models:
        safe_name = (
            model_name.lower()
            .replace(" + ", "_")
            .replace(" ", "_")
        )

        prediction_df[f"{safe_name}_prediction"] = (
            oof_predictions[model_name]
        )

        prediction_df[f"{safe_name}_pd_score"] = (
            oof_scores[model_name]
        )

    prediction_df.to_csv(predictions_path, index=False)

    print("\nPooled out-of-fold comparison")
    print("-----------------------------")
    print(summary_df.round(3).to_string(index=False))

    print("\nFold-level mean and standard deviation")
    print("--------------------------------------")
    print(
        folds_df.groupby("model")[
            ["balanced_accuracy", "roc_auc"]
        ].agg(["mean", "std"]).round(3).to_string()
    )

    print("\nSaved results:")
    print(summary_path)
    print(folds_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
