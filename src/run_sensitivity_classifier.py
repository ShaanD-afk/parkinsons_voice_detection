"""
run_sensitivity_classifier.py
------------------------------
Phase 4: Evaluate the logistic-regression baseline under each of the three
pitch-floor configurations using identical 5-fold stratified cross-validation.

The purpose is to assess robustness, NOT to find the best pitch setting.
Settings must be evaluated on methodological grounds (see pitch-track review),
not on classification performance.

Features used (unchanged from original baseline):
    jitter_local, shimmer_local, hnr_mean_db,
    spectral_centroid_hz, spectral_rolloff_85_hz, spectral_flatness

Cross-validation:
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    -- identical fold assignments across all configurations --

Outputs
-------
data/pitch_sensitivity/cv_fold_assignments.csv
    Participant, fold number, train/test role — same for all configs.

data/pitch_sensitivity/model_comparison_folds.csv
    Balanced accuracy, sensitivity, specificity, ROC-AUC per fold per config.

data/pitch_sensitivity/model_comparison_summary.csv
    Pooled OOF metrics and confusion matrix per configuration.

data/pitch_sensitivity/model_oof_predictions.csv
    Per-participant OOF predictions and PD probabilities per configuration.
"""

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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------
# 1.  Configuration
# -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SENS = DATA / "pitch_sensitivity"

CONFIG_FILES = {
    "floor75_reference": SENS / "floor75_reference.csv",
    "floor60":           SENS / "floor60.csv",
    "floor90":           SENS / "floor90.csv",
}

CLASSIFIER_FEATURES = [
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
]

N_SPLITS    = 5
RANDOM_SEED = 42


# -----------------------------------------------------------------------
# 2.  Build classifier pipeline
# -----------------------------------------------------------------------

def build_model():
    return Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_SEED,
        )),
    ])


# -----------------------------------------------------------------------
# 3.  Load and validate one configuration
# -----------------------------------------------------------------------

def load_config(name, path):
    df = pd.read_csv(path)

    # Normalize label to int via cohort_group.
    df["label"] = (df["cohort_group"] == "PD").astype(int)

    # Restrict to ELDERLY_HC and PD.
    df = df[df["cohort_group"].isin(["ELDERLY_HC", "PD"])].copy()

    if df["participant_folder"].duplicated().any():
        dupes = df.loc[
            df["participant_folder"].duplicated(keep=False),
            "participant_folder",
        ].tolist()
        raise ValueError(f"Duplicate participants in {name}: {dupes}")

    n_hc = int((df["label"] == 0).sum())
    n_pd = int((df["label"] == 1).sum())

    print(f"  {name}: {len(df)} participants (HC={n_hc}, PD={n_pd})")

    for feat in CLASSIFIER_FEATURES:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' missing in {name}")

    return df


# -----------------------------------------------------------------------
# 4.  Cross-validation
# -----------------------------------------------------------------------

def run_cv(df, splits, config_name):
    """
    Run 5-fold CV using the pre-materialized splits.
    Returns fold_records, oof_predictions, oof_proba, y.
    """

    y = df["label"].to_numpy(dtype=int)
    X = df[CLASSIFIER_FEATURES].replace([np.inf, -np.inf], np.nan)

    n = len(df)
    oof_pred  = np.full(n, -1, dtype=int)
    oof_proba = np.full(n, np.nan, dtype=float)

    fold_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model()
        model.fit(X_train, y_train)

        pred  = model.predict(X_test).astype(int)
        proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

        oof_pred[test_idx]  = pred
        oof_proba[test_idx] = proba

        # Fold-level metrics.
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fold_ba   = balanced_accuracy_score(y_test, pred)
        fold_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fold_auc  = roc_auc_score(y_test, proba)

        fold_records.append({
            "config":        config_name,
            "fold":          fold_idx,
            "n_test":        len(test_idx),
            "n_test_hc":     int((y_test == 0).sum()),
            "n_test_pd":     int((y_test == 1).sum()),
            "balanced_accuracy": fold_ba,
            "sensitivity":   fold_sens,
            "specificity":   fold_spec,
            "roc_auc":       fold_auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

        print(
            f"    Fold {fold_idx}: BA={fold_ba:.3f}  "
            f"Sens={fold_sens:.3f}  Spec={fold_spec:.3f}  "
            f"AUC={fold_auc:.3f}"
        )

    # Verify complete coverage.
    if (oof_pred == -1).any():
        raise RuntimeError("Incomplete OOF predictions.")

    return fold_records, oof_pred, oof_proba, y


# -----------------------------------------------------------------------
# 5.  Main
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 4: Classifier evaluation under 3 pitch configurations")
    print("=" * 60)

    # Load all configurations and verify identical participant sets.
    tables = {}
    for name, path in CONFIG_FILES.items():
        tables[name] = load_config(name, path)

    ref_ids = set(tables["floor75_reference"]["participant_folder"])
    for name, df in tables.items():
        ids = set(df["participant_folder"])
        if ids != ref_ids:
            raise ValueError(
                f"Participant mismatch: {name} vs floor75_reference. "
                f"Missing: {ref_ids - ids}  Extra: {ids - ref_ids}"
            )

    # Sort consistently across all configs.
    sort_key = tables["floor75_reference"]["participant_folder"].sort_values().tolist()

    for name in tables:
        tables[name] = (
            tables[name]
            .set_index("participant_folder")
            .loc[sort_key]
            .reset_index()
        )

    ref_df = tables["floor75_reference"]
    y_ref  = ref_df["label"].to_numpy(dtype=int)

    # Materialize fold splits ONCE from the reference ordering.
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    splits = list(cv.split(ref_df[CLASSIFIER_FEATURES], y_ref))

    # Save fold assignments.
    fold_assignment_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        for i in train_idx:
            fold_assignment_rows.append({
                "fold": fold_idx,
                "participant_folder": ref_df.iloc[i]["participant_folder"],
                "cohort_group":       ref_df.iloc[i]["cohort_group"],
                "label":              int(y_ref[i]),
                "role": "train",
            })
        for i in test_idx:
            fold_assignment_rows.append({
                "fold": fold_idx,
                "participant_folder": ref_df.iloc[i]["participant_folder"],
                "cohort_group":       ref_df.iloc[i]["cohort_group"],
                "label":              int(y_ref[i]),
                "role": "test",
            })

    fold_assign_df = pd.DataFrame(fold_assignment_rows)

    # Verify no train/test overlap within a fold.
    for fold_idx in range(1, N_SPLITS + 1):
        fold_sub = fold_assign_df[fold_assign_df["fold"] == fold_idx]
        train_pts = set(fold_sub[fold_sub["role"] == "train"]["participant_folder"])
        test_pts  = set(fold_sub[fold_sub["role"] == "test"]["participant_folder"])
        overlap   = train_pts & test_pts
        if overlap:
            raise RuntimeError(
                f"Train/test overlap in fold {fold_idx}: {overlap}"
            )

    # Verify full OOF coverage.
    test_covered = set(
        fold_assign_df[fold_assign_df["role"] == "test"]["participant_folder"]
    )
    if test_covered != ref_ids:
        raise RuntimeError("Not all participants appear in a test fold.")

    fold_assign_df.to_csv(SENS / "cv_fold_assignments.csv", index=False)
    print(f"\nSaved: {SENS / 'cv_fold_assignments.csv'}")

    print("\nFold composition (reference ordering):")
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y_ref[train_idx]
        y_test  = y_ref[test_idx]
        print(
            f"  Fold {fold_idx}: train n={len(train_idx)} "
            f"[HC={int((y_train==0).sum())}, PD={int((y_train==1).sum())}]  "
            f"test n={len(test_idx)} "
            f"[HC={int((y_test==0).sum())}, PD={int((y_test==1).sum())}]"
        )

    # Run classifier for each configuration.
    all_fold_records = []
    oof_results      = {}

    for config_name, df in tables.items():
        print(f"\nEvaluating: {config_name}")

        # Verify labels are identical across configs (same participants, same order).
        y_this = df["label"].to_numpy(dtype=int)
        if not np.array_equal(y_this, y_ref):
            raise RuntimeError(
                f"Label array differs between floor75_reference and {config_name}."
            )

        fold_records, oof_pred, oof_proba, y = run_cv(
            df, splits, config_name
        )
        all_fold_records.extend(fold_records)
        oof_results[config_name] = {
            "pred":  oof_pred,
            "proba": oof_proba,
            "y":     y,
        }

    # -----------------------------------------------------------------------
    # Save fold-level results.
    # -----------------------------------------------------------------------

    folds_df = pd.DataFrame(all_fold_records)
    folds_df.to_csv(SENS / "model_comparison_folds.csv", index=False)
    print(f"\nSaved: {SENS / 'model_comparison_folds.csv'}")

    # -----------------------------------------------------------------------
    # Pooled OOF summary per configuration.
    # -----------------------------------------------------------------------

    summary_rows = []

    print("\n" + "=" * 60)
    print("Pooled OOF summary (all configurations):")
    print("=" * 60)

    for config_name, res in oof_results.items():
        y     = res["y"]
        pred  = res["pred"]
        proba = res["proba"]

        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        pooled_ba   = balanced_accuracy_score(y, pred)
        pooled_auc  = roc_auc_score(y, proba)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        # Fold-level mean ± SD.
        sub = folds_df[folds_df["config"] == config_name]
        ba_mean = sub["balanced_accuracy"].mean()
        ba_std  = sub["balanced_accuracy"].std(ddof=1)
        auc_mean = sub["roc_auc"].mean()
        auc_std  = sub["roc_auc"].std(ddof=1)

        summary_rows.append({
            "config":                    config_name,
            "n_participants":             int(len(y)),
            "n_hc":                       int((y == 0).sum()),
            "n_pd":                       int((y == 1).sum()),
            "pooled_oof_balanced_accuracy": pooled_ba,
            "pooled_oof_roc_auc":           pooled_auc,
            "pooled_oof_sensitivity":       sensitivity,
            "pooled_oof_specificity":       specificity,
            "true_negatives":               int(tn),
            "false_positives":              int(fp),
            "false_negatives":              int(fn),
            "true_positives":               int(tp),
            "fold_ba_mean":                 ba_mean,
            "fold_ba_sd":                   ba_std,
            "fold_auc_mean":                auc_mean,
            "fold_auc_sd":                  auc_std,
        })

        print(f"\n{config_name}")
        print(f"  Pooled OOF balanced accuracy : {pooled_ba:.3f}")
        print(f"  Pooled OOF ROC-AUC           : {pooled_auc:.3f}")
        print(f"  Sensitivity                  : {sensitivity:.3f}")
        print(f"  Specificity                  : {specificity:.3f}")
        print(f"  Confusion matrix (TN,FP,FN,TP): {tn},{fp},{fn},{tp}")
        print(f"  Fold BA: {ba_mean:.3f} ± {ba_std:.3f}")
        print(f"  Fold AUC: {auc_mean:.3f} ± {auc_std:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SENS / "model_comparison_summary.csv", index=False)
    print(f"\nSaved: {SENS / 'model_comparison_summary.csv'}")

    # -----------------------------------------------------------------------
    # OOF predictions per participant per configuration.
    # -----------------------------------------------------------------------

    oof_pred_rows = []

    for i, participant in enumerate(ref_df["participant_folder"]):
        row = {
            "participant_folder": participant,
            "cohort_group":       ref_df.iloc[i]["cohort_group"],
            "true_label":         int(y_ref[i]),
        }
        # Find which fold this participant was in (test role).
        participant_fold = fold_assign_df[
            (fold_assign_df["participant_folder"] == participant) &
            (fold_assign_df["role"] == "test")
        ]["fold"].iloc[0]
        row["test_fold"] = int(participant_fold)

        for config_name, res in oof_results.items():
            row[f"{config_name}_pred"]  = int(res["pred"][i])
            row[f"{config_name}_proba"] = float(res["proba"][i])

        oof_pred_rows.append(row)

    oof_pred_df = pd.DataFrame(oof_pred_rows)
    oof_pred_df.to_csv(SENS / "model_oof_predictions.csv", index=False)
    print(f"Saved: {SENS / 'model_oof_predictions.csv'}")

    # -----------------------------------------------------------------------
    # Identify prediction disagreements across configurations.
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Participants where configurations disagree:")
    print("=" * 60)

    configs_list = list(CONFIG_FILES.keys())
    pred_cols = [f"{c}_pred" for c in configs_list]

    disagreements = oof_pred_df[
        oof_pred_df[pred_cols].nunique(axis=1) > 1
    ]

    if disagreements.empty:
        print("All configurations agree on every participant's prediction.")
    else:
        print(disagreements[
            ["participant_folder", "cohort_group", "true_label",
             "test_fold"] + pred_cols
        ].to_string(index=False))

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
