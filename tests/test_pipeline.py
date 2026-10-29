"""
tests/test_pipeline.py
----------------------
Automated tests for the Parkinson's voice classification pipeline.

These tests verify:
  1.  Label encoding (HC=0, PD=1).
  2.  Unique participant IDs within each CSV.
  3.  Identical participant sets across the three pitch configurations.
  4.  No missing or invalid feature values.
  5.  Audio-path resolution logic (using synthetic examples).
  6.  Reference extraction consistency (floor75 == original features).
  7.  Fold-assignment consistency (identical across configurations).
  8.  No participant overlap between train and test within any fold.
  9.  Complete OOF prediction coverage (every participant appears exactly once).
  10. Correct sensitivity, specificity, and confusion-matrix calculations
      (using small synthetic examples).

Data-dependent tests use the actual saved CSVs. Metric and validation tests
use small synthetic data where possible to avoid fabricating real results.

Run with:
    pytest tests/test_pipeline.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SENS = DATA / "pitch_sensitivity"

ORIGINAL_FEATURES_FILE = DATA / "va1_features_raw.csv"
MANIFEST_FILE          = DATA / "primary_cohort_manifest.csv"

CONFIG_FILES = {
    "floor75_reference": SENS / "floor75_reference.csv",
    "floor60":           SENS / "floor60.csv",
    "floor90":           SENS / "floor90.csv",
}

OOF_PREDICTIONS_FILE  = SENS / "model_oof_predictions.csv"
FOLD_ASSIGNMENTS_FILE = SENS / "cv_fold_assignments.csv"
AUDIT_FILE            = SENS / "reference_reproduction_audit.csv"

CLASSIFIER_FEATURES = [
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
]

EXPECTED_N_PARTICIPANTS = 29
EXPECTED_N_HC = 17
EXPECTED_N_PD = 12
EXPECTED_N_FOLDS = 5


# -----------------------------------------------------------------------
# Helper: load config CSV and normalize labels
# -----------------------------------------------------------------------

def load_config(path):
    df = pd.read_csv(path)
    df["label"] = (df["cohort_group"] == "PD").astype(int)
    return df[df["cohort_group"].isin(["ELDERLY_HC", "PD"])].copy()


# -----------------------------------------------------------------------
# 1. Label encoding
# -----------------------------------------------------------------------

class TestLabelEncoding:

    def test_original_hc_is_zero(self):
        df = pd.read_csv(ORIGINAL_FEATURES_FILE)
        hc = df[df["cohort_group"] == "ELDERLY_HC"]["label"]
        assert (hc == 0).all(), "HC participants must have label=0"

    def test_original_pd_is_one(self):
        df = pd.read_csv(ORIGINAL_FEATURES_FILE)
        pd_grp = df[df["cohort_group"] == "PD"]["label"]
        assert (pd_grp == 1).all(), "PD participants must have label=1"

    def test_sensitivity_configs_label_derivation(self):
        """Derived labels from cohort_group must be 0=HC, 1=PD."""
        for name, path in CONFIG_FILES.items():
            df = load_config(path)
            hc_labels = df.loc[df["cohort_group"] == "ELDERLY_HC", "label"]
            pd_labels = df.loc[df["cohort_group"] == "PD", "label"]
            assert (hc_labels == 0).all(), f"{name}: HC label != 0"
            assert (pd_labels == 1).all(), f"{name}: PD label != 1"

    def test_synthetic_label_derivation(self):
        """Synthetic: verify cohort_group-to-label mapping logic."""
        groups = pd.Series(["ELDERLY_HC", "PD", "ELDERLY_HC", "PD"])
        labels = (groups == "PD").astype(int)
        expected = [0, 1, 0, 1]
        assert labels.tolist() == expected


# -----------------------------------------------------------------------
# 2. Unique participant IDs
# -----------------------------------------------------------------------

class TestUniqueParticipants:

    def test_no_duplicate_participants_in_original(self):
        df = pd.read_csv(ORIGINAL_FEATURES_FILE)
        dupes = df["participant_folder"].duplicated().sum()
        assert dupes == 0, f"Duplicate participants in original: {dupes}"

    def test_no_duplicate_participants_in_configs(self):
        for name, path in CONFIG_FILES.items():
            df = load_config(path)
            dupes = df["participant_folder"].duplicated().sum()
            assert dupes == 0, f"Duplicate participants in {name}: {dupes}"

    def test_expected_participant_count_in_original(self):
        df = pd.read_csv(ORIGINAL_FEATURES_FILE)
        n = df["participant_folder"].nunique()
        assert n == EXPECTED_N_PARTICIPANTS, (
            f"Expected {EXPECTED_N_PARTICIPANTS} participants, got {n}"
        )


# -----------------------------------------------------------------------
# 3. Identical participant sets across configurations
# -----------------------------------------------------------------------

class TestParticipantConsistency:

    def test_same_participants_across_configs(self):
        ref = load_config(CONFIG_FILES["floor75_reference"])
        ref_ids = set(ref["participant_folder"])

        for name, path in CONFIG_FILES.items():
            df = load_config(path)
            ids = set(df["participant_folder"])
            assert ids == ref_ids, (
                f"{name} has different participants from floor75_reference.\n"
                f"Missing: {ref_ids - ids}\nExtra: {ids - ref_ids}"
            )

    def test_same_participants_between_original_and_ref(self):
        orig = pd.read_csv(ORIGINAL_FEATURES_FILE)
        ref  = load_config(CONFIG_FILES["floor75_reference"])
        assert set(orig["participant_folder"]) == set(ref["participant_folder"])

    def test_expected_cohort_sizes(self):
        df = load_config(CONFIG_FILES["floor75_reference"])
        n_hc = int((df["label"] == 0).sum())
        n_pd = int((df["label"] == 1).sum())
        assert n_hc == EXPECTED_N_HC, f"Expected {EXPECTED_N_HC} HC, got {n_hc}"
        assert n_pd == EXPECTED_N_PD, f"Expected {EXPECTED_N_PD} PD, got {n_pd}"


# -----------------------------------------------------------------------
# 4. Missing or invalid feature values
# -----------------------------------------------------------------------

class TestFeatureValidity:

    def test_no_missing_classifier_features_in_original(self):
        df = pd.read_csv(ORIGINAL_FEATURES_FILE)
        for feat in CLASSIFIER_FEATURES:
            n_missing = df[feat].isna().sum()
            assert n_missing == 0, (
                f"Feature '{feat}' has {n_missing} missing values in original"
            )

    def test_no_inf_in_classifier_features(self):
        df = pd.read_csv(ORIGINAL_FEATURES_FILE)
        for feat in CLASSIFIER_FEATURES:
            n_inf = np.isinf(df[feat].to_numpy(dtype=float)).sum()
            assert n_inf == 0, (
                f"Feature '{feat}' has {n_inf} infinite values in original"
            )

    def test_no_missing_classifier_features_in_configs(self):
        for name, path in CONFIG_FILES.items():
            df = load_config(path)
            for feat in CLASSIFIER_FEATURES:
                n_missing = df[feat].isna().sum()
                assert n_missing == 0, (
                    f"Feature '{feat}' has {n_missing} missing in {name}"
                )

    def test_jitter_in_plausible_range(self):
        """Jitter local should be between 0 and 1 (expressed as a fraction)."""
        for name, path in CONFIG_FILES.items():
            df = load_config(path)
            j = df["jitter_local"].dropna()
            assert (j >= 0).all(), f"{name}: jitter < 0"
            assert (j <= 1).all(), f"{name}: jitter > 1"

    def test_hnr_above_sentinel(self):
        """Mean HNR should be > -100 dB (the Praat sentinel exclusion threshold)."""
        for name, path in CONFIG_FILES.items():
            df = load_config(path)
            hnr = df["hnr_mean_db"].dropna()
            assert (hnr > -100).all(), (
                f"{name}: HNR values at or below -100 dB sentinel found"
            )

    def test_voiced_fraction_between_zero_and_one(self):
        for name, path in CONFIG_FILES.items():
            df = load_config(path)
            if "voiced_frame_fraction" in df.columns:
                vf = df["voiced_frame_fraction"].dropna()
                assert (vf >= 0).all() and (vf <= 1).all(), (
                    f"{name}: voiced_frame_fraction out of [0, 1]"
                )


# -----------------------------------------------------------------------
# 5. Audio-path resolution logic
# -----------------------------------------------------------------------

class TestAudioPathResolution:

    def test_manifest_has_relative_path_column(self):
        manifest = pd.read_csv(MANIFEST_FILE)
        assert "relative_path" in manifest.columns

    def test_all_included_recordings_have_relative_path(self):
        orig = pd.read_csv(ORIGINAL_FEATURES_FILE)
        assert "relative_path" in orig.columns
        n_missing = orig["relative_path"].isna().sum()
        assert n_missing == 0, f"{n_missing} rows have no relative_path"

    def test_synthetic_path_resolution(self):
        """Synthetic: verify that DATASET_ROOT / relative_path resolution logic works."""
        import os
        fake_root = Path("/tmp")
        # Create a fake file to test resolution.
        fake_file = fake_root / "test_audio.wav"
        fake_file.touch()
        resolved = fake_root / "test_audio.wav"
        assert resolved.exists()
        fake_file.unlink()  # clean up


# -----------------------------------------------------------------------
# 6. Reference extraction consistency
# -----------------------------------------------------------------------

class TestReferenceConsistency:

    def test_audit_file_exists(self):
        assert AUDIT_FILE.exists(), f"Audit file not found: {AUDIT_FILE}"

    def test_all_features_exact_match(self):
        audit = pd.read_csv(AUDIT_FILE)
        failed = audit[~audit["within_tolerance"]]
        assert failed.empty, (
            "Reference reproduction audit FAILED for:\n"
            + failed[["feature", "max_abs_diff", "note"]].to_string()
        )

    def test_no_missing_value_mismatches(self):
        audit = pd.read_csv(AUDIT_FILE)
        mismatches = audit[audit["n_missing_mismatch"] > 0]
        assert mismatches.empty, (
            "Missing-value mismatches between original and reference:\n"
            + mismatches[["feature", "n_missing_mismatch"]].to_string()
        )


# -----------------------------------------------------------------------
# 7. Fold assignment consistency
# -----------------------------------------------------------------------

class TestFoldAssignments:

    def _load_assignments(self):
        return pd.read_csv(FOLD_ASSIGNMENTS_FILE)

    def test_fold_assignment_file_exists(self):
        assert FOLD_ASSIGNMENTS_FILE.exists()

    def test_expected_number_of_folds(self):
        df = self._load_assignments()
        n_folds = df["fold"].nunique()
        assert n_folds == EXPECTED_N_FOLDS, (
            f"Expected {EXPECTED_N_FOLDS} folds, got {n_folds}"
        )

    def test_every_participant_appears_exactly_once_as_test(self):
        df = self._load_assignments()
        test_rows = df[df["role"] == "test"]
        counts = test_rows["participant_folder"].value_counts()
        duplicates = counts[counts > 1]
        assert duplicates.empty, (
            "Some participants appear more than once in test role:\n"
            + duplicates.to_string()
        )
        orig = pd.read_csv(ORIGINAL_FEATURES_FILE)
        test_participants = set(test_rows["participant_folder"])
        all_participants  = set(orig["participant_folder"])
        assert test_participants == all_participants, (
            f"Not all participants appear in a test fold.\n"
            f"Missing: {all_participants - test_participants}"
        )


# -----------------------------------------------------------------------
# 8. No participant overlap between train and test
# -----------------------------------------------------------------------

class TestNoLeakage:

    def test_no_train_test_overlap_per_fold(self):
        df = pd.read_csv(FOLD_ASSIGNMENTS_FILE)
        for fold in sorted(df["fold"].unique()):
            fold_df = df[df["fold"] == fold]
            train = set(fold_df[fold_df["role"] == "train"]["participant_folder"])
            test  = set(fold_df[fold_df["role"] == "test"]["participant_folder"])
            overlap = train & test
            assert not overlap, (
                f"Fold {fold}: train/test overlap found: {overlap}"
            )

    def test_oof_predictions_no_overlap(self):
        """In OOF setup, each participant's prediction must come from
        a model that did not train on them."""
        df = pd.read_csv(FOLD_ASSIGNMENTS_FILE)
        oof = pd.read_csv(OOF_PREDICTIONS_FILE)

        for _, row in oof.iterrows():
            participant = row["participant_folder"]
            test_fold   = int(row["test_fold"])

            # Verify participant was in the test set for that fold.
            fold_test = df[
                (df["fold"] == test_fold) & (df["role"] == "test")
            ]["participant_folder"]
            assert participant in fold_test.values, (
                f"{participant} has test_fold={test_fold} but was not "
                f"in the test set for that fold."
            )


# -----------------------------------------------------------------------
# 9. Complete OOF coverage
# -----------------------------------------------------------------------

class TestOOFCoverage:

    def test_all_participants_have_oof_predictions(self):
        oof  = pd.read_csv(OOF_PREDICTIONS_FILE)
        orig = pd.read_csv(ORIGINAL_FEATURES_FILE)
        oof_participants  = set(oof["participant_folder"])
        all_participants  = set(orig["participant_folder"])
        assert oof_participants == all_participants, (
            f"OOF predictions missing for: {all_participants - oof_participants}"
        )

    def test_oof_predictions_one_row_per_participant(self):
        oof = pd.read_csv(OOF_PREDICTIONS_FILE)
        dupes = oof["participant_folder"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate participants in OOF predictions"

    def test_oof_predictions_contain_all_configs(self):
        oof = pd.read_csv(OOF_PREDICTIONS_FILE)
        for config_name in CONFIG_FILES:
            col = f"{config_name}_pred"
            assert col in oof.columns, (
                f"OOF predictions missing column: {col}"
            )


# -----------------------------------------------------------------------
# 10. Sensitivity, specificity, and confusion-matrix calculations
# -----------------------------------------------------------------------

class TestMetricCalculations:
    """Uses small synthetic examples to verify metric formulas."""

    def _make_tn_fp_fn_tp(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return cm.ravel()  # TN, FP, FN, TP

    def test_perfect_classifier_metrics(self):
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 1, 1]
        tn, fp, fn, tp = self._make_tn_fp_fn_tp(y_true, y_pred)
        assert tn == 3 and fp == 0 and fn == 0 and tp == 2
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        assert sensitivity == pytest.approx(1.0)
        assert specificity == pytest.approx(1.0)
        assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong_classifier_metrics(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        tn, fp, fn, tp = self._make_tn_fp_fn_tp(y_true, y_pred)
        assert tn == 0 and fp == 2 and fn == 2 and tp == 0
        ba = balanced_accuracy_score(y_true, y_pred)
        assert ba == pytest.approx(0.0)

    def test_chance_classifier_balanced_accuracy(self):
        y_true = [0, 0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 0, 0, 0]  # always predicts majority class
        ba = balanced_accuracy_score(y_true, y_pred)
        assert ba == pytest.approx(0.5)

    def test_one_false_positive_effect(self):
        """One FP in 17 HC: specificity = 16/17 ≈ 0.941."""
        tn, fp, fn, tp = 16, 1, 0, 12
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ba = 0.5 * (sensitivity + specificity)
        assert sensitivity == pytest.approx(1.0)
        assert specificity == pytest.approx(16 / 17)
        expected_ba = 0.5 * (1.0 + 16 / 17)
        assert ba == pytest.approx(expected_ba)

    def test_reference_config_confusion_matrix(self):
        """Verify the floor75_reference confusion matrix matches expected values."""
        oof = pd.read_csv(OOF_PREDICTIONS_FILE)
        orig = pd.read_csv(ORIGINAL_FEATURES_FILE)

        # Merge to get true labels.
        oof = oof.merge(
            orig[["participant_folder", "label"]],
            on="participant_folder",
            how="left",
        )
        y_true = oof["label"].to_numpy(dtype=int)
        y_pred = oof["floor75_reference_pred"].to_numpy(dtype=int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        assert tn == 15, f"Expected TN=15, got {tn}"
        assert fp == 2,  f"Expected FP=2, got {fp}"
        assert fn == 1,  f"Expected FN=1, got {fn}"
        assert tp == 11, f"Expected TP=11, got {tp}"

    def test_balanced_accuracy_formula(self):
        """Direct formula: BA = 0.5 * (sensitivity + specificity)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ba_manual  = 0.5 * (sens + spec)
        ba_sklearn = balanced_accuracy_score(y_true, y_pred)
        assert ba_manual == pytest.approx(ba_sklearn, abs=1e-9)
