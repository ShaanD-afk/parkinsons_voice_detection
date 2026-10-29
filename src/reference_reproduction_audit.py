"""
reference_reproduction_audit.py
--------------------------------
Phase 3: Formal audit confirming that the floor75_reference configuration
from pitch_sensitivity.py exactly reproduces the original va1_features_raw.csv.

This script does NOT re-extract features (which would take time); it compares
the already-saved floor75_reference.csv against va1_features_raw.csv.

If the maximum absolute difference for any feature is 0.0, the reproduction
is confirmed exact. Any non-zero discrepancy is reported and must be
investigated before continuing to model comparisons.

Output
------
data/pitch_sensitivity/reference_reproduction_audit.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SENS = DATA / "pitch_sensitivity"

ORIGINAL_FILE  = DATA / "va1_features_raw.csv"
REFERENCE_FILE = SENS / "floor75_reference.csv"
AUDIT_OUT      = SENS / "reference_reproduction_audit.csv"

# All features shared between the two files.
AUDIT_FEATURES = [
    "median_f0_hz",
    "f0_std_hz",
    "f0_iqr_hz",
    "voiced_frame_fraction",
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
    "duration_seconds",
    "rms_amplitude",
    "peak_amplitude",
]

# Acceptable numerical tolerance for floating-point reproducibility.
# Exact reproduction should give 0; this is a safeguard.
TOLERANCE = 1e-9


def main():
    print("=" * 60)
    print("Phase 3: Reference reproduction audit")
    print("=" * 60)

    orig = pd.read_csv(ORIGINAL_FILE)
    ref  = pd.read_csv(REFERENCE_FILE)

    # Normalize label column in ref (may be string cohort_group copy).
    ref["label_numeric"] = (ref["cohort_group"] == "PD").astype(int)
    orig["label_numeric"] = orig["label"].astype(int)

    # Merge on participant_folder.
    shared_participants = set(orig["participant_folder"]) & set(ref["participant_folder"])
    n_orig = len(orig)
    n_ref  = len(ref)
    n_shared = len(shared_participants)

    print(f"Original feature file: {n_orig} participants")
    print(f"Reference config file: {n_ref} participants")
    print(f"Shared participant_folder values: {n_shared}")

    if n_shared != n_orig or n_shared != n_ref:
        raise ValueError(
            "Participant mismatch between original and reference files. "
            f"Orig={n_orig}, Ref={n_ref}, Shared={n_shared}"
        )

    merged = orig.merge(
        ref,
        on="participant_folder",
        how="inner",
        suffixes=("_orig", "_ref"),
    )

    # Check label consistency.
    label_mismatch = (merged["label_numeric_orig"] != merged["label_numeric_ref"]).sum()
    print(f"Label mismatches: {label_mismatch}")
    if label_mismatch > 0:
        raise ValueError("Label encoding differs between original and reference files.")

    # Audit each feature.
    rows = []

    for feat in AUDIT_FEATURES:
        if feat not in orig.columns:
            rows.append({
                "feature": feat,
                "n_compared": 0,
                "n_missing_orig": 0,
                "n_missing_ref": 0,
                "n_missing_mismatch": 0,
                "max_abs_diff": np.nan,
                "mean_abs_diff": np.nan,
                "within_tolerance": False,
                "note": "Feature not found in original file",
            })
            continue

        orig_col = feat + "_orig"
        ref_col  = feat + "_ref"

        orig_vals = merged[orig_col]
        ref_vals  = merged[ref_col]

        n_missing_orig = int(orig_vals.isna().sum())
        n_missing_ref  = int(ref_vals.isna().sum())
        # Rows where one is NaN and the other is not.
        missing_mismatch = int(
            (orig_vals.isna() != ref_vals.isna()).sum()
        )

        valid = ~(orig_vals.isna() | ref_vals.isna())
        n_valid = int(valid.sum())

        if n_valid == 0:
            max_diff  = np.nan
            mean_diff = np.nan
            within    = (missing_mismatch == 0)
        else:
            diffs = (orig_vals[valid].to_numpy(dtype=float) -
                     ref_vals[valid].to_numpy(dtype=float))
            abs_diffs = np.abs(diffs)
            max_diff  = float(abs_diffs.max())
            mean_diff = float(abs_diffs.mean())
            within    = (max_diff <= TOLERANCE) and (missing_mismatch == 0)

        rows.append({
            "feature":             feat,
            "n_compared":          n_valid,
            "n_missing_orig":      n_missing_orig,
            "n_missing_ref":       n_missing_ref,
            "n_missing_mismatch":  missing_mismatch,
            "max_abs_diff":        max_diff,
            "mean_abs_diff":       mean_diff,
            "within_tolerance":    within,
            "note": "Exact match" if within else
                    f"DISCREPANCY: max diff = {max_diff:.2e}",
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(AUDIT_OUT, index=False)
    print(f"\nSaved: {AUDIT_OUT}")

    print("\n" + "=" * 60)
    print("Audit results:")
    print("=" * 60)
    print(audit_df[
        ["feature", "n_compared", "max_abs_diff", "mean_abs_diff",
         "n_missing_mismatch", "within_tolerance", "note"]
    ].round(12).to_string(index=False))

    all_within = audit_df["within_tolerance"].all()

    print("\n" + "=" * 60)
    if all_within:
        print("AUDIT PASSED: Reference configuration exactly reproduces")
        print("the original va1_features_raw.csv for all 13 features.")
        print("Tolerance used:", TOLERANCE)
    else:
        failed = audit_df[~audit_df["within_tolerance"]]
        print("AUDIT FAILED: The following features have discrepancies:")
        print(failed[["feature", "max_abs_diff", "note"]].to_string(index=False))
        raise RuntimeError(
            "Reference reproduction audit failed. "
            "Investigate extraction inconsistencies before running classifiers."
        )

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
