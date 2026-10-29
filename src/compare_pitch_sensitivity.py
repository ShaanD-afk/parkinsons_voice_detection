"""
compare_pitch_sensitivity.py
----------------------------
Phase 1: Participant-level comparison of acoustic features across three
pitch-floor configurations (75 Hz reference, 60 Hz, 90 Hz).

Outputs
-------
data/pitch_sensitivity/participant_feature_changes.csv
    Per-participant, per-feature signed and absolute differences relative to
    the 75 Hz reference.

data/pitch_sensitivity/feature_change_summary.csv
    Summary statistics (mean signed diff, median signed diff, mean/median/
    P90/max absolute diff, missing counts) for each feature × configuration.

data/pitch_sensitivity/participant_ranking.csv
    Participants ranked by largest changes in selected features, used to
    guide pitch-track review in Phase 2.

NOTE ON THRESHOLDS
------------------
The 20 Hz exploratory threshold for changes in median F0 or F0 IQR is used
only as a review aid.  It is NOT a validated quality-control standard and
does NOT automatically exclude any participant.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# 1.  Paths
# -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SENS = DATA / "pitch_sensitivity"


CONFIGS = {
    "floor75_reference": SENS / "floor75_reference.csv",
    "floor60":           SENS / "floor60.csv",
    "floor90":           SENS / "floor90.csv",
}

REFERENCE_KEY = "floor75_reference"

# Features subject to pitch-setting sensitivity.
PITCH_FEATURES = [
    "median_f0_hz",
    "f0_std_hz",
    "f0_iqr_hz",
    "voiced_frame_fraction",
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
]

# Exploratory review threshold — for narrative flagging only.
F0_CHANGE_THRESHOLD_HZ = 20.0   # median F0 or F0 IQR change


# -----------------------------------------------------------------------
# 2.  Load and validate
# -----------------------------------------------------------------------

def load_and_validate():
    """Load all configuration tables and verify participant consistency."""

    tables = {}
    for name, path in CONFIGS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")
        tables[name] = pd.read_csv(path)

    # Every table must contain the same 29 participant IDs.
    ref_ids = set(tables[REFERENCE_KEY]["participant_folder"])

    for name, df in tables.items():
        ids = set(df["participant_folder"])
        if ids != ref_ids:
            missing = ref_ids - ids
            extra   = ids - ref_ids
            raise ValueError(
                f"Participant mismatch in {name}.\n"
                f"  Missing from {name}: {sorted(missing)}\n"
                f"  Extra in {name}: {sorted(extra)}"
            )
        if df["participant_folder"].duplicated().any():
            dupes = df.loc[
                df["participant_folder"].duplicated(keep=False),
                "participant_folder",
            ].tolist()
            raise ValueError(
                f"Duplicate participant_folder in {name}: {dupes}"
            )

    # The pitch_sensitivity.py script may write the cohort_group string
    # into the 'label' column rather than a numeric integer. Derive the
    # numeric label from cohort_group and add/overwrite the label column.
    for name, df in tables.items():
        known_groups = set(df["cohort_group"].unique())
        if not known_groups.issubset({"ELDERLY_HC", "PD"}):
            raise ValueError(
                f"Unexpected cohort_group values in {name}: {known_groups}"
            )
        df["label"] = (df["cohort_group"] == "PD").astype(int)
        tables[name] = df

    n = len(tables[REFERENCE_KEY])
    print(f"Validation passed: {n} participants across all 3 configurations.")
    return tables


# -----------------------------------------------------------------------
# 3.  Participant-level differences
# -----------------------------------------------------------------------

def compute_participant_differences(tables):
    """
    For each non-reference configuration and each participant,
    compute signed difference, absolute difference, and relative
    percentage difference relative to the 75 Hz reference.
    """

    ref = tables[REFERENCE_KEY].set_index("participant_folder")
    alt_configs = [k for k in CONFIGS if k != REFERENCE_KEY]

    rows = []

    for participant in sorted(ref.index):
        ref_row = ref.loc[participant]

        for config_name in alt_configs:
            alt = tables[config_name].set_index("participant_folder")
            alt_row = alt.loc[participant]

            record = {
                "participant_folder": participant,
                "cohort_group": ref_row["cohort_group"],
                "label": int(ref_row["label"]),
                "config_compared": config_name,
                "pitch_floor_compared": int(alt_row["pitch_floor_used"]),
                "pitch_floor_reference": int(ref_row["pitch_floor_used"]),
            }

            for feat in PITCH_FEATURES:
                ref_val = ref_row[feat]
                alt_val = alt_row[feat]

                ref_missing = pd.isna(ref_val)
                alt_missing = pd.isna(alt_val)

                if ref_missing or alt_missing:
                    signed_diff = np.nan
                    abs_diff    = np.nan
                    rel_pct     = np.nan
                    either_missing = True
                else:
                    signed_diff    = float(alt_val) - float(ref_val)
                    abs_diff       = abs(signed_diff)
                    if abs(float(ref_val)) > 1e-12:
                        rel_pct = 100.0 * signed_diff / abs(float(ref_val))
                    else:
                        # Avoid division near zero; percentage undefined.
                        rel_pct = np.nan
                    either_missing = False

                record[f"{feat}_ref"]     = ref_val
                record[f"{feat}_alt"]     = alt_val
                record[f"{feat}_diff"]    = signed_diff
                record[f"{feat}_absdiff"] = abs_diff
                record[f"{feat}_relpct"]  = rel_pct
                record[f"{feat}_missing"] = either_missing

            rows.append(record)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# 4.  Feature-level summary statistics
# -----------------------------------------------------------------------

def compute_feature_summary(diff_df):
    """
    For each (feature, config_compared) pair, compute summary stats
    over the 29 participants.
    """

    alt_configs = diff_df["config_compared"].unique()
    rows = []

    for config in sorted(alt_configs):
        sub = diff_df[diff_df["config_compared"] == config]

        for feat in PITCH_FEATURES:
            abs_col  = f"{feat}_absdiff"
            diff_col = f"{feat}_diff"
            miss_col = f"{feat}_missing"

            abs_vals  = sub[abs_col].dropna()
            diff_vals = sub[diff_col].dropna()
            n_missing = int(sub[miss_col].sum())

            rows.append({
                "config_compared":          config,
                "feature":                  feat,
                "mean_signed_diff":         diff_vals.mean() if len(diff_vals) else np.nan,
                "median_signed_diff":       diff_vals.median() if len(diff_vals) else np.nan,
                "mean_abs_diff":            abs_vals.mean()  if len(abs_vals)  else np.nan,
                "median_abs_diff":          abs_vals.median() if len(abs_vals) else np.nan,
                "p90_abs_diff":             float(np.percentile(abs_vals, 90)) if len(abs_vals) else np.nan,
                "max_abs_diff":             abs_vals.max()   if len(abs_vals)  else np.nan,
                "n_missing_or_invalid":     n_missing,
                "n_participants_compared":  len(sub),
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# 5.  Participant ranking for review
# -----------------------------------------------------------------------

RANKING_FEATURES = [
    "median_f0_hz",
    "f0_iqr_hz",
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
]


def compute_participant_ranking(diff_df):
    """
    Rank participants by the magnitude of change in key features.
    Produces one row per participant × feature × configuration,
    sorted by absolute difference descending.
    """

    rows = []

    for _, row in diff_df.iterrows():
        participant = row["participant_folder"]
        cohort      = row["cohort_group"]
        config      = row["config_compared"]

        for feat in RANKING_FEATURES:
            abs_diff = row[f"{feat}_absdiff"]
            ref_val  = row[f"{feat}_ref"]
            alt_val  = row[f"{feat}_alt"]

            # Exploratory flag for large F0 changes.
            flagged = False
            flag_reason = ""
            if feat in ("median_f0_hz", "f0_iqr_hz"):
                if not pd.isna(abs_diff) and abs_diff >= F0_CHANGE_THRESHOLD_HZ:
                    flagged = True
                    flag_reason = (
                        f"|Δ{feat}| ≥ {F0_CHANGE_THRESHOLD_HZ} Hz "
                        f"(exploratory review threshold)"
                    )

            rows.append({
                "participant_folder": participant,
                "cohort_group":       cohort,
                "feature":            feat,
                "config_compared":    config,
                "ref_value":          ref_val,
                "alt_value":          alt_val,
                "abs_diff":           abs_diff,
                "flagged_for_review": flagged,
                "flag_reason":        flag_reason,
            })

    ranking = (
        pd.DataFrame(rows)
        .sort_values("abs_diff", ascending=False)
        .reset_index(drop=True)
    )

    return ranking


# -----------------------------------------------------------------------
# 6.  Main
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 1: Participant-level pitch sensitivity comparison")
    print("=" * 60)

    tables = load_and_validate()

    # Per-participant differences.
    diff_df = compute_participant_differences(tables)

    n_participants = diff_df["participant_folder"].nunique()
    n_configs      = diff_df["config_compared"].nunique()
    print(f"Participants: {n_participants}")
    print(f"Configuration pairs: {n_configs} (vs. 75 Hz reference)")

    out_changes = SENS / "participant_feature_changes.csv"
    diff_df.to_csv(out_changes, index=False)
    print(f"\nSaved: {out_changes}")

    # Feature-level summary.
    summary_df = compute_feature_summary(diff_df)
    out_summary = SENS / "feature_change_summary.csv"
    summary_df.to_csv(out_summary, index=False)
    print(f"Saved: {out_summary}")

    # Participant ranking.
    ranking_df = compute_participant_ranking(diff_df)
    out_ranking = SENS / "participant_ranking.csv"
    ranking_df.to_csv(out_ranking, index=False)
    print(f"Saved: {out_ranking}")

    # Console summary.
    print("\n" + "=" * 60)
    print("Feature change summary (all participants, both configs):")
    print("=" * 60)
    display_cols = [
        "config_compared",
        "feature",
        "mean_signed_diff",
        "mean_abs_diff",
        "max_abs_diff",
        "n_missing_or_invalid",
    ]
    print(summary_df[display_cols].round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("Top 15 largest changes by feature (across all configs):")
    print("=" * 60)
    top = ranking_df.head(15)[
        ["participant_folder", "cohort_group", "feature",
         "config_compared", "ref_value", "alt_value",
         "abs_diff", "flagged_for_review"]
    ]
    print(top.round(4).to_string(index=False))

    flagged = ranking_df[ranking_df["flagged_for_review"]]
    if not flagged.empty:
        print("\n" + "=" * 60)
        print("Participants flagged for review (exploratory F0 threshold):")
        print("=" * 60)
        print(
            flagged[["participant_folder", "cohort_group",
                      "feature", "config_compared",
                      "abs_diff", "flag_reason"]]
            .drop_duplicates(subset=["participant_folder", "feature", "config_compared"])
            .to_string(index=False)
        )
    else:
        print("\nNo participants exceeded the exploratory 20 Hz F0 threshold.")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
