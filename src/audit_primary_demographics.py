
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

features = pd.read_csv(DATA / "va1_features_raw.csv")
manifest = pd.read_csv(DATA / "primary_cohort_manifest.csv")

# The audio path is the key linking the actual feature rows
# to the metadata for the same recording.
required_manifest_columns = [
    "relative_path",
    "participant_folder",
    "cohort_group",
    "possible_task_code",
    "sample_rate",
    "metadata_match_status",
    "metadata_age",
    "metadata_sex",
    "include_primary_recording",
    "exclusion_reason",
]

missing = set(required_manifest_columns) - set(manifest.columns)
if missing:
    raise ValueError(f"Missing manifest columns: {sorted(missing)}")

if features["relative_path"].duplicated().any():
    raise ValueError("Duplicate feature recording paths.")

if manifest["relative_path"].duplicated().any():
    raise ValueError("Duplicate manifest recording paths.")

audit = features.merge(
    manifest[required_manifest_columns],
    on="relative_path",
    how="left",
    suffixes=("_feature", "_manifest"),
    validate="one_to_one",
    indicator=True,
)

if not (audit["_merge"] == "both").all():
    print(audit.loc[audit["_merge"] != "both", ["relative_path"]])
    raise ValueError("Some feature recordings have no manifest match.")

audit = audit.drop(columns="_merge")

# Verify that the key cohort fields agree between files.
for feature_col, manifest_col in [
    ("participant_folder_feature", "participant_folder_manifest"),
    ("cohort_group_feature", "cohort_group_manifest"),
    ("sample_rate_feature", "sample_rate_manifest"),
]:
    if feature_col in audit.columns and manifest_col in audit.columns:
        mismatch = audit[feature_col].astype(str) != audit[manifest_col].astype(str)
        if mismatch.any():
            raise ValueError(f"Mismatch detected in {feature_col}.")

# One participant should have one consistent demographic record.
participant_cols = [
    "participant_folder_feature",
    "cohort_group_feature",
    "metadata_age",
    "metadata_sex",
    "metadata_match_status",
]

participants = audit[participant_cols].drop_duplicates()

if participants["participant_folder_feature"].duplicated().any():
    raise ValueError(
        "Inconsistent participant metadata across recordings."
    )

print("\n=== COHORT SIZE ===")
print("Recordings:", len(audit))
print("Unique participants:", participants["participant_folder_feature"].nunique())

print("\n=== GROUP COUNTS ===")
print(participants["cohort_group_feature"].value_counts(dropna=False))

print("\n=== AGE AVAILABILITY AND SUMMARY ===")
print("Missing ages:", participants["metadata_age"].isna().sum())
print(
    participants.groupby("cohort_group_feature")["metadata_age"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(2)
)

print("\n=== SEX COUNTS BY GROUP ===")
print(
    pd.crosstab(
        participants["cohort_group_feature"],
        participants["metadata_sex"],
        margins=True,
        dropna=False,
    )
)

print("\n=== METADATA MATCH STATUS ===")
print(
    pd.crosstab(
        participants["cohort_group_feature"],
        participants["metadata_match_status"],
        margins=True,
        dropna=False,
    )
)

print("\n=== PARTICIPANT-LEVEL AUDIT TABLE ===")
print(
    participants.sort_values(
        ["cohort_group_feature", "participant_folder_feature"]
    ).to_string(index=False)
)

output = DATA / "primary_demographics_audit.csv"
participants.to_csv(output, index=False)
print(f"\nSaved: {output}")
