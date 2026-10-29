
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

features_path = ROOT / "data" / "va1_features_raw.csv"
predictions_path = ROOT / "data" / "baseline_oof_predictions.csv"

features = pd.read_csv(features_path)
predictions = pd.read_csv(predictions_path)

errors = predictions.loc[
    predictions["true_label"] != predictions["predicted_label"],
    ["participant_folder", "cohort_group",
     "true_label", "predicted_label",
     "predicted_pd_probability"],
]

feature_columns = [
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
]

# Check participant identity before joining.
if features["participant_folder"].duplicated().any():
    raise ValueError("Duplicate participants in feature table.")

if predictions["participant_folder"].duplicated().any():
    raise ValueError("Duplicate participants in prediction table.")

merged = predictions.merge(
    features[["participant_folder"] + feature_columns],
    on="participant_folder",
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not (merged["_merge"] == "both").all():
    raise ValueError("Some predictions have no matching feature row.")

merged = merged.drop(columns="_merge")
errors = merged.loc[
    merged["true_label"] != merged["predicted_label"]
].copy()

print("\n=== MISCLASSIFIED PARTICIPANTS: RAW FEATURES ===")
print(
    errors[
        ["participant_folder", "cohort_group",
         "true_label", "predicted_label",
         "predicted_pd_probability"] + feature_columns
    ].round(4).to_string(index=False)
)

print("\n=== GROUP MEDIANS AND INTERQUARTILE RANGES ===")
print(
    merged.groupby("cohort_group")[feature_columns]
    .agg(["median", lambda x: x.quantile(0.75) - x.quantile(0.25)])
    .round(4).to_string()
)

print("\n=== EACH ERROR VS ITS TRUE GROUP ===")
for _, row in errors.iterrows():
    group = row["cohort_group"]
    reference = merged.loc[merged["cohort_group"] == group, feature_columns]

    print(f"\n{row['participant_folder']} — {group}")
    for feature in feature_columns:
        value = row[feature]
        median = reference[feature].median()
        q1 = reference[feature].quantile(0.25)
        q3 = reference[feature].quantile(0.75)

        position = (
            "below the group's middle 50%"
            if value < q1 else
            "above the group's middle 50%"
            if value > q3 else
            "within the group's middle 50%"
        )

        print(
            f"  {feature}: {value:.4f}; "
            f"group median={median:.4f}; {position}"
        )
