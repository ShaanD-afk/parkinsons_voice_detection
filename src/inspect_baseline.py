
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = ROOT / "data" / "baseline_oof_predictions.csv"

df = pd.read_csv(PREDICTIONS_FILE)

# Match the actual columns in your CSV.
TRUE = "true_label"
PRED = "predicted_label"
PROB = "predicted_pd_probability"
PARTICIPANT = "participant_folder"
GROUP = "cohort_group"

print("Participants:", len(df))

# One out-of-fold prediction per participant is expected.
if df[PARTICIPANT].duplicated().any():
    raise ValueError("Duplicate participant predictions found.")

if df[[TRUE, PRED, PROB]].isna().any().any():
    raise ValueError("Missing labels or probabilities found.")

if not df[TRUE].isin([0, 1]).all():
    raise ValueError("Unexpected values in true_label.")

if not df[PRED].isin([0, 1]).all():
    raise ValueError("Unexpected values in predicted_label.")

if not df[PROB].between(0, 1).all():
    raise ValueError("Probabilities must be between 0 and 1.")

df["correct"] = df[TRUE] == df[PRED]

df["predicted_class_probability"] = np.where(
    df[PRED] == 1,
    df[PROB],
    1 - df[PROB],
)

print("\n=== OVERALL METRICS ===")
print("Balanced accuracy:", round(
    balanced_accuracy_score(df[TRUE], df[PRED]), 3
))
print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(df[TRUE], df[PRED], labels=[0, 1]))

print("\n=== MISCLASSIFIED PARTICIPANTS ===")
errors = df.loc[~df["correct"]].copy()

if errors.empty:
    print("No errors found.")
else:
    cols = [
        PARTICIPANT, GROUP, TRUE, PRED, PROB,
        "predicted_class_probability",
    ]
    print(
        errors[cols]
        .sort_values("predicted_class_probability", ascending=False)
        .to_string(index=False)
    )

print("\n=== ALL PARTICIPANTS, ORDERED BY PD PROBABILITY ===")
cols = [
    PARTICIPANT, GROUP, TRUE, PRED, PROB,
    "predicted_class_probability", "correct",
]
print(
    df[cols]
    .sort_values(PROB)
    .to_string(index=False)
)

print("\n=== PD PROBABILITY BY TRUE GROUP ===")
print(
    df.groupby(GROUP)[PROB]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(3)
)

output_file = ROOT / "data" / "baseline_diagnostic.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")
