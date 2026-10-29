
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

FEATURES_FILE = "data/va1_features_raw.csv"

FEATURES = [
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "spectral_centroid_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
]

df = pd.read_csv(FEATURES_FILE)

# Restrict the analysis to the primary HC-vs-PD cohort.
df = df[
    df["cohort_group"].isin(["ELDERLY_HC", "PD"])
].copy()

# Ensure one row per participant. This should be true for the
# current VA1 cohort; stop if duplicate participants are present.
if df["participant_folder"].duplicated().any():
    duplicates = df.loc[
        df["participant_folder"].duplicated(keep=False),
        "participant_folder",
    ].tolist()
    raise ValueError(f"Duplicate participants found: {duplicates}")

# Label PD=1, elderly healthy controls=0.
y = (df["cohort_group"] == "PD").astype(int)
X = df[FEATURES].replace([np.inf, -np.inf], np.nan)

if y.value_counts().min() < 2:
    raise ValueError("Not enough participants in each class.")

print("Participants:", len(df))
print("Class counts:")
print(df["cohort_group"].value_counts())
print("\nFeatures:", FEATURES)
print("\nMissing values per feature:")
print(X.isna().sum())

# All learned preprocessing is fitted within each training fold.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )),
])

# Five folds are feasible with 12 PD and 17 control participants.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring={
        "balanced_accuracy": "balanced_accuracy",
        "sensitivity": "recall",
        "roc_auc": "roc_auc",
    },
    return_train_score=False,
    error_score="raise",
)

print("\nFive-fold cross-validation results")
for metric in ["balanced_accuracy", "sensitivity", "roc_auc"]:
    values = scores[f"test_{metric}"]
    print(
        f"{metric}: fold values={np.round(values, 3)}, "
        f"mean={values.mean():.3f}, SD={values.std(ddof=1):.3f}"
    )

# Out-of-fold predictions: each participant is predicted by a model
# that was not trained on that participant.
pred = cross_val_predict(model, X, y, cv=cv, method="predict")
prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

print("\nOut-of-fold summary")
print("Balanced accuracy:", round(balanced_accuracy_score(y, pred), 3))
print("ROC-AUC:", round(roc_auc_score(y, prob), 3))

print("\nConfusion matrix (rows=true, columns=predicted; labels 0, 1)")
print(confusion_matrix(y, pred, labels=[0, 1]))

print("\nClassification report")
print(classification_report(
    y,
    pred,
    labels=[0, 1],
    target_names=["ELDERLY_HC", "PD"],
    zero_division=0,
))

# Save out-of-fold predictions for later inspection.
out = df[["participant_folder", "cohort_group"]].copy()
out["true_label"] = y.to_numpy()
out["predicted_label"] = pred
out["predicted_pd_probability"] = prob
out.to_csv("data/baseline_oof_predictions.csv", index=False)

print("\nSaved: data/baseline_oof_predictions.csv")
