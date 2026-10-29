
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from pathlib import Path

ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"
OUT = "data/pitch_settings_comparison.csv"

df = pd.read_csv(MANIFEST)

if "include_primary_recording" in df.columns:
    df = df[
        df["include_primary_recording"].astype(str).str.lower()
        .isin(["true", "1"])
    ]

df = df[
    (df["sample_rate"] == 16000)
    & (df["possible_task_code"].astype(str).str.upper() == "VA1")
].copy()

def get_pitch_features(audio, sr, floor, ceiling, step):
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = sound.to_pitch_ac(
        time_step=step,
        pitch_floor=floor,
        pitch_ceiling=ceiling,
    )

    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]

    if len(voiced) == 0:
        return {
            "median_f0": np.nan,
            "f0_std": np.nan,
            "f0_iqr": np.nan,
            "voiced_fraction": 0.0,
        }

    q25, q75 = np.percentile(voiced, [25, 75])

    return {
        "median_f0": float(np.median(voiced)),
        "f0_std": float(np.std(voiced)),
        "f0_iqr": float(q75 - q25),
        "voiced_fraction": float(len(voiced) / len(values)),
    }

rows = []

for _, row in df.iterrows():
    path = ROOT / row["relative_path"]
    audio, sr = sf.read(path, dtype="float64")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if not np.isfinite(audio).all():
        print("Skipping non-finite audio:", row["participant_folder"])
        continue

    original = get_pitch_features(audio, sr, 75, 500, 0.01)
    alternative = get_pitch_features(audio, sr, 60, 400, 0.005)

    result = {
        "participant_folder": row["participant_folder"],
        "cohort_group": row["cohort_group"],
    }

    for name, features in [
        ("original", original),
        ("alternative", alternative),
    ]:
        for feature, value in features.items():
            result[f"{name}_{feature}"] = value

    rows.append(result)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("Participants compared:", len(out))
print("\nLargest median F0 changes:")
out["median_f0_change_hz"] = (
    out["alternative_median_f0"] - out["original_median_f0"]
)
print(
    out.reindex(
        out["median_f0_change_hz"].abs()
        .sort_values(ascending=False).index
    )[
        ["participant_folder", "cohort_group",
         "original_median_f0", "alternative_median_f0",
         "median_f0_change_hz",
         "original_f0_iqr", "alternative_f0_iqr",
         "original_voiced_fraction",
         "alternative_voiced_fraction"]
    ].head(10).round(3).to_string(index=False)
)

print("\nFull comparison saved to:", OUT)
