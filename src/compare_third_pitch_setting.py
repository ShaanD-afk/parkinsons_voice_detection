
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from pathlib import Path

ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"
OUT = "data/pitch_three_settings.csv"

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

configs = {
    "original": (75, 500, 0.01),
    "alternative": (60, 400, 0.005),
    "third": (90, 400, 0.01),
}

def extract(audio, sr, floor, ceiling, step):
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = sound.to_pitch_ac(
        time_step=step,
        pitch_floor=floor,
        pitch_ceiling=ceiling,
    )
    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]

    if len(voiced) == 0:
        return np.nan, np.nan, np.nan, 0.0

    q25, q75 = np.percentile(voiced, [25, 75])
    return (
        float(np.median(voiced)),
        float(np.std(voiced)),
        float(q75 - q25),
        float(len(voiced) / len(values)),
    )

results = []

for _, row in df.iterrows():
    audio, sr = sf.read(ROOT / row["relative_path"], dtype="float64")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if not np.isfinite(audio).all():
        print("Non-finite audio:", row["participant_folder"])
        continue

    result = {
        "participant_folder": row["participant_folder"],
        "cohort_group": row["cohort_group"],
    }

    for name, (floor, ceiling, step) in configs.items():
        median, std, iqr, voiced_fraction = extract(
            audio, sr, floor, ceiling, step
        )
        result.update({
            f"{name}_median_f0": median,
            f"{name}_f0_std": std,
            f"{name}_f0_iqr": iqr,
            f"{name}_voiced_fraction": voiced_fraction,
        })

    results.append(result)

out = pd.DataFrame(results)
out.to_csv(OUT, index=False)

print("Participants compared:", len(out))
print("\nSelected participants:")
print(
    out[out["participant_folder"].isin([
        "AGNESE P",
        "17-28/Giovanni M",
        "GILDA C",
        "GIOVANNA G",
        "GRAZIA G",
    ])].round(3).to_string(index=False)
)
print("\nSaved:", OUT)
