
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from pathlib import Path

ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"

targets = [
    "GILDA C",
    "GRAZIA G",
    "GIOVANNA G",
    "AGNESE P",
    "17-28/Giovanni M",
]

df = pd.read_csv(MANIFEST)

if "include_primary_recording" in df.columns:
    df = df[
        df["include_primary_recording"].astype(str).str.lower()
        .isin(["true", "1"])
    ]

df = df[
    (df["sample_rate"] == 16000)
    & (df["possible_task_code"].astype(str).str.upper() == "VA1")
]

for participant in targets:
    matches = df[df["participant_folder"] == participant]
    if matches.empty:
        print("Not found:", participant)
        continue

    row = matches.iloc[0]
    audio, sr = sf.read(ROOT / row["relative_path"], dtype="float64")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    pitch = sound.to_pitch_ac(
        time_step=0.01,
        pitch_floor=60,
        pitch_ceiling=400,
    )

    f0 = pitch.selected_array["frequency"]
    valid = f0[f0 > 0]
    below_90 = valid[valid < 90]

    print(f"\n{participant} ({row['cohort_group']})")
    print("Voiced frames:", len(valid))
    print("Detected frames below 90 Hz:", len(below_90))
    print(
        "Percentage of voiced frames below 90 Hz:",
        round(100 * len(below_90) / len(valid), 2)
        if len(valid) else "N/A",
    )
    if len(below_90):
        print("Below-90-Hz percentiles:", np.round(
            np.percentile(below_90, [10, 50, 90]), 2
        ))
