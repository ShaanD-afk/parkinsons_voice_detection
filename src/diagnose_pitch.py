
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from pathlib import Path

MANIFEST = "data/primary_cohort_manifest.csv"
AUDIO_ROOT = Path("/Users/Shaan/Desktop/datasets/italian")

df = pd.read_csv(MANIFEST)

# Keep exactly the same primary cohort and task as feature extraction.
if "include_primary_recording" in df.columns:
    df = df[df["include_primary_recording"].astype(str).str.lower().isin(
        ["true", "1"]
    )]

df = df[
    (df["sample_rate"] == 16000)
    & (df["possible_task_code"].astype(str).str.upper() == "VA1")
].copy()

# Inspect the recordings whose F0 variability was unusually high,
# along with the recordings with unusually low voiced-frame fractions.
targets = {
    "GIOVANNA G",
    "GILDA C",
    "GRAZIA G",
    "17-28/Giustina M",
    "LUIGI P",
    "17-28/Antonia G",
}

df = df[df["participant_folder"].isin(targets)]

for _, row in df.iterrows():
    path = AUDIO_ROOT / row["relative_path"]

    try:
        audio, sr = sf.read(path, dtype="float64")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if not np.isfinite(audio).all():
            print(f"\n{row['participant_folder']}: NON-FINITE AUDIO SAMPLES")
            continue

        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch_ac(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=500,
        )

        values = pitch.selected_array["frequency"]
        voiced = values[values > 0]

        if len(voiced) == 0:
            print(f"\n{row['participant_folder']}: NO VOICED FRAMES")
            continue

        print(f"\n{'=' * 55}")
        print(row["participant_folder"], "|", row["cohort_group"])
        print("Audio duration:", round(len(audio) / sr, 2), "seconds")
        print("Total pitch frames:", len(values))
        print("Voiced frames:", len(voiced))
        print("Voiced fraction:", round(len(voiced) / len(values), 3))
        print("F0 percentiles (Hz):")
        print(np.round(np.percentile(voiced, [5, 25, 50, 75, 95]), 2))
        print("Frames below 85 Hz:", int(np.sum((voiced >= 75) & (voiced < 85))))
        print("Frames above 450 Hz:", int(np.sum(voiced > 450)))

    except Exception as exc:
        print(f"\n{row['participant_folder']}: ERROR: {exc}")
