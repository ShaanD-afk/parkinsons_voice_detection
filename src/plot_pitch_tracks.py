
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
import matplotlib.pyplot as plt
from pathlib import Path

AUDIO_ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"
OUTPUT = Path("data/pitch_track_plots")
OUTPUT.mkdir(parents=True, exist_ok=True)

targets = [
    "GIOVANNA G",
    "GILDA C",
    "GRAZIA G",
    "LUIGI P",
    "17-28/Giustina M",
]

df = pd.read_csv(MANIFEST)

if "include_primary_recording" in df.columns:
    df = df[df["include_primary_recording"].astype(str).str.lower().isin(
        ["true", "1"]
    )]

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
    path = AUDIO_ROOT / row["relative_path"]

    audio, sr = sf.read(path, dtype="float64")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = sound.to_pitch_ac(
        time_step=0.01,
        pitch_floor=75,
        pitch_ceiling=500,
    )

    times = pitch.xs()
    frequencies = pitch.selected_array["frequency"].copy()
    frequencies[frequencies <= 0] = np.nan

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, frequencies, ".", markersize=3)
    ax.set_ylim(60, 350)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Detected F0 (Hz)")
    ax.set_title(f"{participant} — {row['cohort_group']}")
    ax.grid(True, alpha=0.3)

    output_file = OUTPUT / (
        participant.replace("/", "_").replace(" ", "_") + ".png"
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    print(output_file)
