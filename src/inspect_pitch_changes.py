
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"
OUT = Path("data/pitch_change_plots")
OUT.mkdir(parents=True, exist_ok=True)

targets = ["AGNESE P", "17-28/Giovanni M"]

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

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    configs = [
        ("Original: 75–500 Hz", 75, 500, 0.01),
        ("Alternative: 60–400 Hz", 60, 400, 0.005),
    ]

    for ax, (label, floor, ceiling, step) in zip(axes, configs):
        pitch = sound.to_pitch_ac(
            time_step=step,
            pitch_floor=floor,
            pitch_ceiling=ceiling,
        )
        times = pitch.xs()
        f0 = pitch.selected_array["frequency"].copy()
        f0[f0 <= 0] = np.nan

        ax.plot(times, f0, ".", markersize=2)
        ax.set_ylim(50, 350)
        ax.set_ylabel("F0 (Hz)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"{participant} — {row['cohort_group']}")
    fig.tight_layout()

    filename = OUT / (
        participant.replace("/", "_").replace(" ", "_") + ".png"
    )
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print("Saved:", filename)
