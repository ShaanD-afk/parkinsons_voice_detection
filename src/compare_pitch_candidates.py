
import numpy as np
import soundfile as sf
import parselmouth
from pathlib import Path

ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"

import pandas as pd

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

row = df[df["participant_folder"] == "GILDA C"].iloc[0]
audio, sr = sf.read(ROOT / row["relative_path"], dtype="float64")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

for center in [6.443, 9.733]:
    # Use a 200 ms segment centred on the selected time.
    start = max(0, int((center - 0.1) * sr))
    end = min(len(audio), int((center + 0.1) * sr))
    segment = audio[start:end]

    sound = parselmouth.Sound(segment, sampling_frequency=sr)

    pitch = sound.to_pitch_ac(
        time_step=0.005,
        pitch_floor=60,
        pitch_ceiling=400,
    )

    values = pitch.selected_array["frequency"]
    voiced = values[values > 0]

    print(f"\nSegment centred at {center:.3f}s")
    print("Voiced frame count:", len(voiced))
    if len(voiced):
        print("F0 percentiles:", np.round(
            np.percentile(voiced, [10, 25, 50, 75, 90]), 2
        ))

    # Autocorrelation provides an independent view of periodicity.
    x = segment - np.mean(segment)
    x = x * np.hanning(len(x))
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    ac = ac / ac[0]

    min_lag = int(sr / 400)
    max_lag = int(sr / 60)
    lags = np.arange(min_lag, min(max_lag + 1, len(ac)))

    # Show the strongest autocorrelation peaks, excluding zero lag.
    local_peaks = [
        lag for lag in lags[1:-1]
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1]
    ]
    top = sorted(local_peaks, key=lambda lag: ac[lag], reverse=True)[:8]

    print("Strongest autocorrelation peaks (Hz, strength):")
    for lag in sorted(top, key=lambda lag: lag):
        print(f"  {sr / lag:7.2f} Hz  {ac[lag]:.3f}")
