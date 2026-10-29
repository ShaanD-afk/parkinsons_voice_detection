
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"
OUT = Path("data/gilda_audio_diagnostic")
OUT.mkdir(parents=True, exist_ok=True)

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

matches = df[df["participant_folder"] == "GILDA C"]
if matches.empty:
    raise RuntimeError("Could not find GILDA C in the selected cohort.")

row = matches.iloc[0]
path = ROOT / row["relative_path"]

audio, sr = sf.read(path, dtype="float64")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

sound = parselmouth.Sound(audio, sampling_frequency=sr)
pitch = sound.to_pitch_ac(
    time_step=0.01,
    pitch_floor=75,
    pitch_ceiling=500,
)

# Locate voiced frames near each of the two observed pitch bands.
f0 = pitch.selected_array["frequency"]
times = pitch.xs()

targets = [87.0, 175.0]
selected = []

for target in targets:
    valid = np.flatnonzero(f0 > 0)
    idx = valid[np.argmin(np.abs(f0[valid] - target))]
    t = float(times[idx])
    selected.append((target, t, float(f0[idx])))

    print(
        f"Target {target:.0f} Hz: time={t:.3f}s, "
        f"detected F0={f0[idx]:.2f} Hz"
    )

    # Use a 100 ms window centred on the selected frame.
    half_window = 0.05
    start = max(0, int((t - half_window) * sr))
    end = min(len(audio), int((t + half_window) * sr))
    segment = audio[start:end]
    segment_time = np.arange(len(segment)) / sr + start / sr

    # Apply a Hann window before calculating the magnitude spectrum.
    windowed = segment * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1 / sr)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    axes[0].plot(segment_time, segment)
    axes[0].set_title(
        f"GILDA C: waveform near {t:.2f}s "
        f"(Praat F0={f0[idx]:.1f} Hz)"
    )
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs, spectrum)
    axes[1].set_xlim(0, 1200)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Magnitude spectrum")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    filename = OUT / f"gilda_{int(target)}hz.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    print("Saved:", filename)

print("\nAudio sample rate:", sr)
print("Audio duration:", round(len(audio) / sr, 3), "seconds")
