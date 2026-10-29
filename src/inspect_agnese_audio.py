
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
MANIFEST = "data/primary_cohort_manifest.csv"
OUT = Path("data/agnese_audio_diagnostic")
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

matches = df[df["participant_folder"] == "AGNESE P"]
if matches.empty:
    raise RuntimeError("Could not find AGNESE P in the selected cohort.")

row = matches.iloc[0]
path = ROOT / row["relative_path"]

audio, sr = sf.read(path, dtype="float64")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

sound = parselmouth.Sound(audio, sampling_frequency=sr)

# Calculate both pitch tracks over the full recording.
original = sound.to_pitch_ac(
    time_step=0.01, pitch_floor=75, pitch_ceiling=500
)
alternative = sound.to_pitch_ac(
    time_step=0.005, pitch_floor=60, pitch_ceiling=400
)

original_f0 = original.selected_array["frequency"]
original_times = original.xs()
alt_f0 = alternative.selected_array["frequency"]
alt_times = alternative.xs()

# Find a point where the original track is near 217 Hz.
orig_valid = np.flatnonzero(original_f0 > 0)
orig_idx = orig_valid[
    np.argmin(np.abs(original_f0[orig_valid] - 217))
]
t_high = float(original_times[orig_idx])

# Find a point where the alternative track is near 72 Hz.
alt_valid = np.flatnonzero(alt_f0 > 0)
alt_idx = alt_valid[
    np.argmin(np.abs(alt_f0[alt_valid] - 72))
]
t_low = float(alt_times[alt_idx])

print(f"Original track: t={t_high:.3f}s, F0={original_f0[orig_idx]:.2f} Hz")
print(f"Alternative track: t={t_low:.3f}s, F0={alt_f0[alt_idx]:.2f} Hz")

def inspect_segment(center, label):
    # 200 ms segment centred at the selected point.
    start = max(0, int((center - 0.1) * sr))
    end = min(len(audio), int((center + 0.1) * sr))
    segment = audio[start:end]
    segment_time = np.arange(len(segment)) / sr + start / sr

    windowed = segment * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1 / sr)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    axes[0].plot(segment_time, segment)
    axes[0].set_title(f"AGNESE P: {label}, near {center:.3f}s")
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
    filename = OUT / f"agnese_{label}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print("Saved:", filename)

inspect_segment(t_high, "original_217hz")
inspect_segment(t_low, "alternative_72hz")

print("Sample rate:", sr)
print("Duration:", round(len(audio) / sr, 3), "seconds")
