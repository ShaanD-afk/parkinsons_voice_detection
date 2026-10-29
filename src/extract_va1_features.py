
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf


def is_true(value):
    """Safely interpret boolean values read from a CSV."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def extract_features(audio_path):
    """Extract acoustic features from one sustained-vowel recording."""

    # Read audio and verify that it is mono or convert stereo to mono.
    audio, sample_rate = sf.read(audio_path, dtype="float64")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float64)
    audio = audio[np.isfinite(audio)]

    if len(audio) == 0:
        raise ValueError("Empty or invalid audio signal")

    duration = len(audio) / sample_rate
    peak_amplitude = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    if duration < 0.5:
        raise ValueError(f"Recording too short: {duration:.2f}s")

    if rms < 1e-6:
        raise ValueError("Signal is effectively silent")

    # Construct a Praat Sound object.
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)

    # Pitch analysis. These settings are starting values, not universal
    # settings suitable for every speaker.
    pitch = sound.to_pitch_ac(
        time_step=0.01,
        pitch_floor=75,
        pitch_ceiling=500,
    )

    f0 = pitch.selected_array["frequency"]
    voiced_f0 = f0[np.isfinite(f0) & (f0 > 0)]

    if len(voiced_f0) == 0:
        median_f0 = np.nan
        f0_std = np.nan
        f0_iqr = np.nan
        voiced_frame_fraction = 0.0
    else:
        median_f0 = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0, ddof=1)) if len(voiced_f0) > 1 else np.nan
        q25, q75 = np.percentile(voiced_f0, [25, 75])
        f0_iqr = float(q75 - q25)
        voiced_frame_fraction = float(len(voiced_f0) / len(f0))

    
    # Jitter, shimmer and harmonic-to-noise ratio.
    jitter_local = np.nan
    shimmer_local = np.nan
    hnr_mean = np.nan

    try:
        # Praat commands are exposed through parselmouth.praat.call().
        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            75,
            500,
        )

        jitter_local = float(
            parselmouth.praat.call(
                point_process,
                "Get jitter (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
            )
        )

        shimmer_local = float(
            parselmouth.praat.call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
        )

    except Exception as exc:
        warnings.warn(
            f"Jitter/shimmer failed for {audio_path}: {exc}"
        )

    try:
        harmonicity = sound.to_harmonicity_cc(
            time_step=0.01,
            minimum_pitch=75,
            silence_threshold=0.1,
            periods_per_window=1.0,
        )
        hnr_values = harmonicity.values[0]

# Praat uses -200 dB as a sentinel for undefined harmonicity frames.
        hnr_valid = hnr_values[
        np.isfinite(hnr_values) & (hnr_values > -100)
        ]

        if len(hnr_valid) > 0:
            hnr_mean = float(np.mean(hnr_valid))
        else:
            hnr_mean = np.nan

    except Exception as exc:
        warnings.warn(f"HNR failed for {audio_path}: {exc}")


    # Basic spectral features.
    # These are descriptive acoustic measurements, not disease biomarkers.
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)

    # Ignore the DC component.
    spectrum[0] = 0.0
    total_magnitude = float(np.sum(spectrum))

    if total_magnitude > 0:
        spectral_centroid = float(
            np.sum(frequencies * spectrum) / total_magnitude
        )

        cumulative = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(
            cumulative, 0.85 * total_magnitude
        )
        rolloff_idx = min(rolloff_idx, len(frequencies) - 1)
        spectral_rolloff_85 = float(frequencies[rolloff_idx])

        normalized = spectrum / total_magnitude
        positive = normalized > 0
        spectral_flatness = float(
            np.exp(np.mean(np.log(spectrum[positive])))
            / np.mean(spectrum)
        ) if np.any(positive) else np.nan
    else:
        spectral_centroid = np.nan
        spectral_rolloff_85 = np.nan
        spectral_flatness = np.nan

    return {
        "duration_seconds": duration,
        "sample_rate": int(sample_rate),
        "rms_amplitude": rms,
        "peak_amplitude": peak_amplitude,
        "median_f0_hz": median_f0,
        "f0_std_hz": f0_std,
        "f0_iqr_hz": f0_iqr,
        "voiced_frame_fraction": voiced_frame_fraction,
        "jitter_local": jitter_local,
        "shimmer_local": shimmer_local,
        "hnr_mean_db": hnr_mean,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_rolloff_85_hz": spectral_rolloff_85,
        "spectral_flatness": spectral_flatness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", default="data/va1_features_raw.csv")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    audio_root = Path(args.audio_root)
    output_path = Path(args.output)

    df = pd.read_csv(manifest_path)

    required = {
        "relative_path",
        "cohort_group",
        "participant_folder",
        "possible_task_code",
        "sample_rate",
        "include_primary_recording",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    eligible = df[
        df["include_primary_recording"].map(is_true)
        & (pd.to_numeric(df["sample_rate"], errors="coerce") == 16000)
        & (df["possible_task_code"].astype(str).str.strip() == "VA1")
    ].copy()

    print(f"Eligible manifest rows: {len(eligible)}")

    rows = []
    failures = []

    for _, record in eligible.iterrows():
        relative_path = Path(str(record["relative_path"]))
        audio_path = audio_root / relative_path

        # Preserve identity and diagnosis information for analysis.
        group = str(record["cohort_group"]).strip()
        label = 1 if group == "PD" else 0

        base = {
            "relative_path": str(record["relative_path"]),
            "cohort_group": group,
            "label": label,
            "participant_folder": str(record["participant_folder"]),
        }

        if not audio_path.is_file():
            failures.append({
                **base,
                "error": f"File not found: {audio_path}",
            })
            continue

        try:
            features = extract_features(audio_path)
            rows.append({**base, **features})
        except Exception as exc:
            failures.append({
                **base,
                "error": str(exc),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    failure_path = output_path.with_name(
        output_path.stem + "_failures.csv"
    )
    pd.DataFrame(
        failures,
        columns=[
            "relative_path",
            "cohort_group",
            "label",
            "participant_folder",
            "error",
        ],
    ).to_csv(failure_path, index=False)

    print("\nFEATURE EXTRACTION SUMMARY")
    print("==========================")
    print(f"Eligible recordings: {len(eligible)}")
    print(f"Successful extractions: {len(rows)}")
    print(f"Failed extractions: {len(failures)}")
    print(f"Feature table: {output_path}")
    print(f"Failure report: {failure_path}")

    if rows:
        result = pd.DataFrame(rows)
        print("\nParticipants represented by group:")
        print(
            result.groupby("cohort_group")["participant_folder"]
            .nunique()
        )


if __name__ == "__main__":
    main()
