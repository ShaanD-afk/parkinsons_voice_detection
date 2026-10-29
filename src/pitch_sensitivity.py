
from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DATASET_ROOT = Path(
    "/Users/Shaan/Desktop/datasets/italian"
)

MANIFEST_FILE = DATA / "primary_cohort_manifest.csv"
FEATURES_FILE = DATA / "va1_features_raw.csv"

OUTPUT_DIR = DATA / "pitch_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PITCH_CEILING = 500
TIME_STEP = 0.01

CONFIGURATIONS = {
    "floor75_reference": 75,
    "floor60": 60,
    "floor90": 90,
}


# Import the existing extractor for its established
# spectral-feature implementation.
spec = importlib.util.spec_from_file_location(
    "extract_va1_features",
    ROOT / "src" / "extract_va1_features.py",
)

extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


# --------------------------------------------------
# 2. Feature extraction
# --------------------------------------------------

def extract_with_settings(audio_path, pitch_floor):
    """Extract features using a specified pitch floor."""

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float64",
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float64)
    audio = audio[np.isfinite(audio)]

    if len(audio) == 0:
        raise ValueError("Empty or invalid audio")

    duration = len(audio) / sample_rate

    if duration < 0.5:
        raise ValueError(
            f"Recording too short: {duration:.2f}s"
        )

    rms = float(np.sqrt(np.mean(audio**2)))

    if rms < 1e-6:
        raise ValueError("Signal is effectively silent")

    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sample_rate,
    )

    # Pitch analysis
    pitch = sound.to_pitch_ac(
        time_step=TIME_STEP,
        pitch_floor=pitch_floor,
        pitch_ceiling=PITCH_CEILING,
    )

    f0 = pitch.selected_array["frequency"]
    voiced_f0 = f0[
        np.isfinite(f0) & (f0 > 0)
    ]

    if len(voiced_f0) == 0:
        median_f0 = np.nan
        f0_std = np.nan
        f0_iqr = np.nan
    else:
        median_f0 = float(np.median(voiced_f0))

        f0_std = (
            float(np.std(voiced_f0, ddof=1))
            if len(voiced_f0) > 1
            else np.nan
        )

        q25, q75 = np.percentile(
            voiced_f0, [25, 75]
        )
        f0_iqr = float(q75 - q25)

    voiced_fraction = (
        float(len(voiced_f0) / len(f0))
        if len(f0)
        else np.nan
    )

    # Jitter and shimmer
    jitter = np.nan
    shimmer = np.nan

    try:
        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            pitch_floor,
            PITCH_CEILING,
        )

        jitter = float(
            parselmouth.praat.call(
                point_process,
                "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3,
            )
        )

        shimmer = float(
            parselmouth.praat.call(
                [sound, point_process],
                "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6,
            )
        )

    except Exception as exc:
        warnings.warn(
            f"Jitter/shimmer failed for "
            f"{audio_path}: {exc}"
        )

    # Harmonic-to-noise ratio
    hnr_mean = np.nan

    try:
        harmonicity = sound.to_harmonicity_cc(
            time_step=TIME_STEP,
            minimum_pitch=pitch_floor,
            silence_threshold=0.1,
            periods_per_window=1.0,
        )

        hnr_values = harmonicity.values[0]

        # Exclude Praat's -200 dB undefined-value sentinel.
        valid_hnr = (
            np.isfinite(hnr_values)
            & (hnr_values > -100)
        )

        if valid_hnr.any():
            hnr_mean = float(
                np.mean(hnr_values[valid_hnr])
            )

    except Exception as exc:
        warnings.warn(
            f"HNR failed for {audio_path}: {exc}"
        )

    # Reuse the original implementation for spectral features.
    # Its pitch-dependent values are overwritten below.
    original = extractor.extract_features(audio_path)

    original.update({
        "median_f0_hz": median_f0,
        "f0_std_hz": f0_std,
        "f0_iqr_hz": f0_iqr,
        "voiced_frame_fraction": voiced_fraction,
        "jitter_local": jitter,
        "shimmer_local": shimmer,
        "hnr_mean_db": hnr_mean,
        "pitch_floor_used": pitch_floor,
        "pitch_ceiling_used": PITCH_CEILING,
        "time_step_used": TIME_STEP,
    })

    return original


# --------------------------------------------------
# 3. Resolve recording paths
# --------------------------------------------------

def resolve_audio_path(relative_path):
    """
    Resolve paths stored relative to the Italian dataset root.
    Also handles absolute paths and paths already prefixed
    with the dataset directory.
    """

    path = Path(str(relative_path))

    if path.is_absolute() and path.is_file():
        return path

    candidate = DATASET_ROOT / path

    if candidate.is_file():
        return candidate

    # Handle paths that already include the dataset folder name.
    if path.parts and path.parts[0] == DATASET_ROOT.name:
        candidate = DATASET_ROOT.parent / path

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate recording.\n"
        f"Manifest path: {relative_path}\n"
        f"Checked: {DATASET_ROOT / path}"
    )


# --------------------------------------------------
# 4. Load the fixed primary cohort
# --------------------------------------------------

def load_cohort():
    manifest = pd.read_csv(MANIFEST_FILE)
    features = pd.read_csv(FEATURES_FILE)

    required = {
        "relative_path",
        "include_primary_recording",
        "cohort_group",
        "label",
        "participant_folder",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing)}"
        )

    if features["relative_path"].duplicated().any():
        raise ValueError(
            "Feature file contains duplicate recording paths."
        )

    if manifest["relative_path"].duplicated().any():
        raise ValueError(
            "Manifest contains duplicate recording paths."
        )

    existing_paths = set(features["relative_path"])

    included = manifest[
        manifest["include_primary_recording"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    ].copy()

    included = included[
        included["relative_path"].isin(existing_paths)
    ].copy()

    if included.empty:
        raise ValueError(
            "No recordings matched the existing VA1 feature cohort."
        )

    if included["participant_folder"].duplicated().any():
        raise ValueError(
            "Multiple recordings per participant detected."
        )

    return included


# --------------------------------------------------
# 5. Run the sensitivity analysis
# --------------------------------------------------

def main():
    included = load_cohort()

    print(f"Recordings in fixed cohort: {len(included)}")
    print(
        "Groups:",
        included["cohort_group"].value_counts().to_dict(),
    )

    # Resolve paths before starting so path errors are caught early.
    path_map = {}

    for relative_path in included["relative_path"]:
        path_map[relative_path] = resolve_audio_path(
            relative_path
        )

    for config_name, floor in CONFIGURATIONS.items():
        print(
            f"\nConfiguration: {config_name} "
            f"(floor={floor} Hz, "
            f"ceiling={PITCH_CEILING} Hz, "
            f"step={TIME_STEP}s)"
        )

        rows = []
        failures = []

        for _, record in included.iterrows():
            relative_path = record["relative_path"]
            audio_path = path_map[relative_path]

            try:
                result = extract_with_settings(
                    audio_path,
                    pitch_floor=floor,
                )

                result.update({
                    "relative_path": relative_path,
                    "cohort_group": record["cohort_group"],
                    "label": record["label"],
                    "participant_folder": record[
                        "participant_folder"
                    ],
                })

                rows.append(result)

            except Exception as exc:
                failures.append({
                    "relative_path": relative_path,
                    "error": repr(exc),
                })

                print(
                    f"FAILED: {relative_path}: {exc}"
                )

        result_df = pd.DataFrame(rows)
        failures_df = pd.DataFrame(
            failures,
            columns=["relative_path", "error"],
        )

        result_path = (
            OUTPUT_DIR / f"{config_name}.csv"
        )

        failure_path = (
            OUTPUT_DIR / f"{config_name}_failures.csv"
        )

        result_df.to_csv(result_path, index=False)
        failures_df.to_csv(failure_path, index=False)

        print(
            f"Successful: {len(rows)}/{len(included)}"
        )
        print(f"Failures: {len(failures)}")
        print(f"Features saved: {result_path}")
        print(f"Failures saved: {failure_path}")

        if not result_df.empty:
            columns = [
                "median_f0_hz",
                "f0_iqr_hz",
                "voiced_frame_fraction",
                "jitter_local",
                "shimmer_local",
                "hnr_mean_db",
            ]

            print(
                result_df[columns]
                .describe()
                .round(3)
                .to_string()
            )

        if len(rows) != len(included):
            print(
                "WARNING: Not all recordings succeeded. "
                "Do not compare model results until failures "
                "have been investigated."
            )


if __name__ == "__main__":
    main()
