# src/check_audio_quality.py

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
from scipy.io import wavfile


def normalize(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def find_task_code(row):
    """Get the task code from the manifest or the audio filename."""
    task = normalize(row.get("possible_task_code", ""))

    if task:
        return task

    filename = normalize(row.get("filename", ""))
    match = re.match(r"^(VA\d+)", filename)
    return match.group(1) if match else ""


def to_float_audio(data):
    """Convert integer or floating-point WAV data to floating point."""
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        return data.astype(np.float64) / scale

    return data.astype(np.float64)


def inspect_audio(path):
    """Calculate basic signal-quality measures for a WAV file."""
    try:
        sample_rate, raw_audio = wavfile.read(path)

        if raw_audio.size == 0:
            raise ValueError("Audio file is empty")

        audio = to_float_audio(raw_audio)

        # Convert stereo/multichannel audio to mono for quality checks.
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = np.asarray(audio).flatten()

        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains NaN or infinite values")

        duration = len(audio) / sample_rate if sample_rate else 0.0
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio))))

        # These are diagnostic thresholds, not automatic exclusion criteria.
        near_silence_fraction = float(np.mean(np.abs(audio) < 0.001))
        clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))

        flags = []

        if duration < 0.3:
            flags.append("very_short_duration")
        elif duration < 1.0:
            flags.append("short_duration_review")

        if rms < 0.005:
            flags.append("low_signal_level_review")

        if near_silence_fraction > 0.95:
            flags.append("mostly_silent_review")

        if clipping_fraction > 0.001:
            flags.append("possible_clipping_review")

        if peak == 0:
            flags.append("all_zero_signal")

        return {
            "audio_read_status": "ok",
            "actual_sample_rate": int(sample_rate),
            "duration_seconds": round(duration, 4),
            "peak_amplitude": round(peak, 6),
            "rms_amplitude": round(rms, 6),
            "near_silence_fraction": round(near_silence_fraction, 6),
            "clipping_fraction": round(clipping_fraction, 6),
            "quality_flags": ";".join(flags),
            "error": "",
        }

    except Exception as exc:
        return {
            "audio_read_status": "error",
            "actual_sample_rate": pd.NA,
            "duration_seconds": pd.NA,
            "peak_amplitude": pd.NA,
            "rms_amplitude": pd.NA,
            "near_silence_fraction": pd.NA,
            "clipping_fraction": pd.NA,
            "quality_flags": "file_read_error",
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Quality-check VA1 recordings in the primary cohort."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to data/primary_cohort_manifest.csv",
    )
    parser.add_argument(
        "--audio-root",
        required=True,
        help="Root directory containing the original dataset audio folders.",
    )
    parser.add_argument(
        "--output",
        default="data/va1_quality_report.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    audio_root = Path(args.audio_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required = {"relative_path", "possible_task_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    df["task_code_normalized"] = df.apply(find_task_code, axis=1)

    # Only VA1 recordings are in scope for this initial quality check.
    df = df[df["task_code_normalized"] == "VA1"].copy()

    if "include_primary_recording" in df.columns:
        df["include_primary_recording"] = (
            df["include_primary_recording"].fillna(False).astype(bool)
        )
        df = df[df["include_primary_recording"]].copy()
    elif "include_primary_cohort" in df.columns:
        df["include_primary_cohort"] = (
            df["include_primary_cohort"].fillna(False).astype(bool)
        )
        df = df[df["include_primary_cohort"]].copy()
        if "sample_rate" in df.columns:
            df = df[pd.to_numeric(df["sample_rate"], errors="coerce") == 16000]
    else:
        raise ValueError(
            "Manifest has no inclusion column. Run reconcile_metadata.py first."
        )

    if df.empty:
        raise ValueError(
            "No eligible VA1 recordings found. Check the manifest's task codes "
            "and inclusion columns."
        )

    results = []

    for _, row in df.iterrows():
        relative_path = Path(str(row["relative_path"]))
        audio_path = audio_root / relative_path

        result = inspect_audio(audio_path)

        record = row.to_dict()
        record["audio_path_checked"] = str(audio_path)
        record.update(result)

        # The path may be incorrect if the manifest's relative_path includes
        # a parent directory not present under --audio-root.
        record["file_exists"] = audio_path.is_file()

        if not record["file_exists"]:
            record["audio_read_status"] = "file_not_found"
            record["quality_flags"] = "file_not_found"
            record["error"] = "Resolved audio path does not exist"

        results.append(record)

    report = pd.DataFrame(results)
    report.to_csv(output_path, index=False)

    print("\nVA1 AUDIO QUALITY CHECK")
    print("=======================")
    print(f"Eligible VA1 recordings checked: {len(report)}")
    print(f"Files found: {int(report['file_exists'].sum())}")
    print(
        "Files with read errors: "
        f"{int((report['audio_read_status'] != 'ok').sum())}"
    )
    print(
        "Recordings with one or more flags: "
        f"{int(report['quality_flags'].fillna('').ne('').sum())}"
    )
    print(f"Report saved to: {output_path.resolve()}")
    print("\nFlags indicate recordings to review, not automatic exclusions.")


if __name__ == "__main__":
    main()