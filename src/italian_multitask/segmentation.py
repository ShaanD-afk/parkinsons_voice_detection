from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
from scipy.signal import stft

from .audio import frame_rms, mono_for_analysis, read_audio
from .config import FIRST_VOWELS, RANDOM_SEED, TASKS_QC


def rms_vad(audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0):
    levels = frame_rms(audio, sr, frame_ms, hop_ms)
    p95 = float(np.percentile(levels, 95)) if len(levels) else 0.0
    threshold = max(1e-7, p95 * 10 ** (-35 / 20))
    active = levels >= threshold
    # Close one-frame gaps (10 ms) and keep the raw mask for auditability.
    if len(active) > 2:
        for i in range(1, len(active) - 1):
            if not active[i] and active[i - 1] and active[i + 1]:
                active[i] = True
    return levels, active, threshold


def voiced_runs(voiced: np.ndarray):
    edges = np.diff(np.r_[False, voiced, False].astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return list(zip(starts, ends))


def candidate_stable_interval(audio: np.ndarray, sr: int, floor: float = 75.0):
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    voiced = np.isfinite(f0) & (f0 > 0)
    runs = voiced_runs(voiced)
    if not runs:
        return None, f0
    start, end = max(runs, key=lambda x: x[1] - x[0])
    # Remove 10% at each end of the longest voiced run as a provisional transition trim.
    trim = int(round((end - start) * 0.10))
    start2, end2 = start + trim, end - trim
    times = pitch.xs()
    t0, t1 = float(times[start2]), float(times[max(start2, end2 - 1)])
    return (max(0.0, t0), min(len(audio) / sr, t1)), f0


def pilot_rows(manifest: pd.DataFrame, out_dir: Path):
    # One deterministic example per task/group/original-rate stratum; diagnosis key kept separate.
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for (task, group, rate), frame in manifest.groupby(["possible_task_code", "cohort_group", "sample_rate"]):
        pick = frame.iloc[int(rng.integers(0, len(frame)))]
        token = hashlib.sha1(str(pick.relative_path).encode()).hexdigest()[:8].upper()
        rows.append({"blind_id": f"S{token}", "task_code": task, "sample_rate_hz": int(rate),
                     "relative_path": pick.relative_path, "group_key": group,
                     "participant_folder": pick.participant_folder})
    pilot = pd.DataFrame(rows).sort_values(["task_code", "sample_rate_hz", "blind_id"])
    (out_dir / "segmentation").mkdir(parents=True, exist_ok=True)
    pilot[["blind_id", "task_code", "sample_rate_hz", "relative_path"]].to_csv(out_dir / "segmentation" / "blind_pilot_sample.csv", index=False)
    pilot[["blind_id", "group_key", "participant_folder"]].to_csv(out_dir / "segmentation" / "blind_key.csv", index=False)
    return pilot


def plot_recording(row: pd.Series, dataset_root: Path, out_dir: Path):
    data, sr, info = read_audio(dataset_root / row.relative_path)
    audio, _ = mono_for_analysis(data)
    max_seconds = min(len(audio) / sr, 60.0)
    audio = audio[:int(max_seconds * sr)]
    t = np.arange(len(audio)) / sr
    f, tt, z = stft(audio, sr, nperseg=int(0.025 * sr), noverlap=int(0.015 * sr), boundary=None)
    db = 20 * np.log10(np.maximum(np.abs(z), 1e-7))
    levels, active, threshold = rms_vad(audio, sr)
    pitch = parselmouth.Sound(audio, sampling_frequency=sr).to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    pt = pitch.xs()
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    ax[0].plot(t, audio, linewidth=0.35)
    ax[0].set_ylabel("Amplitude")
    ax[1].pcolormesh(tt, f, db, shading="auto", cmap="magma", vmin=-90, vmax=0)
    ax[1].set_ylim(0, min(8000, sr / 2))
    ax[1].set_ylabel("Hz")
    ax[2].plot(np.arange(len(levels)) * 0.01, levels, color="#4c78a8", linewidth=0.8)
    ax[2].axhline(threshold, color="#e45756", linestyle="--", linewidth=0.8)
    ax[2].fill_between(pt, 0, np.nan_to_num(f0, nan=0), alpha=0.4, color="#54a24b")
    ax[2].set_ylabel("RMS / F0")
    ax[2].set_xlabel("Time (s)")
    fig.suptitle(f"{row.task_code} | {row.blind_id} | {sr} Hz")
    path = out_dir / "segmentation" / "plots" / f"{row.blind_id}_{row.task_code}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    active_duration = float(np.sum(active) * 0.01)
    gaps = np.diff(np.r_[False, active, False].astype(int))
    starts, ends = np.where(gaps == 1)[0], np.where(gaps == -1)[0]
    bouts = list(zip(starts, ends))
    pitch_voiced = np.isfinite(f0) & (f0 > 0)
    pitch_runs = voiced_runs(pitch_voiced)
    return {"blind_id": row.blind_id, "task_code": row.task_code, "sample_rate_hz": sr,
            "duration_seconds": info.duration, "active_speech_candidate_seconds": active_duration,
            "rms_vad_bout_count": len(bouts), "voiced_frame_fraction": float(np.mean(pitch_voiced)),
            "pitch_voiced_run_count": len(pitch_runs), "largest_voiced_run_seconds": max((e-s)*0.01 for s,e in pitch_runs) if pitch_runs else 0,
            "pilot_plot": str(path)}


def run_segmentation_pilot(manifest: pd.DataFrame, dataset_root: Path, out_dir: Path):
    pilot = pilot_rows(manifest, out_dir)
    rows = []
    for row in pilot.itertuples(index=False):
        rows.append(plot_recording(pd.Series(row._asdict()), dataset_root, out_dir))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(out_dir / "segmentation" / "pilot_segmentation_diagnostics.csv", index=False)
    family_map = {t: "sustained_vowel_candidate" for t in FIRST_VOWELS + ["VA2", "VE2", "VI2", "VO2", "VU2"]}
    family_map.update({"D1": "syllable", "D2": "syllable", "B1": "reading", "B2": "reading", "FB1": "reading"})
    diagnostics["task_family"] = diagnostics.task_code.map(family_map)
    def failure_reason(row):
        if row.task_family == "sustained_vowel_candidate":
            if row.pitch_voiced_run_count == 0:
                return "no_voiced_run_detected"
            if row.largest_voiced_run_seconds < 0.5:
                return "longest_voiced_run_under_0.5s"
        elif row.active_speech_candidate_seconds < 0.5:
            return "rms_vad_active_duration_under_0.5s"
        return ""
    diagnostics["segmentation_failure_reason"] = diagnostics.apply(failure_reason, axis=1)
    diagnostics["segmentation_flagged"] = diagnostics.segmentation_failure_reason.ne("")
    diagnostics.to_csv(out_dir / "segmentation" / "pilot_segmentation_diagnostics.csv", index=False)
    summary = diagnostics.groupby(["task_family", "task_code"]).agg(
        pilot_recordings=("blind_id", "size"), mean_duration_s=("duration_seconds", "mean"),
        median_active_speech_s=("active_speech_candidate_seconds", "median"),
        median_vowel_pitch_run_s=("largest_voiced_run_seconds", "median"),
        median_rms_vad_bouts=("rms_vad_bout_count", "median"),
        segmentation_flags=("segmentation_flagged", "sum"),
    ).reset_index()
    summary.to_csv(out_dir / "segmentation" / "pilot_family_summary.csv", index=False)
    return diagnostics, summary
