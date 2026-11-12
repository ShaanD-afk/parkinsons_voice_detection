from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

from .audio import frame_rms, mono_for_analysis, read_audio
from .config import PITCH, TASKS_QC


def qc_configuration_hash() -> str:
    raw = json.dumps({"pitch": PITCH, "frame_ms": 25, "hop_ms": 10, "silent_db_below_peak": 50,
                      "clip_threshold": 0.999,"task_aware_silence_and_voicing_flags":True}, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def inspect_recording(row: pd.Series, audio_root: Path) -> dict:
    path = audio_root / str(row.relative_path)
    record = {
        "relative_path": row.relative_path,
        "task_code": row.possible_task_code,
        "group": row.cohort_group,
        "participant_folder": row.participant_folder,
        "sample_rate_manifest_hz": int(row.sample_rate),
        "audio_path_exists": path.exists(),
        "qc_config_sha256": qc_configuration_hash(),
    }
    try:
        if not path.is_file():
            raise FileNotFoundError(str(path))
        info = path.stat()
        data, sr, sf_info = read_audio(path)
        audio, channel_method = mono_for_analysis(data)
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        frame_vals = frame_rms(audio, sr)
        peak_threshold = max(1e-8, peak * 10 ** (-50 / 20))
        silence = frame_vals < peak_threshold
        active_threshold = max(1e-7, float(np.percentile(frame_vals, 95)) * 10 ** (-35 / 20))
        active = frame_vals >= active_threshold
        low_energy = frame_vals < max(1e-7, float(np.percentile(frame_vals, 95)) * 10 ** (-45 / 20))
        p10, p95 = float(np.percentile(frame_vals, 10)), float(np.percentile(frame_vals, 95))
        dynamic_range = float(20 * np.log10(max(p95, 1e-12) / max(p10, 1e-12)))
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch_ac(time_step=PITCH["time_step_s"], pitch_floor=PITCH["floor_hz"], pitch_ceiling=PITCH["ceiling_hz"])
        f0 = pitch.selected_array["frequency"]
        voiced = np.isfinite(f0) & (f0 > 0)
        step = PITCH["time_step_s"]
        flags = []
        if sr != int(row.sample_rate):
            flags.append("manifest_actual_sample_rate_mismatch")
        if peak >= 0.999:
            flags.append("possible_clipping")
        vowel_task = str(row.possible_task_code) in {"VA1","VA2","VE1","VE2","VI1","VI2","VO1","VO2","VU1","VU2"}
        if vowel_task and float(np.mean(silence)) > 0.25:
            flags.append("high_silence_fraction_for_vowel_task")
        if vowel_task and float(np.mean(low_energy)) > 0.5:
            flags.append("high_low_energy_fraction_for_vowel_task")
        if not vowel_task and float(np.mean(active)) < 0.05:
            flags.append("very_low_task_activity_fraction")
        if len(audio) / sr < 0.5:
            flags.append("very_short_recording")
        if vowel_task and not np.any(voiced):
            flags.append("no_voiced_frames_at_reference_pitch_settings")
        elif vowel_task and float(np.mean(voiced)) < 0.2:
            flags.append("low_voiced_frame_fraction")
        if data.shape[1] > 1 and channel_method.startswith("selected_channel"):
            flags.append("stereo_phase_cancellation_risk")
        record.update({
            "status": "ok", "error": "", "file_size_bytes": info.st_size, "file_mtime_ns": info.st_mtime_ns,
            "actual_sample_rate_hz": sr, "sample_rate_matches_manifest": sr == int(row.sample_rate),
            "duration_seconds": len(data) / sr, "channels": data.shape[1], "subtype": sf_info.subtype,
            "bit_depth_bits": {"PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}.get(sf_info.subtype),
            "peak_amplitude": peak, "rms_amplitude": rms, "clipping_fraction": float(np.mean(np.abs(data) >= 0.999)),
            "silence_frame_fraction": float(np.mean(silence)), "low_energy_frame_fraction": float(np.mean(low_energy)),
            "active_frame_fraction_rms_vad": float(np.mean(active)),
            "frame_rms_p10": p10, "frame_rms_p95": p95,
            "rms_dynamic_range_proxy_db_not_snr": dynamic_range,
            "usable_voiced_duration_seconds": float(np.sum(voiced) * step),
            "voiced_frame_fraction_pitch75_500": float(np.mean(voiced)),
            "median_f0_hz_pitch75_500": float(np.median(f0[voiced])) if voiced.any() else np.nan,
            "channel_method_for_analysis": channel_method,
            "qc_flags": ";".join(flags), "qc_flag_count": len(flags),
        })
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "qc_flags": "audio_read_or_analysis_failure", "qc_flag_count": 1})
    return record


def build_qc_manifest(project_root: Path) -> pd.DataFrame:
    full = pd.read_csv(project_root / "data" / "italian_manifest.csv")
    primary = pd.read_csv(project_root / "data" / "primary_cohort_manifest.csv")
    primary = primary[primary.include_primary_cohort.fillna(False).astype(bool) & primary.cohort_group.isin(["ELDERLY_HC", "PD"])]
    primary = primary.assign(cohort_group=primary.cohort_group)
    yhc = full[full.label.eq("YOUNG_HC")].copy().assign(cohort_group="YOUNG_HC")
    selected = pd.concat([primary, yhc], ignore_index=True)
    selected = selected[selected.possible_task_code.isin(TASKS_QC)].copy()
    selected["sample_rate"] = pd.to_numeric(selected.sample_rate, errors="coerce")
    return selected.drop_duplicates("relative_path").reset_index(drop=True)


def run_qc(project_root: Path, dataset_root: Path, out_dir: Path, force: bool = False) -> pd.DataFrame:
    target = out_dir / "qc" / "per_recording_qc.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_qc_manifest(project_root)
    cached = {}
    if target.exists() and not force:
        old = pd.read_csv(target)
        cached = {r.relative_path: r._asdict() for r in old.itertuples(index=False)
                  if getattr(r, "qc_config_sha256", "") == qc_configuration_hash()}
    rows = []
    for i, row in manifest.iterrows():
        path = dataset_root / row.relative_path
        prior = cached.get(row.relative_path)
        if prior and path.exists() and prior.get("file_size_bytes") == path.stat().st_size and prior.get("file_mtime_ns") == path.stat().st_mtime_ns:
            rows.append(prior)
        else:
            rows.append(inspect_recording(row, dataset_root))
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(target, index=False)
            print(f"QC processed {i + 1}/{len(manifest)} recordings")
    result = pd.DataFrame(rows)
    result.to_csv(target, index=False)
    result[result.qc_flag_count.fillna(1).gt(0)].to_csv(out_dir / "qc" / "review_queue.csv", index=False)
    task = result.groupby(["task_code", "group", "actual_sample_rate_hz"], dropna=False).agg(
        recordings=("relative_path", "size"), participants=("participant_folder", "nunique"),
        failed=("status", lambda x: int(x.ne("ok").sum())), flagged_recordings=("qc_flag_count", lambda x: int(x.gt(0).sum())),
        mean_duration_s=("duration_seconds", "mean"), median_duration_s=("duration_seconds", "median"),
        mean_voiced_fraction=("voiced_frame_fraction_pitch75_500", "mean"), mean_clip_fraction=("clipping_fraction", "mean"),
        mean_rms_proxy_db=("rms_dynamic_range_proxy_db_not_snr", "mean"),
    ).reset_index()
    task.to_csv(out_dir / "qc" / "task_qc_summary.csv", index=False)
    group = result.groupby(["group", "task_code"], dropna=False).agg(
        recordings=("relative_path", "size"), participants=("participant_folder", "nunique"),
        failed=("status", lambda x: int(x.ne("ok").sum())), flagged_recordings=("qc_flag_count", lambda x: int(x.gt(0).sum())),
        sample_rates=("actual_sample_rate_hz", lambda x: ";".join(f"{int(k)}Hz:{int(v)}" for k,v in x.value_counts(dropna=False).items())),
    ).reset_index()
    group.to_csv(out_dir / "qc" / "group_qc_summary.csv", index=False)
    return result
