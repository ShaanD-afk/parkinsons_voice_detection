from __future__ import annotations

import numpy as np
import parselmouth

from ..audio import mono_for_analysis, read_audio
from ..config import PITCH, VOWEL_PHYSICAL_RANGES
from ..segmentation import candidate_stable_interval


def _praat_value(obj, command, *args):
    try:
        value = parselmouth.praat.call(obj, command, *args)
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def extract_sustained_vowel(row, audio_path, *, pitch_floor=75.0, stable_region_floor=75.0, resample_to=None):
    data, sr, info = read_audio(audio_path)
    audio, channel_method = mono_for_analysis(data)
    original_sr = sr
    if resample_to is not None and sr != resample_to:
        from ..audio import resample_audio
        audio = resample_audio(audio, sr, resample_to)
        sr = int(resample_to)
    # Keep segmentation fixed at the reference floor during parameter sensitivity runs.
    interval, f0_all = candidate_stable_interval(audio, sr, floor=stable_region_floor)
    if interval is None:
        return {"status": "failed", "failure_reason": "no_voiced_interval", "sample_rate_original_hz": original_sr,
                "sample_rate_analysis_hz": sr, "channels": data.shape[1], "subtype": info.subtype,
                "channel_method": channel_method}
    start, end = interval
    seg = audio[int(start*sr):int(end*sr)]
    if len(seg) < int(0.25 * sr):
        return {"status": "failed", "failure_reason": "stable_voiced_segment_under_250ms", "sample_rate_original_hz": original_sr,
                "sample_rate_analysis_hz": sr, "channels": data.shape[1], "subtype": info.subtype,
                "stable_start_s": start, "stable_end_s": end, "channel_method": channel_method}
    sound = parselmouth.Sound(seg, sampling_frequency=sr)
    pitch = sound.to_pitch_ac(time_step=PITCH["time_step_s"], pitch_floor=pitch_floor, pitch_ceiling=PITCH["ceiling_hz"])
    f0 = pitch.selected_array["frequency"]
    voiced = np.isfinite(f0) & (f0 > 0)
    point = None
    point_failure = ""
    try:
        point = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", pitch_floor, PITCH["ceiling_hz"])
    except Exception as exc:
        point_failure = f"point_process_failed:{type(exc).__name__}"
    jitter = _praat_value(point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) if point is not None else np.nan
    shimmer = _praat_value([sound, point], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) if point is not None else np.nan
    try:
        harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=pitch_floor,
                                              silence_threshold=0.1, periods_per_window=1.0)
        hnr_values = harmonicity.values[0]
        hnr_values = hnr_values[np.isfinite(hnr_values) & (hnr_values > -100)]
        hnr = float(np.mean(hnr_values)) if len(hnr_values) else np.nan
    except Exception:
        hnr = np.nan

    # Formant estimates use one fixed setting across groups. Wide validity bounds flag,
    # rather than discard, possible tracking errors. The task/vowel remains explicit.
    try:
        formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                         maximum_formant=5000.0, window_length=0.025,
                                         pre_emphasis_from=50.0)
    except Exception:
        formants = None
    sample_times = np.arange(0.1 * len(seg) / sr, 0.9 * len(seg) / sr, 0.02)
    formant_values = {k: [] for k in (1, 2, 3)}
    if formants is not None:
        for t in sample_times:
            for k in formant_values:
                try:
                    val = float(formants.get_value_at_time(k, float(t)))
                    if np.isfinite(val) and val > 0:
                        formant_values[k].append(val)
                except Exception:
                    pass
    result = {
        "status": "ok", "failure_reason": "", "sample_rate_original_hz": original_sr,
        "sample_rate_analysis_hz": sr, "channels": data.shape[1], "subtype": info.subtype,
        "channel_method": channel_method, "stable_start_s": start, "stable_end_s": end,
        "stable_voiced_duration_s": end-start, "voiced_frame_fraction": float(np.mean(voiced)),
        "usable_voiced_duration_s": float(np.sum(voiced) * PITCH["time_step_s"]),
        "median_f0_hz": float(np.median(f0[voiced])) if voiced.any() else np.nan,
        "f0_iqr_hz": float(np.percentile(f0[voiced], 75)-np.percentile(f0[voiced], 25)) if voiced.any() else np.nan,
        "f0_std_hz": float(np.std(f0[voiced], ddof=1)) if voiced.sum() > 1 else np.nan,
        "jitter_local": jitter, "shimmer_local": shimmer, "hnr_mean_db": hnr,
        "point_process_failure": point_failure,
    }
    for k, name in [(1,"f1_hz"),(2,"f2_hz"),(3,"f3_hz")]:
        vals = np.asarray(formant_values[k], dtype=float)
        result[name] = float(np.median(vals)) if len(vals) else np.nan
        result[name.replace("_hz", "_iqr_hz")] = float(np.percentile(vals,75)-np.percentile(vals,25)) if len(vals)>1 else np.nan
        lo, hi = VOWEL_PHYSICAL_RANGES[name]
        result[name.replace("_hz", "_range_valid")] = bool(len(vals) and lo <= result[name] <= hi)

    # Frame-level spectral summaries on stable interval; unnormalized waveform.
    frame = max(1, int(0.025 * sr)); hop = max(1, int(0.01 * sr))
    feats = []
    if len(seg) >= frame:
        for pos in range(0, len(seg)-frame+1, hop):
            x = seg[pos:pos+frame] * np.hanning(frame)
            spectrum = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(frame, 1/sr)
            spectrum[0] = 0
            total = spectrum.sum()
            if total <= 0: continue
            centroid = float(np.sum(freqs*spectrum)/total)
            roll_idx = min(int(np.searchsorted(np.cumsum(spectrum), 0.85*total)), len(freqs)-1)
            eps = max(float(np.max(spectrum))*1e-12, 1e-15)
            flatness = float(np.exp(np.mean(np.log(spectrum+eps)))/(np.mean(spectrum)+eps))
            feats.append((centroid, float(freqs[roll_idx]), flatness))
    for j, name in enumerate(["spectral_centroid_hz", "spectral_rolloff_85_hz", "spectral_flatness"]):
        result[name] = float(np.nanmedian(np.array(feats)[:,j])) if feats else np.nan
    result["cpps"] = np.nan
    result["cpps_status"] = "deferred_no_validated_implementation_in_installed_environment"
    validation=[]
    if np.isfinite(result["median_f0_hz"]) and not (pitch_floor <= result["median_f0_hz"] <= PITCH["ceiling_hz"]):
        validation.append("median_f0_outside_configured_range")
    if not (0 <= result["voiced_frame_fraction"] <= 1): validation.append("voiced_fraction_outside_0_1")
    if np.isfinite(jitter) and not (0 <= jitter <= 0.1): validation.append("jitter_outside_broad_plausibility_range")
    if np.isfinite(shimmer) and not (0 <= shimmer <= 1): validation.append("shimmer_outside_broad_plausibility_range")
    if np.isfinite(hnr) and not (-100 <= hnr <= 60): validation.append("hnr_outside_broad_plausibility_range")
    if np.isfinite(result["spectral_flatness"]) and not (0 <= result["spectral_flatness"] <= 1): validation.append("spectral_flatness_outside_0_1")
    result["feature_validation_flags"]=";".join(validation)
    return result
