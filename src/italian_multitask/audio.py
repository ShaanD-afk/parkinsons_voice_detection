from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def read_audio(path):
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    if data.size == 0:
        raise ValueError("empty audio")
    if not np.isfinite(data).all():
        raise ValueError("audio contains non-finite samples")
    return data, int(sr), sf.info(path)


def mono_for_analysis(data: np.ndarray) -> tuple[np.ndarray, str]:
    if data.shape[1] == 1:
        return data[:, 0].astype(np.float64), "mono"
    centered = data - data.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    if norms.min() > 0:
        corr = np.corrcoef(centered.T)
        if np.nanmin(corr[np.triu_indices_from(corr, 1)]) < -0.85:
            # Avoid cancellation: use highest-energy channel, preserve original channel count.
            channel = int(np.argmax(np.mean(centered**2, axis=0)))
            return data[:, channel], f"selected_channel_{channel}_due_to_phase_risk"
    return data.mean(axis=1), "arithmetic_mean_channels"


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    if source_rate == target_rate:
        return audio.copy()
    divisor = gcd(int(source_rate), int(target_rate))
    return resample_poly(audio, target_rate // divisor, source_rate // divisor)


def frame_rms(audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0):
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if len(audio) < frame:
        return np.array([float(np.sqrt(np.mean(audio**2)))])
    starts = np.arange(0, len(audio) - frame + 1, hop)
    return np.array([np.sqrt(np.mean(audio[s:s + frame] ** 2)) for s in starts])
