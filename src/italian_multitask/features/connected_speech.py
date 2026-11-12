"""Transcript-free connected-speech timing utilities."""

import numpy as np

from ..segmentation import rms_vad


def describe_connected_speech(audio, sample_rate):
    """Return candidate RMS-VAD timing; not a transcript or validated alignment."""
    levels, active, threshold = rms_vad(audio, sample_rate)
    step = 0.01
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    bouts = list(zip(starts, ends))
    return {"rms_vad_active_duration_s": float(active.sum()*step),
            "rms_vad_bout_count_candidate": len(bouts),
            "rms_vad_threshold_rms": threshold,
            "interpretation": "candidate timing only; no transcript, phrase alignment, or speaking rate inferred"}
