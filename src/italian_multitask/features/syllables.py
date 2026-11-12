"""Syllable-task utilities; task-specific timing remains gated on protocol review."""

from ..segmentation import rms_vad


def describe_syllable_signal(audio, sample_rate):
    """Return generic activity structure without inferring repetition or syllable meaning."""
    levels, active, threshold = rms_vad(audio, sample_rate)
    return {"rms_vad_frame_count": int(len(levels)), "active_frame_count": int(active.sum()),
            "active_fraction": float(active.mean()) if len(active) else 0.0,
            "activity_threshold_rms": threshold,
            "interpretation": "generic activity only; /pa/ vs /ta/ production pattern not inferred"}
