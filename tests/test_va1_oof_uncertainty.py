import numpy as np
import pandas as pd
import pytest

from src.va1_oof_uncertainty import analyze


def sample():
    return pd.DataFrame({
        "participant_folder": [f"HC/{i}" for i in range(4)] + [f"PD/{i}" for i in range(4)],
        "true_label": [0] * 4 + [1] * 4,
        "predicted_label": [0, 0, 1, 0, 1, 1, 0, 1],
        "predicted_pd_probability": [.1, .2, .8, .4, .9, .7, .3, .6],
    })


def test_stratified_bootstrap_is_deterministic_and_bounded():
    a, da = analyze(sample(), replicates=500, seed=7)
    b, db = analyze(sample(), replicates=500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(da, db)
    assert set(a.metric) == {"balanced_accuracy", "roc_auc", "sensitivity", "specificity"}
    assert ((a.ci_lower >= 0) & (a.ci_upper <= 1)).all()
    assert (a.ci_lower <= a.estimate).all() and (a.estimate <= a.ci_upper).all()


def test_duplicate_participant_is_rejected():
    frame = sample()
    frame.loc[1, "participant_folder"] = frame.loc[0, "participant_folder"]
    with pytest.raises(ValueError, match="one OOF row"):
        analyze(frame, replicates=100)


def test_invalid_probabilities_are_rejected():
    frame = sample()
    frame.loc[0, "predicted_pd_probability"] = 1.2
    with pytest.raises(ValueError, match="in \[0, 1\]"):
        analyze(frame, replicates=100)
