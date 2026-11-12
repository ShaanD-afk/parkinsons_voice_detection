from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.italian_multitask.audio import resample_audio
from src.italian_multitask.config import FIRST_VOWELS, GROUPS_PRIMARY
from src.italian_multitask.features.vowels import extract_sustained_vowel
from src.italian_multitask.segmentation import rms_vad
from src.italian_multitask.statistics import _bootstrap_effect, _cliffs_delta, holm_adjust


ROOT = Path(__file__).resolve().parents[1]


def test_polyphase_resampler_length_and_rate():
    sr0, sr1 = 44100, 16000
    x = np.sin(2*np.pi*220*np.arange(sr0)/sr0)
    y = resample_audio(x, sr0, sr1)
    assert abs(len(y)-sr1) <= 1
    assert np.isfinite(y).all()


def test_rms_vad_returns_bounded_activity():
    sr=16000
    silence=np.zeros(sr)
    tone=0.2*np.sin(2*np.pi*220*np.arange(sr)/sr)
    x=np.r_[silence[:sr//4],tone,silence[:sr//4]]
    levels,active,threshold=rms_vad(x,sr)
    assert len(levels)==len(active)
    assert 0 < active.mean() < 1
    assert threshold > 0


def test_sustained_vowel_extraction_synthetic_tone(tmp_path):
    sr=16000
    t=np.arange(sr*2)/sr
    x=0.25*np.sin(2*np.pi*220*t)
    path=tmp_path/"tone.wav"
    sf.write(path,x,sr,subtype="PCM_16")
    row=pd.Series({"relative_path":"tone.wav","possible_task_code":"VA1","group":"ELDERLY_HC",
                   "participant_folder":"SYNTHETIC","sample_rate":sr})
    result=extract_sustained_vowel(row,path)
    assert result["status"]=="ok"
    assert 200 < result["median_f0_hz"] < 240
    assert 0 <= result["voiced_frame_fraction"] <= 1
    assert np.isnan(result["cpps"])


def test_holm_adjustment_monotonic_and_bounded():
    adjusted=holm_adjust([0.01,0.04,0.03])
    assert np.all((adjusted>=0)&(adjusted<=1))
    assert np.all(adjusted[np.argsort([0.01,0.04,0.03])] == np.sort(adjusted))


def test_cliffs_delta_and_bootstrap_interval_bounded():
    x=np.array([1.0,2.0,3.0,4.0]);y=np.array([2.0,3.0,4.0,5.0])
    assert -1<=_cliffs_delta(x,y)<=1
    low,high=_bootstrap_effect(x,y,_cliffs_delta,42)
    assert -1<=low<=high<=1


def test_multitask_output_contract_after_run():
    out=ROOT/"results"/"italian_multitask_v1"
    feature_path=out/"features"/"primary_native16_first_vowels.csv"
    assert feature_path.exists(), "run `python -m src.italian_multitask --dataset-root ...` first"
    f=pd.read_csv(feature_path)
    assert len(f)==145
    assert set(f.task_code)==set(FIRST_VOWELS)
    assert set(f.group)==set(GROUPS_PRIMARY)
    assert set(f.sample_rate_original_hz)=={16000}
    assert not f.duplicated(["group","participant_folder","task_code"]).any()
    assert int(f.group.eq("ELDERLY_HC").sum())==85
    assert int(f.group.eq("PD").sum())==60
    if "spectral_flatness" in f:
        assert f.spectral_flatness.dropna().between(0,1).all()
    failures=pd.read_csv(out/"features"/"feature_extraction_failures.csv")
    assert failures.empty
    integrity=pd.read_json(out/"reliability"/"baseline_integrity.json",typ="series")
    assert bool(integrity["unchanged"])
    checks=pd.read_csv(out/"validation_checks.csv")
    assert checks.status.eq("PASS").all()


def test_qc_output_row_count_matches_input_manifest_after_run():
    out=ROOT/"results"/"italian_multitask_v1"
    path=out/"qc"/"per_recording_qc.csv"
    assert path.exists(), "run QC pipeline first"
    qc=pd.read_csv(path)
    assert not qc.relative_path.duplicated().any()
    assert qc.task_code.nunique()==15

