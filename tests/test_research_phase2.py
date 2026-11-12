from pathlib import Path
import pandas as pd

from src.italian_research_phase2 import _event_features, _vad_features, holm
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/italian_research_phase2'


def test_holm_adjustment_is_monotone_in_sorted_order():
    p=np.array([.04,.001,.02,.5])
    adj=holm(p)
    order=np.argsort(p)
    assert np.all(np.diff(adj[order])>=-1e-12)
    assert np.all(adj>=p)
    assert np.all(adj<=1)


def test_event_features_are_deterministic_and_bounded():
    sr=16000;t=np.arange(sr*3)/sr
    pulse=np.zeros_like(t)
    for start in [.3,.8,1.3,1.8,2.3]:
        pulse[(t>=start)&(t<start+.2)]=.4*np.sin(2*np.pi*180*t[(t>=start)&(t<start+.2)])
    a=_event_features(pulse,sr);b=_event_features(pulse,sr)
    assert a==b
    assert 0<=a['pause_fraction_thresholded']<=1
    assert a['candidate_energy_peaks_n']>=1


def test_connected_vad_features_are_deterministic_and_bounded():
    sr=16000;t=np.arange(sr*4)/sr
    x=np.zeros_like(t);mask=(t>.5)&(t<2.5);x[mask]=.2*np.sin(2*np.pi*180*t[mask])
    a=_vad_features(x,sr,-35);b=_vad_features(x,sr,-35)
    assert a==b
    assert 0<=a['active_speech_proportion']<=1
    assert a['cpps_status']=='deferred_no_validated_implementation'


def test_vi1_shimmer_leave_one_out_direction_survives_all_single_case_deletions():
    x=pd.read_csv(OUT/'sensitivity/vi1_shimmer_leave_one_out.csv')
    for cohort,g in x.groupby('cohort'):
        assert g.mean_diff_hc_minus_pd_loo.min()>0, cohort


def test_second_variant_outputs_keep_codes_separate():
    x=pd.read_csv(OUT/'second_variants/second_and_first_variant_signal_characterization.csv')
    assert len(x)==355
    assert set(x.task_code)=={'VA1','VA2','VE1','VE2','VI1','VI2','VO1','VO2','VU1','VU2'}
    assert x.groupby(['participant_folder','task_code']).size().max()==1


def test_d1d2_characterization_has_recording_rows_not_inferred_syllable_labels():
    x=pd.read_csv(OUT/'d1d2/recording_characterization.csv')
    assert len(x)==72
    assert set(x.task_code)=={'D1','D2'}
    assert x.interpretation.str.contains('not validated syllable').all()


def test_review_queue_is_separate_and_has_no_auto_exclusion():
    s=pd.read_csv(OUT/'review/review_status.csv').iloc[0]
    q=pd.read_csv(OUT/'review/human_review_queue.csv')
    a=pd.read_csv(OUT/'review/manual_annotations.csv')
    assert len(q)==s.queue_items
    assert s.automated_exclusions==0
    assert len(a)==0
    assert q.source.str.contains('D1_D2_protocol_review').any()


def test_connected_speech_has_three_vad_thresholds_and_no_extraction_failures():
    x=pd.read_csv(OUT/'connected_speech/vad_threshold_sensitivity_long.csv')
    assert len(x)==312
    assert set(x.vad_threshold_db_below_p95.dropna())=={-30,-35,-40}
    assert x.failure.isna().all() if 'failure' in x else True

