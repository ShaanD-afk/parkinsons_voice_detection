from pathlib import Path
import json
import pandas as pd
from src.cross_language_generalisation.pipeline import CORE4,CORE5

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cross_language_generalisation_v1'


def test_harmonised_features_are_one_row_per_primary_participant():
    f=pd.read_csv(OUT/'tables/harmonised_raw_audio_features.csv')
    assert set(CORE4).issubset(f.columns) and set(CORE5).issubset(f.columns)
    assert len(f)==110
    assert not f.duplicated(['dataset','task','participant_id']).any()
    assert f.status.eq('ok').all()


def test_external_zero_shot_table_has_no_scored_invalid_task_pairs():
    ext=pd.read_csv(OUT/'tables/external_classifier_results.csv')
    invalid=pd.read_csv(OUT/'tables/invalid_cross_dataset_directions.csv')
    assert len(ext)==24 and ext.training_test_id_overlap_count.eq(0).all()
    assert len(invalid)==4 and invalid.evaluation_status.eq('invalid_not_comparable').all()
    assert invalid[['balanced_accuracy','roc_auc']].isna().all().all()


def test_nested_split_audit_has_no_participant_overlap():
    s=pd.read_csv(OUT/'tables/within_cv_split_audit.csv')
    assert s.participant_overlap_count.eq(0).all()
    for r in s.itertuples():
        train=set(json.loads(r.train_participant_ids));test=set(json.loads(r.test_participant_ids))
        assert train.isdisjoint(test)


def test_master_summary_contains_internal_external_and_invalid_statuses():
    m=pd.read_csv(OUT/'tables/master_classifier_summary.csv')
    assert {'within_dataset_nested_cv','external_zero_shot',
            'external_not_run_invalid_task_match'}.issubset(set(m.evaluation))
    assert 'external_zero_shot_rate_sensitivity' in set(m.evaluation_status)
    assert {'training_dataset','test_dataset','task_compatibility','model','features',
            'balanced_accuracy','sensitivity','specificity','roc_auc','evaluation_status'}.issubset(m.columns)


def test_cross_language_validation_passed():
    summary=json.loads((OUT/'validation_summary.json').read_text())
    assert summary['all_passed'] and summary['failed']==0
    checks=pd.read_csv(OUT/'validation_checks.csv')
    assert checks.passed.all()


def test_dataset_identity_cv_has_no_participant_leakage():
    s=pd.read_csv(OUT/'tables/dataset_identity_split_audit.csv')
    assert len(s)==30*3*5
    assert s.participant_overlap_count.eq(0).all()
    for r in s.itertuples():
        assert set(json.loads(r.train_participant_ids)).isdisjoint(json.loads(r.test_participant_ids))


def test_dataset_identity_ablation_outputs_complete_feature_sets():
    summary=pd.read_csv(OUT/'tables/dataset_identity_feature_ablation_summary.csv')
    assert len(summary)==11
    assert 'core5_all' in set(summary.feature_set)
    assert (OUT/'figures/fig6_dataset_identity_feature_ablation.png').exists()
    assert (OUT/'figures/fig6_dataset_identity_feature_ablation.pdf').exists()
