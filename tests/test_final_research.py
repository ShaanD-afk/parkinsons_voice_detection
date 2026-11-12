from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.final_research.analysis import compare_groups, holm, load_uci301

ROOT=Path(__file__).resolve().parents[1]


def test_holm_adjustment_known_values_and_monotonicity():
    p=np.array([.01,.04,.03])
    adjusted=holm(p)
    assert np.allclose(adjusted,[.03,.06,.06])
    assert np.all((adjusted>=p)&(adjusted<=1))


def test_group_comparison_rejects_duplicate_participant_rows():
    data=pd.DataFrame({'participant_id':['a','a','b','c'],'group':['HC','HC','PD','PD'],'x':[1,2,3,4]})
    with pytest.raises(ValueError,match='duplicate'):
        compare_groups(data,['x'],dataset='test',task='test',seed=7)


def test_group_comparison_uses_one_row_per_participant():
    data=pd.DataFrame({'participant_id':['h1','h2','p1','p2'],'group':['HC','HC','PD','PD'],'x':[1,2,3,4]})
    out=compare_groups(data,['x'],dataset='test',task='test',seed=7)
    assert len(out)==1 and out[0]['n_hc']==2 and out[0]['n_pd']==2
    assert out[0]['participant_unit_verified'] is True


def test_uci301_local_archive_partition_structure():
    train,test=load_uci301()
    assert train.shape[0]==1040 and train.participant_id.nunique()==40
    assert train.groupby('participant_id').size().eq(26).all()
    assert train.groupby('participant_id').group.nunique().eq(1).all()
    assert test.shape[0]==168 and test.participant_id.nunique()==28
    assert test.groupby('participant_id').size().eq(6).all()
    assert test.group.eq('PD').all()


def test_final_package_validation_has_no_unresolved_checks():
    p=ROOT/'results/english_research_20261031/validation_summary.json'
    assert p.exists(), 'run `python -m src.final_research.validate_package` first'
    summary=json.loads(p.read_text())
    assert summary['all_passed'] and summary['failed']==0 and summary['total']==28
    checks=pd.read_csv(p.parent/'validation_checks.csv')
    assert checks.passed.all()
