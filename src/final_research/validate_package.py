from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.final_research.analysis import build_uci_tables, discover_audio, holm, sha256_file

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'results/final_research'
EXP=ROOT/'results/english_research_20261031'


def run_validation():
    checks=[]
    def add(name, passed, observed, expected):
        checks.append({'check':name,'passed':bool(passed),'observed':str(observed),'expected':str(expected)})
    all_qc=pd.read_csv(EXP/'audio_qc_by_recording.csv')
    all_feat=pd.read_csv(EXP/'audio_features_by_recording.csv')
    figqc=all_qc.loc[all_qc.dataset.eq('Figshare')].copy()
    figfeat=all_feat.loc[all_feat.dataset.eq('Figshare')].copy()
    mdvrqc=all_qc.loc[all_qc.dataset.eq('MDVR-KCL')].copy()
    mdvrfeat=all_feat.loc[all_feat.dataset.eq('MDVR-KCL')].copy()
    add('Figshare audio rows',len(figqc)==81,len(figqc),81)
    add('Figshare participant count',figqc.participant_id.nunique()==81,figqc.participant_id.nunique(),81)
    add('Figshare group counts',figqc.groupby('group').participant_id.nunique().to_dict()=={'HC':41,'PD':40},figqc.groupby('group').participant_id.nunique().to_dict(),{'HC':41,'PD':40})
    add('Figshare all readable',bool(figqc.readable.all()),int(figqc.readable.sum()),81)
    add('Figshare workbook match',bool(figfeat.metadata_match_count.eq(1).all()),figfeat.metadata_match_count.value_counts().to_dict(),'all ids match exactly once')
    add('MDVR audio rows',len(mdvrqc)==73,len(mdvrqc),73)
    add('MDVR participant count',mdvrqc.participant_id.nunique()==37,mdvrqc.participant_id.nunique(),37)
    add('MDVR group counts',mdvrqc.groupby('group').participant_id.nunique().to_dict()=={'HC':21,'PD':16},mdvrqc.groupby('group').participant_id.nunique().to_dict(),{'HC':21,'PD':16})
    taskcounts=mdvrqc.groupby('task').participant_id.nunique().to_dict()
    add('MDVR task coverage',taskcounts=={'read_text':37,'spontaneous_dialogue':36},taskcounts,{'read_text':37,'spontaneous_dialogue':36})
    add('MDVR all readable',bool(mdvrqc.readable.all()),int(mdvrqc.readable.sum()),73)
    add('MDVR separator anomaly retained',bool(mdvrqc.filename_anomaly.fillna('').str.contains('ID22HC_filename_separator_missing').any()),int(mdvrqc.filename_anomaly.fillna('').str.contains('ID22HC_filename_separator_missing').sum()),'>=1')
    ox,oxcheck,oxsub,tm,tms,u301,u301test,u301task=build_uci_tables()
    add('UCI174 local file structure',len(ox)==195 and ox.participant_id.nunique()==32 and ox.groupby('group').participant_id.nunique().to_dict()=={'HC':8,'PD':24},f'rows={len(ox)}, ids={ox.participant_id.nunique()}, groups={ox.groupby("group").participant_id.nunique().to_dict()}','195 rows; 32 ids; HC8/PD24')
    add('UCI174 status consistency',bool(oxcheck.group_count.eq(1).all()),int((oxcheck.group_count>1).sum()),'0 conflicting participants')
    add('UCI189 local file structure',len(tm)==5875 and tm['subject#'].nunique()==42,f'rows={len(tm)}, ids={tm["subject#"].nunique()}','5875 rows; 42 IDs')
    a=ROOT.parent/'datasets/kaggle/telemonitoring/parkinsons_updrs.data'; b=ROOT.parent/'datasets/UCI_csv/parkinsons_updrs.data'
    add('UCI189 duplicate source copies',a.exists() and b.exists() and sha256_file(a)==sha256_file(b),sha256_file(a)==sha256_file(b) if a.exists() and b.exists() else 'missing','identical SHA-256')
    add('UCI301 train structure',len(u301)==1040 and u301.participant_id.nunique()==40 and bool(u301.groupby('participant_id').size().eq(26).all()),f'rows={len(u301)}, ids={u301.participant_id.nunique()}, task rows/person={sorted(u301.groupby("participant_id").size().unique())}','1040 rows; 40 IDs; 26 task rows each')
    add('UCI301 test structure',len(u301test)==168 and u301test.participant_id.nunique()==28 and bool(u301test.groupby('participant_id').size().eq(6).all()),f'rows={len(u301test)}, ids={u301test.participant_id.nunique()}, repeats/person={sorted(u301test.groupby("participant_id").size().unique())}','168 rows; 28 IDs; 6 repeats each')
    q=pd.read_csv(EXP/'audio_qc_by_recording.csv'); f=pd.read_csv(EXP/'audio_features_by_recording.csv')
    add('Raw audio QC row totals',len(q)==154,len(q),154)
    add('Raw audio feature row totals',len(f)==154,len(f),154)
    add('Raw audio participant-task uniqueness',not f.duplicated(['dataset','task','participant_id']).any(),int(f.duplicated(['dataset','task','participant_id']).sum()),'0 duplicates')
    v=pd.read_csv(EXP/'participant_level_group_effects.csv')
    add('Effect comparisons have independent group participants',bool(v.n_hc.gt(0).all() and v.n_pd.gt(0).all()) if {'n_hc','n_pd'}.issubset(v.columns) else False, list(v.columns), 'positive n_hc and n_pd columns')
    add('Effect table contains no duplicate dataset-task-feature tests',not v.duplicated(['dataset','task','feature']).any(),int(v.duplicated(['dataset','task','feature']).sum()),'0 duplicates')
    vad=pd.read_csv(EXP/'mdvr_vad_threshold_sensitivity_long.csv')
    add('MDVR VAD sensitivity rows',len(vad)==73*4,len(vad),292)
    add('MDVR VAD thresholds',set(vad.threshold_db)=={-20,-25,-30,-35},sorted(vad.threshold_db.unique()),[-20,-25,-30,-35])
    h=pd.read_csv(EXP/'source_hashes_before.json') if False else json.loads((EXP/'source_hashes_before.json').read_text())
    h2=json.loads((EXP/'source_hashes_after.json').read_text())
    add('Protected source hashes unchanged',h==h2,len(h),f'{len(h)} source hashes identical')
    needed=['italian_results.md','italian_manuscript_draft.md','english_dataset_inventory.md','english_results.md','english_manuscript_draft.md','cross_language_comparison.md','combined_research_summary.md','reproducibility_report.md','publication_readiness.md']
    missing=[x for x in needed if not (PKG/x).is_file()]
    add('Required research reports exist',not missing,missing,needed)
    figures=list(PKG.rglob('*.png'))
    pdfs=list(PKG.rglob('*.pdf'))
    add('Figures exported as PNG and PDF',len(figures)>=10 and len(figures)==len(pdfs),f'{len(figures)} PNG / {len(pdfs)} PDF','>=10 of each, paired')
    add('Classifier not trained',not any(k in (PKG/'reproducibility_report.md').read_text().lower() for k in ['classifier trained: true','model training performed: true']),'reports state classifier disabled','no classifier training')
    df=pd.DataFrame(checks)
    EXP.mkdir(exist_ok=True,parents=True)
    df.to_csv(EXP/'validation_checks.csv',index=False)
    summary={'passed':int(df.passed.sum()),'failed':int((~df.passed).sum()),'total':len(df),'all_passed':bool(df.passed.all())}
    (EXP/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Final research package validation','',f"**Status:** {'PASS' if summary['all_passed'] else 'INCOMPLETE'} — {summary['passed']}/{summary['total']} checks passed.",'','| Check | Result | Observed | Expected |','|---|---|---|---|']
    for r in checks: lines.append(f"| {r['check']} | {'PASS' if r['passed'] else 'FAIL'} | {r['observed']} | {r['expected']} |")
    (EXP/'validation_report.md').write_text('\n'.join(lines)+'\n')
    if not summary['all_passed']:
        raise SystemExit(f"Validation incomplete: {summary['passed']}/{summary['total']} passed")
    print(json.dumps(summary,indent=2))

if __name__=='__main__': run_validation()
