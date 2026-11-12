from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from src.cross_language_generalisation.pipeline import OUT, metrics


def run():
    checks=[]
    def add(name,ok,observed,expected):checks.append({'check':name,'passed':bool(ok),'observed':str(observed),'expected':str(expected)})
    T=OUT/'tables'
    raw=pd.read_csv(T/'harmonised_raw_audio_features.csv')
    it=raw.loc[raw.dataset.eq('ItalianPVS_VA1')].copy();fs=raw.loc[raw.dataset.eq('Figshare_sustained_a')].copy()
    add('Common raw audio feature rows',len(raw)==110,len(raw),110)
    add('Italian VA1 participants and groups',len(it)==29 and it.groupby('group').participant_id.nunique().to_dict()=={'HC':17,'PD':12},it.groupby('group').participant_id.nunique().to_dict(),'29 unique; HC 17 / PD 12')
    add('Figshare participants and groups',len(fs)==81 and fs.groupby('group').participant_id.nunique().to_dict()=={'HC':41,'PD':40},fs.groupby('group').participant_id.nunique().to_dict(),'81 unique; HC 41 / PD 40')
    add('One raw feature row per participant/task',not raw.duplicated(['dataset','task','participant_id']).any(),int(raw.duplicated(['dataset','task','participant_id']).sum()),'zero duplicates')
    add('No raw acoustic extraction failures',raw.status.eq('ok').all(),int(raw.status.ne('ok').sum()),0)
    add('Figshare workbook labels match folders',fs.metadata_label_matches_folder.fillna(False).all(),int((~fs.metadata_label_matches_folder.fillna(False)).sum()),'all labels agree')
    add('Native sample rates retained',it.sample_rate_hz.eq(16000).all() and fs.sample_rate_hz.eq(8000).all(),f"Italian={sorted(it.sample_rate_hz.unique())}, Figshare={sorted(fs.sample_rate_hz.unique())}",'Italian 16 kHz; Figshare 8 kHz')
    copy=pd.read_csv(T/'italian_va1_resampled8k_feature_table.csv')
    add('Italian 8 kHz analysis copies',len(copy)==29 and copy.sample_rate_hz.eq(8000).all() and copy.participant_id.nunique()==29,
        f'rows={len(copy)}, rates={sorted(copy.sample_rate_hz.dropna().unique())}, IDs={copy.participant_id.nunique()}','29 separate derived 8 kHz copies')
    sens=pd.read_csv(T/'feature_setting_summary.csv')
    add('Pitch-floor extraction sensitivity settings',set(sens.pitch_floor_hz)=={60,75,90},sorted(sens.pitch_floor_hz.unique()),[60,75,90])
    add('No automatic outlier exclusions',not pd.read_csv(T/'quality_outlier_flags.csv').automatic_exclusion.any(),'all outlier flags retained, no exclusions','no automatic exclusions')
    within=pd.read_csv(T/'within_dataset_classifier_results.csv');external=pd.read_csv(T/'external_classifier_results.csv')
    add('Within-dataset model count',len(within)==36,len(within),36)
    add('External and rate-sensitivity model count',len(external)==24,len(external),24)
    add('External classifier targets fixed',external.target_labels_used_for_any_fitting_or_tuning.eq(False).all(),external.target_labels_used_for_any_fitting_or_tuning.value_counts(dropna=False).to_dict(),False)
    add('External participant-key overlap',external.training_test_id_overlap_count.eq(0).all(),int(external.training_test_id_overlap_count.max()),0)
    split=pd.read_csv(T/'within_cv_split_audit.csv')
    noover=True;coverage=True
    for key,g in split.groupby(['dataset','feature_set','model']):
        seen=[]
        for r in g.itertuples():
            tr=set(json.loads(r.train_participant_ids));te=set(json.loads(r.test_participant_ids))
            noover &= not (tr&te) and int(r.participant_overlap_count)==0
            seen.extend(te)
        pp=pd.read_csv(T/'classifier_participant_predictions.csv')
        pg=pp.loc[pp.evaluation.eq('within_dataset_nested_cv')&pp.dataset.eq(key[0])&pp.feature_set.eq(key[1])&pp.model.eq(key[2])]
        coverage &= sorted(seen)==sorted(pg.participant_id.astype(str).tolist()) and len(seen)==pg.participant_id.nunique()
    add('Nested CV participant split disjointness',noover,not noover,'zero participant overlap each fold')
    add('Nested CV test folds cover each participant once',coverage,coverage,'each participant exactly once per model/feature set')
    predictions=pd.read_csv(T/'classifier_participant_predictions.csv')
    recompute_rows=[];recompute_ok=True
    for r in within.itertuples():
        g=predictions.loc[predictions.evaluation.eq('within_dataset_nested_cv')&predictions.dataset.eq(r.test_dataset)&predictions.model.eq(r.model)&predictions.feature_set.eq(r.feature_set)]
        m=metrics(g.y_true.to_numpy(int),g.y_pred.to_numpy(int),g.decision_score.to_numpy(float))
        ok=(len(g)==r.participants and all(np.isclose(m[k],getattr(r,k),equal_nan=True,rtol=1e-10,atol=1e-10) for k in ['balanced_accuracy','sensitivity','specificity','roc_auc','precision','f1']) and
            all(m[k]==getattr(r,k) for k in ['tn','fp','fn','tp']))
        recompute_ok &= ok;recompute_rows.append({'evaluation':'within','dataset':r.test_dataset,'model':r.model,'feature_set':r.feature_set,'predictions':len(g),'match':ok})
    for r in external.itertuples():
        g=predictions.loc[predictions.training_dataset.eq(r.training_dataset)&predictions.test_dataset.eq(r.test_dataset)&predictions.model.eq(r.model)&predictions.feature_set.eq(r.feature_set)]
        m=metrics(g.y_true.to_numpy(int),g.y_pred.to_numpy(int),g.decision_score.to_numpy(float))
        ok=(len(g)==r.test_participants and all(np.isclose(m[k],getattr(r,k),equal_nan=True,rtol=1e-10,atol=1e-10) for k in ['balanced_accuracy','sensitivity','specificity','roc_auc','precision','f1']) and
            all(m[k]==getattr(r,k) for k in ['tn','fp','fn','tp']))
        recompute_ok &= ok;recompute_rows.append({'evaluation':'external','training_dataset':r.training_dataset,'test_dataset':r.test_dataset,'model':r.model,'feature_set':r.feature_set,'predictions':len(g),'match':ok})
    pd.DataFrame(recompute_rows).to_csv(OUT/'tables/metric_recomputation_audit.csv',index=False)
    add('Metrics independently recomputed from saved predictions',recompute_ok,f'{sum(x["match"] for x in recompute_rows)}/{len(recompute_rows)} match','all confusion counts and six metrics match')
    # Construct explicit source/test participant lists for every zero-shot direction from the frozen tables.
    train_tables={'ItalianPVS_VA1 16k':raw.loc[raw.dataset.eq('ItalianPVS_VA1')],
      'Figshare_sustained_a 8k':raw.loc[raw.dataset.eq('Figshare_sustained_a')],
      'ItalianPVS_VA1 resampled 8k':copy}
    ext_audit=[];disjoint=True
    for r in external.itertuples():
        tr=train_tables[r.training_dataset].participant_id.astype(str).tolist()
        te=train_tables[r.test_dataset].participant_id.astype(str).tolist()
        overlap=sorted(set(tr)&set(te));disjoint &= not overlap
        ext_audit.append({'training_dataset':r.training_dataset,'test_dataset':r.test_dataset,'model':r.model,'feature_set':r.feature_set,
          'training_participant_ids':json.dumps(tr),'test_participant_ids':json.dumps(te),'training_n':len(tr),'test_n':len(te),
          'participant_key_overlap_count':len(overlap),'target_labels_used_for_fit':False})
    pd.DataFrame(ext_audit).to_csv(OUT/'tables/external_split_audit.csv',index=False)
    add('External source/test participant manifests disjoint',disjoint,not disjoint,'zero overlap by dataset-scoped participant key')
    ident=pd.read_csv(T/'dataset_identity_matched_cv.csv')
    add('Dataset identity matching and repetitions',len(ident)==90 and ident.matched_participants.eq(58).all() and ident.n_italian_hc.eq(17).all() and ident.n_italian_pd.eq(12).all(),f'{len(ident)} rows, {sorted(ident.matched_participants.unique())} each; exact diagnosis counts retained','30 draws × 3 models; n=58 matched per draw')
    identpred=pd.read_csv(T/'dataset_identity_oof_predictions.csv')
    identity_splits=[];identity_disjoint=True;identity_coverage=True
    for (rep,model),g in identpred.groupby(['repeat','model']):
        folds=sorted(g.fold.unique());seen=[]
        for fold in folds:
            testids=set(g.loc[g.fold.eq(fold),'participant_id'].astype(str))
            trainids=set(g.loc[~g.fold.eq(fold),'participant_id'].astype(str))
            identity_disjoint &= not (trainids & testids)
            seen.extend(testids)
            identity_splits.append({'repeat':rep,'model':model,'fold':fold,
              'train_participant_ids':json.dumps(sorted(trainids)),'test_participant_ids':json.dumps(sorted(testids)),
              'participant_overlap_count':len(trainids & testids)})
        identity_coverage &= len(seen)==58 and len(set(seen))==58
    pd.DataFrame(identity_splits).to_csv(OUT/'tables/dataset_identity_split_audit.csv',index=False)
    add('Dataset identity CV participant split disjointness',identity_disjoint,identity_disjoint,'zero train/test participant overlap')
    add('Dataset identity CV covers each matched participant once per draw',identity_coverage,identity_coverage,'58 unique test participants per model/draw')
    ab=pd.read_csv(T/'dataset_identity_feature_ablation.csv')
    absum=pd.read_csv(T/'dataset_identity_feature_ablation_summary.csv')
    abpred=pd.read_csv(T/'dataset_identity_feature_ablation_oof.csv')
    expected_sets={'core5_all',*[f'core5_without_{x}' for x in ['median_f0_hz','f0_iqr_hz','jitter_local','shimmer_local','hnr_mean_db']],
                   *[f'only_{x}' for x in ['median_f0_hz','f0_iqr_hz','jitter_local','shimmer_local','hnr_mean_db']]}
    add('Dataset identity feature ablation coverage',len(ab)==330 and set(absum.feature_set)==expected_sets and len(abpred)==30*11*58,
        f'{len(ab)} repeated scores, {len(absum)} feature sets, {len(abpred)} OOF predictions','30 draws × 11 pre-specified feature sets × 58 people')
    ab_splits=[];ab_disjoint=True;ab_cover=True
    for (rep,feature_set),g in abpred.groupby(['repeat','feature_set']):
        seen=[]
        for fold in sorted(g.fold.unique()):
            te=set(g.loc[g.fold.eq(fold),'participant_id'].astype(str));tr=set(g.loc[~g.fold.eq(fold),'participant_id'].astype(str))
            ab_disjoint &= not (te&tr);seen.extend(te)
            ab_splits.append({'repeat':rep,'feature_set':feature_set,'fold':fold,
              'train_participant_ids':json.dumps(sorted(tr)),'test_participant_ids':json.dumps(sorted(te)),
              'participant_overlap_count':len(te&tr)})
        ab_cover &= len(seen)==58 and len(set(seen))==58
    pd.DataFrame(ab_splits).to_csv(OUT/'tables/dataset_identity_ablation_split_audit.csv',index=False)
    add('Dataset identity ablation split disjointness',ab_disjoint,ab_disjoint,'zero train/test overlap')
    add('Dataset identity ablation covers matched participants',ab_cover,ab_cover,'58 unique test participants per draw/feature set')
    invalid=pd.read_csv(T/'invalid_cross_dataset_directions.csv')
    add('Invalid raw-audio task transfers documented',len(invalid)==4 and invalid.evaluation_status.eq('invalid_not_comparable').all(),len(invalid),4)
    adjusted=pd.read_csv(T/'age_sex_adjusted_domain_models.csv')
    add('Age/sex adjusted domain model cohort',adjusted.n_participants.eq(110).all(),sorted(adjusted.n_participants.unique()),110)
    hashes_before=json.loads((OUT/'source_hashes_before.json').read_text());hashes_after=json.loads((OUT/'source_hashes_after.json').read_text())
    add('Pre-existing source hashes unchanged',hashes_before==hashes_after,len(hashes_before),f'{len(hashes_before)} hashes identical')
    integrity=json.loads((OUT/'integrity_check.json').read_text())
    add('Integrity manifest reports no modified source',integrity.get('all_preexisting_sources_unchanged') and not integrity.get('changed_paths'),integrity.get('changed_paths'),[]) 
    figs=list((OUT/'figures').glob('*.png'));pdfs=list((OUT/'figures').glob('*.pdf'))
    add('Figure export pairs',len(figs)==6 and len(pdfs)==6,f'{len(figs)} PNG / {len(pdfs)} PDF','6 paired figures')
    master=pd.concat([within,external,invalid],ignore_index=True,sort=False)
    master['participants']=master.get('participants',pd.Series(np.nan,index=master.index)).fillna(master.get('test_participants',pd.Series(np.nan,index=master.index)))
    master.to_csv(T/'master_classifier_summary.csv',index=False)
    required=['README.md','dataset_compatibility.md','feature_harmonisation.md','statistical_comparison.md','classifier_results.md','generalisation_analysis.md','limitations_and_failed_experiments.md','experiment_decisions.md']
    missing=[x for x in required if not (OUT/x).exists()]
    add('Required experiment documents exist',not missing,missing,required)
    df=pd.DataFrame(checks);df.to_csv(OUT/'validation_checks.csv',index=False)
    summary={'passed':int(df.passed.sum()),'failed':int((~df.passed).sum()),'total':len(df),'all_passed':bool(df.passed.all())}
    (OUT/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Cross-language generalisation validation','',f"**Status:** {'PASS' if summary['all_passed'] else 'INCOMPLETE'} — {summary['passed']}/{summary['total']} checks passed.",'','| Check | Result | Observed | Expected |','|---|---|---|---|']
    for r in checks:lines.append(f"| {r['check']} | {'PASS' if r['passed'] else 'FAIL'} | {r['observed']} | {r['expected']} |")
    (OUT/'validation_report.md').write_text('\n'.join(lines)+'\n')
    if not summary['all_passed']:raise SystemExit(f"Validation incomplete: {summary['passed']}/{summary['total']} checks passed")
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
