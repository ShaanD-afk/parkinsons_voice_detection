from __future__ import annotations
import json, os, platform, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.final_research.analysis import build_uci_tables
from src.cross_language_generalisation.pipeline import *


def save_csv(df,name):
    p=OUT/'tables'/name;p.parent.mkdir(exist_ok=True,parents=True);df.to_csv(p,index=False);return p


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    logdir=OUT/'logs';logdir.mkdir(exist_ok=True,parents=True)
    start=time.time()
    try:
        gs=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True,check=False).stdout
        lines=[x for x in gs.splitlines() if 'cross_language_generalisation_v1' not in x and 'src/cross_language_generalisation/' not in x]
        (logdir/'git_status_before.txt').write_text('\n'.join(lines)+'\n')
    except Exception as e:(logdir/'git_status_before.txt').write_text(f'git status unavailable: {e}\n')
    before=capture_hashes();(OUT/'source_hashes_before.json').write_text(json.dumps(before,indent=2,sort_keys=True))
    print(f'Hashed {len(before)} source files',flush=True)
    # Common, identical feature code is rerun against raw Italian VA1 and Figshare /a/ audio.
    raw,qc=load_raw_vowel_cohorts()
    print(f'Primary raw task extraction complete: {len(raw)} rows',flush=True)
    sensitivity=[]
    for floor in [60.,90.]:
        subset=[]
        for r in raw.loc[raw.pitch_floor_hz.eq(75)].itertuples(index=False):
            path=(ITALIAN_AUDIO/r.source_relative_path) if r.dataset=='ItalianPVS_VA1' else (FIGSHARE/r.source_relative_path)
            x=stable_features(path,r.dataset,r.task,r.group,r.participant_id,r.source_relative_path,
                              floor,float(r.age_years) if pd.notna(r.age_years) else np.nan,r.sex)
            x['rate_origin']=r.rate_origin;subset.append(x)
        sensitivity.extend(subset)
        print(f'Feature setting floor {floor:g} Hz complete',flush=True)
    sens_df=pd.DataFrame(sensitivity)
    full_settings=pd.concat([raw,sens_df],ignore_index=True,sort=False)
    setrows=[]
    for (d,f),g in full_settings.groupby(['dataset','pitch_floor_hz']):
        b=g.loc[g.status.eq('ok')]
        for feat in CORE5:
            x=b[feat].dropna()
            setrows.append({'dataset':d,'pitch_floor_hz':f,'feature':feat,'n_rows':len(g),'n_valid':len(x),
                            'failure_count':int(g.status.ne('ok').sum()),'median':x.median(),'q1':x.quantile(.25),'q3':x.quantile(.75)})
    setting_summary=pd.DataFrame(setrows)
    setting_compare=[]
    for d,g in full_settings.groupby('dataset'):
        base=g.loc[g.pitch_floor_hz.eq(75)].set_index('participant_id')
        for floor in [60.,90.]:
            alt=g.loc[g.pitch_floor_hz.eq(floor)].set_index('participant_id')
            for feat in CORE5:
                z=base[[feat]].join(alt[[feat]].rename(columns={feat:'alt'})).dropna()
                if not len(z):continue
                setting_compare.append({'dataset':d,'pitch_floor_hz':floor,'feature':feat,'n_paired':len(z),
                 'median_abs_change':(z[feat]-z.alt).abs().median(),'median_relative_abs_change':((z[feat]-z.alt).abs()/z[feat].abs().clip(lower=1e-10)).median(),
                 'spearman_base_vs_alt':pd.Series(z[feat]).corr(pd.Series(z.alt),method='spearman')})
    save_csv(setting_summary,'feature_setting_summary.csv');save_csv(pd.DataFrame(setting_compare),'feature_setting_sensitivity.csv')
    raw75=raw.loc[raw.pitch_floor_hz.eq(75)].copy()
    it8,sr_sens=make_resampled_italian(raw75)
    save_csv(it8,'italian_va1_resampled8k_feature_table.csv');save_csv(sr_sens,'italian_resampling_sensitivity.csv')
    save_csv(raw75,'harmonised_raw_audio_features.csv');save_csv(full_settings,'all_pitch_floor_extractions.csv');save_csv(qc,'raw_audio_qc.csv')
    # Quality and plausibility flags do not automatically remove participants.
    plaus=[]
    bounds={'median_f0_hz':(50,600),'f0_iqr_hz':(0,300),'jitter_local':(0,.2),'shimmer_local':(0,.5),'hnr_mean_db':(-20,60)}
    for _,r in raw75.iterrows():
        for f,(lo,hi) in bounds.items():
            v=r.get(f,np.nan);plaus.append({'dataset':r.dataset,'participant_id':r.participant_id,'feature':f,
              'value':v,'implausible_range_flag':bool(pd.notna(v) and not (lo<=v<=hi)),
              'robust_iqr_outlier_flag':False,'automatic_exclusion':False})
    outlier=pd.DataFrame(plaus)
    for (d,f),g in outlier.groupby(['dataset','feature']):
        vals=pd.to_numeric(g.value,errors='coerce').dropna();q1,q3=vals.quantile([.25,.75]);iqr=q3-q1
        if iqr>0:
            mask=(pd.to_numeric(g.value,errors='coerce')<q1-3*iqr)|(pd.to_numeric(g.value,errors='coerce')>q3+3*iqr)
            outlier.loc[g.index,'robust_iqr_outlier_flag']=mask.fillna(False).to_numpy()
    save_csv(outlier,'quality_outlier_flags.csv')
    # Participant-level group effects, corrected within each dataset/task family.
    stat_frames=[]
    for d,g in raw75.loc[raw75.status.eq('ok')].groupby('dataset'):
        stat_frames.append(effect_table(g,CORE5,d,'sustained_a'))
    # Separate candidate and precomputed datasets; never cross-merge their feature definitions.
    prev=pd.read_csv(ROOT/'results/final_research/english_tables/audio_features_by_recording.csv')
    rd=prev.loc[prev.dataset.eq('MDVR-KCL')&prev.task.eq('read_text')&prev.status.eq('ok')].copy()
    if len(rd):stat_frames.append(effect_table(rd,READ_FEATURES,'MDVR-KCL','read_text'))
    ox,oxcheck,oxsub,tm,tms,u301,u301test,u301task=build_uci_tables()
    stat_frames.append(effect_table(oxsub,UCI174_FEATURES,'UCI174_feature_table','sustained_vowel_a'))
    for vowel in ['a','o','u']:
        v=u301.loc[u301.task.eq(vowel)&u301.sample_order.eq({'a':1,'o':2,'u':3}[vowel])].copy()
        stat_frames.append(effect_table(v,UCI301_FEATURES,'UCI301_feature_table',f'sustained_{vowel}'))
    stats_table=pd.concat(stat_frames,ignore_index=True,sort=False);save_csv(stats_table,'participant_level_group_effects.csv')
    adjusted=adjusted_domain_models(raw75.loc[raw75.status.eq('ok')],CORE5);save_csv(adjusted,'age_sex_adjusted_domain_models.csv')
    # Feature distributions, missingness, and balanced group/task counts.
    rows=[]
    datasets=[('ItalianPVS_VA1',raw75.loc[raw75.dataset.eq('ItalianPVS_VA1')&raw75.status.eq('ok')],CORE5,'VA1'),
      ('Figshare_sustained_a',raw75.loc[raw75.dataset.eq('Figshare_sustained_a')&raw75.status.eq('ok')],CORE5,'sustained_a'),
      ('MDVR-KCL_read_text',rd,READ_FEATURES,'read_text'),
      ('UCI174',oxsub,UCI174_FEATURES,'sustained_vowel_a')]
    for vowel in ['a','o','u']:
        datasets.append((f'UCI301_{vowel}',u301.loc[u301.task.eq(vowel)&u301.sample_order.eq({'a':1,'o':2,'u':3}[vowel])],UCI301_FEATURES,f'sustained_{vowel}'))
    for name,g,features,task in datasets:
        for group,sub in g.groupby('group'):
            for f in features:
                v=pd.to_numeric(sub[f],errors='coerce').dropna()
                rows.append({'dataset_task':name,'task':task,'group':group,'feature':f,
                 'n_participants':sub.participant_id.nunique(),'n_nonmissing':len(v),'missing_n':len(sub)-len(v),
                 'mean':v.mean(),'sd':v.std(ddof=1),'median':v.median(),'q1':v.quantile(.25),'q3':v.quantile(.75),
                 'minimum':v.min(),'maximum':v.max()})
    dist=pd.DataFrame(rows);save_csv(dist,'feature_distributions_and_missingness.csv')
    # Dataset compatibility matrix; external classifier candidates are restricted to sustained /a/.
    compatibility=pd.DataFrame([
      ['ItalianPVS VA1','Italian confirmed at dataset level','sustained vowel; FILE CODES wording: “2 phonation of the vocal a”','/a/ target; exact duration/repetition unspecified','16 kHz mono primary cohort','dataset recording chain; demographics available','17 HC / 12 PD','one included VA1 recording/person','same raw feature code applied here','Comparable with explicit limitations','Eligible for zero-shot cross-dataset; not proof of cross-language transfer'],
      ['Figshare','language not established; do not label English','prolonged /a/ by telephone per source item','/a/ target','8 kHz mono FLOAT; telephone','telephone recordings; age/sex workbook available','41 HC / 40 PD','one waveform/person in local copy','same raw feature code applied here; 8 kHz limits bandwidth','Comparable with explicit limitations','Eligible for zero-shot cross-dataset; domain, device, rate and possibly language differ'],
      ['MDVR-KCL ReadText','English source task','connected reading text','lexical text; not sustained /a/','44.1 kHz mono PCM24 original; candidate features use 16k copy','reading setup differs; age/sex not in local files','21 HC / 16 PD','one local read recording/person','candidate VAD/prosody; human mask review absent','Not suitable for direct cross-dataset test against sustained vowels','Within-dataset classifier exploratory only'],
      ['MDVR-KCL SpontaneousDialogue','English source task','spontaneous dialogue; examiner may appear','connected spontaneous speech','44.1 kHz mono PCM24 original','phone dialogue; speaker attribution unresolved','21 HC / 15 PD','one local dialogue/person','candidate speech activity/prosody; unvalidated','Not suitable for direct cross-dataset test against sustained vowels','No classifier used: speaker attribution/VAD validity unresolved'],
      ['UCI #174','language metadata not used for cross-language claim','precomputed sustained voice features','prompt not available in local table','no local raw audio','covariates limited in local feature table','8 HC / 24 PD','195 rows / 32 coded participants; repeats collapsed to participant medians','historical MDVP features; not re-extracted','Not suitable for raw-audio external validation','Separate within-dataset exploratory classification'],
      ['UCI #301 train','language not established from local source metadata','precomputed vowels/numbers/sentences/words','task rows grouped per source order','no local raw audio','covariates limited','20 HC / 20 PD','26 task rows/person; classifier uses one vowel row/person/task','precomputed extractor; HTN not same as HNR','Not suitable for raw-audio external validation','Separate within-dataset exploratory classification'],
      ['UCI #189','language not needed for current scope','longitudinal precomputed voice measures','home voice records','no local raw audio','42 PD; no HC','0 HC / 42 PD','5,875 repeated rows; subject-level only','precomputed table','Not suitable for binary group classifier','No HC comparison']
    ],columns=['dataset','language_certainty','task_and_phonation','phonetic_content','sample_rate_and_bandwidth','device_environment_and_covariates','participants_by_class','recordings_per_participant','feature_compatibility','comparison_classification','decision'])
    save_csv(compatibility,'dataset_compatibility_matrix.csv')
    # Prepare all classifier cohorts at participant-level. This includes one explicitly provisional MDVR reading baseline.
    fig=raw75.loc[raw75.dataset.eq('Figshare_sustained_a')&raw75.status.eq('ok')].copy()
    it=raw75.loc[raw75.dataset.eq('ItalianPVS_VA1')&raw75.status.eq('ok')].copy()
    it8ok=it8.loc[it8.status.eq('ok')].copy()
    md=rd.copy()
    mdl=[]
    for name,g,features,task,kind in [
       ('ItalianPVS_VA1',it,CORE5,'VA1','raw_audio_sustained'),
       ('Figshare_sustained_a',fig,CORE5,'sustained_a','raw_audio_sustained'),
       ('MDVR-KCL_read_text',md,READ_FEATURES,'read_text','candidate_connected_speech'),
       ('UCI174_feature_table',oxsub,UCI174_FEATURES,'sustained_vowel_a','precomputed_feature_table')]:mdl.append((name,g,features,task,kind))
    for vowel in ['a','o','u']:
        v=u301.loc[u301.task.eq(vowel)&u301.sample_order.eq({'a':1,'o':2,'u':3}[vowel])].copy()
        mdl.append((f'UCI301_{vowel}_feature_table',v,UCI301_FEATURES,f'sustained_{vowel}','precomputed_feature_table'))
    registry=model_registry();within=[];split_frames=[];pred_frames=[];model_log=[]
    for d,g,features,task,kind in mdl:
        featuresets=[('core4',CORE4)] if kind=='raw_audio_sustained' else [('task_specific',features)]
        if kind=='raw_audio_sustained':featuresets.append(('core5',CORE5))
        for fsname,fs in featuresets:
            for modelname,(est,grid) in registry.items():
                print(f'Nested CV: {d} {fsname} {modelname}',flush=True)
                row,splits,preds=evaluate_nested(g,fs,modelname,est,grid,SEED+len(within))
                row.update({'training_dataset':d,'test_dataset':d,'task_compatibility':'within_dataset','feature_set':fsname,
                            'data_type':kind,'evaluation_status':'exploratory_internal_validation'})
                within.append(row);splits['dataset']=d;splits['feature_set']=fsname;split_frames.append(splits)
                preds['dataset']=d;preds['evaluation']='within_dataset_nested_cv';preds['feature_set']=fsname;pred_frames.append(preds)
                model_log.append({'dataset':d,'task':task,'features':fs,'model':modelname,'grid':grid,'feature_set':fsname})
    # True external tests: harmonized sustained /a/ only. Sensitivity uses the Italian analysis copy at 8 kHz.
    cross=[];external_predictions=[];cross_specs=[
      ('ItalianPVS_VA1 16k',it,'Figshare_sustained_a 8k',fig,'core5',CORE5),
      ('Figshare_sustained_a 8k',fig,'ItalianPVS_VA1 16k',it,'core5',CORE5),
      ('ItalianPVS_VA1 resampled 8k',it8ok,'Figshare_sustained_a 8k',fig,'core5',CORE5),
      ('Figshare_sustained_a 8k',fig,'ItalianPVS_VA1 resampled 8k',it8ok,'core5',CORE5),
    ]
    # Compare 4-feature and 5-feature set on primary-rate transfers; rate sensitivity is core5 only.
    for spec_i,(srcname,trgname) in enumerate([('ItalianPVS_VA1 16k','Figshare_sustained_a 8k'),('Figshare_sustained_a 8k','ItalianPVS_VA1 16k')]):
        for fsname,features in [('core4',CORE4),('core5',CORE5)]:
            tr=it if srcname.startswith('Italian') else fig;te=fig if trgname.startswith('Figshare') else it
            for modelname,(est,grid) in registry.items():
                print(f'Zero-shot: {srcname} -> {trgname} {fsname} {modelname}',flush=True)
                row,preds=evaluate_external(tr,te,features,modelname,est,grid,srcname,trgname,SEED+100+len(cross))
                row.update({'task_compatibility':'matched sustained /a/ with rate/device/cohort differences','feature_set':fsname,
                  'data_type':'raw_audio','evaluation_status':'external_zero_shot','target_labels_used_for_any_fitting_or_tuning':False})
                cross.append(row);preds['feature_set']=fsname;external_predictions.append(preds)
    for i,(srcname,tr,tarname,te,fsname,features) in enumerate(cross_specs[2:],start=0):
        for modelname,(est,grid) in registry.items():
            print(f'Rate sensitivity zero-shot: {srcname} -> {tarname} {modelname}',flush=True)
            row,preds=evaluate_external(tr,te,features,modelname,est,grid,srcname,tarname,SEED+300+len(cross))
            row.update({'task_compatibility':'matched sustained /a/; Italian signal resampled to 8 kHz sensitivity','feature_set':fsname,
              'data_type':'raw_audio','evaluation_status':'external_zero_shot_rate_sensitivity','target_labels_used_for_any_fitting_or_tuning':False})
            cross.append(row);preds['feature_set']=fsname;external_predictions.append(preds)
    invalid=[]
    possible=[('ItalianPVS_VA1','MDVR-KCL_read_text'),('MDVR-KCL_read_text','ItalianPVS_VA1'),
              ('Figshare_sustained_a','MDVR-KCL_read_text'),('MDVR-KCL_read_text','Figshare_sustained_a')]
    counts={d:(len(g),int(g.group.eq('HC').sum()),int(g.group.eq('PD').sum())) for d,g,_,_,_ in mdl}
    for a,b in possible:
        invalid.append({'evaluation':'external_not_run_invalid_task_match','training_dataset':a,'test_dataset':b,'task_compatibility':'sustained phonation vs connected reading','model':'not_run','features':'not comparable','training_participants':counts.get(a,(np.nan,)*3)[0],
          'training_n_hc':counts.get(a,(np.nan,)*3)[1],'training_n_pd':counts.get(a,(np.nan,)*3)[2],
          'test_participants':counts.get(b,(np.nan,)*3)[0],'test_n_hc':counts.get(b,(np.nan,)*3)[1],'test_n_pd':counts.get(b,(np.nan,)*3)[2],
          'balanced_accuracy':np.nan,'sensitivity':np.nan,'specificity':np.nan,'roc_auc':np.nan,'evaluation_status':'invalid_not_comparable'})
    # Dataset identity classifier on 30 diagnosis-matched samples. This estimates domain detectability, not language.
    identity,identity_preds=dataset_identity_experiment(raw75);save_csv(identity,'dataset_identity_matched_cv.csv');save_csv(identity_preds,'dataset_identity_oof_predictions.csv')
    # Aggregate classifier tables and predictions.
    within_df=pd.DataFrame(within);cross_df=pd.DataFrame(cross);invalid_df=pd.DataFrame(invalid)
    master=pd.concat([within_df,cross_df,invalid_df],ignore_index=True,sort=False)
    save_csv(within_df,'within_dataset_classifier_results.csv');save_csv(cross_df,'external_classifier_results.csv')
    save_csv(pd.concat([within_df,cross_df],ignore_index=True,sort=False),'master_classifier_summary.csv')
    save_csv(invalid_df,'invalid_cross_dataset_directions.csv')
    save_csv(pd.concat(pred_frames+external_predictions,ignore_index=True,sort=False),'classifier_participant_predictions.csv')
    save_csv(pd.concat(split_frames,ignore_index=True,sort=False),'within_cv_split_audit.csv')
    save_csv(pd.DataFrame(model_log),'model_search_configuration.csv')
    # Dataset origin model adjusted for diagnosis-count imbalance via exact within-diagnosis matching.
    # plots use raw row-level points but never train/evaluate a disease classifier on them.
    plot_all(raw75,stats_table,master.loc[master.evaluation.isin(['within_dataset_nested_cv','external_zero_shot','external_zero_shot_rate_sensitivity'])].copy(),identity,OUT/'figures')
    # Config, explicit run record, outputs and hash integrity.
    config={'experiment_id':'CROSS-LANGUAGE-GENERALISATION-V1-20261031','seed':SEED,'timestamp_local':'2026-10-31 Asia/Kolkata',
      'scope':'exploratory cross-dataset evaluation; Figshare language not established','primary_task':'ItalianPVS VA1 sustained /a/ vs Figshare prolonged /a/',
      'core_features':{'core4':CORE4,'core5':CORE5},'feature_settings_pitch_floors_hz':[60,75,90],
      'sample_rate_sensitivity':'Italian native 16 kHz vs stored polyphase 8 kHz analysis copies; Figshare native 8 kHz retained',
      'models':{k:{'estimator':str(v[0]),'grid':v[1]} for k,v in registry.items()},
      'nested_cv':{'outer':'StratifiedKFold 5; inner StratifiedKFold 3; grid search scoring balanced accuracy; no feature selection; model selection inside training folds'},
      'external_evaluation':'source-only GridSearchCV (up to 5 folds) then refit on all source; target cohort never used for tuning, scaling, imputation, selection or threshold setting; default decision threshold',
      'class_imbalance':'class_weight=balanced for LR/SVM/RF; participant-level class-stratified bootstrap CI conditional on held-out predictions',
      'dataset_classifier':'30 matched subsamples with Figshare sampled within HC/PD to Italian counts; 5-fold OOF, no participant overlap',
      'no_gradient_boosting_reason':'smallest cohort n=29 with only five pre-specified acoustic features; classical LR/SVM/RF favored; avoid unstable additional model family',
      'no_deep_learning':True,'no_external_adaptation':True,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
      'scipy':scipy.__version__,'sklearn':sklearn.__version__,'soundfile':sf.__version__,'parselmouth':parselmouth.__version__,
      'git_status_before_file':'logs/git_status_before.txt','source_file_count_before':len(before)}
    (OUT/'configs/experiment_config.json').write_text(json.dumps(config,indent=2,sort_keys=True,default=str))
    (logdir/'commands.txt').write_text('MPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.cross_language_generalisation.run_experiment\nMPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.cross_language_generalisation.validate_experiment\nMPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/ -q\n')
    after=capture_hashes();(OUT/'source_hashes_after.json').write_text(json.dumps(after,indent=2,sort_keys=True))
    changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
    integrity={'source_hash_count_before':len(before),'source_hash_count_after':len(after),'all_preexisting_sources_unchanged':not changed,'changed_paths':changed}
    (OUT/'integrity_check.json').write_text(json.dumps(integrity,indent=2))
    # Machine-readable source summary with main cohort sizes and failures.
    summary={'raw_audio_feature_rows':len(raw75),'raw_audio_extraction_failures':int(raw75.status.ne('ok').sum()),
      'italian_primary':raw75.loc[raw75.dataset.eq('ItalianPVS_VA1')].group.value_counts().to_dict(),
      'figshare':raw75.loc[raw75.dataset.eq('Figshare_sustained_a')].group.value_counts().to_dict(),
      'within_models':len(within_df),'external_models':len(cross_df),'invalid_cross_directions':len(invalid_df),
      'dataset_identity_models':identity.model.nunique(),'hashes_unchanged':not changed,'elapsed_seconds':round(time.time()-start,1)}
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
