from __future__ import annotations
import hashlib, json, platform, sys, warnings
from pathlib import Path
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
import scipy
from scipy import stats
from scipy.signal import resample_poly
import sklearn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.final_research.analysis import (DATASETS, FIGSHARE, audio_qc, sha256_file)
from src.italian_multitask.audio import mono_for_analysis
from src.italian_multitask.segmentation import voiced_runs

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cross_language_generalisation_v1'
ITALIAN_AUDIO=DATASETS/'italian'
SEED=20261031
CORE4=['median_f0_hz','jitter_local','shimmer_local','hnr_mean_db']
CORE5=CORE4[:1]+['f0_iqr_hz']+CORE4[1:]
READ_FEATURES=['active_speech_fraction','pause_count_candidate','pause_fraction_candidate',
 'median_f0_active_hz','f0_iqr_active_hz','voiced_fraction_active','hnr_median_active_db',
 'relative_intensity_iqr_db','spectral_centroid_active_hz','spectral_rolloff_85_active_hz']
UCI174_FEATURES=['median_f0_hz','jitter_local','shimmer_local','hnr_mean_db']
UCI301_FEATURES=['jitter_local','shimmer_local','htn','median_pitch_hz','unvoiced_fraction']


def participant_key(dataset, raw_id):
    return f'{dataset}_{hashlib.sha256((dataset+"|"+str(raw_id)).encode()).hexdigest()[:10]}'


def stable_features(path:Path, dataset:str, task:str, group:str, pid:str, relative:str,
                    floor:float=75.0, age=np.nan, sex='') -> dict:
    r={'dataset':dataset,'task':task,'group':group,'participant_id':pid,'source_file':Path(relative).name,
       'source_relative_path':str(relative),'sample_rate_hz':np.nan,'pitch_floor_hz':floor,
       'age_years':age,'sex':sex,'status':'failed','failure_reason':''}
    try:
        x,sr=sf.read(path,dtype='float64',always_2d=True)
        mono,channel_method=mono_for_analysis(x)
        snd0=parselmouth.Sound(mono,sampling_frequency=int(sr))
        pitch0=snd0.to_pitch_ac(time_step=.01,pitch_floor=floor,pitch_ceiling=500)
        f0_0=pitch0.selected_array['frequency']; voiced0=np.isfinite(f0_0)&(f0_0>0)
        runs=voiced_runs(voiced0)
        if not runs: raise ValueError('no voiced interval at requested pitch floor')
        a,b=max(runs,key=lambda ab:ab[1]-ab[0]); trim=int(round((b-a)*.10)); a+=trim;b-=trim
        if b<=a: raise ValueError('voiced run empty after 10% edge trim')
        times=pitch0.xs(); t0=float(times[a]);t1=float(times[max(a,b-1)])
        segment=mono[max(0,int(t0*sr)):min(len(mono),int(t1*sr))]
        if len(segment)<int(.25*sr): raise ValueError('stable segment shorter than 250 ms')
        snd=parselmouth.Sound(segment,sampling_frequency=int(sr))
        pitch=snd.to_pitch_ac(time_step=.01,pitch_floor=floor,pitch_ceiling=500)
        f0=pitch.selected_array['frequency'];v=np.isfinite(f0)&(f0>0)
        pp=parselmouth.praat.call(snd,'To PointProcess (periodic, cc)',floor,500)
        jitter=float(parselmouth.praat.call(pp,'Get jitter (local)',0,0,.0001,.02,1.3))
        shimmer=float(parselmouth.praat.call([snd,pp],'Get shimmer (local)',0,0,.0001,.02,1.3,1.6))
        hn=snd.to_harmonicity_cc(time_step=.01,minimum_pitch=floor,silence_threshold=.1,periods_per_window=1.0).values[0]
        hn=hn[np.isfinite(hn)&(hn>-100)]
        r.update({'status':'ok','failure_reason':'','sample_rate_hz':int(sr),'channel_method':channel_method,
          'stable_duration_s':(b-a)*.01,'median_f0_hz':float(np.median(f0[v])) if v.any() else np.nan,
          'f0_iqr_hz':float(np.percentile(f0[v],75)-np.percentile(f0[v],25)) if v.any() else np.nan,
          'voiced_fraction':float(v.mean()) if len(v) else np.nan,'jitter_local':jitter,
          'shimmer_local':shimmer,'hnr_mean_db':float(np.mean(hn)) if len(hn) else np.nan})
    except Exception as e:
        r.update({'failure_reason':f'{type(e).__name__}: {e}'})
    return r


def source_paths():
    paths=[]
    for p in [ROOT/'data/va1_features_raw.csv',ROOT/'data/baseline_oof_predictions.csv',
              ROOT/'data/italian_manifest.csv',ROOT/'data/primary_cohort_manifest.csv',
              ROOT/'data/primary_cohort_participants.csv',ROOT/'src/train_baseline.py',
              ROOT/'src/final_research/analysis.py',ROOT/'src/italian_multitask/segmentation.py',
              ROOT/'src/italian_multitask/features/vowels.py',FIGSHARE/'Demographics_age_sex.xlsx',
              DATASETS/'kaggle/parkinsons.data',DATASETS/'UCI_csv/parkinsons_updrs.data',
              DATASETS/'kaggle/telemonitoring/parkinsons_updrs.data',DATASETS/'uci_csv2.rar']:
        if p.exists(): paths.append(p)
    for folder in [ROOT/'results/italian_multitask_v1',ROOT/'results/italian_research_phase2',
                   ROOT/'results/final_research',ROOT/'results/english_research_20261031',
                   ROOT/'results/va1_oof_uncertainty_20261031',ROOT/'reports/italian_task_audit',
                   ROOT/'data']:
        if folder.exists(): paths += [p for p in folder.rglob('*') if p.is_file()]
    for folder in [FIGSHARE,ITALIAN_AUDIO,DATASETS/'MDVR-KCL']:
        if folder.exists(): paths += [p for p in folder.rglob('*.wav') if p.is_file()]
    paths += [p for p in ITALIAN_AUDIO.rglob('*.xlsx') if p.is_file()]
    return sorted(set(p for p in paths if p.is_file()))


def capture_hashes():
    return {str(p):sha256_file(p) for p in source_paths()}


def load_raw_vowel_cohorts():
    manifest=pd.read_csv(ROOT/'data/primary_cohort_manifest.csv')
    it=manifest.loc[manifest.possible_task_code.eq('VA1')&manifest.include_primary_recording.astype(bool)].copy()
    rows=[];qcs=[]
    for row in it.itertuples(index=False):
        pid=participant_key('ItalianPVS_VA1',row.participant_folder)
        path=ITALIAN_AUDIO/row.relative_path
        qc=audio_qc(path,'ItalianPVS','VA1',row.cohort_group,pid,row.relative_path)
        qcs.append(qc)
        r=stable_features(path,'ItalianPVS_VA1','VA1',row.cohort_group.replace('ELDERLY_HC','HC'),pid,
                          row.relative_path,75,float(row.metadata_age) if pd.notna(row.metadata_age) else np.nan,
                          str(row.metadata_sex) if pd.notna(row.metadata_sex) else '')
        r['task_definition_source']='FILE CODES: 2 phonation of the vocal “a”; duration/repetition protocol not specified'
        r['rate_origin']='native';rows.append(r)
    meta=pd.read_excel(FIGSHARE/'Demographics_age_sex.xlsx',sheet_name='Parselmouth')
    meta=meta.rename(columns={'Sample ID':'sample_id','Label':'group','Age':'age_years','Sex':'sex'})
    meta['sample_id']=meta.sample_id.astype(str);meta['group']=meta.group.replace({'PwPD':'PD','HC':'HC'})
    files=sorted(FIGSHARE.glob('*_AH/*.wav'))
    for path in files:
        sample=path.stem; m=meta.loc[meta.sample_id.eq(sample)]
        group='HC' if path.parent.name=='HC_AH' else 'PD'
        pid=participant_key('Figshare_sustained_a',sample)
        relative=path.relative_to(FIGSHARE).as_posix()
        age=float(m.iloc[0].age_years) if len(m)==1 and pd.notna(m.iloc[0].age_years) else np.nan
        sex=str(m.iloc[0].sex) if len(m)==1 else ''
        qc=audio_qc(path,'Figshare','sustained_a',group,pid,relative)
        metadata_group=str(m.iloc[0].group) if len(m)==1 else ''
        qc['metadata_match_count']=len(m);qc['metadata_group']=metadata_group
        qc['metadata_label_matches_folder']=bool(metadata_group==group) if metadata_group else False
        qcs.append(qc)
        r=stable_features(path,'Figshare_sustained_a','sustained_a',group,pid,relative,75,age,sex)
        r['metadata_group']=metadata_group;r['metadata_match_count']=len(m);r['metadata_label_matches_folder']=bool(metadata_group==group) if metadata_group else False
        r['task_definition_source']='Source item: prolonged /a/ phonation by telephone; language not established'
        r['rate_origin']='native';rows.append(r)
    return pd.DataFrame(rows),pd.DataFrame(qcs)


def make_resampled_italian(it_features):
    mani=pd.read_csv(ROOT/'data/primary_cohort_manifest.csv')
    mani=mani.loc[mani.possible_task_code.eq('VA1')&mani.include_primary_recording.astype(bool)].copy()
    out=[]; qrows=[]
    dest=OUT/'analysis_copies/italian_va1_8khz'
    for row in mani.itertuples(index=False):
        pid=participant_key('ItalianPVS_VA1',row.participant_folder); src=ITALIAN_AUDIO/row.relative_path
        x,sr=sf.read(src,dtype='float64',always_2d=True);mono,_=mono_for_analysis(x)
        if int(sr)!=16000: raise ValueError(f'Expected frozen primary VA1 16 kHz, found {sr}: {row.relative_path}')
        down=resample_poly(mono,1,2)
        destpath=dest/f'{pid}.wav';sf.write(destpath,down,8000,subtype='FLOAT')
        rr=stable_features(destpath,'ItalianPVS_VA1','VA1',row.cohort_group.replace('ELDERLY_HC','HC'),pid,
                           row.relative_path,75,float(row.metadata_age) if pd.notna(row.metadata_age) else np.nan,
                           str(row.metadata_sex) if pd.notna(row.metadata_sex) else '')
        rr.update({'rate_origin':'polyphase_16k_to_8k_analysis_copy','analysis_copy':str(destpath.relative_to(OUT))})
        out.append(rr)
    out=pd.DataFrame(out)
    base=it_features.loc[it_features.dataset.eq('ItalianPVS_VA1')&it_features.pitch_floor_hz.eq(75)].set_index('participant_id')
    low=out.set_index('participant_id');sens=[]
    for feat in CORE5:
        z=base[[feat,'group']].join(low[[feat]].rename(columns={feat:'resampled_8k'}),how='inner').dropna()
        diff=z[feat]-z.resampled_8k
        rng=np.random.default_rng(SEED+len(sens));boots=np.array([np.median(rng.choice(diff.to_numpy(),len(diff),replace=True)) for _ in range(5000)])
        try:p=float(stats.wilcoxon(z[feat],z.resampled_8k).pvalue)
        except Exception:p=np.nan
        sens.append({'feature':feat,'n_paired_participants':len(z),'native16_median':z[feat].median(),
          'resampled8_median':z.resampled_8k.median(),'median_native_minus_resampled':diff.median(),
          'median_diff_boot_ci95_low':np.percentile(boots,2.5),'median_diff_boot_ci95_high':np.percentile(boots,97.5),
          'spearman_native_resampled':stats.spearmanr(z[feat],z.resampled_8k).statistic,'wilcoxon_p':p})
    sens=pd.DataFrame(sens);sens['holm_p_5_features']=holm_adjust(sens.wilcoxon_p)
    return out,sens


def holm_adjust(pvalues):
    p=np.asarray(pvalues,float);n=len(p);order=np.argsort(p);adj=np.empty(n);running=0.
    for rank,idx in enumerate(order):
        running=max(running,(n-rank)*p[idx]);adj[idx]=min(1,running)
    return adj


def effect_table(df, features, dataset, task, seed=SEED):
    rows=[]
    for i,f in enumerate(features):
        z=df[['participant_id','group',f]].dropna()
        if z.duplicated('participant_id').any(): raise ValueError(f'Duplicate participant rows: {dataset}/{task}/{f}')
        hc=z.loc[z.group.eq('HC'),f].to_numpy(float);pdv=z.loc[z.group.eq('PD'),f].to_numpy(float)
        if not len(hc) or not len(pdv): continue
        diff=hc.mean()-pdv.mean();rng=np.random.default_rng(seed+i)
        bh=hc[rng.integers(0,len(hc),(5000,len(hc)))];bp=pdv[rng.integers(0,len(pdv),(5000,len(pdv)))]
        boot=bh.mean(axis=1)-bp.mean(axis=1)
        pooled=np.sqrt(((len(hc)-1)*hc.var(ddof=1)+(len(pdv)-1)*pdv.var(ddof=1))/(len(hc)+len(pdv)-2))
        correction=1-3/(4*(len(hc)+len(pdv))-9)
        g=correction*(diff/pooled) if pooled else np.nan
        # Participant bootstrap Hedges' g interval.
        gboot=[]
        for x,y in zip(bh,bp):
            pool=np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/(len(x)+len(y)-2))
            gboot.append(correction*((x.mean()-y.mean())/pool) if pool else np.nan)
        mw=stats.mannwhitneyu(hc,pdv,alternative='two-sided')
        rows.append({'dataset':dataset,'task':task,'feature':f,'n_hc':len(hc),'n_pd':len(pdv),
          'mean_hc':hc.mean(),'mean_pd':pdv.mean(),'median_hc':np.median(hc),'median_pd':np.median(pdv),
          'mean_difference_hc_minus_pd':diff,'mean_diff_boot_ci95_low':np.percentile(boot,2.5),
          'mean_diff_boot_ci95_high':np.percentile(boot,97.5),'hedges_g_hc_minus_pd':g,
          'hedges_g_boot_ci95_low':np.nanpercentile(gboot,2.5),'hedges_g_boot_ci95_high':np.nanpercentile(gboot,97.5),
          'mann_whitney_p':mw.pvalue,'participant_unit_verified':True})
    out=pd.DataFrame(rows)
    if len(out):out['holm_p_within_dataset_task']=holm_adjust(out.mann_whitney_p)
    return out


def model_registry():
    def build(model,scale):
        steps=[('imputer',SimpleImputer(strategy='median'))]
        if scale:steps.append(('scaler',StandardScaler()))
        steps.append(('model',model))
        return Pipeline(steps)
    return {
      'Logistic_scaled':(build(LogisticRegression(class_weight='balanced',max_iter=5000,solver='liblinear',random_state=SEED),True),
                         {'model__C':[.01,.1,1,10]}),
      'Logistic_unscaled':(build(LogisticRegression(class_weight='balanced',max_iter=5000,solver='liblinear',random_state=SEED),False),
                           {'model__C':[.01,.1,1,10]}),
      'SVM_RBF_scaled':(build(SVC(class_weight='balanced'),True),
                        {'model__C':[.1,1,10],'model__gamma':['scale','auto']}),
      'RandomForest':(build(RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=SEED,n_jobs=1),False),
                      {'model__max_depth':[2,4,None],'model__min_samples_leaf':[1,2]})}


def score_value(fit,x):
    if hasattr(fit,'predict_proba'):return fit.predict_proba(x)[:,1]
    return fit.decision_function(x)


def metrics(y, pred, score):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'balanced_accuracy':balanced_accuracy_score(y,pred),'sensitivity':recall_score(y,pred,zero_division=0),
      'specificity':tn/(tn+fp) if tn+fp else np.nan,'roc_auc':roc_auc_score(y,score) if len(np.unique(y))==2 else np.nan,
      'precision':precision_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def bootstrap_metrics(y,pred,score,seed=SEED,nboot=5000):
    y=np.asarray(y,int);pred=np.asarray(pred,int);score=np.asarray(score,float);rng=np.random.default_rng(seed)
    ix0=np.where(y==0)[0];ix1=np.where(y==1)[0];draw=[]
    for _ in range(nboot):
        idx=np.r_[rng.choice(ix0,len(ix0),replace=True),rng.choice(ix1,len(ix1),replace=True)]
        draw.append(metrics(y[idx],pred[idx],score[idx]))
    return {f'{k}_ci95_low':float(np.nanpercentile([r[k] for r in draw],2.5)) for k in ['balanced_accuracy','sensitivity','specificity','roc_auc','precision','f1']} | \
           {f'{k}_ci95_high':float(np.nanpercentile([r[k] for r in draw],97.5)) for k in ['balanced_accuracy','sensitivity','specificity','roc_auc','precision','f1']}


def evaluate_nested(df,features,model_name,model,grid,seed=SEED,outer_splits=5):
    X=df[features].to_numpy(float); y=df.group.map({'HC':0,'PD':1}).to_numpy(int); ids=df.participant_id.astype(str).to_numpy()
    n=min(outer_splits,int(np.bincount(y).min()))
    if n<3: raise ValueError('fewer than three members in minority class')
    outer=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
    pred=np.full(len(df),-1,int);score=np.full(len(df),np.nan);fold_id=np.full(len(df),-1,int);split=[];paramrows=[]
    for fold,(tr,te) in enumerate(outer.split(X,y),1):
        inner_n=min(3,int(np.bincount(y[tr]).min()))
        inner=StratifiedKFold(n_splits=inner_n,shuffle=True,random_state=seed+fold)
        gs=GridSearchCV(clone(model),grid,scoring='balanced_accuracy',cv=inner,n_jobs=1,refit=True,error_score='raise')
        gs.fit(X[tr],y[tr]); fit=gs.best_estimator_
        pred[te]=fit.predict(X[te]);score[te]=score_value(fit,X[te]);fold_id[te]=fold
        trids=ids[tr].tolist();teids=ids[te].tolist()
        overlap=sorted(set(trids)&set(teids))
        split.append({'model':model_name,'fold':fold,'train_participant_ids':json.dumps(trids),
                      'test_participant_ids':json.dumps(teids),'participant_overlap_count':len(overlap),
                      'best_params':json.dumps(gs.best_params_,sort_keys=True),'inner_best_balanced_accuracy':gs.best_score_})
    met=metrics(y,pred,score)|bootstrap_metrics(y,pred,score,seed+123)
    row={'evaluation':'within_dataset_nested_cv','model':model_name,'features':','.join(features),
         'participants':len(df),'n_hc':int((y==0).sum()),'n_pd':int((y==1).sum()),
         'outer_folds':n,'outer_repeats':1,'score_threshold':'model default (0 decision boundary); no threshold tuning',
         'preprocessing':'median imputation and optional fold-fitted standardization',**met}
    pred_df=pd.DataFrame({'participant_id':ids,'group':df.group.to_numpy(),'y_true':y,'y_pred':pred,
                          'decision_score':score,'fold':fold_id,'model':model_name,'features':','.join(features)})
    return row,pd.DataFrame(split),pred_df


def evaluate_external(train,test,features,model_name,model,grid,source_name,target_name,seed=SEED):
    X=train[features].to_numpy(float);y=train.group.map({'HC':0,'PD':1}).to_numpy(int)
    xt=test[features].to_numpy(float);yt=test.group.map({'HC':0,'PD':1}).to_numpy(int)
    n=min(5,int(np.bincount(y).min()))
    if n<3:raise ValueError('training cohort too small for source-only tuning')
    cv=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
    gs=GridSearchCV(clone(model),grid,scoring='balanced_accuracy',cv=cv,n_jobs=1,refit=True,error_score='raise')
    gs.fit(X,y);fit=gs.best_estimator_;pred=fit.predict(xt);score=score_value(fit,xt)
    met=metrics(yt,pred,score)|bootstrap_metrics(yt,pred,score,seed+99)
    trids=train.participant_id.astype(str).tolist();teids=test.participant_id.astype(str).tolist()
    row={'evaluation':'external_zero_shot','training_dataset':source_name,'test_dataset':target_name,
         'model':model_name,'features':','.join(features),'training_participants':len(train),
         'training_n_hc':int((y==0).sum()),'training_n_pd':int((y==1).sum()),
         'test_participants':len(test),'test_n_hc':int((yt==0).sum()),'test_n_pd':int((yt==1).sum()),
         'source_only_cv_folds':n,'best_source_params':json.dumps(gs.best_params_,sort_keys=True),
         'training_test_id_overlap_count':len(set(trids)&set(teids)),
         'score_threshold':'source-fit model default; external labels not used for thresholding',**met}
    preds=pd.DataFrame({'participant_id':test.participant_id.to_numpy(),'group':test.group.to_numpy(),
       'y_true':yt,'y_pred':pred,'decision_score':score,'training_dataset':source_name,'test_dataset':target_name,
       'model':model_name,'features':','.join(features)})
    return row,preds


def adjusted_domain_models(df,features):
    rows=[]
    z=df.dropna(subset=['age_years','sex',*features]).copy()
    pd_indicator=z.group.eq('PD').astype(float).to_numpy();fs=z.dataset.eq('Figshare_sustained_a').astype(float).to_numpy()
    age=pd.to_numeric(z.age_years).to_numpy(float);age=age-age.mean();male=z.sex.astype(str).str.upper().str.startswith('M').astype(float).to_numpy()
    X=np.column_stack([np.ones(len(z)),pd_indicator,fs,pd_indicator*fs,age,male]);n,k=X.shape
    if n<=k+2: return pd.DataFrame()
    inv=np.linalg.inv(X.T@X)
    outs=[]
    for f in features:
        y=z[f].to_numpy(float);beta=inv@X.T@y;res=y-X@beta;lev=np.sum(X@inv*X,axis=1)
        meat=X.T@(((res/np.maximum(1-lev,1e-8))**2)[:,None]*X);cov=inv@meat@inv
        dfree=n-k;crit=stats.t.ppf(.975,dfree)
        for label,vec in [('ItalianPVS PD_minus_HC',np.array([0,1,0,0,0,0.])),
                          ('Figshare PD_minus_HC',np.array([0,1,0,1,0,0.])),
                          ('dataset_by_diagnosis_interaction',np.array([0,0,0,1,0,0.]))]:
            est=float(vec@beta);se=float(np.sqrt(vec@cov@vec));t=est/se if se else np.nan;p=float(2*stats.t.sf(abs(t),dfree)) if np.isfinite(t) else np.nan
            outs.append({'feature':f,'contrast':label,'estimate':est,'hc3_se':se,'ci95_low':est-crit*se,
                         'ci95_high':est+crit*se,'p_unadjusted':p,'n_participants':n,'covariates':'age, sex, dataset, diagnosis, dataset×diagnosis',
                         'assumption_note':'linear age; HC3; dataset is entangled with recording rate/device/cohort'})
    out=pd.DataFrame(outs)
    out['holm_p_within_contrast_family']=np.nan
    for contrast,g in out.groupby('contrast'):
        out.loc[g.index,'holm_p_within_contrast_family']=holm_adjust(g.p_unadjusted)
    return out


def dataset_identity_experiment(raw):
    # Match Figshare down to all Italian participants within each diagnosis to remove the observed class-proportion imbalance.
    it=raw.loc[raw.dataset.eq('ItalianPVS_VA1')&raw.pitch_floor_hz.eq(75)&raw.status.eq('ok')].copy()
    fs=raw.loc[raw.dataset.eq('Figshare_sustained_a')&raw.pitch_floor_hz.eq(75)&raw.status.eq('ok')].copy()
    counts=it.group.value_counts().to_dict();registry=model_registry();rows=[];preds=[]
    for rep in range(30):
        selected=[]
        for grp,n_take in counts.items():
            selected.append(fs.loc[fs.group.eq(grp)].sample(n=n_take,random_state=SEED+rep+len(selected)))
        sample=pd.concat([it,*selected],ignore_index=True)
        sample['dataset_target']=sample.dataset.eq('Figshare_sustained_a').astype(int)
        X=sample[CORE5].to_numpy(float); y=sample.dataset_target.to_numpy(int)
        strata=(sample.dataset.astype(str)+'_'+sample.group.astype(str)).to_numpy()
        folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+1000+rep)
        for name,(model,grid) in registry.items():
            # Fixed, modest configurations; all transformations are fit inside each split.
            if name=='Logistic_scaled': fit_template=clone(model).set_params(model__C=1)
            elif name=='SVM_RBF_scaled':fit_template=clone(model).set_params(model__C=1,model__gamma='scale')
            elif name=='RandomForest':fit_template=clone(model).set_params(model__max_depth=2,model__min_samples_leaf=2)
            else:continue
            pr=np.full(len(sample),-1);sc=np.full(len(sample),np.nan)
            for fold,(tr,te) in enumerate(folds.split(X,strata),1):
                fit=clone(fit_template).fit(X[tr],y[tr]);pr[te]=fit.predict(X[te]);sc[te]=score_value(fit,X[te])
                preds.extend({'repeat':rep+1,'model':name,'participant_id':sample.iloc[j].participant_id,
                   'dataset':sample.iloc[j].dataset,'group':sample.iloc[j].group,'y_true_dataset':int(y[j]),
                   'y_pred_dataset':int(pr[j]),'decision_score':float(sc[j]),'fold':fold} for j in te)
            m=metrics(y,pr,sc)
            rows.append({'repeat':rep+1,'model':name,'matched_participants':len(sample),
             'n_italian_hc':counts.get('HC',0),'n_italian_pd':counts.get('PD',0),**m})
    return pd.DataFrame(rows),pd.DataFrame(preds)


def plot_all(raw,effects,master,identity,figdir):
    figdir.mkdir(parents=True,exist_ok=True)
    colors={'HC':'#287C8E','PD':'#D45B4B'}
    good=raw.loc[raw.pitch_floor_hz.eq(75)&raw.status.eq('ok')]
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,f in zip(axs.flat,CORE5):
        for i,d in enumerate(['ItalianPVS_VA1','Figshare_sustained_a']):
            for j,g in enumerate(['HC','PD']):
                v=good.loc[good.dataset.eq(d)&good.group.eq(g),f].dropna().to_numpy(float)
                rng=np.random.default_rng(SEED+i*100+j*11)
                x=i+(j-.5)*.24+rng.normal(0,.025,len(v))
                ax.scatter(x,v,s=24,alpha=.72,color=colors[g],edgecolor='white',linewidth=.3,label=g if i==0 else None)
        ax.set_xticks([0,1],['Italian VA1\n16 kHz','Figshare /a/\n8 kHz']);ax.set_title(f,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.2)
    axs.flat[-1].axis('off')
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower right',frameon=False)
    fig.suptitle('Participant-level sustained /a/ features by dataset and diagnosis',fontsize=14,fontweight='bold')
    fig.text(.5,-.015,'Each point is one participant. Figshare language is not identified; dataset, device, and sample rate remain confounded.',ha='center',fontsize=8)
    savefig(fig,figdir,'fig1_common_vowel_distributions')
    # Effect-size forest for shared tasks only.
    sub=effects.loc[effects.dataset.isin(['ItalianPVS_VA1','Figshare_sustained_a'])]
    fig,ax=plt.subplots(figsize=(9,5.5),constrained_layout=True)
    feats=CORE5;labels=[];ys=[]
    for i,f in enumerate(feats):
        for j,d in enumerate(['ItalianPVS_VA1','Figshare_sustained_a']):
            r=sub.loc[sub.feature.eq(f)&sub.dataset.eq(d)]
            if not len(r):continue
            r=r.iloc[0];ypos=i*2+j;ys.append(ypos);labels.append(('Italian VA1' if j==0 else 'Figshare')+' · '+f)
            ax.errorbar(r.hedges_g_hc_minus_pd,ypos,xerr=[[r.hedges_g_hc_minus_pd-r.hedges_g_boot_ci95_low],[r.hedges_g_boot_ci95_high-r.hedges_g_hc_minus_pd]],fmt='o',capsize=3,color='#24333D' if j==0 else '#D39B32')
    ax.set_yticks(ys,labels);ax.axvline(0,color='#7A858B',lw=.8);ax.invert_yaxis();ax.set_xlabel('Hedges g (HC − PD), participant bootstrap 95% CI');ax.grid(axis='x',alpha=.2)
    ax.set_title('Shared sustained /a/ effects: independent cohorts',loc='left',fontweight='bold')
    fig.text(.5,-.02,'Direction and intervals are dataset-specific; overlapping features are not proof of measurement or disease-effect equivalence.',ha='center',fontsize=8)
    savefig(fig,figdir,'fig2_common_vowel_effect_forest')
    # Compact primary core-5 performance plot: internal cohorts beside native-rate zero-shot transfers.
    x=master.loc[master.feature_set.eq('core5') & master.evaluation.isin(['within_dataset_nested_cv','external_zero_shot'])].copy()
    models=['Logistic_scaled','SVM_RBF_scaled','RandomForest','Logistic_unscaled']
    fig,axs=plt.subplots(1,2,figsize=(13,7),constrained_layout=True,sharex=True)
    panels=[(axs[0],x.loc[x.evaluation.eq('within_dataset_nested_cv')],['ItalianPVS_VA1','Figshare_sustained_a'],
             {'ItalianPVS_VA1':'Italian internal','Figshare_sustained_a':'Figshare internal'},'Nested internal CV'),
            (axs[1],x.loc[x.evaluation.eq('external_zero_shot')&x.training_dataset.isin(['ItalianPVS_VA1 16k','Figshare_sustained_a 8k'])],
             ['ItalianPVS_VA1 16k','Figshare_sustained_a 8k'],
             {'ItalianPVS_VA1 16k':'Italian → Figshare','Figshare_sustained_a 8k':'Figshare → Italian'},'Zero-shot external')]
    nice_model={'Logistic_scaled':'Logistic (scaled)','Logistic_unscaled':'Logistic (raw units)',
                'SVM_RBF_scaled':'RBF SVM','RandomForest':'Random Forest'}
    for ax,part,datasets,names,title in panels:
        vals=[];labels=[];los=[];his=[]
        for d in datasets:
            for mod in models:
                if title=='Nested internal CV': z=part.loc[part.training_dataset.eq(d)&part.model.eq(mod)]
                else:z=part.loc[part.training_dataset.eq(d)&part.model.eq(mod)]
                if z.empty:continue
                r=z.iloc[0];vals.append(r.balanced_accuracy);labels.append(f"{names[d]} · {nice_model[mod]}")
                los.append(r.balanced_accuracy-r.balanced_accuracy_ci95_low);his.append(r.balanced_accuracy_ci95_high-r.balanced_accuracy)
        yy=np.arange(len(vals));ax.errorbar(vals,yy,xerr=[los,his],fmt='o',capsize=2,color='#287C8E')
        ax.axvline(.5,color='#D45B4B',ls='--',lw=1);ax.set_yticks(yy,labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.set_xlabel('Balanced accuracy (participant bootstrap 95% CI)');ax.grid(axis='x',alpha=.2);ax.set_title(title,loc='left',fontweight='bold')
    fig.suptitle('Within-dataset and cross-dataset performance on shared sustained /a/ features',fontsize=13,fontweight='bold')
    fig.text(.5,-.02,'Core-5 only; native 16 kHz Italian and 8 kHz Figshare. Rate-resampling sensitivity and other tasks are in the CSV tables.',ha='center',fontsize=8)
    savefig(fig,figdir,'fig3_classifier_performance')
    # PCA for core sustained /a/, descriptive only.
    d=good.loc[good.dataset.isin(['ItalianPVS_VA1','Figshare_sustained_a'])].dropna(subset=CORE5).copy()
    X=StandardScaler().fit_transform(d[CORE5]);pca=PCA(n_components=2,random_state=SEED);z=pca.fit_transform(X)
    fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
    marks={'ItalianPVS_VA1':'o','Figshare_sustained_a':'s'}
    for (ds,g),idx in d.groupby(['dataset','group']).groups.items():
        ix=d.index.get_indexer(idx);ax.scatter(z[ix,0],z[ix,1],label=f'{ds} {g}',marker=marks[ds],color=colors[g],alpha=.75,edgecolor='white',linewidth=.4)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)');ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)');ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.2)
    ax.set_title('Exploratory PCA of common sustained /a/ features',loc='left',fontweight='bold')
    fig.text(.5,-.02,'PCA is descriptive; dataset separation cannot be attributed specifically to language or disease.',ha='center',fontsize=8)
    savefig(fig,figdir,'fig4_common_vowel_pca')
    # dataset identity classification, matched HC/PD subsampling.
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for i,model in enumerate(identity.model.unique()):
        g=identity.loc[identity.model.eq(model)]
        x0=np.full(len(g),i)+np.linspace(-.12,.12,len(g));ax.scatter(x0,g.balanced_accuracy,alpha=.6,s=20)
        ax.plot([i-.18,i+.18],[g.balanced_accuracy.median()]*2,color='#24333D',lw=2)
    ax.axhline(.5,color='#D45B4B',ls='--');ax.set_xticks(range(identity.model.nunique()),identity.model.unique(),rotation=15,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Dataset identity balanced accuracy');ax.grid(axis='y',alpha=.2)
    ax.set_title('Can acoustic features identify dataset after matching diagnosis counts?',loc='left',fontweight='bold')
    fig.text(.5,-.02,'30 matched Figshare subsamples (17 HC, 12 PD) with participant-stratified 5-fold CV; variation is subsampling sensitivity, not independent replications.',ha='center',fontsize=8)
    savefig(fig,figdir,'fig5_dataset_identity')


def savefig(fig,folder,stem):
    folder.mkdir(exist_ok=True,parents=True)
    fig.savefig(folder/f'{stem}.png',dpi=320,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/f'{stem}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
