from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.cross_language_generalisation.pipeline import OUT,SEED,CORE5,savefig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    raw=pd.read_csv(OUT/'tables/harmonised_raw_audio_features.csv')
    it=raw.loc[raw.dataset.eq('ItalianPVS_VA1')&raw.pitch_floor_hz.eq(75)&raw.status.eq('ok')].copy()
    fs=raw.loc[raw.dataset.eq('Figshare_sustained_a')&raw.pitch_floor_hz.eq(75)&raw.status.eq('ok')].copy()
    counts=it.group.value_counts().to_dict()
    sets={'core5_all':CORE5}
    for f in CORE5:sets[f'core5_without_{f}']=[x for x in CORE5 if x!=f]
    for f in CORE5:sets[f'only_{f}']=[f]
    rows=[];predrows=[];coefrows=[]
    for rep in range(30):
        chosen=[it]
        for ix,(grp,n_take) in enumerate(counts.items()):chosen.append(fs.loc[fs.group.eq(grp)].sample(n=n_take,random_state=SEED+rep+ix))
        sample=pd.concat(chosen,ignore_index=True);sample['domain']=sample.dataset.eq('Figshare_sustained_a').astype(int)
        strata=(sample.dataset.astype(str)+'_'+sample.group.astype(str)).to_numpy()
        folds=StratifiedKFold(5,shuffle=True,random_state=SEED+1000+rep)
        for name,features in sets.items():
            y=sample.domain.to_numpy(int);pred=np.full(len(sample),-1,int);score=np.full(len(sample),np.nan)
            for fold,(tr,te) in enumerate(folds.split(sample[features],strata),1):
                pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),
                  ('model',LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=5000,random_state=SEED))])
                pipe.fit(sample.iloc[tr][features],y[tr]);pred[te]=pipe.predict(sample.iloc[te][features]);score[te]=pipe.predict_proba(sample.iloc[te][features])[:,1]
                if name=='core5_all':
                    for f,beta in zip(features,pipe.named_steps['model'].coef_[0]):
                        coefrows.append({'repeat':rep+1,'fold':fold,'feature':f,'standardized_logistic_coefficient':float(beta)})
                predrows.extend({'repeat':rep+1,'fold':fold,'feature_set':name,'participant_id':sample.iloc[j].participant_id,
                  'group':sample.iloc[j].group,'dataset':sample.iloc[j].dataset,'y_true_domain':int(y[j]),
                  'y_pred_domain':int(pred[j]),'score':float(score[j])} for j in te)
            from sklearn.metrics import balanced_accuracy_score,roc_auc_score
            bacc=float(balanced_accuracy_score(y,pred));auc=float(roc_auc_score(y,score))
            rows.append({'repeat':rep+1,'feature_set':name,'features':','.join(features),'n_per_dataset':29,
              'n_hc_each':counts.get('HC',0),'n_pd_each':counts.get('PD',0),'balanced_accuracy':bacc,'roc_auc':auc})
    d=pd.DataFrame(rows);base=d.loc[d.feature_set.eq('core5_all'),['repeat','balanced_accuracy','roc_auc']].rename(columns={'balanced_accuracy':'baseline_bacc','roc_auc':'baseline_auc'})
    d=d.merge(base,on='repeat');d['delta_balanced_accuracy_vs_core5']=d.balanced_accuracy-d.baseline_bacc
    d['delta_roc_auc_vs_core5']=d.roc_auc-d.baseline_auc
    d.to_csv(OUT/'tables/dataset_identity_feature_ablation.csv',index=False)
    d.groupby('feature_set',as_index=False).agg(features=('features','first'),median_bacc=('balanced_accuracy','median'),
      q1_bacc=('balanced_accuracy',lambda x:x.quantile(.25)),q3_bacc=('balanced_accuracy',lambda x:x.quantile(.75)),
      median_roc_auc=('roc_auc','median'),median_delta_bacc=('delta_balanced_accuracy_vs_core5','median'),
      delta_bacc_min=('delta_balanced_accuracy_vs_core5','min'),delta_bacc_max=('delta_balanced_accuracy_vs_core5','max')).to_csv(OUT/'tables/dataset_identity_feature_ablation_summary.csv',index=False)
    pd.DataFrame(coefrows).to_csv(OUT/'tables/dataset_identity_logistic_coefficients.csv',index=False)
    pd.DataFrame(predrows).to_csv(OUT/'tables/dataset_identity_feature_ablation_oof.csv',index=False)
    summ=pd.read_csv(OUT/'tables/dataset_identity_feature_ablation_summary.csv')
    order=['core5_all']+[f'core5_without_{x}' for x in CORE5]+[f'only_{x}' for x in CORE5]
    summ['ord']=summ.feature_set.map({x:i for i,x in enumerate(order)});summ=summ.sort_values('ord')
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    y=np.arange(len(summ));ax.errorbar(summ.median_bacc,y,xerr=[summ.median_bacc-summ.q1_bacc,summ.q3_bacc-summ.median_bacc],fmt='o',color='#287C8E',capsize=2)
    ax.axvline(.5,color='#D45B4B',ls='--');ax.set_yticks(y,[x.replace('core5_without_','− ').replace('only_','only ') for x in summ.feature_set]);ax.invert_yaxis();ax.set_xlim(.3,.9)
    ax.set_xlabel('Dataset identity balanced accuracy (median and IQR across 30 matched draws)');ax.grid(axis='x',alpha=.2)
    ax.set_title('Diagnosis-matched dataset identity: feature ablation',loc='left',fontweight='bold')
    fig.text(.5,-.02,'Fixed standardized logistic model; fold-local imputation/scaling; 17 HC and 12 PD per dataset in every draw. Repeated draws are sensitivity analyses.',ha='center',fontsize=8)
    savefig(fig,OUT/'figures','fig6_dataset_identity_feature_ablation')
    print(summ[['feature_set','median_bacc','q1_bacc','q3_bacc','median_roc_auc','median_delta_bacc']].to_string(index=False))

if __name__=='__main__':run()
