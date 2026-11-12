from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import soundfile as sf
import parselmouth

from src.final_research.analysis import (
    FIGSHARE, MDVR, DATASETS, ARCHIVE_301, VOICE_FEATURES, MDVR_FEATURES,
    UCI174_MAP, UCI301_FEATURE_NAMES, adjusted_figshare_effect, audio_qc,
    compare_groups, discover_audio, holm, load_uci301, protected_files,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT/'results/english_research_20261031'
PKG = ROOT/'results/final_research'
FIGI = PKG/'italian_figures'; FIGE = PKG/'english_figures'; FIGT = PKG/'italian_tables'; TABLEE = PKG/'english_tables'
FIGC = PKG/'cross_language_figures'
SEED = 20261031
COLORS={'HC':'#287C8E','PD':'#D45B4B','other':'#888888','dark':'#24333D','light':'#DCE7EA','gold':'#D39B32'}
TASKS=['VA1','VE1','VI1','VO1','VU1']


def md_table(df: pd.DataFrame, columns=None, precision=3) -> str:
    z=df if columns is None else df[columns]
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v,(float,np.floating)): return f'{v:.{precision}f}'
        return str(v).replace('|','\\|').replace('\n',' ')
    head='| '+' | '.join(map(str,z.columns))+' |'
    sep='| '+' | '.join(['---']*len(z.columns))+' |'
    return '\n'.join([head,sep,*('| '+' | '.join(fmt(v) for v in row)+' |' for row in z.itertuples(index=False,name=None))])


def sha_manifest(paths):
    return {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):sha256_file(p) for p in paths if p.exists()}


def savefig(fig, folder: Path, stem: str):
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{stem}.png',dpi=320,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/f'{stem}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def bootstrap_stat_diff(hc,pd_values,statistic,seed,replicates=10000):
    rng=np.random.default_rng(seed); x=np.asarray(hc,float); y=np.asarray(pd_values,float)
    bx=x[rng.integers(0,len(x),(replicates,len(x)))]; by=y[rng.integers(0,len(y),(replicates,len(y)))]
    d=statistic(bx,axis=1)-statistic(by,axis=1)
    return float(np.percentile(d,2.5)),float(np.percentile(d,97.5))


def record_feature_tables(fig_qc,fig_feat,mdvr_qc,mdvr_feat,italian,expout):
    # Per-recording source audio QC, with flags retained and no exclusions.
    qc=pd.concat([fig_qc,mdvr_qc],ignore_index=True,sort=False)
    qc.to_csv(expout/'audio_qc_by_recording.csv',index=False)
    features=pd.concat([fig_feat,mdvr_feat],ignore_index=True,sort=False)
    features.to_csv(expout/'audio_features_by_recording.csv',index=False)
    feat_summary=features.groupby(['dataset','task','group'],dropna=False).agg(
        recordings=('participant_id','size'),participants=('participant_id','nunique'),
        sample_rate_hz_median=('sample_rate_hz','median'),feature_rows_ok=('status',lambda s:int((s=='ok').sum()))
    ).reset_index()
    feat_summary.to_csv(expout/'feature_extraction_coverage.csv',index=False)
    qsum=qc.groupby(['dataset','task','group'],dropna=False).agg(
        recordings=('participant_id','size'),participants=('participant_id','nunique'),
        readable=('readable','sum'),rate_hz=('sample_rate_hz','median'),
        duration_median_s=('duration_s','median'),duration_q1_s=('duration_s',lambda x:x.quantile(.25)),
        duration_q3_s=('duration_s',lambda x:x.quantile(.75)),possible_qc_flags=('qc_flag',lambda s:int(s.fillna('').ne('').sum())),
        exact_duplicate_hashes=('exact_duplicate_in_dataset','sum')
    ).reset_index()
    qsum.to_csv(expout/'audio_qc_summary.csv',index=False)
    return qc,features,feat_summary,qsum


def stats_and_tables(fig_feat,mdvr_feat,ox_subject,tm,tm_subject,u301,u301_test,expout,italian_stats):
    all_effects=[]
    fig=fig_feat.loc[fig_feat.status.eq('ok')].copy()
    fig_group=fig[['participant_id','group',*VOICE_FEATURES]].copy()
    fe=compare_groups(fig_group,VOICE_FEATURES,dataset='Figshare',task='sustained_a',seed=SEED)
    if fe: fe=pd.DataFrame(fe); fe['holm_p_within_task_4_features']=holm(fe.mann_whitney_p); all_effects.extend(fe.to_dict('records'))
    adjusted=pd.DataFrame([adjusted_figshare_effect(fig_group.assign(age=fig_feat.set_index('participant_id').age.reindex(fig_group.participant_id).to_numpy(),
                                                    sex=fig_feat.set_index('participant_id').sex.reindex(fig_group.participant_id).to_numpy()),f)
                           for f in VOICE_FEATURES])
    adjusted['holm_p_within_task_4_features']=holm(adjusted.p_value_unadjusted)
    adjusted.to_csv(expout/'figshare_age_sex_adjusted_effects.csv',index=False)
    # MDVR text reading and spontaneous dialogue are analyzed independently.
    for task,sub in mdvr_feat.loc[mdvr_feat.status.eq('ok')].groupby('task'):
        cols=['participant_id','group',*MDVR_FEATURES]
        frame=sub[cols].copy()
        ef=compare_groups(frame,MDVR_FEATURES,dataset='MDVR-KCL',task=task,seed=SEED+len(all_effects))
        if ef:
            e=pd.DataFrame(ef); e['holm_p_within_task_10_features']=holm(e.mann_whitney_p); all_effects.extend(e.to_dict('records'))
    # UCI 174 raw feature table: collapse repeats to the participant before inference.
    ox_effects=compare_groups(ox_subject[['participant_id','group',*VOICE_FEATURES]],VOICE_FEATURES,
                              dataset='UCI174_feature_table',task='sustained_vowel_a',seed=SEED+2)
    if ox_effects:
        e=pd.DataFrame(ox_effects);e['holm_p_within_task_4_features']=holm(e.mann_whitney_p);all_effects.extend(e.to_dict('records'))
    # UCI 301 train table: the official within-subject sample order defines three sustained vowels.
    for task in ['a','o','u']:
        v=u301.loc[u301.task.eq(task)&u301.sample_order.eq({'a':1,'o':2,'u':3}[task])].copy()
        cols=['jitter_local','shimmer_local','htn','median_pitch_hz','unvoiced_fraction']
        ef=compare_groups(v[['participant_id','group',*cols]],cols,dataset='UCI301_feature_table',task=f'sustained_{task}',seed=SEED+ord(task))
        if ef:
            e=pd.DataFrame(ef);e['holm_p_within_task_5_features']=holm(e.mann_whitney_p);all_effects.extend(e.to_dict('records'))
    effects=pd.DataFrame(all_effects)
    pcols=[c for c in effects if c.startswith('holm_p_within_task_')]
    effects['holm_adjusted_p']=effects[pcols].bfill(axis=1).iloc[:,0] if pcols else np.nan
    effects.to_csv(expout/'participant_level_group_effects.csv',index=False)
    # UCI 189 has only PD participants; summarize within-person feature/UPDRS correlations.
    tele=[]
    for feat in ['Jitter(%)','Shimmer','HNR','PPE']:
        for outcome in ['motor_UPDRS','total_UPDRS']:
            rr=[]
            for pid,g in tm.groupby('subject#'):
                if g[feat].nunique()<2 or g[outcome].nunique()<2: continue
                rr.append({'participant_id':str(pid),'rho':float(stats.spearmanr(g[feat],g[outcome]).statistic),
                           'n_repeated_recordings':len(g)})
            rhos=np.array([x['rho'] for x in rr],float)
            rng=np.random.default_rng(SEED+len(tele)); b=np.median(rng.choice(rhos,(10000,len(rhos)),replace=True),axis=1)
            try: p=float(stats.wilcoxon(rhos).pvalue) if np.any(rhos) else 1.0
            except Exception: p=np.nan
            tele.append({'feature':feat,'outcome':outcome,'n_participants':len(rhos),'median_within_person_spearman_rho':float(np.median(rhos)),
                         'rho_q1':float(np.percentile(rhos,25)),'rho_q3':float(np.percentile(rhos,75)),
                         'median_rho_boot_ci95_low':float(np.percentile(b,2.5)),'median_rho_boot_ci95_high':float(np.percentile(b,97.5)),
                         'participants_positive':int((rhos>0).sum()),'wilcoxon_p_two_sided':p})
    tele=pd.DataFrame(tele); tele['holm_p_8_tests']=holm(tele.wilcoxon_p_two_sided)
    tele.to_csv(expout/'uci189_within_participant_correlations.csv',index=False)
    # UCI 301 independent test set: repeat spread is descriptive only, participant is unit.
    repeat=[]
    for task,g in u301_test.groupby('task'):
        if task not in ['a','o']: continue
        for feature in ['jitter_local','shimmer_local','median_pitch_hz']:
            for pid,pg in g.groupby('participant_id'):
                vals=pg[feature].dropna().to_numpy(float)
                if len(vals)>=2:
                    repeat.append({'task':task,'feature':feature,'participant_id':pid,'group':'PD','n_repetitions':len(vals),
                                   'mean_value':float(np.mean(vals)),'within_person_sd':float(np.std(vals,ddof=1)),
                                   'within_person_cv':float(np.std(vals,ddof=1)/abs(np.mean(vals))) if np.mean(vals)!=0 else np.nan})
    repeat=pd.DataFrame(repeat); repeat.to_csv(expout/'uci301_pd_test_repeat_spread.csv',index=False)
    return effects,adjusted,tele,repeat


def italian_tables(figi,fige,figc,tablei,tablee):
    # Cohort flow and demographic summaries are based on the already reconciled manifests.
    manifest=pd.read_csv(ROOT/'data/italian_manifest.csv')
    primary=pd.read_csv(ROOT/'data/primary_cohort_manifest.csv')
    primary=primary.loc[primary.include_primary_cohort.astype(bool)].copy()
    participants=pd.read_csv(ROOT/'data/primary_cohort_participants.csv')
    included=participants.loc[participants.include_primary_cohort.astype(bool)].copy()
    cohort=[]
    for name,df,pcol,gcol in [('full_dataset',manifest,'participant_folder','label'),('reconciled_HC_PD',primary,'participant_folder','cohort_group')]:
        for group,g in df.groupby(gcol):
            cohort.append({'population':name,'group':group,'participants':g[pcol].nunique(),
                           'recordings':len(g),'mean_age_years':np.nan,'female_n':np.nan,'male_n':np.nan})
    for group,g in included.groupby('cohort_group'):
        ages=pd.to_numeric(g.metadata_age,errors='coerce')
        cohort.append({'population':'reconciled_participant_roster','group':group,'participants':g.participant_folder.nunique(),
                       'recordings':np.nan,'mean_age_years':float(ages.mean()),
                       'female_n':int(g.metadata_sex.eq('F').sum()),'male_n':int(g.metadata_sex.eq('M').sum())})
    base=pd.read_csv(ROOT/'data/va1_features_raw.csv')
    if 'group' in base.columns:
        for group,g in base.groupby('cohort_group'):
            cohort.append({'population':'VA1_reference_baseline','group':group,'participants':g.participant_folder.nunique(),
                           'recordings':len(g),'mean_age_years':np.nan,'female_n':np.nan,'male_n':np.nan})
    cohort=pd.DataFrame(cohort);cohort.to_csv(tablei/'cohort_characteristics.csv',index=False)
    # Task counts at the reconciled and native-16k population levels.
    coverage=[]
    for pop,df in [('reconciled_HC_PD',primary),('native_16k_primary_subset',pd.read_csv(ROOT/'data/primary_cohort_recordings_included.csv'))]:
        taskcol='possible_task_code'; groupcol='cohort_group'
        for (task,group),g in df.groupby([taskcol,groupcol]):
            coverage.append({'population':pop,'task_code':task,'group':group,'recordings':len(g),
                             'participants':g.participant_folder.nunique(),'sample_rate_hz':'mixed' if pop=='reconciled_HC_PD' else 16000})
    coverage=pd.DataFrame(coverage);coverage.to_csv(tablei/'task_recording_coverage.csv',index=False)
    featuredefs=pd.read_csv(ROOT/'results/italian_multitask_v1/configs/feature_definitions.csv')
    featuredefs.to_csv(tablei/'acoustic_feature_extraction_summary.csv',index=False)
    comps=pd.read_csv(ROOT/'results/italian_multitask_v1/statistics/native16_group_comparisons_effect_sizes_ci.csv')
    primary_feats=comps.loc[comps.task_code.isin(TASKS)&comps.feature.isin(VOICE_FEATURES)].copy()
    primary_feats.to_csv(tablei/'italian_primary_effects.csv',index=False)
    sens=pd.read_csv(ROOT/'results/italian_research_phase2/sensitivity/group_contrast_sensitivity.csv')
    vi=sens.loc[sens.cohort.eq('native16')&sens.task_code.eq('VI1')&sens.feature.eq('shimmer_local')].copy()
    hist=pd.read_csv(ROOT/'results/italian_research_phase2/sensitivity/vi1_shimmer_bootstrap_interval_audit.csv')
    for c in ['new_participant_bootstrap_percentile_ci_low','new_participant_bootstrap_percentile_ci_high','new_participant_bootstrap_bca_ci_low','new_participant_bootstrap_bca_ci_high']:
        vi[c]=hist.loc[0,c]
    vi.to_csv(tablei/'vi1_shimmer_sensitivity.csv',index=False)
    evidence=pd.DataFrame([
        ['VA1/VE1/VI1/VO1/VU1 sustained vowels','participant-level primary family','VI1 shimmer only survives original Holm 20-test correction','stronger exploratory evidence','small n; Yuen adjusted p=0.061; human review pending'],
        ['VI1 HNR and other shimmer contexts','secondary vowel-specific observations','not surviving primary-family correction','suggestive only','wide uncertainty; sensitive to task/context'],
        ['Second vowels','descriptive signal QC and paired summaries','shorter durations observed; protocol undefined','inconclusive','cannot call repetitions or test-retest reliability'],
        ['D1/D2','all-recording signal characterization','repeated CV-like bursts in most plots','human validation required','candidate peaks are not validated syllable counts or VOT'],
        ['B1/B2/FB1','candidate generic VAD/prosody/voice quality','threshold-sensitive speech masks; no inferential tests','human validation required','no transcript alignment; FB1 material unavailable'],
        ['VA1 cross-validation baseline','saved 29-person OOF predictions','BA=.900, AUC=.917','exploratory benchmark only','not independent or clinical validation'],
        ['Frame/cycle or recording-level ML','not performed','would inflate n or invite leakage without participant controls','exclude from inferential conclusions','no classifiers trained in this package'],
    ],columns=['task_or_analysis','unit_and_scope','observed_result','evidence_class','limitation'])
    evidence.to_csv(tablei/'evidence_classification_and_limitations.csv',index=False)
    return manifest,primary,participants,coverage,featuredefs,primary_feats,vi,evidence


def make_italian_figures(manifest,primary,coverage,italian_features,italian_effects,sensitivity,connected,figdir):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    # Figure 1: flow from supplied manifest to the analyzed cohort.
    fig,ax=plt.subplots(figsize=(11,6));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    boxes=[(.35,.82,.30,.12,'Full ItalianPVS manifest\n831 recordings · 65 participants',COLORS['light']),
           (.35,.60,.30,.12,'Reconciled elderly-HC vs PD cohort\n567 recordings · 36 participants', '#E7EEF4'),
           (.06,.34,.36,.15,'Elderly healthy controls\n17 participants · native 16 kHz', '#DCEEF0'),
           (.58,.34,.36,.15,'Parkinson’s disease\n12 of 19 reconciled participants · 16 kHz', '#F7E5E1'),
           (.19,.08,.62,.15,'Primary first-vowel analysis\n145 participant-task records · 17 HC / 12 PD per vowel\nVA1 baseline: 29 participants · 17 HC / 12 PD', '#EAF1E4')]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.02',facecolor=color,edgecolor='#61717A',linewidth=1.1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.5,color=COLORS['dark'])
    for a,b in [((.5,.82),(.5,.72)),((.42,.60),(.24,.49)),((.58,.60),(.76,.49)),((.24,.34),(.40,.23)),((.76,.34),(.60,.23))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#667780',linewidth=1.1))
    ax.text(.5,.98,'Figure 1. ItalianPVS participant flow and analysis cohorts',ha='center',va='top',fontsize=13,fontweight='bold')
    ax.text(.5,.015,'Reconciliation decisions unchanged; sample-rate-restricted PD count is smaller because additional PD recordings are 44.1 kHz.',ha='center',fontsize=8,color='#566')
    savefig(fig,figdir,'fig1_italian_cohort_flow')
    # Figure 2: group-level task coverage.
    task_order=['VA1','VA2','VE1','VE2','VI1','VI2','VO1','VO2','VU1','VU2','D1','D2','B1','B2','FB1']
    prim=pd.read_csv(ROOT/'data/primary_cohort_manifest.csv')
    prim=prim.loc[prim.include_primary_cohort.astype(bool)].copy()
    native=pd.read_csv(ROOT/'data/primary_cohort_recordings_included.csv')
    mats=[]
    for df,title in [(prim,'Reconciled mixed-rate cohort'),(native,'Native 16 kHz cohort')]:
        mat=np.zeros((2,len(task_order)))
        for i,g in enumerate(['ELDERLY_HC','PD']):
            sub=df[df.cohort_group.eq(g)]
            for j,t in enumerate(task_order): mat[i,j]=sub.loc[sub.possible_task_code.eq(t),'participant_folder'].nunique()
        mats.append((mat,title))
    fig,axs=plt.subplots(2,1,figsize=(12,5.4),sharex=True,constrained_layout=True)
    for ax,(mat,title) in zip(axs,mats):
        im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=max(mat.max(),1),aspect='auto')
        ax.set_yticks([0,1],['Elderly HC','PD']);ax.set_title(title,loc='left',fontweight='bold')
        for i in range(2):
            for j in range(len(task_order)):
                ax.text(j,i,f'{int(mat[i,j])}',ha='center',va='center',fontsize=7,
                        color='white' if mat[i,j]>.55*mat.max() else COLORS['dark'])
    axs[-1].set_xticks(range(len(task_order)),task_order,rotation=0)
    fig.colorbar(im,ax=axs,shrink=.82,label='Unique participants with task recording')
    fig.suptitle('Figure 2. ItalianPVS participant task coverage by group and sample-rate population',fontsize=13,fontweight='bold')
    savefig(fig,figdir,'fig2_italian_task_coverage')
    # Figure 3: per-person feature distributions with bootstrap mean intervals.
    feats=['jitter_local','shimmer_local','hnr_mean_db','median_f0_hz']
    labels={'jitter_local':'Local jitter','shimmer_local':'Local shimmer','hnr_mean_db':'HNR (dB)','median_f0_hz':'Median F0 (Hz)'}
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,feature in zip(axs.flat,feats):
        for i,t in enumerate(TASKS):
            for group,offset in [('HC',-.16),('PD',.16)]:
                v=italian_features.loc[italian_features.task_code.eq(t)&italian_features.group.eq({'HC':'ELDERLY_HC','PD':'PD'}[group]),feature].dropna().to_numpy(float)
                rng=np.random.default_rng(SEED+i+len(feature)+(0 if group=='HC' else 100))
                ax.scatter(i+offset+rng.normal(0,.035,len(v)),v,s=25,alpha=.76,color=COLORS[group],edgecolor='white',linewidth=.35,zorder=2)
                if len(v):
                    boot=np.mean(rng.choice(v,(5000,len(v)),replace=True),axis=1)
                    mean=np.mean(v);lo,hi=np.percentile(boot,[2.5,97.5])
                    ax.errorbar(i+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',ms=4,color=COLORS[group],capsize=2,lw=1.1,zorder=3)
        ax.set_title(labels[feature],loc='left',fontweight='bold');ax.set_xticks(range(5),TASKS)
        ax.set_ylabel(labels[feature]);ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS[g],label=l,markersize=7) for g,l in [('HC','Elderly HC'),('PD','PD')]],frameon=False)
    fig.suptitle('Figure 3. Participant-level first-vowel acoustic distributions (native 16 kHz)',fontsize=13,fontweight='bold')
    fig.text(.5,-.015,'Dots are individual participants; diamonds and whiskers show group means and percentile bootstrap 95% intervals (n=17 HC, 12 PD in every vowel).',ha='center',fontsize=8)
    savefig(fig,figdir,'fig3_italian_acoustic_distributions')
    # Figure 4: primary effect forest plot.
    ordered_features=['median_f0_hz','jitter_local','shimmer_local','hnr_mean_db']
    ftitle={'median_f0_hz':'Median F0','jitter_local':'Local jitter','shimmer_local':'Local shimmer','hnr_mean_db':'HNR'}
    fig,axs=plt.subplots(4,1,figsize=(9.2,10),sharex=True,constrained_layout=True)
    for ax,f in zip(axs,ordered_features):
        sub=italian_effects[italian_effects.feature.eq(f)].set_index('task_code').reindex(TASKS).reset_index()
        for y,row in sub.iterrows():
            sig=pd.notna(row.get('holm_p_primary_20_comparisons')) and row.holm_p_primary_20_comparisons<.05
            c=COLORS['PD'] if sig else COLORS['dark']
            ax.errorbar(row.hedges_g_hc_minus_pd,y,xerr=[[row.hedges_g_hc_minus_pd-row.hedges_g_ci95_low],[row.hedges_g_ci95_high-row.hedges_g_hc_minus_pd]],fmt='o',color=c,ms=5,capsize=2,lw=1.1)
            ax.text(2.68,y,('Holm p='+f'{row.holm_p_primary_20_comparisons:.3f}') if sig else '',va='center',fontsize=7,color=COLORS['PD'])
        ax.axvline(0,color='#737E85',lw=.8);ax.set_yticks(range(5),TASKS);ax.invert_yaxis();ax.set_title(ftitle[f],loc='left',fontweight='bold');ax.grid(axis='x',alpha=.2)
    axs[-1].set_xlabel('Hedges’ g (HC minus PD), participant bootstrap 95% interval')
    axs[0].set_xlim(-2.2,3.2)
    fig.suptitle('Figure 4. First-vowel group effect sizes across five vowel contexts',fontsize=13,fontweight='bold')
    fig.text(.5,-.014,'Red marks the single primary-family result below adjusted p=.05 (VI1 shimmer, Holm p=.012). Intervals are wide; effects are exploratory.',ha='center',fontsize=8)
    savefig(fig,figdir,'fig4_italian_vowel_effect_forest')
    # Figure 5: VI1 shimmer observations, influence and method sensitivity.
    sub=italian_features.loc[italian_features.task_code.eq('VI1')].copy()
    sens=sensitivity.loc[sensitivity.cohort.eq('native16')&sensitivity.task_code.eq('VI1')&sensitivity.feature.eq('shimmer_local')].iloc[0]
    boot=pd.read_csv(ROOT/'results/italian_research_phase2/sensitivity/vi1_shimmer_bootstrap_interval_audit.csv').iloc[0]
    loo=pd.read_csv(ROOT/'results/italian_research_phase2/sensitivity/vi1_shimmer_leave_one_out.csv')
    hc=sub.loc[sub.group.eq('ELDERLY_HC'),'shimmer_local'].to_numpy(float); pdv=sub.loc[sub.group.eq('PD'),'shimmer_local'].to_numpy(float)
    metrics=[('Mean',np.mean, sens.mean_diff_boot_pct_low,sens.mean_diff_boot_pct_high),
             ('Median',np.median,sens.median_diff_boot_pct_low,sens.median_diff_boot_pct_high),
             ('20% trimmed mean',lambda a,axis=None:stats.trim_mean(a,.2,axis=axis),np.nan,np.nan),
             ('10% winsorized mean',lambda a,axis=None:np.mean(np.clip(a,np.quantile(a,.1,axis=axis,keepdims=True),np.quantile(a,.9,axis=axis,keepdims=True)),axis=axis),np.nan,np.nan)]
    bootsumm=[]
    for j,(name,stat,lo,hi) in enumerate(metrics):
        if not np.isfinite(lo): lo,hi=bootstrap_stat_diff(hc,pdv,stat,SEED+420+j)
        bootsumm.append({'summary':name,'estimate':float(stat(hc)-stat(pdv)),'ci_low':lo,'ci_high':hi})
    bsum=pd.DataFrame(bootsumm)
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
    rng=np.random.default_rng(SEED)
    for group,vals,x in [('HC',hc,0),('PD',pdv,1)]:
        axs[0].scatter(x+rng.normal(0,.045,len(vals)),vals,s=32,alpha=.78,color=COLORS[group],edgecolor='white',linewidth=.4)
        m=vals.mean();lo,hi=np.percentile([np.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(4000)],[2.5,97.5])
        axs[0].errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt='D',color=COLORS['dark'],capsize=3,ms=4)
    axs[0].set_xticks([0,1],['Elderly HC\nn=17','PD\nn=12']);axs[0].set_ylabel('Local shimmer');axs[0].set_title('Participant observations',loc='left',fontweight='bold');axs[0].grid(axis='y',alpha=.2)
    loo=loo.loc[loo.cohort.eq('native16')].reset_index(drop=True)
    colors=[COLORS['HC'] if g=='ELDERLY_HC' else COLORS['PD'] for g in loo.left_out_group]
    axs[1].scatter(np.arange(1,len(loo)+1),loo.mean_diff_hc_minus_pd_loo,c=colors,s=22)
    mean=float(sens.mean_diff_hc_minus_pd)
    axs[1].axhline(mean,color=COLORS['dark'],lw=1.3,label='Full estimate');axs[1].axhline(0,color='#888',lw=.8)
    axs[1].set_title('Leave-one-participant-out',loc='left',fontweight='bold');axs[1].set_xlabel('Participant omitted (index)');axs[1].set_ylabel('HC − PD mean difference');axs[1].legend(frameon=False,fontsize=7);axs[1].grid(axis='y',alpha=.2)
    for y,row in enumerate(bsum.itertuples(index=False)):
        axs[2].errorbar(row.estimate,y,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt='o',color=COLORS['dark'],capsize=3)
    axs[2].axvline(0,color='#888',lw=.8);axs[2].set_yticks(range(len(bsum)),bsum.summary);axs[2].invert_yaxis();axs[2].set_xlabel('HC − PD difference (95% bootstrap interval)');axs[2].set_title('Alternative summaries',loc='left',fontweight='bold');axs[2].grid(axis='x',alpha=.2)
    fig.suptitle('Figure 5. VI1 shimmer: directional robustness and method-dependent inference',fontsize=13,fontweight='bold')
    fig.text(.5,-.025,f"Original Holm Mann–Whitney p={sens.holm_mw_p_within_cohort_20:.3f}; Holm Yuen p={sens.holm_yuen_p_within_cohort_20:.3f}. All leave-one-out estimates positive; bootstrap intervals do not replace protocol or external validation.",ha='center',fontsize=8)
    savefig(fig,figdir,'fig5_vi1_shimmer_robustness')
    # Figure 6: threshold sensitivity for candidate connected-speech features.
    v=pd.read_csv(ROOT/'results/italian_research_phase2/connected_speech/vad_threshold_sensitivity_long.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    thresholds=[-30,-35,-40]
    metrics=[('active_speech_proportion','Candidate active-speech proportion'),('internal_pause_count','Candidate internal pause count')]
    styles={('B1','ELDERLY_HC'):(COLORS['HC'],'o'),('B1','PD'):(COLORS['PD'],'o'),
            ('B2','ELDERLY_HC'):(COLORS['HC'],'s'),('B2','PD'):(COLORS['PD'],'s'),
            ('FB1','ELDERLY_HC'):(COLORS['HC'],'^'),('FB1','PD'):(COLORS['PD'],'^')}
    for ax,(metric,title) in zip(axs,metrics):
        for (task,group),g in v.groupby(['task_code','group']):
            c,m=styles.get((task,group),(COLORS['other'],'o')); med=[];low=[];high=[]
            for th in thresholds:
                a=g.loc[g.vad_threshold_db_below_p95.eq(th),metric].dropna().to_numpy(float)
                med.append(np.median(a) if len(a) else np.nan);low.append(np.percentile(a,25) if len(a) else np.nan);high.append(np.percentile(a,75) if len(a) else np.nan)
            n=int(g.relative_path.nunique())
            ax.errorbar(range(3),med,yerr=[np.array(med)-np.array(low),np.array(high)-np.array(med)],
                        color=c,marker=m,ms=4,capsize=2,lw=1.1,alpha=.88,label=f'{task} {group} (n={n})')
        ax.set_xticks(range(3),[f'{t} dB' for t in thresholds]);ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Fraction of recording');axs[1].set_ylabel('Pauses per recording')
    axs[1].legend(ncol=2,fontsize=7,frameon=False,loc='best')
    fig.suptitle('Figure 6. Italian connected-speech candidates vary with RMS-VAD threshold',fontsize=13,fontweight='bold')
    fig.text(.5,-.025,'Medians and IQR across recordings; thresholds are dB below each recording’s 95th-percentile RMS. VAD masks are not human-validated; pause counts are candidate only.',ha='center',fontsize=8)
    savefig(fig,figdir,'fig6_italian_vad_threshold_sensitivity')
    return bsum


def make_english_figures(figfeat,mdvrfeat,effects,vad_sensitivity,figdir):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    features=[('median_f0_hz','Median F0 (Hz)'),('jitter_local','Local jitter'),('shimmer_local','Local shimmer'),('hnr_mean_db','HNR (dB)')]
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    valid=figfeat.loc[figfeat.status.eq('ok')]
    for ax,(feature,label) in zip(axs.flat,features):
        for i,group in enumerate(['HC','PD']):
            x=valid.loc[valid.group.eq(group),feature].dropna().to_numpy(float)
            rng=np.random.default_rng(SEED+i+len(label))
            ax.scatter(i+rng.normal(0,.055,len(x)),x,s=32,color=COLORS[group],alpha=.76,edgecolor='white',linewidth=.45)
            if len(x):
                b=np.mean(rng.choice(x,(8000,len(x)),replace=True),axis=1);m=np.mean(x);lo,hi=np.percentile(b,[2.5,97.5])
                ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='D',color=COLORS['dark'],ms=4,capsize=3)
        ax.set_xticks([0,1],[f'HC\nn={int(valid.group.eq("HC").sum())}',f'PD\nn={int(valid.group.eq("PD").sum())}'])
        ax.set_ylabel(label);ax.set_title(label,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Figure E1. Figshare sustained /a/: participant-level acoustic distributions',fontsize=13,fontweight='bold')
    fig.text(.5,-.015,'One local waveform per workbook Sample ID (41 HC, 40 PD); group differences are unadjusted descriptive comparisons, with age/sex-adjusted models tabulated separately.',ha='center',fontsize=8)
    savefig(fig,figdir,'figE1_figshare_vowel_distributions')
    # MDVR task-specific forest plot, visually separate read and spontaneous tasks.
    ef=effects.loc[effects.dataset.eq('MDVR-KCL')].copy()
    if not ef.empty:
        feat_order=['active_speech_fraction','pause_count_candidate','pause_fraction_candidate','median_f0_active_hz','f0_iqr_active_hz','voiced_fraction_active','hnr_median_active_db','relative_intensity_iqr_db','spectral_centroid_active_hz','spectral_rolloff_85_active_hz']
        nice={'active_speech_fraction':'Active proportion','pause_count_candidate':'Pause count','pause_fraction_candidate':'Pause fraction',
              'median_f0_active_hz':'Median F0','f0_iqr_active_hz':'F0 IQR','voiced_fraction_active':'Voiced proportion',
              'hnr_median_active_db':'HNR','relative_intensity_iqr_db':'Intensity IQR','spectral_centroid_active_hz':'Centroid','spectral_rolloff_85_active_hz':'Rolloff'}
        fig,axs=plt.subplots(1,2,figsize=(13,6.8),constrained_layout=True,sharex=False)
        for ax,task in zip(axs,['read_text','spontaneous_dialogue']):
            sub=ef.loc[ef.task.eq(task)].set_index('feature').reindex(feat_order).dropna(subset=['hedges_g_hc_minus_pd'])
            for i,row in enumerate(sub.itertuples()):
                ax.errorbar(row.hedges_g_hc_minus_pd,i,xerr=[[row.hedges_g_hc_minus_pd-row.hedges_g_boot_ci95_low],[row.hedges_g_boot_ci95_high-row.hedges_g_hc_minus_pd]],fmt='o',color=COLORS['dark'],capsize=2,ms=4)
            ax.axvline(0,color='#77838A',lw=.8);ax.set_yticks(range(len(sub)),[nice[x] for x in sub.index]);ax.invert_yaxis()
            ax.set_title('Reading text' if task=='read_text' else 'Spontaneous dialogue',loc='left',fontweight='bold')
            ax.set_xlabel('Hedges’ g (HC − PD)');ax.grid(axis='x',alpha=.2)
        fig.suptitle('Figure E2. MDVR-KCL candidate connected-speech feature effects by task',fontsize=13,fontweight='bold')
        fig.text(.5,-.02,'One recording per participant-task; n=21 HC/16 PD reading and 21 HC/15 PD dialogue. VAD, pitch and HNR depend on mixed-speaker segmentation and require validation.',ha='center',fontsize=8)
        savefig(fig,figdir,'figE2_mdvr_task_effects')
    if not vad_sensitivity.empty:
        fig,axs=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
        ths=[-20,-25,-30,-35]
        marks={'read_text':'o','spontaneous_dialogue':'s'}
        # Plot active proportion and pause counts separately across thresholds and strata.
        for ax,metric,title in [(axs[0],'active_speech_fraction','Candidate active-speech proportion'),
                                (axs[1],'pause_count_candidate','Candidate internal pause count')]:
            for (task,group),g in vad_sensitivity.groupby(['task','group']):
                med=[];lo=[];hi=[]
                for threshold in ths:
                    v=g.loc[g.threshold_db.eq(threshold),metric].dropna().to_numpy(float)
                    med.append(np.median(v) if len(v) else np.nan);lo.append(np.percentile(v,25) if len(v) else np.nan);hi.append(np.percentile(v,75) if len(v) else np.nan)
                color=COLORS['HC'] if group=='HC' else COLORS['PD']
                ax.errorbar(range(len(ths)),med,yerr=[np.asarray(med)-np.asarray(lo),np.asarray(hi)-np.asarray(med)],
                            marker=marks.get(task,'o'),color=color,capsize=2,lw=1.1,ms=4,label=f'{task.replace("_"," ")} {group}')
            ax.set_xticks(range(len(ths)),[f'{x} dB' for x in ths]);ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.2)
        axs[0].set_ylabel('Fraction of recording');axs[1].set_ylabel('Pauses >=250 ms per recording')
        axs[1].legend(ncol=2,fontsize=7,frameon=False)
        fig.suptitle('Figure E3. MDVR candidate VAD summaries are threshold-sensitive',fontsize=13,fontweight='bold')
        fig.text(.5,-.02,'Fixed 25 ms / 10 ms RMS frames on the polyphase-resampled 16 kHz analysis signal; thresholds relative to each file p95. −25 dB is provisional; all masks remain unvalidated.',ha='center',fontsize=8)
        savefig(fig,figdir,'figE3_mdvr_vad_threshold_sensitivity')


def make_cross_figure(it_effects,all_effects,figdir):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    aliases={'median_f0_hz':'median_f0_hz','jitter_local':'jitter_local','shimmer_local':'shimmer_local','hnr_mean_db':'hnr_mean_db'}
    # Only sustained /a/ contrasts are shown. Speech-text tasks are not pooled with phonation.
    parts=[]
    for t in ['VA1']:
        z=it_effects.loc[it_effects.task_code.eq(t)&it_effects.feature.isin(aliases)].copy()
        z['dataset_label']='ItalianPVS VA1 /a/';z['feature_key']=z.feature
        z=z.rename(columns={'hedges_g_hc_minus_pd':'g','hedges_g_ci95_low':'lo','hedges_g_ci95_high':'hi'})
        parts.append(z[['dataset_label','feature_key','g','lo','hi']])
    e=all_effects.copy()
    specs=[('Figshare','sustained_a','Figshare /a/'),('UCI174_feature_table','sustained_vowel_a','UCI 174 /a/'),('UCI301_feature_table','sustained_a','UCI 301 /a/')]
    for ds,task,label in specs:
        z=e.loc[e.dataset.eq(ds)&e.task.eq(task)].copy()
        rename={'median_pitch_hz':'median_f0_hz'}
        z['feature_key']=z.feature.replace(rename);z['dataset_label']=label
        z=z.rename(columns={'hedges_g_hc_minus_pd':'g','hedges_g_boot_ci95_low':'lo','hedges_g_boot_ci95_high':'hi'})
        z=z.loc[z.feature_key.isin(aliases)]
        parts.append(z[['dataset_label','feature_key','g','lo','hi']])
    data=pd.concat(parts,ignore_index=True)
    if data.empty:return
    labs=['median_f0_hz','jitter_local','shimmer_local','hnr_mean_db'];pretty={'median_f0_hz':'Median F0','jitter_local':'Jitter','shimmer_local':'Shimmer','hnr_mean_db':'HNR'}
    ds_order=['ItalianPVS VA1 /a/','Figshare /a/','UCI 174 /a/','UCI 301 /a/']
    colors={ds:col for ds,col in zip(ds_order,['#24333D','#287C8E','#9A7B37','#8C6BAA'])}
    fig,axs=plt.subplots(4,1,figsize=(10,8.6),sharex=True,constrained_layout=True)
    for ax,f in zip(axs,labs):
        z=data.loc[data.feature_key.eq(f)]
        for j,ds in enumerate(ds_order):
            r=z.loc[z.dataset_label.eq(ds)]
            if r.empty:continue
            row=r.iloc[0]
            ax.errorbar(row.g,j,xerr=[[row.g-row.lo],[row.hi-row.g]],fmt='o',color=colors[ds],capsize=3,ms=5,label=ds)
        ax.axvline(0,color='#76838A',lw=.8);ax.set_yticks(range(4),ds_order);ax.invert_yaxis();ax.set_title(pretty[f],loc='left',fontweight='bold');ax.grid(axis='x',alpha=.18)
    axs[-1].set_xlabel('Hedges’ g (HC − PD), with dataset-specific participant bootstrap 95% intervals')
    fig.suptitle('Figure C1. Sustained-vowel feature contrasts across independent cohorts',fontsize=13,fontweight='bold')
    fig.text(.5,-.014,'Directional/effect-size comparison only: the datasets differ in language, microphones, sample rates, feature implementations and cohort composition. UCI tables contain precomputed features; this is not harmonized validation.',ha='center',fontsize=8)
    savefig(fig,figdir,'figC1_sustained_vowel_cross_dataset_effects')


def derive_inventory_and_checks(fig_qc,fig_feat,mdvr_qc,mdvr_feat,ox,oxcheck,oxsub,tm,tm_sub,u301,u301test):
    inventory=pd.DataFrame([
        ['Figshare voice samples',True,81,81,81,41,40,0,'sustained /a/','8 kHz mono FLOAT WAV','participant age/sex/label in xlsx; one Sample ID per local file','CC BY 4.0 (item page)','raw audio usable; vowel-only task; language not explicitly specified','Source item 23849127; local IDs match workbook, one file per ID'],
        ['MDVR-KCL',True,73,73,37,21,16,0,'reading text + spontaneous phone dialogue','44.1 kHz mono PCM_24 WAV','participant IDs and PD scores partly encoded in filenames; age not locally available','Zenodo item; CC BY 4.0 source records','raw audio usable; task speaker diarization not present','One anomalous ID22 dialogue filename; one PD participant lacks local reading file'],
        ['UCI Parkinsons / Oxford #174',True,195,0,32,8,24,0,'sustained vowel voice measurements; exact prompt not in local table','precomputed CSV; no audio','name field contains participant/record IDs; group constant by participant','UCI CC BY 4.0','source page says 31 people/23 PD; local values contain 32/24 PD and 8 HC','recording-level rows aggregated to participant medians'],
        ['UCI Parkinson’s Telemonitoring #189',True,5875,0,42,0,42,0,'home telemonitoring voice measures; prompt documentation limited','precomputed CSV; no audio','subject number, age, sex, time, UPDRS and features','UCI CC BY 4.0','PD-only severity monitoring; not binary diagnostic dataset','two local CSV copies verified by checksum'],
        ['UCI Parkinson Speech Multiple Types #301',True,1040,0,40,20,20,0,'3 vowels, 10 numbers, 4 sentences, 9 words','precomputed feature RAR; no raw audio','subject ID/task order/class/UPDRS in training archive','UCI CC BY 4.0','local archive includes 1,040 train rows + 168 PD-only vowel-repeat test rows; language not explicitly documented by the UCI page','subject-task aggregation used for inference; independent test portion kept separate'],
        ['UCI Parkinson’s Disease Classification #470',False,0,0,0,np.nan,np.nan,0,'source describes sustained /a/ x3','not locally available; UCI publishes 756×754 feature data','source records contain participant IDs; not inspected locally','UCI CC BY 4.0','no local raw audio or feature file located','documentation reviewed only; no download performed'],
        ['mPower voice',False,0,0,np.nan,np.nan,np.nan,0,'mobile phonation / voice tasks; exact subtask depends on data release','no local audio/data found','coded IDs accessible after account/data access steps','Synapse controlled/qualified research access; terms and data use statement required','not available in local workspace; no credentialed access attempted','not analyzed; account/permission required']
    ],columns=['dataset','local_data_present','recordings_or_feature_rows_local','readable_raw_audio_local','participants_observed_or_reported','hc_participants','pd_participants','other_groups','task','local_format_or_rate','metadata_and_ids','license_or_access','suitability_and_unknowns','local_audit_note'])
    # Reconcile local counts for datasets with audio and feature data.
    inventory.loc[inventory.dataset.eq('Figshare voice samples'),'recordings_or_feature_rows_local']=len(fig_qc)
    inventory.loc[inventory.dataset.eq('Figshare voice samples'),'readable_raw_audio_local']=int(fig_qc.readable.sum())
    inventory.loc[inventory.dataset.eq('Figshare voice samples'),'participants_observed_or_reported']=fig_feat.participant_id.nunique()
    inventory.loc[inventory.dataset.eq('MDVR-KCL'),'readable_raw_audio_local']=int(mdvr_qc.readable.sum())
    inventory.loc[inventory.dataset.eq('MDVR-KCL'),'participants_observed_or_reported']=mdvr_qc.participant_id.nunique()
    inventory.loc[inventory.dataset.eq('UCI Parkinsons / Oxford #174'),'participants_observed_or_reported']=oxcheck.participant_id.nunique()
    inventory.loc[inventory.dataset.eq('UCI Parkinsons / Oxford #174'),'pd_participants']=int((oxcheck.group=='PD').sum())
    inventory.loc[inventory.dataset.eq('UCI Parkinsons / Oxford #174'),'hc_participants']=int((oxcheck.group=='HC').sum())
    inventory.to_csv(TABLEE/'english_dataset_inventory.csv',index=False)
    # English task-level recording/participant coverage.
    cov=[]
    for name,qc in [('Figshare',fig_qc),('MDVR-KCL',mdvr_qc)]:
        for (task,group),g in qc.groupby(['task','group']):
            cov.append({'dataset':name,'task':task,'group':group,'recordings':len(g),'participants':g.participant_id.nunique(),
                        'sample_rate_hz':int(g.sample_rate_hz.median()),'recordings_readable':int(g.readable.sum())})
    coverage=pd.DataFrame(cov);coverage.to_csv(TABLEE/'english_task_participant_coverage.csv',index=False)
    # Feature extraction methods and explicit deferrals.
    methods=pd.DataFrame([
        ['Figshare sustained /a/','median_f0_hz','Praat autocorrelation; voiced stable interval','75–500 Hz; 10 ms step; longest voiced run; 10% end trim','candidate; pitch-floor sensitivity needed'],
        ['Figshare sustained /a/','jitter_local','Praat local cycle period perturbation','stable interval; 0.0001–0.02 s; period factor 1.3','cycle validity depends on stable periodic phonation'],
        ['Figshare sustained /a/','shimmer_local','Praat local cycle amplitude perturbation','stable interval; 0.0001–0.02 s; period factor 1.3; amplitude factor 1.6','sensitive to phone gain/noise; exploratory'],
        ['Figshare sustained /a/','hnr_mean_db','Praat CC harmonicity; average valid frames','10 ms; minimum pitch 75 Hz; silence threshold .1; one period/window','bandwidth/device effects; exploratory'],
        ['MDVR reading/dialogue','speech activity/pause candidates','25 ms RMS frames, 10 ms hop; provisional −25 dB relative to each file p95; sensitivity at −20/−25/−30/−35; close 10 ms holes; pauses >=250 ms','original 44.1 kHz PCM24 downsampled with polyphase to 16 kHz for analysis; no transcript','not speaker-specific and not human-validated; downsampling does not restore bandwidth'],
        ['MDVR reading/dialogue','F0 and voicing','Praat autocorrelation in active RMS-VAD frames','16 kHz analysis; 75–500 Hz; 10 ms step','speaker mixture possible in dialogue; no individual voice attribution'],
        ['MDVR reading/dialogue','HNR','Praat CC harmonicity within candidate active windows','16 kHz analysis; 10 ms; min pitch 75 Hz; periods/window 1','candidate voiced-active frames only; human review absent'],
        ['MDVR reading/dialogue','relative intensity dynamics','IQR of active-frame RMS in dB','16 kHz analysis; 25 ms; 10 ms hop; relative within recording','not calibrated sound pressure'],
        ['MDVR reading/dialogue','jitter/shimmer/formants','not extracted','no stable utterance-level segmentation or validated speaker-only periods','deferred'],
        ['All English connected speech','CPP/CPPS','not extracted','no validated implementation in this pipeline','deferred']
    ],columns=['task','feature','definition','settings','validity_and_limits'])
    methods.to_csv(TABLEE/'english_feature_extraction_summary.csv',index=False)
    return inventory,coverage,methods


def reports_and_tables(context):
    (finalout:=PKG).mkdir(parents=True,exist_ok=True)
    # Direct copies of final tables for manuscript-ready location.
    for p in (EXP).glob('*.csv'):
        shutil.copy2(p,TABLEE/p.name)
    # Populate reports after all verified counts/statistics are available.
    info=context
    effects=info['effects']; italian=info['italian_effects']; adj=info['adjusted']; tele=info['tele']; repeat=info['repeat']
    inventory=info['inventory']; coverage=info['english_coverage']; itcoverage=info['it_coverage']; evidence=info['evidence']
    srow=info['vi'].iloc[0]
    figstats=effects.loc[effects.dataset.eq('Figshare')]
    u174=effects.loc[effects.dataset.eq('UCI174_feature_table')]
    u301a=effects.loc[effects.dataset.eq('UCI301_feature_table')&effects.task.eq('sustained_a')]
    mdread=effects.loc[effects.dataset.eq('MDVR-KCL')&effects.task.eq('read_text')]
    mdsp=effects.loc[effects.dataset.eq('MDVR-KCL')&effects.task.eq('spontaneous_dialogue')]
    # Italian synthesis
    main=italian.loc[italian.holm_p_primary_20_comparisons.lt(.05)]
    main_result=main.iloc[0] if len(main) else None
    it_cols=['task_code','feature','n_hc','n_pd','mean_difference_hc_minus_pd','mean_diff_ci95_low','mean_diff_ci95_high',
             'hedges_g_hc_minus_pd','hedges_g_ci95_low','hedges_g_ci95_high','mann_whitney_p_unadjusted','holm_p_primary_20_comparisons']
    italian_effect_table=md_table(italian[it_cols],precision=4)
    engcols=['dataset','task','feature','n_hc','n_pd','mean_difference_hc_minus_pd','mean_diff_boot_ci95_low','mean_diff_boot_ci95_high',
             'hedges_g_hc_minus_pd','hedges_g_boot_ci95_low','hedges_g_boot_ci95_high','mann_whitney_p','holm_adjusted_p']
    ecols=[c for c in engcols if c in effects.columns]
    english_effect_table=md_table(effects[ecols],precision=4)
    adj_table=md_table(adj[['feature','n','adjusted_pd_minus_hc','hc3_ci95_low','hc3_ci95_high','p_value_unadjusted','holm_p_within_task_4_features']],precision=4)
    significant=effects.loc[effects.holm_adjusted_p.lt(.05)] if 'holm_adjusted_p' in effects else effects.iloc[0:0]
    significant_text='No analysis in the separate exploratory families survived their task-specific Holm correction.' if significant.empty else 'Corrected signals: '+', '.join(f"{r.dataset}/{r.task}/{r.feature} (g={r.hedges_g_hc_minus_pd:.2f}, adjusted p={r.holm_adjusted_p:.3f})" for r in significant.itertuples())+'. These remain exploratory.'
    sections=f'''# ItalianPVS research synthesis

**Synthesis date:** 31 October 2026. **Analysis sources:** preserved experiments `ITALIAN-MULTITASK-V1-20261030` and `ITALIAN-PHASE2-REMAINING-20261030`; this synthesis did not rerun historical feature extraction or alter their outputs.

## What we investigated

The work examined the reconciled elderly healthy-control (HC) versus Parkinson’s disease (PD) cohort, task coverage and sample-rate restrictions; five sustained vowels (VA1/VE1/VI1/VO1/VU1); second-vowel signal structure; D1/D2 `/pa/` and `/ta/` recordings; B1/B2 reading and FB1 phrase recordings; a 29-person VA1 cross-validation baseline; and sensitivity of the main vowel results to participant influence, alternative summaries and resampling.

Primary group comparisons use native 16 kHz: **17 HC and 12 PD**. The first-vowel table has 145 task-participant rows (29 people per vowel). The mixed-rate reconciled cohort has 17 HC and 19 PD, but the additional seven PD participants with 44.1 kHz recordings do not have HC counterparts at that rate.

## Measurements and principal results

The sustained-vowel pipeline measured median F0, F0 IQR/SD, voiced fraction, local jitter, local shimmer, HNR, F1–F3 tracks and spectral centroid, 85% rolloff and flatness. The prespecified primary family was median F0, local jitter, local shimmer and HNR in each of five vowels (20 tests). Vowel contexts are separate phonetic conditions; they are not repeatability trials.

**VI1 local shimmer is the only primary-family contrast surviving the original Holm correction** (HC-minus-PD mean difference 0.04439, Hedges’ g 0.889, historical participant-bootstrap 95% interval [0.01853, 0.07745], Mann–Whitney p=0.000600, Holm p=0.01199; 17/12 participants). It suggests a group-associated difference in cycle-to-cycle amplitude perturbation during the Italian `/i/` sustained task in this sample. The direction is notable because it persists when any one participant is omitted (range 0.03335–0.04739) and under median/trimmed/winsorized summaries; independent percentile and BCa participant-bootstrap intervals are positive. However, the 20%-trimmed Yuen test’s Holm-adjusted p≈0.061 does not cross .05. It is an exploratory signal, not a definitive biological or diagnostic effect. Present participant dots, HC-minus-PD estimate, bootstrap intervals, leave-one-out estimates, and both adjusted inferential results together (Figure 5); avoid a single p-value headline.

Several other unadjusted vowel-specific contrasts are suggestive, including lower HNR in VI1 (mean HC-minus-PD ≈−7.18 dB; primary-family Holm p≈0.0815), and shimmer estimates in VA1, VO1 and VU1. These do not survive the 20-test correction. Median F0 group estimates are imprecise, and the other primary contrasts are inconclusive. The full table supplies means, participant-level effect sizes, intervals and adjusted p-values.

### Prespecified 20-comparison family

{italian_effect_table}

### Evidence classification

**Stronger exploratory evidence:** VI1 shimmer has a large observed standardized contrast and stable positive direction across individual deletion and alternative summaries. Its adjusted inferential status is analysis-dependent.

**Suggestive:** selected HNR/shimmer vowel contrasts and the pattern that voice measures depend on vowel context. These estimates have wide uncertainty, multiple comparisons and small groups; they do not establish disease-specific effects.

**Inconclusive:** F0 contrasts and most primary tests; external validation of any Italian association; resampled group contrasts because extra observations are PD-only.

**Requires human validation:** formant overlays and the open 159-recording review queue; D1/D2 onset segmentation; connected-speech VAD masks.

**Do not include as main inferential results:** automated D1/D2 event counts/intervals, connected-speech pause or speech-activity estimates, second-vowel paired differences as repeatability, frame-level observations as independent n, or classifier metrics as clinical evidence.

## Tasks and scientific contribution

VA1/VE1/VI1/VO1/VU1 are the only tasks with participant-level acoustic group analysis and matched per-vowel coverage. The VA1 cross-validation reference used 29 participants and reported BA 0.900, AUC 0.917, sensitivity 0.917 and specificity 0.882. A later uncertainty analysis on the fixed saved OOF predictions gives BA 95% interval [0.775,1.000] and AUC [0.789,1.000]; it conditions on fitted fold predictions and omits model refitting/fold uncertainty. This is internal exploratory cross-validation, not external validation or clinical diagnostic performance.

Second vowels have markedly shorter duration distributions than first variants, but blank protocol descriptions prevent a repetition or reliability interpretation. Most D1/D2 plots show repeated CV-like bursts, not enough to validate syllable counts, voice-onset time or per-participant timing. B1/B2/FB1 generic connected-speech features are threshold-dependent candidates, not validated speech-activity measures; FB1 materials are unavailable, and no transcript-based rate/alignment was attempted.

The contribution is a transparent, participant-level, multi-task characterization that establishes coverage and where the data do and do not support inference; it identifies an exploratory VI1 shimmer association and quantifies its sensitivity instead of treating model CV as the only result.

## What the Results Actually Tell Us

Cycle amplitude perturbation in Italian `/i/` deserves a targeted independent replication. It is not yet a general PD voice marker: the effect may depend on vowel, sample composition, age/sex, recording device, or analysis choice. HNR in `/i/` is also worth checking, but it did not pass the prespecified family correction. More vowel contexts did not produce more validated evidence. The second variants currently add protocol uncertainty rather than test-retest information.

Connected speech may eventually add timing and prosody information, but current VAD settings materially change pauses and active proportions; there is no validated segmentation or speaker attribution. Until reviewed, it adds diagnostic plots and candidate variables, not a defensible inferential claim. Avoid developing classifiers or large feature sets on these cohorts. Deprioritize text-specific analyses until verified materials and alignments are available.

## Recommended Next Research Experiments

1. **Independent sustained `/i/` shimmer replication** (highest scientific value; moderate feasibility): collect a larger, demographically matched cohort using controlled microphone distance, fixed sample rate, repeat tokens and blinded quality review; prespecify one primary shimmer definition and interval. Include `/a/` as a planned contrast, not a fishing family.
2. **Italian audio adjudication and task validity** (high feasibility): complete the open review queue; annotate D1/D2 onsets and connected-speech masks; estimate inter-rater agreement and detector error before any participant summaries.
3. **Protocol recovery for vowel variants and reading** (low cost): obtain the collection sheet, exact texts and sample-rate/hardware details; only then decide which paired or text-dependent analyses are meaningful.
4. **External data access for mPower or an explicitly English task dataset** (moderate effort, access-dependent): run a preregistered protocol-matched sustained-phonation study only after access, language/task, label and demographic eligibility are verified.
5. Reconsider model evaluation only after validated measures and adequate independent participant counts exist; participant-level nested/external validation should precede any deep model.

## Limitations

n=17/12 at native 16 kHz; no power-based confirmatory claim is established. Age/sex and device/session differences may confound group contrasts. All additional 44.1 kHz task recordings are PD. Repeated extraction verifies software determinism, not biological reliability. Formant broad ranges do not constitute expert validation. 159 manual reviews are open. D1/D2 protocol/count, second vowel definitions, FB1 text, FB2 absence, PR1/PR11 and some roster identities remain unresolved. The resampling sensitivity does not remove sample-rate/group confounding. No clinical claims are supported.

## Data and figure notes

The participant-flow and task-coverage figures use the frozen manifests and reconciliation. Acoustic distributions and per-vowel effects use the saved native-16-kHz table and historical comparison table. The VI1 robustness figure joins the new sensitivity audit. The Italian connected-speech figure uses unadjudicated −30/−35/−40 dB RMS-VAD summaries; the MDVR figure uses −20/−25/−30/−35 dB sensitivity summaries with a provisional −25 dB feature mask. Both remain explicitly exploratory.

References and protocol sources are listed in the [manuscript draft](italian_manuscript_draft.md). The original published dataset/protocol sources include Dimauro et al. (2016, 2017) and the 2022 task paper; see the cited DOI links in the manuscript.
'''
    (PKG/'italian_results.md').write_text(sections)
    # English dataset inventory and evidence
    (PKG/'english_dataset_inventory.md').write_text(f'''# English-language and related voice datasets: local audit

Audit date: 31 October 2026. Counts below were verified from local audio, workbook rows, archive contents, and UCI source descriptions. “Recordings” are not treated as participants.

## Local datasets and disposition

{md_table(inventory[['dataset','local_data_present','recordings_or_feature_rows_local','readable_raw_audio_local','participants_observed_or_reported','hc_participants','pd_participants','task','local_format_or_rate','license_or_access']])}

**Figshare:** the local directory has 81 readable mono 8 kHz FLOAT WAV files (41 HC / 40 PD by folder). The workbook has 81 unique Sample IDs and labels/ages/sexes. Every filename ID matches one row and group; thus the current local copy represents 81 distinct coded participants, one file each. The record identifies prolonged vowel `/a/` recordings made by participants’ telephones; it does not state a linguistic-language task. The current Figshare item page lists **CC BY 4.0**; attribute the dataset and source publication. Age and sex are present but groups may not be balanced on them. The source report identifies 81 recordings, not an explicit unique-participant figure, so this local one-to-one reconciliation supports n=81 for this copy only. [Figshare record](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127); [source article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11972358/).

**MDVR-KCL:** 73/73 files decode as mono 44.1 kHz PCM_24. Local folder labels and filename IDs produce 37 participants (21 HC/16 PD). There are 37 reading files and 36 dialogue files: all 21 HC have both local tasks; 15 PD have both; one PD has only reading. The dialogue filename `ID22hc_0_0_0.wav` lacks a separator; its ID was normalized to ID22 using digits and the HC folder label, with anomaly retained. The source documentation gives reading (one of two text materials) and spontaneous phone dialogue; the exact reading text assignment/wording and demographics are not present in the local files. Filename strings encode PD severity items, not age/sex. Analysis kept reading and dialogue separate, and dialogue measures are especially vulnerable to examiner voice. [Zenodo record](https://doi.org/10.5281/zenodo.2867216); [task descriptions in source study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934449/).

## UCI datasets are not interchangeable

- **UCI 174, Oxford Parkinsons:** local `parkinsons.data` has 195 feature rows. Parsing the `name` key yields 32 participants (24 PD, 8 HC), with status constant within person. UCI’s page says 31 participants/23 PD while its row count is 197 in one page field and 195 in its text; the local actual file resolves to 32/24/8 and 195. It contains precomputed sustained-voice measurements (including F0, jitter, shimmer, HNR) and no raw audio. We aggregated repeated rows to one median per participant. This is usable as a cautious feature-table comparison, not acoustic re-extraction. [UCI 174 record](https://archive.ics.uci.edu/dataset/174/parkinsons); a published audit also reports the distributed 32-ID discrepancy ([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6789029/)).
- **UCI 189, Telemonitoring:** local dataset has 5,875 precomputed recording rows for 42 PD participants, longitudinal test time, age/sex and interpolated motor/total UPDRS targets. There are no HC labels and no audio. It is a separate severity/progression resource, not a binary diagnostic dataset. The two local copies were checksum-identical. We computed per-participant longitudinal feature/UPDRS correlations as supplementary descriptive evidence, never row-wise diagnosis tests. [UCI 189](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).
- **UCI 301, Parkinson Speech with Multiple Types:** `uci_csv2.rar` contains training feature table 1,040×29 (40 participants, 20 PD/20 HC, 26 task rows each) and an independent 168-row PD-only test table (28 IDs × 6 vowel repetitions). The order gives sustained `/a/`, `/o/`, `/u/`, numbers, sentences and words. It is a feature archive, not raw audio. The original cohort was collected in Istanbul; task-language metadata are not explicit on the UCI page, so it is not treated as English. Training inference aggregates the participant-task row; the PD-only test is held separate and used only for within-person repeat spread. [UCI 301](https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings); [original study DOI](https://doi.org/10.1109/JBHI.2013.2245674).
- **UCI 470, Parkinson’s Disease Classification:** the UCI page describes 188 PD/64 HC, 756 recordings, 44.1 kHz acquisition and 754 precomputed features from three sustained `/a/` phonations/person. No local archive, table or audio was found, so it was not included in analysis or downloaded. [UCI 470](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification).
- **mPower:** no local data were found. Voice/phonation records are available only to qualified Synapse users who complete access requirements and accept the applicable data-use terms. No account-dependent access attempt or data download was made. It is a plausible future sustained-phonation source only after controlled access and label/task inspection. [Synapse mPower voice activity](https://www.synapse.org/Synapse:syn5511444/tables/); [access description](https://pmc.ncbi.nlm.nih.gov/articles/4776701/).

### Other local-file checks

`kaggle/telemonitoring/parkinsons_updrs.data` and `UCI_csv/parkinsons_updrs.data` are byte-identical copies of UCI 189. The file `kaggle/parkinsons.data` is UCI 174, not telemonitoring. The RAR archive is UCI 301, not UCI 470. No `uci_csv2` files were extracted into the shared data directory.

## Raw-audio suitability

Suitable for independent **exploratory** raw-audio work: Figshare sustained `/a/` and MDVR-KCL read/dialogue tasks. Figshare IDs are resolved one-to-one in the local workbook, although demographic confounding and 8 kHz telephone processing limit interpretation. MDVR has modest n and incomplete PD reading coverage; dialogue has a second speaker and no diarization. Neither dataset supports clinical validation. UCI 174/189/301 can support distinct participant-level analyses on precomputed features, but cannot test a common raw-audio pipeline.
''')
    # English results
    (PKG/'english_results.md').write_text(f'''# Independent English and related-dataset analyses

Experiment ID: `ENGLISH-VOICE-DATASET-AUDIT-20261031`. The experiment is isolated in `results/english_research_20261031/`; local source data were read only. Raw-audio analysis covered Figshare sustained `/a/` and MDVR-KCL reading/dialogue. UCI tables and archive data were analyzed as a separate precomputed-feature investigation. No classifier was fitted.

## Acoustic findings from raw audio

### Figshare sustained `/a/`

The workbook-to-file join verified 81 unique Sample IDs (41 HC/40 PD), one waveform/person, with group labels consistent between workbook and directory. We extracted median F0, F0 IQR, voiced fraction, Praat local jitter, local shimmer and mean HNR in the longest candidate voiced segment (75–500 Hz pitch range; 10 ms step; 10% trimmed at both ends). No participant was excluded based on QC. Group difference estimates, participant bootstrap intervals and four-feature Holm correction are in the table below; age/sex-adjusted HC3 estimates are reported separately. Given 8 kHz telephone waveforms, group-age/sex imbalance risk, small n, and unverified protocol duration, any difference is dataset-specific exploratory evidence.

### MDVR-KCL reading and spontaneous dialogue

All 73 files were readable: 21 HC/16 PD reading and 21 HC/15 PD dialogue; one PD reading file is absent and one dialogue filename has a separator anomaly. Original 44.1 kHz waveforms were polyphase-resampled to 16 kHz for analysis. Candidate connected-speech measurements used 25 ms RMS frames, 10 ms hop, provisional −25 dB threshold below each file’s p95 RMS, and a 10 ms gap closure. Sensitivity at −20/−25/−30/−35 dB is reported; thresholds substantially alter candidate activity and pause estimates, and none of the masks has been human-validated. Features were active-speech fraction, pauses >=250 ms, pause fraction, active-window median F0/IQR and voiced fraction, active-window HNR, relative RMS IQR, spectral centroid and 85% rolloff. Group tests use one participant-task row per person and Holm-adjust each task’s feature family. There is no transcript or human VAD/speaker diarization. The examiner’s voice may appear in dialogue, so its F0/HNR/activity summaries are not confidently participant-specific. Read and dialogue results are not combined.

**Figure E2 reports candidate effect sizes, not validated speech biomarkers.** The analysis is not powered to distinguish subtle effects; p-values are exploratory, and wide effect intervals should guide any interpretation. Jitter/shimmer and formants were not extended to full connected-speech recordings.

### Participant-level group effects

{english_effect_table}

Task-specific Holm-corrected findings: {significant_text}

### Figshare age/sex-adjusted association checks

{adj_table}

## Supplementary precomputed-feature analyses

UCI 174’s local 195 rows resolve to 32 subject IDs (24 PD/8 HC), not the repository text count of 31/23/8. We summarized repeated rows by participant median before group tests. This retains participant as the unit, but the discrepancy needs upstream clarification before publication claims or re-use of labels. UCI 301 train data support participant-level contrasts across `/a/`, `/o/`, `/u/` (20 PD/20 HC); the independent test file contains 28 PD participants and three repetitions each for `/a/` and `/o/`, allowing only a within-PD repeat-spread summary. These Turkish-clinic/no-explicit-language, precomputed-feature results are not English replication and are not merged with raw-audio features.

UCI 189 provides no HC group and no audio. For each of 42 PD participants, we computed within-person Spearman correlations between four voice features and motor/total UPDRS over repeated visits, then summarized correlations across people. These are supplementary severity association summaries; they do not estimate diagnostic differences or raw-audio effects.

## What the English evidence does and does not show

The strongest data assets are the one-to-one Figshare sustained-vowel set and the two-task MDVR set. They support independent group-level descriptive effect estimation at the participant level. The local analysis identifies candidate group directions, but the cohorts are small, recordings are observational and protocol/channel differences are large. The provided precomputed UCI datasets add independent participant structures and task-specific features, not repeatable common measurements. We found no dataset adequate for externally validating Italian VI1 `/i/` shimmer: Figshare and UCI 174 provide `/a/`, UCI 301 provides `/a,o,u/`, and MDVR has no sustained `/i/` task.

No frame-, task-row-, or repeated recording-level inferential pseudo-replication was used. No pooled cross-dataset classifier was trained. See the inventory, effect tables, audio QC and reproducibility report for complete counts and methods.
''')
    # structured package manuscripts and cross-language report written below.
    return


def create_additional_deliverables(context,hash_before,hash_after):
    effects=context['effects']; italian=context['italian_effects']; inventory=context['inventory']
    coverage=context['english_coverage']; features=context['english_features']; qc=context['english_qc']
    tele=context['tele']; repeat=context['repeat']; source_integrity=hash_before==hash_after
    # Cohort characteristics for local English audio and UCI 189, without deriving missing demographics.
    figmeta=pd.read_excel(FIGSHARE/'Demographics_age_sex.xlsx',sheet_name='Parselmouth').rename(
        columns={'Sample ID':'participant_id','Label':'group','Age':'age','Sex':'sex'})
    figmeta['participant_id']=figmeta.participant_id.astype(str)
    figmeta['group']=figmeta.group.replace({'PwPD':'PD'})
    chars=[]
    for group,g in figmeta.groupby('group'):
        chars.append({'dataset':'Figshare','group':group,'participants':g.participant_id.nunique(),'age_mean':g.age.mean(),
                      'age_sd':g.age.std(ddof=1),'age_median':g.age.median(),'age_q1':g.age.quantile(.25),'age_q3':g.age.quantile(.75),
                      'female_n':int(g.sex.eq('F').sum()),'male_n':int(g.sex.eq('M').sum()),'unknown_fields':''})
    for dataset,group,n,unknown in [('MDVR-KCL','HC',21,'age/sex absent from local bundle'),('MDVR-KCL','PD',16,'age/sex absent from local bundle'),
                                    ('UCI174 feature table','HC',8,'age/sex absent from local feature table'),('UCI174 feature table','PD',24,'age/sex absent from local feature table'),
                                    ('UCI301 training table','HC',20,'age/sex not in local archive; UCI reports sex counts'),('UCI301 training table','PD',20,'age/sex not in local archive; UCI reports sex counts'),
                                    ('UCI189 telemonitoring','PD',42,'HC group absent')]:
        chars.append({'dataset':dataset,'group':group,'participants':n,'age_mean':np.nan,'age_sd':np.nan,'age_median':np.nan,'age_q1':np.nan,'age_q3':np.nan,'female_n':np.nan,'male_n':np.nan,'unknown_fields':unknown})
    chars=pd.DataFrame(chars);chars.to_csv(TABLEE/'participant_cohort_characteristics.csv',index=False)
    # Task overlap for MDVR, with the anomalous filename normalization documented.
    md=qc.loc[qc.dataset.eq('MDVR-KCL')]
    piv=md.pivot_table(index=['group','participant_id'],columns='task',values='readable',aggfunc='first').reset_index()
    for task in ['read_text','spontaneous_dialogue']:
        if task not in piv:piv[task]=False
    piv['has_both_tasks']=piv.read_text.fillna(False)&piv.spontaneous_dialogue.fillna(False)
    piv.to_csv(TABLEE/'mdvr_participant_task_overlap.csv',index=False)
    # Cross-dataset compatibility at the feature/task level.
    compat=pd.DataFrame([
        ['Median F0','Yes: Praat pitch on stable vowel; 75–500 Hz','Yes: same Praat candidate stable-vowel definition','Yes: active voiced frames','Yes: active voiced frames','Precomputed MDVP:Fo; definition differs','Precomputed median pitch; source task conditions differ','Task and sample rate comparable only for sustained phonation; utterance-level meanings differ'],
        ['F0 variability','Yes: IQR/SD positive-pitch frames','Yes: IQR/SD positive-pitch frames','Yes: voiced active-frame IQR','Yes: voiced active-frame IQR','Range/variability columns; definition differs','Pitch SD/range features; precomputed','Feature family compatible; numerical harmonization requires exact definitions'],
        ['Jitter','Praat local jitter','Same Praat local jitter','Not extracted: connected speech','Not extracted: speaker diarization absent','MDVP jitter, repeated-record median','Dataset local jitter feature; original algorithm/protocol','Candidate compatible within sustained vowel only; not a numeric replication without source methods'],
        ['Shimmer','Praat local shimmer','Same Praat local shimmer','Not extracted','Not extracted','MDVP shimmer; participant median','Local shimmer feature in precomputed archive','Potential sustained-vowel association comparison; differing hardware/extractor confounds'],
        ['HNR','Praat CC HNR','Praat CC HNR','Praat CC in active voiced frames','Praat CC; mixed voices possible','MDVP HNR','Historical HTN/NTH not identical to HNR','Same concept, but bandwidth/window/implementation differ for feature-table sources'],
        ['Formants','Praat Burg F1–F3','Not extracted; 8 kHz bandwidth limits upper formants','Not extracted','Not extracted','Not available in local raw audio','Not available as verified raw measurements','No cross-dataset formant comparison'],
        ['Spectral centroid/rolloff/flatness','Frame spectra on stable vowel, 16 kHz source','Not used in primary cross-dataset comparison','Candidate frame spectra at 44.1 kHz','Candidate frame spectra at 44.1 kHz','Not raw/reproducible','No exact equivalent specified','Acquisition bandwidth and task strongly limit comparison'],
        ['Speech activity / pauses','Candidate RMS-VAD; threshold sensitivity; review open','Not applicable to sustained vowel','Candidate RMS-VAD, −25 dB provisional; no manual validation','Candidate RMS-VAD, examiner voice possible','No audio','No task-resolved raw measure available','Not comparable across different prompts, segmentation and speaker content'],
        ['Jitter/shimmer variability reliability','No test-retest; deterministic extraction only','One file/person; no repeat task','Repeated task recordings absent','No clean participant-only turns','Repeated records but no raw audio','UCI301 PD test has 3 same-vowel feature rows/person','UCI301 repeat spread is within-PD feature-table stability only; not biological day-to-day reliability']
    ],columns=['feature_family','ItalianPVS','Figshare','MDVR_read_text','MDVR_spontaneous','UCI174','UCI301','interpretation'])
    compat.to_csv(PKG/'cross_language_feature_compatibility.csv',index=False)
    # Tables referenced by manuscript / paper-ready package.
    if 'holm_adjusted_p' in effects:
        effects.loc[:,['dataset','task','feature','n_hc','n_pd','mean_hc','mean_pd','mean_difference_hc_minus_pd','mean_diff_boot_ci95_low','mean_diff_boot_ci95_high','hedges_g_hc_minus_pd','hedges_g_boot_ci95_low','hedges_g_boot_ci95_high','mann_whitney_p','holm_adjusted_p']].to_csv(TABLEE/'participant_level_group_effects.csv',index=False)
    # Cross-language report with nuanced separation of association, compatibility and validation.
    it_aa=italian.loc[italian.task_code.eq('VA1')&italian.feature.isin(VOICE_FEATURES)].copy()
    eng_aa=effects.loc[((effects.dataset.eq('Figshare')&effects.task.eq('sustained_a'))|
                        (effects.dataset.eq('UCI174_feature_table')&effects.task.eq('sustained_vowel_a'))|
                        (effects.dataset.eq('UCI301_feature_table')&effects.task.eq('sustained_a')))]
    ctext=f'''# Italian–English and cross-dataset comparison

## Evidence boundary

This is a comparison of independent, task-labelled cohort estimates, not a pooled model and not proof of language transfer. Only ItalianPVS, Figshare and MDVR-KCL have raw local audio; Figshare is a sustained `/a/` telephone task with language not explicitly identified in its metadata, while MDVR-KCL includes English reading and spontaneous dialogue. UCI 174, 189 and 301 are precomputed feature tables; UCI 301 is from an Istanbul clinic and its exact task language is not identified in the UCI record. UCI 470 and mPower were documented but are not locally analyzed.

## Feature compatibility matrix

{md_table(compat,precision=2)}

## Sustained-vowel effect directions

Figure C1 compares standardized HC-minus-PD effects for `/a/`-labelled tasks across ItalianPVS VA1, Figshare, UCI 174 and UCI 301. This figure visualizes direction and uncertainty only. ItalianPVS / Figshare were extracted with matching local Praat stable-vowel settings, but differ in language/phonetic realization, 16 vs 8 kHz bandwidth, phone/hardware and demographics. UCI features use historical software definitions and participant aggregation, so their intervals are not a raw-audio harmonization.

The Italian VI1 `/i/` shimmer observation has no direct task-matched replication in the local English datasets. The Figshare corpus and UCI 174 provide `/a/`; UCI 301’s training data provide `/a,o,u/`; MDVR has no sustained `/i/`. A same-name shimmer feature for a different vowel is not a replication of the VI1 contrast.

## What is and is not consistent

**Acoustic feature compatibility:** median F0, local jitter, local shimmer and HNR can be defined consistently on sustained vowels in raw audio, but absolute values remain channel, bandwidth and implementation dependent. For the two raw-audio `/a/` corpora, the same Praat definition is applied here. UCI table definitions are not identical enough for pooled absolute values. Pause/active-speech features share a broad concept but are not interchangeable across task materials and VAD validity.

**Directional consistency:** inspect the plotted signs and table values per feature. A shared sign across independent tables is only directional agreement, especially where language/task, age and recording methods differ. The precomputed UCI sets have independent participants but historical feature extraction and task metadata limitations.

**Effect-size consistency:** standardized effects and intervals are displayed by dataset; overlap/non-overlap of small-sample intervals is descriptive and does not establish equivalence. We do not rank unrelated tasks.

**Generalization:** no feature set can validate cross-language transfer in this work package. The small and task-mismatched corpora do not support a pooled classifier or dataset-held-out participant experiment with common task, labels and sufficient covariate balance. mPower may provide a future controlled-access sustained phonation cohort after task and consent filtering.

## Recommended cross-language next step

Pre-register a sustained-vowel protocol that captures `/a/` and `/i/` (plus repeated tokens) in Italian and English speakers using the same microphone, calibrated distance, sample rate, quiet-room instructions and feature code. Balance or model age/sex, retain participant identifiers, and reserve a completely independent cohort for replication. Treat `/i/` shimmer as a targeted hypothesis and `/a/` as the bridge task. Do not attempt domain adaptation until raw measurement comparability and cohort labels are verified.
'''
    (PKG/'cross_language_comparison.md').write_text(ctext)
    # Manuscript drafts.
    figure_refs='''Figures: `italian_figures/fig1_italian_cohort_flow`, `fig2_italian_task_coverage`, `fig3_italian_acoustic_distributions`, `fig4_italian_vowel_effect_forest`, `fig5_vi1_shimmer_robustness`, and `fig6_italian_vad_threshold_sensitivity`.'''
    (PKG/'italian_manuscript_draft.md').write_text(f'''# Task-specific voice measurements in ItalianPVS: an exploratory participant-level study

## Abstract

**Background:** Parkinsonian speech analysis is vulnerable to small samples, repeated recordings, task differences and sample-rate confounding. **Objective:** characterize task coverage and estimate participant-level acoustic group differences across five sustained Italian vowels, while assessing sensitivity and documenting unsupported task measures. **Methods:** the reconciled elderly-HC/PD cohort included 17 HC and 19 PD across rates; the primary native-16-kHz feature cohort included 17 HC and 12 PD. The saved VA1 reference contains 29 participants. Five first vowels were measured with Praat-derived median F0, jitter, shimmer, HNR, formants and spectral descriptors. Twenty prespecified group contrasts used Mann–Whitney tests with Holm correction and participant-bootstrap intervals. An independent robustness analysis evaluated VI1 shimmer using leave-one-participant-out, robust summaries, and percentile/BCa bootstrap intervals. D1/D2 and B1/B2/FB1 were characterized but not treated as validated participant-level outcomes. **Results:** VI1 local shimmer was the only primary contrast below the original 20-test Holm threshold (HC-minus-PD mean difference 0.0444; n=17/12; adjusted p≈0.012); its direction remained positive under individual deletion and robust summaries, but the adjusted trimmed-mean/Yuen result was ≈0.061. VA1 OOF baseline performance was BA .900/AUC .917 on 29 participants, with substantial internal-validation uncertainty. D1/D2 event candidates and connected-speech VAD remained unvalidated. **Conclusion:** a vowel-specific shimmer association merits independent replication, but the study is exploratory and not clinically validated.

## Introduction and research gap

Acoustic outcomes depend on speech task and recording conditions. A credible small-corpus study should preserve participant as the inferential unit, report uncertainty, distinguish sustained phonation from connected speech, and avoid turning recordings into independent sample size. ItalianPVS provides several vowel and speech tasks but has sample-rate imbalance and unresolved protocols for some tasks. This analysis prioritizes task-specific measurement and uncertainty over model complexity.

## Research objectives

1. Summarize participant/task coverage under the existing reconciliation and native-16-kHz restriction.
2. Estimate HC–PD first-vowel acoustic contrasts at participant level.
3. Reassess the VI1 shimmer observation under influence, robust-statistic and bootstrap sensitivity.
4. Characterize second-vowel, D1/D2 and reading signals without assigning unknown protocols or claiming unvalidated measures.

## Dataset and participants

The supplied manifest contains 831 recordings from 65 group-qualified participant folders. The preserved reconciliation yields 567 HC/PD recordings from 36 people (17 HC, 19 PD). The included native-16-kHz subset is 455 recordings from 29 people (17 HC, 12 PD). The first-vowel statistical table contains 145 task-participant observations. The 29-person VA1 reference contains one recording per participant. YHC are not in primary disease-group contrasts. Reconciliation decisions are unchanged.

## Methods

### Audio and measurements

Audio files were analyzed without modification. The primary group comparison used native 16 kHz because extra 44.1 kHz recordings belonged to PD participants only. The vowel feature pipeline used a candidate longest voiced interval at a 75 Hz reference floor, trimmed by 10% at both ends. Praat autocorrelation pitch used 75–500 Hz and 10 ms step; cycle perturbation used the configured local jitter/shimmer settings; HNR used Praat CC; formants used Burg with fixed 5 kHz maximum-formant setting; spectral descriptors used 25 ms Hann windows and 10 ms hop. Formant tracks are candidate values, not expert validated. Full definitions and settings appear in the feature table.

### Statistical analysis

Participant was the unit. Prespecified primary features were median F0, local jitter, local shimmer and mean HNR for VA1/VE1/VI1/VO1/VU1 (20 tests). The historical analysis reported mean differences, Hedges’ g, Cliff’s delta, participant-bootstrap intervals and Mann–Whitney p-values, with Holm correction over the 20 tests. New VI1 sensitivity work directly rechecked the source feature CSV; used 10,000 participant-bootstrap replicates for percentile/BCa intervals; computed median, 20%-trimmed and 10%-winsorized differences; and deleted each participant in turn. The saved VA1 OOF results were not refit. No classifiers were added.

## Results

The only primary-family contrast surviving Holm correction was VI1 local shimmer (17 HC/12 PD, mean difference 0.04439, historical 95% interval [0.01853, 0.07745], Hedges’ g 0.889, Mann–Whitney p 0.000600, Holm p 0.01199). The independent percentile interval was [0.01818, 0.07563] and BCa interval [0.02246, 0.08619]. All single-participant deletion estimates remained positive (0.03335–0.04739). The 20%-trimmed estimate remained positive, but its adjusted Yuen p≈0.061; therefore significance depends on analysis choice even though the direction is stable. VI1 HNR showed an HC-minus-PD difference around −7.18 dB but Holm p≈0.0815. Other primary contrasts did not survive correction. Refer to the 20-contrast table and Figure 4 for all estimates/intervals.

The VA1 baseline retained its previously reported 29-participant OOF results: BA .900, AUC .917, sensitivity .917 and specificity .882. Fixed-prediction participant bootstrap intervals were BA [.775,1.000] and AUC [.789,1.000], but do not include refitting or fold variability and are not external validation.

Second-vowel distributions were shorter than first variants, but codebook wording is blank. D1/D2 plots often showed separated CV-like events, but event peaks are not validated syllable onsets. B1/B2/FB1 activity/pause candidates changed with VAD threshold and remain unreviewed. No syllable count, VOT, transcript timing, or connected-speech group inference is reported.

## Discussion

The VI1 shimmer direction persists across participant deletion and robust summaries and is therefore less likely to be the consequence of one influential observation. Its Holm-adjusted p-value does not persist under a 20%-trimmed Yuen test. The signal may be task-specific, but a small n, repeated analysis choices, age/sex/device differences and the fact that all extra-rate recordings are PD prevent disease-specific or confirmatory interpretation. The current corpora do not allow direct replication of `/i/` shimmer.

### What the Results Actually Tell Us

The most actionable result is a prespecified, sensitivity-tested hypothesis: replicate local shimmer during sustained Italian `/i/` under a controlled, repeated-token protocol. The five vowel contexts do not support a general claim that every vowel captures the same group signal. Vowel context changes the measurements, and there is no validated second-variant reliability estimate. Connected speech could add pause/prosody information, but current threshold-sensitive masks do not yet support it. The study’s present contribution is transparent task-specific coverage and measurement uncertainty, not clinical discrimination.

### Recommended Next Research Experiments

1. Independent, blinded `/i/` shimmer replication with matched age/sex, fixed recording chain and repeated tokens.
2. Complete manual review and quantify inter-rater agreement for D1/D2 onset and connected-speech masks.
3. Recover protocol sheets and exact reading material before implementing task-specific timing or text alignment.
4. Continue only with measures whose segmentation and measurement validity can be demonstrated. Defer complex classifiers until larger independent cohorts exist.

## Limitations

The native-16-kHz cohort is 17 HC/12 PD. This does not establish adequate power for small effects. Recording rate is group-associated; resampled PD-only additions cannot solve it. Sample/device and demographic confounding remain. The stable vowel segment is a candidate algorithm. No inter-rater or biological test-retest reliability has been measured. 159 items await human review. D1/D2 count protocol and second-vowel definitions are unresolved; exact FB1 text is unavailable. The VA1 estimate is internal cross-validation and not external or clinical evidence.

## Conclusion

VI1 shimmer is an exploratory, directionally robust but method-dependent signal. It should guide a focused replication, not a clinical claim. Other tasks should remain descriptive until human validation and protocol documentation support participant-level measurements.

## References

1. Dimauro G, Caivano D, Bevilacqua V, Girardi F, Napoletano V. VoxTester, software for digital evaluation of speech changes in Parkinson disease. IEEE MeMeA, 2016. [doi:10.1109/MeMeA.2016.7533761](https://doi.org/10.1109/MeMeA.2016.7533761).
2. Dimauro G, Di Nicola V, Bevilacqua V, Caivano D, Girardi F. Assessment of Speech Intelligibility in Parkinson’s Disease Using a Speech-To-Text System. IEEE Access, 2017. [doi:10.1109/ACCESS.2017.2762475](https://doi.org/10.1109/ACCESS.2017.2762475).
3. Exploring robust computer-aided diagnosis of Parkinson’s disease based on various voice signals. Frontiers in Physics, 2022. [doi:10.3389/fphy.2022.1048833](https://doi.org/10.3389/fphy.2022.1048833).
4. Boersma P, Weenink D. Praat, a system for doing phonetics by computer. Glot International, 2001;5(9/10):341–345. [Praat](https://www.fon.hum.uva.nl/praat/).
5. Study dataset citation and task source documentation are maintained in the local audit reports and referenced above; see [task-audit report](../..//reports/italian_task_audit/task_audit_report.md).

{figure_refs}
''')
    # English analysis manuscript draft.
    (PKG/'english_manuscript_draft.md').write_text(f'''# Independent voice-feature analyses across English-language and related Parkinson datasets

## Abstract

This exploratory work audited local Parkinson voice datasets and analyzed participant-level acoustic differences where raw audio and labels were available. Figshare supplied 81 sustained `/a/` phone recordings linked one-to-one with demographic workbook IDs. MDVR-KCL supplied 73 mono 44.1 kHz recordings from 37 people (21 HC, 16 PD), in reading and spontaneous dialogue. Praat-derived sustained-vowel F0, jitter, shimmer and HNR were measured in Figshare; candidate activity, pause, F0, voicing, HNR and spectral summaries were measured separately for MDVR tasks. UCI 174, 189 and 301 were audited as precomputed feature datasets; their repeated records were aggregated or modeled within participants. The MDVR dialogue and pause measures remain unvalidated, UCI 174 local participant counts differ from the source description, and UCI feature tables do not permit raw-audio harmonization. These data support independent exploratory estimates, not external clinical validation or a pooled cross-language model.

## Introduction and objectives

The purpose was to determine whether locally available non-Italian voice resources can supply independent participant-level acoustic evidence and which are suitable for comparison with ItalianPVS. Analyses were limited to task-appropriate measurements and did not treat multiple recordings as independent participants.

## Dataset and participants

Figshare: 81 WAVs, 41 HC/40 PD, one workbook-linked Sample ID per file, 8 kHz mono FLOAT. The source describes prolonged `/a/` by telephone. MDVR-KCL: 73 WAVs, 37 unique speakers, 21 HC/16 PD, 44.1 kHz mono PCM_24; 37 reading and 36 dialogue files, with one PD absent from the local reading task. The dialogue filename ID22 separator anomaly is documented. UCI 174: 195 feature rows, 32 local participant IDs (24 PD/8 HC); source discrepancy versus the UCI page is retained. UCI 189: 5,875 longitudinal PD voice-feature rows and 42 people. UCI 301: 1,040 training task rows, 40 people (20/20), plus a separate 168-row PD-only repeated vowel table. No raw audio exists locally for these UCI files. UCI 470 and mPower were not locally analyzed.

## Methods

Figshare sustained vowel measurements used the same candidate stable-voiced interval and Praat definitions as the Italian vowel pipeline: 75 Hz floor/500 Hz ceiling, 10 ms pitch step, longest voiced run with 10% end trimming; local jitter and shimmer; Praat CC HNR. The 8 kHz native signal was not upsampled. Age/sex-adjusted linear associations used HC3 standard errors as a confounding sensitivity check.

MDVR read and spontaneous tasks were analyzed separately. Original 44.1 kHz audio was polyphase-resampled to 16 kHz for the analysis copy. RMS VAD used 25 ms frames, 10 ms hop, a provisional −25 dB threshold relative to the recording p95 frame RMS, with 10 ms hole closure; sensitivity at −20/−25/−30/−35 dB is reported. Pauses were gaps of at least 250 ms. F0 and HNR were summarized only in candidate active/voiced windows; intensity IQR and spectral centroid/rolloff were computed on active windows. We did not extract sustained-phonation jitter/shimmer from dialogue or reading.

UCI 174 repeated feature rows were reduced to participant medians. UCI 301 training contrasts used one subject-task row for sustained `/a/`, `/o/`, `/u/`; tasks remained separate. The independent UCI 301 test partition was PD-only and used only for descriptive within-person repetition spread. UCI 189 used each participant’s repeated measurements to compute within-person Spearman feature/UPDRS association; the distribution across people, not 5,875 rows, was summarized. Group tests report effect sizes and participant-bootstrap intervals with within-dataset/task Holm correction. No models were trained.

## Results

See the participant-level effect table in `english_results.md` and full CSV outputs under `english_tables/`. The Figshare and MDVR group contrasts are task-specific exploratory estimates with wide intervals and potential demographic/channel confounding. Dialogue results may mix participant and examiner speech. UCI 174 can inform historical feature-table direction only, and its local count conflict (32 IDs vs source page 31) requires resolution. UCI 301 training data offer separate `/a,o,u/` precomputed-feature contrasts; the repeated PD-only test file cannot estimate group differences. UCI 189 adds severity-association context but no HC comparison. No local UCI 470 data or mPower data were available.

## Discussion and limitations

Raw-audio Figshare and MDVR are suitable for careful exploratory independent analysis. Figshare is not a confirmed language-specific English reading/speech corpus; it is a sustained phone `/a/` task. MDVR supports English reading and dialogue but is small and lacks detailed demographics locally; dialogue voice attribution is not solved. The UCI sets add sample structures but use precomputed features and cannot reproduce a common extraction method. The UCI 174 participant-count discrepancy, UCI 301 language/protocol metadata gaps, incomplete task availability, age/sex/device effects and no clinical holdout preclude definitive inference. The local analysis did not externally validate Italian `/i/` shimmer; no comparable `/i/` task exists.

## Conclusion

The current English-language evidence is useful for separate, low-dimensional participant-level effect estimation and method planning. It is not yet adequate to claim replication or generalization. A common controlled protocol and fully documented independent cohort are required for cross-language validation.

## References

1. Figshare. Voice Samples for Patients with Parkinson’s Disease and Healthy Controls. [Dataset item 23849127](https://doi.org/10.6084/m9.figshare.23849127); license CC BY 4.0.
2. Jaeger H, Trivedi D, Stadtschnitzer M. Mobile Device Voice Recordings at King’s College London (MDVR-KCL). Zenodo, 2019. [doi:10.5281/zenodo.2867216](https://doi.org/10.5281/zenodo.2867216).
3. Little MA. Parkinsons [Dataset]. UCI Machine Learning Repository, 2007. [doi:10.24432/C59C74](https://doi.org/10.24432/C59C74).
4. Tsanas A, Little MA, McSharry PE, Ramig LO. Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests. IEEE TBME, 2010. [UCI 189 dataset](https://doi.org/10.24432/C5ZS3N).
5. Erdogdu Sakar B et al. Collection and Analysis of a Parkinson Speech Dataset With Multiple Types of Sound Recordings. IEEE JBHI, 2013. [doi:10.1109/JBHI.2013.2245674](https://doi.org/10.1109/JBHI.2013.2245674); [UCI 301](https://doi.org/10.24432/C5NC8M).
6. Sakar C et al. Parkinson’s Disease Classification [Dataset]. UCI, 2018. [doi:10.24432/C5MS4X](https://doi.org/10.24432/C5MS4X).
''')
    # Combined summary.
    (PKG/'combined_research_summary.md').write_text('''# Combined research summary

## ItalianPVS

The project established task and participant coverage, preserved the 29-person VA1 baseline, and completed participant-level analysis of five first sustained vowels. Native-16-kHz HC-versus-PD n is 17/12. VI1 shimmer is the only original 20-comparison Holm-supported observation (mean difference 0.0444; adjusted p≈.012) and is directionally robust to individual deletion, but significance changes under a trimmed/Yuen test (adjusted p≈.061). It remains exploratory. Second vowel, D1/D2 and connected-speech outputs are not inferentially validated.

## English and related datasets

Local Figshare audio yields 81 coded participants (41 HC/40 PD) on sustained `/a/`. MDVR-KCL has 37 participants (21 HC/16 PD), 73 raw files across reading and dialogue. UCI 174/189/301 are precomputed feature datasets; 189 is longitudinal PD-only, and UCI 174 has an unresolved local-versus-documentation participant discrepancy. UCI 470 and mPower are not local. Participant-level exploratory analyses were performed where data supported them. No classifier was trained.

## Cross-dataset evidence

Sustained-vowel measures make an `/a/` bridge feasible for acoustic definition comparisons, but language/accent, acquisition hardware/bandwidth, participant composition and historic UCI feature algorithms differ. MDVR reading/dialogue does not match the Italian text tasks. Italian VI1 `/i/` shimmer has no same-vowel English replication in the local data. No independent cross-language validation or true external clinical validation was achieved.

## Publication contribution and requirements

The strongest contribution is a reproducible participant-level multi-task measurement and uncertainty study, with explicit separation of usable and unvalidated tasks and a sensitivity-tested `/i/` shimmer hypothesis. Further requirements include manual audio adjudication, protocol recovery, demographic/device control, a larger independent `/i/` replication, clarification of the UCI 174 roster discrepancy and a prospectively matched cross-language cohort.
''')
    # Readiness assessment.
    (PKG/'publication_readiness.md').write_text('''# Publication readiness assessment

**Assessment:** research-package quality is publication-oriented; the current evidence is **not ready for a journal submission claiming confirmed disease-specific or cross-language validation**.

| Domain | Assessment | Reason / action |
|---|---|---|
| Research contribution | Promising exploratory question | Italian multi-task measurement and targeted VI1 shimmer sensitivity are coherent; novelty requires focused literature review before submission. |
| Italian dataset suitability | Limited, usable for hypothesis generation | Native primary cohort is 17 HC/12 PD; recording rate, demographics and acquisition may confound. |
| Statistical strength | Transparent, not confirmatory | Participant-level effects, bootstrap intervals and Holm correction are reported; one result changes threshold under Yuen analysis. |
| Measurement validity | Partial | First vowels extracted reproducibly; no biological/inter-rater reliability. 159 audio review items remain open. |
| Task protocol validity | Incomplete | D1/D2 repetition instructions and second-vowel definitions are unknown; FB1 text unavailable; reading VAD unreviewed. |
| English external evidence | Exploratory | Figshare and MDVR provide independent audio but small/task/channel differences; no matched `/i/` replication. |
| UCI support | Supplementary only | No raw audio; UCI 174 local participant-count discrepancy requires clarification; UCI 189 is PD-only severity data. |
| External validation | Not achieved | No fully held-out compatible cohort tested with locked protocol. Cross-dataset effect plots are not external model validation. |
| Clinical validity | None claimed | No diagnostic or clinical claims are justified. |
| Reproducibility | Good for the analyzed package | Configs, file hashes, participant-level tables, tests and figures are included; source data themselves remain external and are not redistributed. |
| Manuscript completeness | Structured draft, not submission-ready | Requires human review results, protocol details, refined literature gap/novelty, author/source permissions, journal-specific reporting checklist and independent statistical review. |

## Specific blockers before submission

1. Complete or formally scope the 159-item Italian review, especially all 72 D1/D2 recordings, flagged QC/pitch items, and stratified reading masks. Add inter-rater agreement where two reviewers are available.
2. Recover authoritative D1/D2 and second-vowel protocol information; otherwise exclude these as exploratory signal characterization and state the limitation.
3. Address demographic and acquisition confounding and replicate the prespecified VI1 shimmer contrast in a larger, matched participant cohort with repeated tokens.
4. Resolve the local UCI 174 ID discrepancy (32 IDs/24 PD in the file versus 31/23 stated in the UCI narrative) before using it as publication-level replication evidence.
5. Avoid describing Figshare as language-specific English reading: its local task is prolonged `/a/` telephone phonation and language is not specified.
6. Establish an external, task-matched participant cohort before using “validation” beyond directional replication.

## Realistic publication direction

The strongest near-term manuscript would be an exploratory task-specific ItalianPVS acoustics and reproducibility report centered on participant-level vowel-context estimates and the sensitivity-tested VI1 shimmer hypothesis. Treat D1/D2 and connected speech as a transparent measurement/coverage appendix until reviewed. The English resources belong in a separate companion analysis or a later replication paper after task/protocol and participant covariates are aligned. A pooled language classifier would be scientifically premature.
''')
    # Reproducibility report.
    hashes_before=json.dumps(hash_before,sort_keys=True,indent=2)
    hashes_after=json.dumps(hash_after,sort_keys=True,indent=2)
    (PKG/'reproducibility_report.md').write_text(f'''# Reproducibility and preservation report

**Experiment IDs:** `ENGLISH-VOICE-DATASET-AUDIT-20261031`, `FINAL-RESEARCH-PACKAGE-20261031`.

## Analysis environment

- Python {platform.python_version()}; NumPy {np.__version__}; pandas {pd.__version__}; SciPy {scipy.__version__}; SoundFile {sf.__version__}; Parselmouth {parselmouth.__version__}; Matplotlib {matplotlib.__version__}.
- Deterministic participant bootstraps use seed {SEED}; effect-interval analyses use 5,000 resamples; figure-only distribution intervals use 5,000–8,000 resamples. The VI1 sensitivity itself is re-used from the independently completed 10,000-resample audit and is not rerun here.
- No data were downloaded, merged into Italian manifests or written back to original sources. The UCI 301 RAR was read with bsdtar stream output; no extraction to the shared dataset directory occurred.

## Reproduce

From project root:

```bash
MPLCONFIGDIR=results/italian_research_phase2/configs/matplotlib_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.final_research.build_package
MPLCONFIGDIR=results/italian_research_phase2/configs/matplotlib_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.final_research.validate_package
MPLCONFIGDIR=results/italian_research_phase2/configs/matplotlib_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/ -q
```

## Preservation check

SHA-256 hashes were captured before and after package generation for every existing file under VA1 baseline data outputs, the completed `italian_multitask_v1` and `italian_research_phase2` experiment directories, task-audit reports, and all local English source data files. Match status: **{source_integrity}**. Machine-readable before/after values are in `english_research_20261031/source_hashes_before.json` and `source_hashes_after.json`; changed paths are listed in its `integrity_check.json`.

## Statistical and table controls

- Every independent group test uses one participant per dataset/task row. UCI 174 repeats are collapsed to participant medians. UCI 301 uses one subject-task row in the training file; its test subset remains separate and PD-only. Telemonitoring correlations are first computed within participant and then summarized across 42 people.
- No classifier and no train/test split were created; participant-leakage validation is therefore not applicable. Participant uniqueness checks are included in automated validation.
- Holm adjustments are grouped within dataset/task families and independently tested. Figure source CSVs and table source CSVs are preserved in `english_research_20261031/` and copied into final table folders where useful.
- Figures are exported as 320 dpi PNG and vector PDF. Figure datasets and sample counts are labeled; Figure 6/E2/C1 explicitly identify candidate or cross-dataset limitations.

## Preservation values

Before:

```json
{hashes_before}
```

After:

```json
{hashes_after}
```

The package does not claim that open human-review questions are automated validation passes. See the validation CSV/JSON for each exact check and status.
''')
    return compat


def main():
    for p in [EXP,PKG,FIGI,FIGE,FIGT,TABLEE,FIGC]: p.mkdir(parents=True,exist_ok=True)
    for p in [EXP/'logs',EXP/'configs']: p.mkdir(parents=True,exist_ok=True)
    # Snapshot all preserved Italian references and all located English source files.
    srcs=protected_files(ROOT)
    srcs += [p for folder in [FIGSHARE,MDVR] for p in folder.rglob('*') if p.is_file()]
    srcs += [DATASETS/'kaggle/parkinsons.data',DATASETS/'kaggle/parkinsons.names',
             DATASETS/'kaggle/telemonitoring/parkinsons_updrs.data',DATASETS/'kaggle/telemonitoring/parkinsons_updrs.names',
             DATASETS/'UCI_csv/parkinsons_updrs.data',DATASETS/'UCI_csv/parkinsons_updrs.names',ARCHIVE_301]
    hash_before=sha_manifest(sorted(set(p for p in srcs if p.exists())))
    (EXP/'source_hashes_before.json').write_text(json.dumps(hash_before,indent=2,sort_keys=True))
    # Read/extract independent audio tables; no source audio is rewritten.
    fig_qc,fig_feat,mdvr_qc,mdvr_feat=discover_audio()
    audio_qc_all,audio_features,feature_coverage,qc_summary=record_feature_tables(fig_qc,fig_feat,mdvr_qc,mdvr_feat,None,EXP)
    # Load each tabular resource and verify its recorded unit/partition structure.
    ox,oxcheck,oxsub,tm,tm_sub,u301,u301test,u301_task=(
        __import__('src.final_research.analysis',fromlist=['build_uci_tables']).build_uci_tables())
    oxcheck.to_csv(EXP/'uci174_participant_status_inventory.csv',index=False)
    oxsub.to_csv(EXP/'uci174_participant_median_features.csv',index=False)
    tm_sub.to_csv(EXP/'uci189_participant_median_summary.csv',index=False)
    u301.to_csv(EXP/'uci301_training_task_feature_table.csv',index=False)
    u301test.to_csv(EXP/'uci301_test_repeated_vowel_features.csv',index=False)
    # Participant-task rows, class consistency and archive file inventory.
    classcons=u301.groupby('participant_id').agg(group_count=('group','nunique'),tasks=('task','size'),group=('group','first')).reset_index()
    classcons.to_csv(EXP/'uci301_training_participant_structure.csv',index=False)
    row_count=pd.DataFrame([
        ['UCI174_local',len(ox),ox.participant_id.nunique(),int((ox.groupby('participant_id').status.nunique()>1).sum())],
        ['UCI189_local',len(tm),tm['subject#'].nunique(),0],
        ['UCI301_train',len(u301),u301.participant_id.nunique(),int((classcons.group_count>1).sum())],
        ['UCI301_test',len(u301test),u301test.participant_id.nunique(),int((u301test.groupby('participant_id')['class'].nunique()>1).sum())]
    ],columns=['dataset_partition','rows','participants','participants_with_conflicting_group_labels'])
    row_count.to_csv(EXP/'feature_dataset_row_counts.csv',index=False)
    # Tabular and participant-level inferential summaries.
    effects,adjusted,tele,repeat=stats_and_tables(fig_feat,mdvr_feat,oxsub,tm,tm_sub,u301,u301test,EXP,None)
    # Italian manuscript-ready tables and source data for visualizations.
    manifest,primary,participants,itcoverage,featuredefs,italian_effects,vi,evidence=italian_tables(FIGI,FIGE,FIGC,FIGT,TABLEE)
    it_features=pd.read_csv(ROOT/'results/italian_multitask_v1/features/primary_native16_first_vowels.csv')
    sens=pd.read_csv(ROOT/'results/italian_research_phase2/sensitivity/group_contrast_sensitivity.csv')
    connected=pd.read_csv(ROOT/'results/italian_research_phase2/connected_speech/vad_threshold_sensitivity_long.csv')
    # Save figure-input rows and report tables in the isolated experiment folder.
    it_features.to_csv(EXP/'italian_figure_source_first_vowels_readonly_copy.csv',index=False)
    italian_effects.to_csv(EXP/'italian_figure_source_primary_effects_readonly_copy.csv',index=False)
    vi.to_csv(EXP/'italian_vi1_sensitivity_summary.csv',index=False)
    itcoverage.to_csv(EXP/'italian_task_recording_coverage.csv',index=False)
    # Final experiment effect table with all feature-family Holm values normalized.
    effects.to_csv(TABLEE/'participant_level_group_effects.csv',index=False)
    adjusted.to_csv(TABLEE/'figshare_age_sex_adjusted_effects.csv',index=False)
    tele.to_csv(TABLEE/'uci189_within_participant_correlations.csv',index=False)
    repeat.to_csv(TABLEE/'uci301_pd_test_repeat_spread.csv',index=False)
    # Tables summarizing Italian dataset for final paper package.
    make_italian_figures(manifest,primary,itcoverage,it_features,italian_effects,sens,connected,FIGI)
    vad_rows=[]
    for _,row in mdvr_feat.iterrows():
        for th in [-20,-25,-30,-35]:
            suf=f'{th}db'
            vad_rows.append({'dataset':'MDVR-KCL','task':row['task'],'group':row['group'],'participant_id':row['participant_id'],
                             'relative_path':row['relative_path'],'threshold_db':th,
                             'active_speech_fraction':row.get(f'active_speech_fraction_{suf}'),
                             'pause_count_candidate':row.get(f'pause_count_{suf}'),
                             'pause_fraction_candidate':row.get(f'pause_fraction_{suf}')})
    vad_sensitivity=pd.DataFrame(vad_rows)
    vad_sensitivity.to_csv(EXP/'mdvr_vad_threshold_sensitivity_long.csv',index=False)
    make_english_figures(fig_feat,mdvr_feat,effects,vad_sensitivity,FIGE)
    make_cross_figure(italian_effects,effects,FIGC)
    inventory,engcoverage,methods=derive_inventory_and_checks(fig_qc,fig_feat,mdvr_qc,mdvr_feat,ox,oxcheck,oxsub,tm,tm_sub,u301,u301test)
    # Tabular inventory/report exports and provenance.
    for source,target in [(DATASETS/'UCI_csv/parkinsons_updrs.data',EXP/'uci189_raw_feature_table.csv'),
                          (DATASETS/'kaggle/parkinsons.data',EXP/'uci174_raw_feature_table.csv')]:
        shutil.copy2(source,target)
    (EXP/'configs/experiment_config.json').write_text(json.dumps({
        'experiment_id':'ENGLISH-VOICE-DATASET-AUDIT-20261031','package_id':'FINAL-RESEARCH-PACKAGE-20261031',
        'created':'2026-10-31 Asia/Kolkata','analysis_roots':{'figshare':str(FIGSHARE),'mdvr_kcl':str(MDVR),'datasets':str(DATASETS)},
        'raw_audio_features':VOICE_FEATURES,'mdvr_features':MDVR_FEATURES,
        'praat_pitch_floor_hz':75,'praat_pitch_ceiling_hz':500,'pitch_step_s':.01,'vowel_stable_interval':'longest 75Hz-floor voiced run; trim 10% each edge',
        'mdvr_vad':{'original_rate_hz':44100,'analysis_rate_hz':16000,'resampling':'scipy.signal.resample_poly','frame_ms':25,'hop_ms':10,'threshold_db_below_p95':-25,'one_frame_gap_closure':True,'minimum_internal_pause_ms':250},
        'participant_bootstraps':5000,'random_seed':SEED,'group_test':'two-sided Mann-Whitney U; one participant per dataset-task row; Holm within dataset/task feature family',
        'confidence_intervals':'participant bootstrap percentile for mean differences and Hedges g',
        'UCI174_repeated_rows':'participant median aggregation before inference',
        'UCI301':'one row per participant-task for training comparisons; PD-only test partition separate',
        'UCI189':'within-participant Spearman feature/UPDRS, then participant-level summary',
        'classifier_trained':False,'source_files_sha256_before':len(hash_before),
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
        'soundfile':sf.__version__,'parselmouth':parselmouth.__version__,'matplotlib':matplotlib.__version__
    },indent=2))
    # Reports and cross-dataset matrices.
    effects.to_csv(EXP/'participant_level_group_effects.csv',index=False)
    context={'effects':effects,'italian_effects':italian_effects,'adjusted':adjusted,'tele':tele,'repeat':repeat,
             'inventory':inventory,'english_coverage':engcoverage,'english_features':audio_features,'english_qc':audio_qc_all,
             'it_coverage':itcoverage,'evidence':evidence,'vi':vi}
    reports_and_tables(context)
    # Source checksum re-check after all work so changes outside the new results tree are visible.
    hash_after=sha_manifest(sorted(set(p for p in srcs if p.exists())))
    changed=sorted(k for k in set(hash_before)|set(hash_after) if hash_before.get(k)!=hash_after.get(k))
    (EXP/'source_hashes_after.json').write_text(json.dumps(hash_after,indent=2,sort_keys=True))
    integrity={'all_preexisting_sources_unchanged':not changed,'changed_paths':changed,
               'source_file_count_before':len(hash_before),'source_file_count_after':len(hash_after)}
    (EXP/'integrity_check.json').write_text(json.dumps(integrity,indent=2))
    create_additional_deliverables(context,hash_before,hash_after)
    # Replace the earlier copied source tables with the complete final table set.
    for p in EXP.glob('*.csv'):
        if p.name not in {x.name for x in TABLEE.glob('*.csv')}:
            shutil.copy2(p,TABLEE/p.name)
    for p in [EXP/'source_hashes_before.json',EXP/'source_hashes_after.json',EXP/'integrity_check.json',EXP/'configs/experiment_config.json']:
        if p.exists(): shutil.copy2(p,PKG/p.name)
    print(json.dumps({'experiment':'ENGLISH-VOICE-DATASET-AUDIT-20261031','figshare_audio_rows':len(fig_qc),
                      'mdvr_audio_rows':len(mdvr_qc),'english_effects':len(effects),'italian_checks_source_files_unchanged':not changed,
                      'outputs':str(PKG),'experiment_outputs':str(EXP)},indent=2))


if __name__=='__main__':
    main()

