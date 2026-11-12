from __future__ import annotations

import hashlib
import io
import re
import subprocess
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from scipy import stats
from scipy.signal import resample_poly

from src.italian_multitask.audio import mono_for_analysis
from src.italian_multitask.segmentation import candidate_stable_interval, voiced_runs


FIGSHARE = Path('/Users/Shaan/Desktop/datasets/figshare')
MDVR = Path('/Users/Shaan/Desktop/datasets/MDVR-KCL')
DATASETS = Path('/Users/Shaan/Desktop/datasets')
ARCHIVE_301 = DATASETS / 'uci_csv2.rar'
VOICE_FEATURES = ['median_f0_hz', 'jitter_local', 'shimmer_local', 'hnr_mean_db']
MDVR_FEATURES = [
    'active_speech_fraction', 'pause_count_candidate', 'pause_fraction_candidate',
    'median_f0_active_hz', 'f0_iqr_active_hz', 'voiced_fraction_active',
    'hnr_median_active_db', 'relative_intensity_iqr_db',
    'spectral_centroid_active_hz', 'spectral_rolloff_85_active_hz',
]
UCI174_MAP = {
    'median_f0_hz': 'MDVP:Fo(Hz)', 'jitter_local': 'MDVP:Jitter(%)',
    'shimmer_local': 'MDVP:Shimmer', 'hnr_mean_db': 'HNR',
}
UCI301_FEATURE_NAMES = [
    'jitter_local', 'jitter_local_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
    'shimmer_local', 'shimmer_local_db', 'shimmer_apq3', 'shimmer_apq5',
    'shimmer_apq11', 'shimmer_dda', 'ac_12', 'nth', 'htn', 'median_pitch_hz',
    'mean_pitch_hz', 'pitch_sd_hz', 'min_pitch_hz', 'max_pitch_hz', 'pulse_count',
    'period_count', 'mean_period_s', 'period_sd_s', 'unvoiced_fraction',
    'voice_break_count', 'degree_voice_breaks',
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protected_files(root: Path) -> list[Path]:
    paths = [root / 'data/va1_features_raw.csv', root / 'data/baseline_oof_predictions.csv',
             root / 'data/primary_cohort_manifest.csv', root / 'data/italian_manifest.csv']
    for folder in [root / 'results/italian_multitask_v1', root / 'results/italian_research_phase2',
                   root / 'results/va1_oof_uncertainty_20261031', root / 'reports/italian_task_audit']:
        if folder.exists():
            paths.extend(p for p in folder.rglob('*') if p.is_file())
    return sorted(set(p for p in paths if p.exists()))


def compute_stable_vowel(path: Path, dataset: str, task: str, group: str, participant_id: str,
                         relative_path: str, **metadata) -> dict:
    result = {'dataset': dataset, 'task': task, 'group': group, 'participant_id': participant_id,
              'relative_path': relative_path, **metadata}
    try:
        data, sr = sf.read(path, dtype='float64', always_2d=True)
        mono, channel_method = mono_for_analysis(data)
        interval, _ = candidate_stable_interval(mono, int(sr), floor=75.0)
        if interval is None:
            raise ValueError('no 75-Hz-floor voiced interval')
        start, end = interval
        segment = mono[max(0, int(start * sr)):min(len(mono), int(end * sr))]
        if len(segment) < int(.25 * sr):
            raise ValueError('voiced interval shorter than 250 ms')
        snd = parselmouth.Sound(segment, sampling_frequency=sr)
        pitch = snd.to_pitch_ac(time_step=.01, pitch_floor=75, pitch_ceiling=500)
        f0 = pitch.selected_array['frequency']
        voiced = np.isfinite(f0) & (f0 > 0)
        point = parselmouth.praat.call(snd, 'To PointProcess (periodic, cc)', 75, 500)
        jitter = float(parselmouth.praat.call(point, 'Get jitter (local)', 0, 0, .0001, .02, 1.3))
        shimmer = float(parselmouth.praat.call([snd, point], 'Get shimmer (local)', 0, 0, .0001, .02, 1.3, 1.6))
        hn = snd.to_harmonicity_cc(time_step=.01, minimum_pitch=75,
                                   silence_threshold=.1, periods_per_window=1.0).values[0]
        hn = hn[np.isfinite(hn) & (hn > -100)]
        result.update({
            'status': 'ok', 'failure_reason': '', 'sample_rate_hz': int(sr),
            'stable_duration_s': float(end - start), 'channel_method': channel_method,
            'median_f0_hz': float(np.median(f0[voiced])) if voiced.any() else np.nan,
            'f0_iqr_hz': float(np.percentile(f0[voiced], 75)-np.percentile(f0[voiced], 25)) if voiced.any() else np.nan,
            'voiced_fraction': float(voiced.mean()) if len(voiced) else np.nan,
            'jitter_local': jitter, 'shimmer_local': shimmer,
            'hnr_mean_db': float(np.mean(hn)) if len(hn) else np.nan,
        })
    except Exception as exc:
        result.update({'status': 'failed', 'failure_reason': f'{type(exc).__name__}: {exc}',
                       'sample_rate_hz': np.nan, 'stable_duration_s': np.nan,
                       'channel_method': '', **{f: np.nan for f in VOICE_FEATURES[0:]},
                       'f0_iqr_hz': np.nan, 'voiced_fraction': np.nan})
    return result


def audio_qc(path: Path, dataset: str, task: str, group: str, participant_id: str,
             relative_path: str, filename_anomaly: str = '') -> dict:
    base = {'dataset': dataset, 'task': task, 'group': group, 'participant_id': participant_id,
            'relative_path': relative_path, 'filename_anomaly': filename_anomaly}
    try:
        info = sf.info(path)
        x, sr = sf.read(path, dtype='float64', always_2d=True)
        mono, _ = mono_for_analysis(x)
        peak = float(np.max(np.abs(x)))
        frame = max(1, int(round(sr * .025))); hop = max(1, int(round(sr * .01)))
        if len(mono) < frame:
            rms = np.array([np.sqrt(np.mean(mono**2))])
        else:
            rms = np.array([np.sqrt(np.mean(mono[s:s+frame]**2))
                            for s in range(0, len(mono)-frame+1, hop)])
        p95 = float(np.percentile(rms, 95)) if len(rms) else 0
        silence = float(np.mean(rms < max(1e-12, p95 * .01))) if len(rms) else np.nan
        full = np.abs(x) >= .999
        full_fraction = float(full.mean())
        flattened = full.reshape(-1)
        edges = np.diff(np.r_[False, flattened, False].astype(np.int8))
        runs = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        longest = int(np.max(ends-runs)) if len(runs) else 0
        qc_flag = 'possible_clipping_plateau' if longest >= 3 or full_fraction >= .001 else ''
        if silence >= .5:
            qc_flag = ';'.join(filter(None, [qc_flag, 'high_low_energy_fraction']))
        return {**base, 'readable': True, 'read_error': '', 'sample_rate_hz': int(sr),
                'channels': int(info.channels), 'subtype': info.subtype, 'duration_s': float(info.duration),
                'peak_abs': peak, 'rms': float(np.sqrt(np.mean(x*x))), 'full_scale_fraction_ge_0_999': full_fraction,
                'longest_full_scale_run_samples': longest, 'low_energy_frame_fraction_below_minus40dB_p95': silence,
                'qc_flag': qc_flag, 'sha256': sha256_file(path)}
    except Exception as exc:
        return {**base, 'readable': False, 'read_error': f'{type(exc).__name__}: {exc}',
                'sample_rate_hz': np.nan, 'channels': np.nan, 'subtype': '', 'duration_s': np.nan,
                'peak_abs': np.nan, 'rms': np.nan, 'full_scale_fraction_ge_0_999': np.nan,
                'longest_full_scale_run_samples': np.nan, 'low_energy_frame_fraction_below_minus40dB_p95': np.nan,
                'qc_flag': 'read_error', 'sha256': ''}


def mdvr_features(path: Path, task: str, group: str, pid: str, relative: str) -> dict:
    result = {'dataset': 'MDVR-KCL', 'task': task, 'group': group, 'participant_id': pid,
              'relative_path': relative, 'vad_threshold_db_below_p95': -25,
              'vad_frame_ms': 25, 'vad_hop_ms': 10}
    try:
        data, original_sr = sf.read(path, dtype='float64', always_2d=True)
        x, channel_method = mono_for_analysis(data)
        sr=16000
        if original_sr != sr:
            from math import gcd
            divisor=gcd(int(original_sr),int(sr))
            x=resample_poly(x,sr//divisor,int(original_sr)//divisor)
        frame = max(1, int(round(sr*.025))); hop = max(1, int(round(sr*.01)))
        if len(x) < frame:
            levels = np.array([np.sqrt(np.mean(x*x))])
        else:
            levels = np.array([np.sqrt(np.mean(x[s:s+frame]**2)) for s in range(0, len(x)-frame+1, hop)])
        p95 = float(np.percentile(levels, 95))
        duration_s = len(x)/sr
        sensitivity={}
        masks={}
        for threshold_db in [-20,-25,-30,-35]:
            mask=levels >= max(1e-12,p95*10**(threshold_db/20))
            if len(mask)>2:
                gaps=np.where((~mask[1:-1])&mask[:-2]&mask[2:])[0]+1
                mask[gaps]=True
            edges0=np.diff(np.r_[False,mask,False].astype(np.int8))
            s0,e0=np.where(edges0==1)[0],np.where(edges0==-1)[0]
            bouts0=list(zip(s0,e0))
            pause0=[b-a for (_,b),(a,_) in zip(bouts0[:-1],bouts0[1:]) if a-b>=25]
            masks[threshold_db]=mask
            sensitivity[f'active_speech_fraction_{threshold_db}db']=float(mask.sum()*.01/duration_s)
            sensitivity[f'pause_count_{threshold_db}db']=len(pause0)
            sensitivity[f'pause_fraction_{threshold_db}db']=float(sum(pause0)*.01/(len(x)/sr))
        active=masks[-25]
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        starts, ends = np.where(edges==1)[0], np.where(edges==-1)[0]
        bouts = list(zip(starts, ends))
        internal_pauses = [b-a for (_, b), (a, _) in zip(bouts[:-1], bouts[1:]) if a-b >= 25]
        snd = parselmouth.Sound(x, sampling_frequency=sr)
        pitch = snd.to_pitch_ac(time_step=.01, pitch_floor=75, pitch_ceiling=500)
        f0 = pitch.selected_array['frequency']; pt = pitch.xs()
        inds = np.clip(np.rint(pt/.01).astype(int), 0, len(active)-1)
        pitch_active = active[inds]
        voiced_active = pitch_active & np.isfinite(f0) & (f0 > 0)
        f0v = f0[voiced_active]
        harmonicity = snd.to_harmonicity_cc(time_step=.01, minimum_pitch=75,
                                            silence_threshold=.1, periods_per_window=1.0)
        hvals = harmonicity.values[0]; ht = harmonicity.xs()
        hi = np.clip(np.rint(ht/.01).astype(int), 0, len(active)-1)
        valid_h = active[hi] & np.isfinite(hvals) & (hvals > -100)
        # Frame spectra use original rate and are interpreted within MDVR only.
        centroids=[]; rolloffs=[]; intensity=[]
        for i, is_active in enumerate(active[:len(levels)]):
            if not is_active: continue
            pos = i*hop
            chunk = x[pos:pos+frame]
            if len(chunk) < frame: continue
            win = chunk*np.hanning(frame); sp=np.abs(np.fft.rfft(win)); hz=np.fft.rfftfreq(frame, 1/sr)
            sp[0]=0; total=sp.sum()
            if total <= 0: continue
            centroids.append(float(np.sum(hz*sp)/total))
            rolloffs.append(float(hz[min(np.searchsorted(np.cumsum(sp), .85*total),len(hz)-1)]))
            intensity.append(20*np.log10(max(levels[i],1e-12)))
        active_duration = float(active.sum()*.01)
        internal_pause_s = float(sum(internal_pauses)*.01)
        result.update({'status':'ok','failure_reason':'','sample_rate_hz':int(original_sr),
                       'analysis_sample_rate_hz':int(sr),'duration_s':duration_s,
                       'channel_method':channel_method,'active_speech_fraction':active_duration/duration_s,
                       'active_speech_duration_s':active_duration,'candidate_speech_bouts':len(bouts),
                       'pause_count_candidate':len(internal_pauses),
                       'pause_fraction_candidate':internal_pause_s/duration_s,
                       'median_pause_s_candidate':float(np.median(internal_pauses)*.01) if internal_pauses else 0.0,
                       'median_f0_active_hz':float(np.median(f0v)) if len(f0v) else np.nan,
                       'f0_iqr_active_hz':float(np.percentile(f0v,75)-np.percentile(f0v,25)) if len(f0v) else np.nan,
                       'voiced_fraction_active':float(voiced_active.sum()/max(1,pitch_active.sum())),
                       'hnr_median_active_db':float(np.median(hvals[valid_h])) if valid_h.any() else np.nan,
                       'relative_intensity_iqr_db':float(np.percentile(intensity,75)-np.percentile(intensity,25)) if intensity else np.nan,
                       'spectral_centroid_active_hz':float(np.median(centroids)) if centroids else np.nan,
                       'spectral_rolloff_85_active_hz':float(np.median(rolloffs)) if rolloffs else np.nan,
                       **sensitivity})
    except Exception as exc:
        result.update({'status':'failed','failure_reason':f'{type(exc).__name__}: {exc}',
                       **{f:np.nan for f in MDVR_FEATURES}, 'sample_rate_hz':np.nan,'duration_s':np.nan,
                       'analysis_sample_rate_hz':np.nan,'channel_method':'',
                       **{f'active_speech_fraction_{t}db':np.nan for t in [-20,-25,-30,-35]},
                       **{f'pause_count_{t}db':np.nan for t in [-20,-25,-30,-35]},
                       **{f'pause_fraction_{t}db':np.nan for t in [-20,-25,-30,-35]}})
    return result


def _cliffs_delta(x, y):
    u = stats.mannwhitneyu(x, y, alternative='two-sided').statistic
    return float(2*u/(len(x)*len(y))-1)


def _hedges_g(x, y):
    if len(x)<2 or len(y)<2: return np.nan
    dof=len(x)+len(y)-2
    pooled=np.sqrt(((len(x)-1)*np.var(x,ddof=1)+(len(y)-1)*np.var(y,ddof=1))/dof)
    if pooled<=0: return np.nan
    return float((1-3/(4*dof-1))*(np.mean(x)-np.mean(y))/pooled)


def holm(p):
    p=np.asarray(p,float); out=np.full(len(p),np.nan); ix=np.flatnonzero(np.isfinite(p))
    order=ix[np.argsort(p[ix])]; run=0
    for j,i in enumerate(order):
        run=max(run,(len(order)-j)*p[i]); out[i]=min(1.0,run)
    return out


def compare_groups(frame: pd.DataFrame, features: list[str], *, dataset: str, task: str,
                   group_col: str='group', participant_col: str='participant_id', seed: int=61031,
                   bootstraps: int=5000) -> list[dict]:
    # Every input row must already be one independent participant for this task.
    if frame[participant_col].duplicated().any():
        raise ValueError(f'{dataset}/{task}: duplicate participant rows supplied to group test')
    rows=[]; rng=np.random.default_rng(seed)
    for feature in features:
        if feature not in frame: continue
        hc=frame.loc[frame[group_col].eq('HC'),feature].dropna().astype(float).to_numpy()
        pdv=frame.loc[frame[group_col].eq('PD'),feature].dropna().astype(float).to_numpy()
        if len(hc)<2 or len(pdv)<2: continue
        boots=np.mean(rng.choice(hc,(bootstraps,len(hc)),replace=True),axis=1)-np.mean(rng.choice(pdv,(bootstraps,len(pdv)),replace=True),axis=1)
        hx=rng.choice(hc,(bootstraps,len(hc)),replace=True); px=rng.choice(pdv,(bootstraps,len(pdv)),replace=True)
        hd=len(hc)+len(pdv)-2; corr=1-3/(4*hd-1)
        hvar=np.var(hx,axis=1,ddof=1); pvar=np.var(px,axis=1,ddof=1)
        pooled=np.sqrt(((len(hc)-1)*hvar+(len(pdv)-1)*pvar)/hd)
        gboot=np.divide(corr*(hx.mean(axis=1)-px.mean(axis=1)),pooled,
                        out=np.full(bootstraps,np.nan),where=pooled>0)
        mw=stats.mannwhitneyu(hc,pdv,alternative='two-sided')
        rows.append({'dataset':dataset,'task':task,'feature':feature,'n_hc':len(hc),'n_pd':len(pdv),
                     'mean_hc':float(np.mean(hc)),'mean_pd':float(np.mean(pdv)),
                     'median_hc':float(np.median(hc)),'median_pd':float(np.median(pdv)),
                     'mean_difference_hc_minus_pd':float(np.mean(hc)-np.mean(pdv)),
                     'mean_diff_boot_ci95_low':float(np.percentile(boots,2.5)),
                     'mean_diff_boot_ci95_high':float(np.percentile(boots,97.5)),
                     'hedges_g_hc_minus_pd':_hedges_g(hc,pdv),
                     'hedges_g_boot_ci95_low':float(np.nanpercentile(gboot,2.5)),
                     'hedges_g_boot_ci95_high':float(np.nanpercentile(gboot,97.5)),
                     'cliffs_delta_hc_minus_pd':_cliffs_delta(hc,pdv),
                     'mann_whitney_p':float(mw.pvalue),'participant_unit_verified':True})
    return rows


def adjusted_figshare_effect(df: pd.DataFrame, feature: str) -> dict:
    z=df[['group','age','sex',feature]].dropna().copy()
    y=z[feature].to_numpy(float)
    # PD coefficient adjusted for age and sex; HC is the reference.
    X=np.column_stack([np.ones(len(z)),z.group.eq('PD').astype(float),
                       z.age.to_numpy(float)-z.age.mean(),z.sex.eq('M').astype(float)])
    inv=np.linalg.inv(X.T@X); beta=inv@X.T@y; resid=y-X@beta
    leverage=np.sum(X@inv*X,axis=1)
    meat=X.T@((resid/np.maximum(1-leverage,1e-8))[:,None]**2*X)
    cov=inv@meat@inv; se=float(np.sqrt(cov[1,1])); dfree=len(y)-X.shape[1]
    tval=float(beta[1]/se); p=float(2*stats.t.sf(abs(tval),dfree))
    crit=float(stats.t.ppf(.975,dfree))
    return {'feature':feature,'n':len(y),'adjusted_pd_minus_hc':float(beta[1]),
            'hc3_ci95_low':float(beta[1]-crit*se),'hc3_ci95_high':float(beta[1]+crit*se),
            'hc3_se':se,'p_value_unadjusted':p,'covariates':'age, sex; HC reference',
            'interpretation':'exploratory linear association, HC3 heteroskedasticity-robust CI'}


def load_uci301() -> tuple[pd.DataFrame,pd.DataFrame]:
    def read(name):
        proc=subprocess.run(['bsdtar','-xOf',str(ARCHIVE_301),name],check=True,capture_output=True)
        return pd.read_csv(io.BytesIO(proc.stdout),header=None)
    train=read('train_data.txt'); test=read('test_data.txt')
    if train.shape[1] != 29 or test.shape[1] != 28:
        raise ValueError(f'UCI301 archive column-count mismatch: train={train.shape}, test={test.shape}')
    train.columns=['participant_id',*UCI301_FEATURE_NAMES,'updrs','class']
    test.columns=['participant_id',*UCI301_FEATURE_NAMES,'class']
    train['participant_id']=train.participant_id.astype(str)
    test['participant_id']=test.participant_id.astype(str)
    train['group']=np.where(train['class'].eq(1),'PD','HC')
    test['group']=np.where(test['class'].eq(1),'PD','HC')
    train['sample_order']=train.groupby('participant_id').cumcount()+1
    test['repeat_order']=test.groupby('participant_id').cumcount()+1
    train['task']=train.sample_order.map(lambda k: 'a' if k==1 else 'o' if k==2 else 'u' if k==3 else 'number' if k<=13 else 'sentence' if k<=17 else 'word')
    test['task']=np.where(test.repeat_order<=3,'a','o')
    return train,test


def discover_audio() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    fig_meta=pd.read_excel(FIGSHARE/'Demographics_age_sex.xlsx',sheet_name='Parselmouth')
    meta=fig_meta.rename(columns={'Sample ID':'sample_id','Label':'group','Age':'age','Sex':'sex'}).copy()
    meta['sample_id']=meta.sample_id.astype(str)
    meta['group']=meta.group.replace({'PwPD':'PD','HC':'HC'})
    meta_counts=meta.sample_id.value_counts()
    rows=[]; features=[]; hashes={}
    figfiles=sorted(FIGSHARE.glob('*_AH/*.wav'))
    for index,p in enumerate(figfiles,1):
        if index==1 or index%20==0 or index==len(figfiles): print(f'Figshare audio features {index}/{len(figfiles)}',flush=True)
        stem=p.stem; sample_id=stem
        folder_group='PD' if p.parent.name=='PD_AH' else 'HC'
        m=meta.loc[meta.sample_id.eq(sample_id)]
        group=str(m.iloc[0].group) if len(m)==1 else folder_group
        rel=str(p.relative_to(FIGSHARE))
        filename_anomaly='' if len(m)==1 and group==folder_group else 'workbook_id_or_group_mismatch'
        qc=audio_qc(p,'Figshare','sustained_a',group,sample_id,rel,filename_anomaly)
        qc['metadata_match_count']=len(m); qc['metadata_group']=str(m.iloc[0].group) if len(m)==1 else ''
        qc['exact_duplicate_in_dataset']=qc.get('sha256','') in hashes if qc.get('sha256') else False
        if qc.get('sha256'): hashes[qc['sha256']]=rel
        rows.append(qc)
        md={'age':float(m.iloc[0].age) if len(m)==1 and pd.notna(m.iloc[0].age) else np.nan,
            'sex':str(m.iloc[0].sex) if len(m)==1 else ''}
        features.append(compute_stable_vowel(p,'Figshare','sustained_a',group,sample_id,rel,**md))
    fig_feature=pd.DataFrame(features)
    fig_feature['metadata_match_count']=fig_feature.participant_id.map(meta_counts).fillna(0).astype(int)
    # A dataset-specific QC flag, not an exclusion.
    mdvr_rows=[]; mdvr_feats=[]; mdvr_hashes={}
    mdvrfiles=sorted(MDVR.rglob('*.wav'))
    for index,p in enumerate(mdvrfiles,1):
        if index==1 or index%10==0 or index==len(mdvrfiles): print(f'MDVR-KCL audio features {index}/{len(mdvrfiles)}',flush=True)
        rel=str(p.relative_to(MDVR)); parts=p.relative_to(MDVR).parts
        task_folder,folder_group=parts[0],parts[1]
        task='read_text' if task_folder=='ReadText' else 'spontaneous_dialogue'
        group='HC' if folder_group=='HC' else 'PD'
        stem=p.stem
        match=re.match(r'(ID\d+)',stem,re.I)
        pid=match.group(1).upper() if match else stem
        anomaly=''
        if stem.startswith('ID22hc'):
            pid='ID22'; anomaly='ID22HC_filename_separator_missing; participant key normalized from ID digits and verified against HC folder'
        # Verify folder group and filename's embedded label where one is parseable.
        lab=re.search(r'ID\d+[_-]?([a-zA-Z]+)',stem)
        if lab:
            embedded=lab.group(1).lower()
            if embedded in {'pd','hc'} and embedded.upper()!=group:
                anomaly=';'.join(filter(None,[anomaly,'filename_group_conflicts_with_folder']))
        qc=audio_qc(p,'MDVR-KCL',task,group,pid,rel,anomaly)
        qc['exact_duplicate_in_dataset']=qc.get('sha256','') in mdvr_hashes if qc.get('sha256') else False
        if qc.get('sha256'): mdvr_hashes[qc['sha256']]=rel
        mdvr_rows.append(qc); mdvr_feats.append(mdvr_features(p,task,group,pid,rel))
    return pd.DataFrame(rows),fig_feature,pd.DataFrame(mdvr_rows),pd.DataFrame(mdvr_feats)


def build_uci_tables():
    # UCI 174: one participant-level median across repeated recordings.
    ox=pd.read_csv(DATASETS/'kaggle/parkinsons.data')
    ox['participant_id']=ox['name'].str.extract(r'(S\d+)',flags=re.I)[0].str.upper()
    ox['group']=np.where(ox.status.eq(1),'PD','HC')
    ox_consistency=ox.groupby('participant_id').agg(recordings=('name','size'),group_count=('group','nunique'),group=('group','first')).reset_index()
    ox_subject=ox.groupby(['participant_id','group'],as_index=False).agg(**{v:(col,'median') for v,col in UCI174_MAP.items()})
    # Telemonitoring duplicate source copies are checked by SHA before use.
    tm=pd.read_csv(DATASETS/'UCI_csv/parkinsons_updrs.data')
    tm['participant_id']=tm['subject#'].astype(str)
    tm_subject=tm.groupby('participant_id',as_index=False).agg(age=('age','first'),sex=('sex','first'),
        motor_UPDRS=('motor_UPDRS','median'),total_UPDRS=('total_UPDRS','median'),
        test_time_span=('test_time',lambda x:float(np.max(x)-np.min(x))),n_recordings=('test_time','size'),
        jitter_local=('Jitter(%)','median'),shimmer_local=('Shimmer','median'),hnr_mean_db=('HNR','median'),
        ppe=('PPE','median'),rpde=('RPDE','median'))
    u301,test301=load_uci301()
    # The subject id occurs once per each of the 26 documented recording types.
    u301_task=u301.loc[u301.sample_order.eq(1)].copy()
    return ox,ox_consistency,ox_subject,tm,tm_subject,u301,test301,u301_task

