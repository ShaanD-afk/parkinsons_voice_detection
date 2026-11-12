from __future__ import annotations

"""Reproducible non-classification analyses for the remaining ItalianPVS tasks."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import scipy
from scipy.signal import find_peaks, stft
from scipy.stats import (bootstrap, mannwhitneyu, median_abs_deviation,
                         ttest_ind, trim_mean, wilcoxon)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "italian_research_phase2"
DATASET_DEFAULT = Path("/Users/Shaan/Desktop/datasets/italian")
sys.path.insert(0, str(ROOT))
from src.italian_multitask.audio import mono_for_analysis, read_audio
from src.italian_multitask.segmentation import rms_vad
from src.italian_multitask.features.vowels import extract_sustained_vowel

SEED = 42026
GROUP_HC, GROUP_PD = "ELDERLY_HC", "PD"
FEATURES = ["median_f0_hz", "jitter_local", "shimmer_local", "hnr_mean_db"]


def primary_manifest():
    m = pd.read_csv(ROOT / "data/primary_cohort_manifest.csv")
    m = m[m.include_primary_cohort.fillna(False).astype(bool) & m.cohort_group.isin([GROUP_HC, GROUP_PD])].copy()
    m["sample_rate"] = pd.to_numeric(m.sample_rate, errors="coerce")
    return m


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    out = np.empty(len(p), float)
    running = 0.0
    n = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[idx]))
        out[idx] = running
    return out


def _hedges(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / max(nx + ny - 2, 1))
    if not pooled:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / pooled
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return float(correction * d)


def sensitivity_analysis():
    base = pd.read_csv(ROOT / "results/italian_multitask_v1/features/primary_native16_first_vowels.csv")
    combined = pd.read_csv(ROOT / "results/italian_multitask_v1/sampling_rate_sensitivity/combined_native16_plus_resampled_features.csv")
    combined["sample_rate_original_hz"] = combined.sample_rate_original_hz.astype(int)
    frames = [("native16", base), ("combined_resampled_sensitivity", combined)]
    out_rows, loo_rows = [], []
    rng = np.random.default_rng(SEED)
    for cohort_label, frame in frames:
        for task in ["VA1", "VE1", "VI1", "VO1", "VU1"]:
            for feature in FEATURES:
                sub = frame[frame.task_code.eq(task)]
                x = sub.loc[sub.group.eq(GROUP_HC), feature].dropna().to_numpy(float)
                y = sub.loc[sub.group.eq(GROUP_PD), feature].dropna().to_numpy(float)
                if min(len(x), len(y)) < 2:
                    continue
                # Independent participant bootstrap; percentile intervals are reported explicitly.
                bx = rng.choice(x, size=(10000, len(x)), replace=True)
                by = rng.choice(y, size=(10000, len(y)), replace=True)
                boot_mean = bx.mean(axis=1) - by.mean(axis=1)
                boot_median = np.median(bx, axis=1) - np.median(by, axis=1)
                mw = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
                tt = ttest_ind(x, y, equal_var=False)
                trim_t = ttest_ind(x, y, equal_var=False, trim=0.2)
                out_rows.append({
                    "cohort": cohort_label, "task_code": task, "feature": feature,
                    "n_hc": len(x), "n_pd": len(y), "mean_diff_hc_minus_pd": np.mean(x)-np.mean(y),
                    "mean_diff_boot_pct_low": np.quantile(boot_mean, .025), "mean_diff_boot_pct_high": np.quantile(boot_mean, .975),
                    "median_diff_hc_minus_pd": np.median(x)-np.median(y),
                    "median_diff_boot_pct_low": np.quantile(boot_median, .025), "median_diff_boot_pct_high": np.quantile(boot_median, .975),
                    "trimmed_mean_diff_20pct": trim_mean(x, .2)-trim_mean(y, .2),
                    "winsorized_mean_diff_10pct": np.mean(np.clip(x, *np.quantile(x,[.1,.9])))-np.mean(np.clip(y,*np.quantile(y,[.1,.9]))),
                    "hedges_g": _hedges(x,y), "mann_whitney_p": mw.pvalue,
                    "welch_p": tt.pvalue, "yuen_20pct_p": trim_t.pvalue,
                    "hc_median_abs_deviation": median_abs_deviation(x, scale="normal"),
                    "pd_median_abs_deviation": median_abs_deviation(y, scale="normal"),
                    "ci_method": "independent participant percentile bootstrap; 10,000 replicates; seed 42026",
                })
                for group, vals in [(GROUP_HC,x),(GROUP_PD,y)]:
                    for i, value in enumerate(vals):
                        reduced = vals[np.arange(len(vals)) != i]
                        other = y if group == GROUP_HC else x
                        loo_rows.append({"cohort":cohort_label,"task_code":task,"feature":feature,"left_out_group":group,
                                         "left_out_index":i,"left_out_value":value,"mean_diff_hc_minus_pd_loo":
                                         (reduced.mean()-other.mean()) if group==GROUP_HC else (other.mean()-reduced.mean()),
                                         "hedges_g_loo":_hedges(reduced,other) if group==GROUP_HC else _hedges(other,reduced)})
    res = pd.DataFrame(out_rows)
    # Multiplicity correction within each declared 20-contrast family and per sensitivity family.
    res["holm_mw_p_within_cohort_20"] = res.groupby("cohort").mann_whitney_p.transform(holm)
    res["holm_yuen_p_within_cohort_20"] = res.groupby("cohort").yuen_20pct_p.transform(holm)
    res.to_csv(OUT / "sensitivity/group_contrast_sensitivity.csv", index=False)
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(OUT / "sensitivity/leave_one_participant_out.csv", index=False)
    # Direct VI1 shimmer robustness table at the statistic level, including influential cases.
    vi = loo[(loo.task_code == "VI1") & (loo.feature == "shimmer_local")]
    vi.to_csv(OUT / "sensitivity/vi1_shimmer_leave_one_out.csv", index=False)
    # Original-rate comparison: 44.1 kHz stratum is PD-only, so only descriptive PD shifts are estimable.
    native44 = pd.read_csv(ROOT / "results/italian_multitask_v1/sampling_rate_sensitivity/native_44100_first_vowels.csv")
    rate_rows = []
    for task in ["VA1", "VE1", "VI1", "VO1", "VU1"]:
        for f in FEATURES:
            a = base.loc[base.task_code.eq(task) & base.group.eq(GROUP_PD), f].dropna().to_numpy(float)
            b = native44.loc[native44.task_code.eq(task) & native44.group.eq(GROUP_PD), f].dropna().to_numpy(float)
            rate_rows.append({"task_code":task,"feature":f,"pd_native16_n":len(a),"pd_native44100_n":len(b),
                              "pd_native16_median":np.median(a),"pd_native44100_median":np.median(b),
                              "pd_native16_mean":np.mean(a),"pd_native44100_mean":np.mean(b),
                              "between_group_contrast_estimable_at_44100":False,
                              "reason":"No HC recordings are available at 44.1 kHz."})
    pd.DataFrame(rate_rows).to_csv(OUT / "sensitivity/original_rate_strata.csv", index=False)
    # Verify the headline signal and interval construction against the saved raw participant table.
    vi=base[base.task_code.eq("VI1")]
    vx=vi.loc[vi.group.eq(GROUP_HC),"shimmer_local"].dropna().to_numpy(float)
    vy=vi.loc[vi.group.eq(GROUP_PD),"shimmer_local"].dropna().to_numpy(float)
    def mean_difference(a,b,axis): return np.mean(a,axis=axis)-np.mean(b,axis=axis)
    bca=bootstrap((vx,vy),mean_difference,vectorized=True,paired=False,axis=0,n_resamples=10000,
                  confidence_level=.95,method="BCa",rng=np.random.default_rng(SEED+1))
    orig=pd.read_csv(ROOT/"results/italian_multitask_v1/statistics/native16_group_comparisons_effect_sizes_ci.csv")
    old=orig[(orig.task_code=="VI1")&(orig.feature=="shimmer_local")].iloc[0]
    sens_row=res[(res.cohort=="native16")&(res.task_code=="VI1")&(res.feature=="shimmer_local")].iloc[0]
    pd.DataFrame([{"task_code":"VI1","feature":"shimmer_local","n_hc":len(vx),"n_pd":len(vy),
       "raw_mean_difference":float(np.mean(vx)-np.mean(vy)),"historical_mean_difference":float(old.mean_difference_hc_minus_pd),
       "raw_mann_whitney_p":float(mannwhitneyu(vx,vy,alternative="two-sided",method="asymptotic").pvalue),
       "historical_mann_whitney_p":float(old.mann_whitney_p_unadjusted),
       "historical_primary_holm_p":float(old.holm_p_primary_20_comparisons),
       "new_participant_bootstrap_percentile_ci_low":float(sens_row.mean_diff_boot_pct_low),
       "new_participant_bootstrap_percentile_ci_high":float(sens_row.mean_diff_boot_pct_high),
       "new_participant_bootstrap_bca_ci_low":float(bca.confidence_interval.low),
       "new_participant_bootstrap_bca_ci_high":float(bca.confidence_interval.high),
       "original_ci_method":"percentile participant bootstrap (5,000 replicates in historical config)",
       "new_ci_method":"independent HC/PD participant bootstrap, 10,000 replicates; percentile and BCa; seed 42027",
       "bca_interval_excludes_zero":bool(bca.confidence_interval.low>0 or bca.confidence_interval.high<0)}]).to_csv(
           OUT/"sensitivity/vi1_shimmer_bootstrap_interval_audit.csv",index=False)
    return res, loo


def second_variants(dataset_root: Path):
    m = primary_manifest()
    codes = ["VA1","VA2","VE1","VE2","VI1","VI2","VO1","VO2","VU1","VU2"]
    rows = m[m.possible_task_code.isin(codes)].copy()
    feat_rows=[]; qc_rows=[]
    for row in rows.itertuples(index=False):
        path=dataset_root/row.relative_path
        try:
            x,sr,info=read_audio(path); audio,_=mono_for_analysis(x)
            duration=len(audio)/sr
            rms=np.sqrt(np.mean(np.square(audio))) if len(audio) else np.nan
            peak=np.max(np.abs(audio)) if len(audio) else np.nan
            pcm=parselmouth.Sound(audio,sampling_frequency=sr)
            pitch=pcm.to_pitch_ac(time_step=.01,pitch_floor=75,pitch_ceiling=500)
            f0=pitch.selected_array["frequency"]
            voiced=np.isfinite(f0)&(f0>0)
            runs=np.diff(np.r_[False,voiced,False].astype(int))
            starts=np.where(runs==1)[0]; ends=np.where(runs==-1)[0]
            n_runs=len(starts)
            # Descriptive complete-file measures; deliberately no inferential repeatability claims.
            feat={"relative_path":row.relative_path,"participant_folder":row.participant_folder,"group":row.cohort_group,
                  "task_code":row.possible_task_code,"sample_rate_hz":int(sr),"duration_s":duration,
                  "rms_amplitude":rms,"peak_amplitude":peak,"voiced_fraction_pitch75_500":float(np.mean(voiced)) if len(voiced) else np.nan,
                  "voiced_run_count":n_runs,"longest_voiced_run_s":float(max((e-s for s,e in zip(starts,ends)),default=0)*.01),
                  "median_f0_hz":float(np.median(f0[voiced])) if voiced.any() else np.nan,
                  "median_f0_iqr_hz":float(np.subtract(*np.percentile(f0[voiced],[75,25]))) if voiced.any() else np.nan}
            feat_rows.append(feat)
            qc_rows.append({**feat,"signal_read":"PASS","signal_structure":"descriptive_only_no_trial_interpretation"})
        except Exception as e:
            qc_rows.append({"relative_path":row.relative_path,"participant_folder":row.participant_folder,"group":row.cohort_group,
                            "task_code":row.possible_task_code,"signal_read":"FAIL","failure":f"{type(e).__name__}: {e}"})
    f=pd.DataFrame(feat_rows)
    f.to_csv(OUT/"second_variants/second_and_first_variant_signal_characterization.csv",index=False)
    # Common participant paired descriptive differences, same vowel only; no repeatability interpretation.
    paired=[]
    for stem in "AEIOU":
        first,second=f"V{stem}1",f"V{stem}2"
        a=f[f.task_code.eq(first)].set_index(["group","participant_folder"])
        b=f[f.task_code.eq(second)].set_index(["group","participant_folder"])
        for feature in ["duration_s","voiced_fraction_pitch75_500","voiced_run_count","longest_voiced_run_s","median_f0_hz"]:
            common=a.index.intersection(b.index)
            for key in common:
                if pd.notna(a.loc[key,feature]) and pd.notna(b.loc[key,feature]):
                    paired.append({"vowel_context":stem.lower(),"first_code":first,"second_code":second,
                                   "group":key[0],"participant_folder":key[1],"feature":feature,
                                   "first_value":a.loc[key,feature],"second_value":b.loc[key,feature],
                                   "second_minus_first":b.loc[key,feature]-a.loc[key,feature],
                                   "interpretation":"descriptive code contrast; variant relationship unknown"})
    pd.DataFrame(paired).to_csv(OUT/"second_variants/paired_variant_descriptives.csv",index=False)
    return f


def _event_features(audio, sr):
    # Generic acoustic event diagnostic; threshold/peak count is not validated syllable count.
    frame=int(.010*sr); hop=frame
    pad=np.pad(audio,(0,max(0,frame-len(audio))))
    n=max(1,1+(len(pad)-frame)//hop)
    rms=np.array([np.sqrt(np.mean(pad[i*hop:i*hop+frame]**2)) for i in range(n)])
    smooth=np.convolve(rms,np.ones(5)/5,mode="same")
    threshold=max(np.quantile(smooth,.35)*1.8, np.max(smooth)*.025,1e-5)
    active=smooth>=threshold
    edges=np.diff(np.r_[False,active,False].astype(int)); starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]
    # Candidate local rises from a smoothed energy envelope with 120 ms minimum spacing.
    peaks,_=find_peaks(smooth,distance=max(1,int(.12*sr/hop)),prominence=max(np.std(smooth)*.35,threshold*.15))
    duration=len(audio)/sr
    return {"duration_s":duration,"event_threshold_rms":threshold,"candidate_energy_peaks_n":int(len(peaks)),
            "active_segments_n":int(len(starts)),"active_duration_s":float(np.sum(active)*hop/sr),
            "pause_fraction_thresholded":float(1-np.mean(active)) if len(active) else np.nan,
            "candidate_event_times_s":";".join(f"{t*hop/sr:.3f}" for t in peaks),
            "event_intervals_s":";".join(f"{dt*hop/sr:.3f}" for dt in np.diff(peaks))}


def d1d2_characterize(dataset_root: Path):
    m=primary_manifest(); m=m[m.possible_task_code.isin(["D1","D2"])].copy()
    rows=[]
    for r in m.itertuples(index=False):
        try:
            x,sr,_=read_audio(dataset_root/r.relative_path); audio,_=mono_for_analysis(x)
            f=_event_features(audio,sr)
            snd=parselmouth.Sound(audio,sampling_frequency=sr)
            p=snd.to_pitch_ac(time_step=.01,pitch_floor=75,pitch_ceiling=500).selected_array["frequency"]
            f["pitch_voiced_fraction"]=float(np.mean(np.isfinite(p)&(p>0))) if len(p) else np.nan
            f["median_f0_hz"]=float(np.median(p[np.isfinite(p)&(p>0)])) if np.any(np.isfinite(p)&(p>0)) else np.nan
            # Event interval variability is diagnostic only until manual reference exists.
            ints=np.diff(np.asarray([float(v) for v in f["candidate_event_times_s"].split(";") if v])) if f["candidate_event_times_s"] else np.array([])
            f["candidate_interval_median_s"]=float(np.median(ints)) if len(ints) else np.nan
            f["candidate_interval_cv"]=float(np.std(ints,ddof=1)/np.mean(ints)) if len(ints)>1 and np.mean(ints)>0 else np.nan
            rows.append({"relative_path":r.relative_path,"participant_folder":r.participant_folder,"group":r.cohort_group,
                         "task_code":r.possible_task_code,"sample_rate_hz":int(sr),**f,
                         "interpretation":"automated envelope events; not validated syllable/onset counts"})
        except Exception as e:
            rows.append({"relative_path":r.relative_path,"participant_folder":r.participant_folder,"group":r.cohort_group,
                         "task_code":r.possible_task_code,"failure":f"{type(e).__name__}: {e}"})
    d=pd.DataFrame(rows); d.to_csv(OUT/"d1d2/recording_characterization.csv",index=False)
    summary=d.groupby(["task_code","group","sample_rate_hz"],dropna=False).agg(
        recordings=("relative_path","count"),median_duration_s=("duration_s","median"),median_candidate_peaks=("candidate_energy_peaks_n","median"),
        median_active_segments=("active_segments_n","median"),median_pause_fraction=("pause_fraction_thresholded","median"),
        median_pitch_voiced_fraction=("pitch_voiced_fraction","median")).reset_index()
    summary.to_csv(OUT/"d1d2/coverage_summary.csv",index=False)
    return d,summary


def _vad_features(audio, sr, db_below_p95):
    levels,_,_=rms_vad(audio,sr)
    p95=float(np.percentile(levels,95)) if len(levels) else 0
    threshold=max(1e-7,p95*10**(db_below_p95/20))
    active=levels>=threshold
    # 10 ms isolated dropout closure as used by the original VAD pilot.
    if len(active)>2:
        mask=active.copy()
        for i in range(1,len(active)-1):
            if not active[i] and active[i-1] and active[i+1]: mask[i]=True
        active=mask
    edges=np.diff(np.r_[False,active,False].astype(int)); starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]
    hop=.01; total=len(active)*hop
    durations=(ends-starts)*hop
    pauses=np.diff(np.r_[0,starts, len(active)]) if False else None
    # Quiet gaps bounded by speech; exclude leading/trailing non-speech from internal pause count.
    gaps=[(starts[i+1]-ends[i])*hop for i in range(max(0,len(starts)-1))]
    snd=parselmouth.Sound(audio,sampling_frequency=sr)
    pitch=snd.to_pitch_ac(time_step=.01,pitch_floor=75,pitch_ceiling=500)
    f0=pitch.selected_array["frequency"]
    voiced=np.isfinite(f0)&(f0>0)
    hnr_vals=[]
    for s,e in zip(starts,ends):
        if e-s < 15: continue
        t0=s*hop; t1=min(e*hop,len(audio)/sr)
        try:
            part=parselmouth.Sound(audio[int(t0*sr):int(t1*sr)],sampling_frequency=sr)
            harm=part.to_harmonicity_cc(time_step=.01,minimum_pitch=75,silence_threshold=.1,periods_per_window=1.0)
            hv=harm.values[0]; hnr_vals.extend(hv[np.isfinite(hv)&(hv>-100)].tolist())
        except Exception: pass
    frame_intensity=np.log10(np.maximum(levels,1e-12))*20
    act_i=frame_intensity[active]
    return {"vad_threshold_db_below_p95":db_below_p95,"vad_threshold_rms":threshold,
            "duration_s":len(audio)/sr,"active_speech_duration_s":float(np.sum(durations)),
            "internal_pause_count":len(gaps),"internal_pause_duration_s":float(np.sum(gaps)),
            "internal_pause_median_s":float(np.median(gaps)) if gaps else np.nan,
            "internal_pause_rate_per_active_min":float(len(gaps)/(np.sum(durations)/60)) if np.sum(durations)>0 else np.nan,
            "speech_to_internal_pause_ratio":float(np.sum(durations)/np.sum(gaps)) if np.sum(gaps)>0 else np.nan,
            "active_speech_proportion":float(np.sum(durations)/total) if total>0 else np.nan,
            "median_f0_hz":float(np.median(f0[voiced])) if voiced.any() else np.nan,
            "f0_iqr_hz":float(np.subtract(*np.percentile(f0[voiced],[75,25]))) if voiced.any() else np.nan,
            "f0_range_hz":float(np.max(f0[voiced])-np.min(f0[voiced])) if voiced.any() else np.nan,
            "voiced_frame_proportion":float(voiced.mean()) if len(voiced) else np.nan,
            "pitch_variability_sd_hz":float(np.std(f0[voiced],ddof=1)) if voiced.sum()>1 else np.nan,
            "relative_intensity_iqr_db":float(np.subtract(*np.percentile(act_i,[75,25]))) if len(act_i) else np.nan,
            "relative_intensity_sd_db":float(np.std(act_i,ddof=1)) if len(act_i)>1 else np.nan,
            "hnr_voiced_active_median_db":float(np.median(hnr_vals)) if hnr_vals else np.nan,
            "cpps_status":"deferred_no_validated_implementation"}


def connected_speech(dataset_root: Path):
    m=primary_manifest(); m=m[m.possible_task_code.isin(["B1","B2","FB1"])].copy()
    rows=[]
    for r in m.itertuples(index=False):
        try:
            x,sr,_=read_audio(dataset_root/r.relative_path); audio,_=mono_for_analysis(x)
            for db in [-30,-35,-40]:
                vals=_vad_features(audio,sr,db)
                rows.append({"relative_path":r.relative_path,"participant_folder":r.participant_folder,"group":r.cohort_group,
                             "task_code":r.possible_task_code,"sample_rate_hz":int(sr),**vals})
        except Exception as e:
            rows.append({"relative_path":r.relative_path,"participant_folder":r.participant_folder,"group":r.cohort_group,
                         "task_code":r.possible_task_code,"failure":f"{type(e).__name__}: {e}"})
    d=pd.DataFrame(rows); d.to_csv(OUT/"connected_speech/vad_threshold_sensitivity_long.csv",index=False)
    default=d[d.vad_threshold_db_below_p95.eq(-35)].copy()
    default.to_csv(OUT/"connected_speech/participant_recording_features.csv",index=False)
    agg=default.groupby(["task_code","group","sample_rate_hz"],dropna=False).agg(
        recordings=("relative_path","count"),participants=("participant_folder","nunique"),
        median_active_speech_duration_s=("active_speech_duration_s","median"),median_pause_count=("internal_pause_count","median"),
        median_pause_fraction=("active_speech_proportion",lambda s:1-s.median()),median_f0_hz=("median_f0_hz","median"),
        median_voiced_proportion=("voiced_frame_proportion","median"),median_hnr_voiced_db=("hnr_voiced_active_median_db","median")).reset_index()
    agg.to_csv(OUT/"connected_speech/task_group_summary.csv",index=False)
    # Within-person B1 -> B2 description only, pair matched original sample rate and participant.
    pair=[]
    features=["active_speech_duration_s","internal_pause_count","internal_pause_median_s","speech_to_internal_pause_ratio",
              "active_speech_proportion","median_f0_hz","f0_iqr_hz","voiced_frame_proportion","relative_intensity_iqr_db","hnr_voiced_active_median_db"]
    key=["group","participant_folder","sample_rate_hz"]
    a=default[default.task_code.eq("B1")].set_index(key); b=default[default.task_code.eq("B2")].set_index(key)
    for ix in a.index.intersection(b.index):
        for f in features:
            av,bv=a.loc[ix,f],b.loc[ix,f]
            if pd.notna(av) and pd.notna(bv):
                pair.append({"group":ix[0],"participant_folder":ix[1],"sample_rate_hz":ix[2],"feature":f,"B1_first_reading":av,
                             "B2_second_reading":bv,"B2_minus_B1":bv-av,"interpretation":"paired order contrast; practice/fatigue not separable"})
    pd.DataFrame(pair).to_csv(OUT/"connected_speech/b1_b2_paired_descriptives.csv",index=False)
    return d,default,agg


def dataset_compatibility():
    # Empirical local inventories are recorded separately from literature-level documented facts.
    import soundfile as sf
    root=Path("/Users/Shaan/Desktop/datasets")
    rows=[]
    for name,folder in [("ItalianPVS",root/"italian"),("Figshare_AH",root/"figshare"),("MDVR-KCL",root/"MDVR-KCL")]:
        wavs=sorted(folder.rglob("*.wav")) if folder.exists() else []
        byrate={}; details=[]
        for p in wavs:
            try:
                info=sf.info(p); byrate[str(info.samplerate)]=byrate.get(str(info.samplerate),0)+1
                details.append((info.channels,info.subtype,info.duration))
            except Exception: pass
        rows.append({"dataset":name,"local_audio_files":len(wavs),"readable_audio_files":len(details),
                     "observed_sample_rates_hz_json":json.dumps(byrate,sort_keys=True),
                     "observed_channel_count_values":json.dumps(sorted({x[0] for x in details})),
                     "observed_subtypes":json.dumps(sorted({x[1] for x in details})),
                     "local_access_note":"inventory only; local copy was not merged or used for modeling"})
    # UCI tables are feature-only and have longitudinal repeated measures.
    uci=pd.read_csv(root/"UCI_csv/parkinsons_updrs.data")
    rows.append({"dataset":"UCI_Parkinsons_Telemonitoring","local_audio_files":0,"readable_audio_files":0,
                 "observed_sample_rates_hz_json":"not applicable: derived feature table; audio unavailable",
                 "observed_channel_count_values":"not available","observed_subtypes":"CSV numeric features",
                 "local_access_note":f"{len(uci)} feature rows; {uci['subject#'].nunique()} subjects; {uci.shape[1]} columns; PD-only longitudinal dataset"})
    pd.DataFrame(rows).to_csv(OUT/"compatibility/local_dataset_inventory.csv",index=False)
    # Cross-corpus task compatibility matrix based on known task documentation/local metadata.
    compat=[
      ["ItalianPVS","Italian","65 total speakers in source; reconciled cohort 17 HC/19 PD pre-rate, 17/12 at native 16 kHz","sustained vowels; /pa/ /ta/ five seconds; text readings; phonemically balanced phrases","16 kHz and 44.1 kHz; mono PCM16 observed in prior QC","participant folder identifiers; roster reconciliation exists","local audio and codebooks supplied","same-language task labels but task protocol partly unresolved"],
      ["Figshare AH vowel set","English","41 HC and 40 PD; 81 local WAVs; demographics workbook has 81 unique Sample IDs, one file per ID","sustained /a/ phonation; remote telephone capture reported in source paper","8 kHz, mono, FLOAT WAV observed locally","Sample ID in workbook matches one filename in group folder","Figshare item 23849127; item-specific license not independently verified because item page blocks automated access","phonation type broadly comparable; 8 kHz channel and remote acquisition differ; check item license before redistribution"],
      ["MDVR-KCL","English","37 people (21 HC/16 PD); 73 WAVs (31 PD/42 HC task recordings) locally","read text and spontaneous phone dialogue","44.1 kHz mono documented; local files decode as PCM_24, a bit-depth discrepancy","ID-based names and task folders","Zenodo record 2867216; open access, CC BY 4.0","reading task type comparable but text differs; spontaneous dialogue is not equivalent to ItalianPVS reading"],
      ["UCI Parkinson's Telemonitoring","English","42 PD; 5,875 rows","repeated home-recorded voice measurements; exact prompts unavailable in local table","not available from feature CSV; raw audio unavailable","subject# longitudinal identifier","CC BY 4.0 feature table at UCI; not raw voice files","cannot recompute harmonized acoustic measures; PD-only severity cohort, not binary HC-PD"],
      ["UCI Parkinson's Disease Classification (Sakar)","Turkish","252 people: 188 PD/64 HC; 756 recordings, 3 per person","three sustained /a/ phonations; UCI download provides 754 extracted features","44.1 kHz documented; local audio not found","subject identifier in feature records","CC BY 4.0 per UCI","phonation task is broadly comparable, but Turkish language and unavailable audio prevent harmonized extraction; not English integration candidate"],
      ["UCI Parkinsons / Oxford #174","Language not documented in UCI record","31 reported, 23 PD/8 HC; 195 voice-feature rows","repeated voice records per person; biomedical measures; no raw audio","not available","name field includes subject and recording number","CC BY 4.0 feature table at UCI","not usable for raw-audio harmonization or connected speech; possible feature-method comparison only"],
      ["mPower","English comprehension required; US iPhone app study","large longitudinal cohort; available participant/task counts depend on consent and data filtering","sustained /a/ up to 10 s; mobile app; repeated daily samples for some PD participants","mobile device formats/rates vary; inspect after approved access","coded subject IDs and repeated tasks","Synapse access requires validated account/certification, intended-use statement and data-specific terms; not downloaded","sustained /a/ potentially comparable; self-report labels, device/home conditions and medication timing need subset controls"],
    ]
    pd.DataFrame(compat,columns=["dataset","language","participants","tasks","sample_rate_audio","participant_ids","access_license","italian_compatibility"]).to_csv(OUT/"compatibility/cross_dataset_compatibility_matrix.csv",index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",type=Path,default=DATASET_DEFAULT); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"configs/experiment_config.json").write_text(json.dumps({"experiment_id":"ITALIAN-PHASE2-REMAINING-20261030",
       "source_experiment":"ITALIAN-MULTITASK-V1-20261030","seed":SEED,"sample_rates":"native; no classifier",
       "vad_thresholds_db_below_p95":[-30,-35,-40],"bootstrap_replicates":10000,"python":sys.version,
       "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"parselmouth":parselmouth.__version__},indent=2))
    sensitivity_analysis()
    second_variants(args.dataset_root)
    d1d2_characterize(args.dataset_root)
    connected_speech(args.dataset_root)
    dataset_compatibility()
    print(f"Completed initial analyses under {OUT}")


if __name__=="__main__": main()
