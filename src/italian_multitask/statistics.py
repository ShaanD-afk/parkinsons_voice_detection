from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY_FEATURES = ["median_f0_hz", "jitter_local", "shimmer_local", "hnr_mean_db"]
FEATURES = ["median_f0_hz","f0_iqr_hz","f0_std_hz","voiced_frame_fraction","usable_voiced_duration_s",
            "jitter_local","shimmer_local","hnr_mean_db","f1_hz","f2_hz","f3_hz",
            "f1_iqr_hz","f2_iqr_hz","f3_iqr_hz","spectral_centroid_hz","spectral_rolloff_85_hz","spectral_flatness"]
BOOTSTRAPS = 2000
SEED = 42


def _cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y): return np.nan
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2*u/(len(x)*len(y))-1)


def _hedges_g(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or len(y) < 2: return np.nan
    dof = len(x)+len(y)-2
    pooled = np.sqrt(((len(x)-1)*np.var(x,ddof=1)+(len(y)-1)*np.var(y,ddof=1))/dof)
    if pooled == 0: return 0.0
    correction = 1-3/(4*dof-1) if dof > 1 else 1
    return float(correction*(np.mean(x)-np.mean(y))/pooled)


def _bootstrap_diff(x, y, statistic, seed):
    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y): return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    xb=x[rng.integers(0,len(x),(BOOTSTRAPS,len(x)))]
    yb=y[rng.integers(0,len(y),(BOOTSTRAPS,len(y)))]
    diffs=statistic(xb,axis=1)-statistic(yb,axis=1)
    return tuple(np.percentile(diffs, [2.5,97.5]).astype(float))


def _bootstrap_effect(x, y, effect, seed):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<2 or len(y)<2: return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    xb=x[rng.integers(0,len(x),(BOOTSTRAPS,len(x)))]
    yb=y[rng.integers(0,len(y),(BOOTSTRAPS,len(y)))]
    if effect is _hedges_g:
        nx,ny=len(x),len(y); dof=nx+ny-2
        vx=np.var(xb,axis=1,ddof=1);vy=np.var(yb,axis=1,ddof=1)
        pooled=np.sqrt(((nx-1)*vx+(ny-1)*vy)/dof)
        correction=1-3/(4*dof-1) if dof>1 else 1
        values=np.divide((np.mean(xb,axis=1)-np.mean(yb,axis=1))*correction,pooled,
                         out=np.zeros(BOOTSTRAPS),where=pooled>0)
    else:
        gt=xb[:,:,None]>yb[:,None,:]
        lt=xb[:,:,None]<yb[:,None,:]
        values=(gt.sum(axis=(1,2))-lt.sum(axis=(1,2)))/(len(x)*len(y))
    values=values[np.isfinite(values)]
    return tuple(np.percentile(values,[2.5,97.5]).astype(float)) if len(values) else (np.nan,np.nan)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid): return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*p[idx])
        out[idx] = min(1.0, running)
    return out


def descriptive_and_group_comparison(features: pd.DataFrame, output_dir, prefix=""):
    output_dir.mkdir(parents=True, exist_ok=True)
    id_cols = ["task_code", "group", "participant_folder", "relative_path", "sample_rate_original_hz"]
    numeric = [c for c in FEATURES if c in features]
    desc = []
    compare = []
    outliers = []
    analysis_features=features.copy()
    for c in ["f1_hz","f2_hz","f3_hz"]:
        valid_col=c.replace("_hz","_range_valid")
        if c in analysis_features and valid_col in analysis_features:
            analysis_features.loc[~analysis_features[valid_col].fillna(False).astype(bool),c]=np.nan
    for (task, feature), frame in analysis_features.melt(id_vars=id_cols, value_vars=numeric,
                                                 var_name="feature", value_name="value").groupby(["task_code","feature"]):
        for group in ["ELDERLY_HC", "PD"]:
            z = frame[(frame.group==group) & frame.value.notna()].value.astype(float).to_numpy()
            desc.append({"task_code":task,"feature":feature,"group":group,"n_valid":len(z),
                         "n_missing":int(((frame.group==group)&frame.value.isna()).sum()),
                         "mean":float(np.mean(z)) if len(z) else np.nan,
                         "sd":float(np.std(z,ddof=1)) if len(z)>1 else np.nan,
                         "median":float(np.median(z)) if len(z) else np.nan,
                         "q1":float(np.percentile(z,25)) if len(z) else np.nan,
                         "q3":float(np.percentile(z,75)) if len(z) else np.nan})
        x=frame[(frame.group=="ELDERLY_HC")&frame.value.notna()].value.astype(float).to_numpy()
        y=frame[(frame.group=="PD")&frame.value.notna()].value.astype(float).to_numpy()
        if len(x)>=2 and len(y)>=2:
            ci_mean=_bootstrap_diff(x,y,np.mean,SEED+len(compare))
            ci_median=_bootstrap_diff(x,y,np.median,SEED+len(compare)+1000)
            mw=float(stats.mannwhitneyu(x,y,alternative="two-sided").pvalue)
            if np.ptp(np.r_[x,y]) <= np.finfo(float).eps*max(1.0,float(np.max(np.abs(np.r_[x,y])))):
                welch=np.nan
                welch_status="not_estimable_constant_feature"
            else:
                import warnings
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always",RuntimeWarning)
                    welch=float(stats.ttest_ind(x,y,equal_var=False).pvalue)
                welch_status="precision_warning_near_constant" if caught else "ok"
            g=_hedges_g(x,y); delta=_cliffs_delta(x,y)
            gci=_bootstrap_effect(x,y,_hedges_g,SEED+len(compare)+2000)
            dci=_bootstrap_effect(x,y,_cliffs_delta,SEED+len(compare)+3000)
            compare.append({"task_code":task,"feature":feature,"n_hc":len(x),"n_pd":len(y),
                           "mean_hc":float(np.mean(x)),"sd_hc":float(np.std(x,ddof=1)),
                           "median_hc":float(np.median(x)),"q1_hc":float(np.percentile(x,25)),"q3_hc":float(np.percentile(x,75)),
                           "mean_pd":float(np.mean(y)),"sd_pd":float(np.std(y,ddof=1)),
                           "median_pd":float(np.median(y)),"q1_pd":float(np.percentile(y,25)),"q3_pd":float(np.percentile(y,75)),
                           "mean_difference_hc_minus_pd":float(np.mean(x)-np.mean(y)),"mean_diff_ci95_low":ci_mean[0],"mean_diff_ci95_high":ci_mean[1],
                           "median_difference_hc_minus_pd":float(np.median(x)-np.median(y)),"median_diff_ci95_low":ci_median[0],"median_diff_ci95_high":ci_median[1],
                           "hedges_g_hc_minus_pd":g,"hedges_g_ci95_low":gci[0],"hedges_g_ci95_high":gci[1],
                           "cliffs_delta_hc_minus_pd":delta,"cliffs_delta_ci95_low":dci[0],"cliffs_delta_ci95_high":dci[1],
                           "mann_whitney_p_unadjusted":mw,"welch_t_p_unadjusted":welch,"welch_test_status":welch_status})
        vals=frame[frame.value.notna()]
        for group, gf in vals.groupby("group"):
            q1,q3=gf.value.quantile([.25,.75]); iqr=q3-q1
            suspicious=gf[(gf.value<q1-3*iqr)|(gf.value>q3+3*iqr)] if iqr>0 else gf.iloc[0:0]
            for r in suspicious.itertuples():
                outliers.append({"task_code":task,"feature":feature,"group":group,"participant_folder":r.participant_folder,"value":r.value,"review_reason":"beyond 3x sample IQR; flag only, no exclusion"})
    desc=pd.DataFrame(desc); comp=pd.DataFrame(compare)
    if not comp.empty:
        comp["holm_p_within_feature_across_vowels"] = np.nan
        # Multiplicity family: same feature across five vowel contexts (primary or exploratory feature).
        primary_ix=comp[comp.feature.isin(PRIMARY_FEATURES)&comp.task_code.isin(["VA1","VE1","VI1","VO1","VU1"])].index
        comp.loc[primary_ix,"holm_p_primary_20_comparisons"] = holm_adjust(comp.loc[primary_ix,"mann_whitney_p_unadjusted"].to_numpy())
        for feature, ix in comp[comp.task_code.isin(["VA1","VE1","VI1","VO1","VU1"])].groupby("feature").groups.items():
            comp.loc[ix,"holm_p_within_feature_across_vowels"] = holm_adjust(comp.loc[ix,"mann_whitney_p_unadjusted"].to_numpy())
    desc.to_csv(output_dir/f"{prefix}feature_descriptive_statistics.csv",index=False)
    comp.to_csv(output_dir/f"{prefix}group_comparisons_effect_sizes_ci.csv",index=False)
    pd.DataFrame(outliers).to_csv(output_dir/f"{prefix}outliers_for_review.csv",index=False)
    return desc, comp, pd.DataFrame(outliers)


def paired_vowel_analysis(features: pd.DataFrame, output_dir, prefix=""):
    output_dir.mkdir(parents=True,exist_ok=True)
    vowels=["VA1","VE1","VI1","VO1","VU1"]
    candidate=["median_f0_hz","f0_iqr_hz","jitter_local","shimmer_local","hnr_mean_db","f1_hz","f2_hz"]
    pair_rows=[]; omnibus=[]
    for feature in candidate:
        if feature not in features: continue
        for group in ["ELDERLY_HC","PD"]:
            sub=features[features.group.eq(group)&features.task_code.isin(vowels)]
            wide=sub.pivot_table(index="participant_folder",columns="task_code",values=feature,aggfunc="first").reindex(columns=vowels)
            complete=wide.dropna()
            if len(complete)>=3:
                try: fr=float(stats.friedmanchisquare(*[complete[t].to_numpy() for t in vowels]).pvalue)
                except Exception: fr=np.nan
            else: fr=np.nan
            omnibus.append({"feature":feature,"group":group,"n_complete_all_five_vowels":len(complete),"friedman_p_unadjusted":fr})
            local=[]
            for i,a in enumerate(vowels):
                for b in vowels[i+1:]:
                    pair=wide[[a,b]].dropna(); d=(pair[a]-pair[b]).to_numpy()
                    if len(d):
                        rng=np.random.default_rng(SEED+len(pair_rows))
                        boot=np.array([np.median(rng.choice(d,size=len(d),replace=True)) for _ in range(BOOTSTRAPS)])
                        try: p=float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0
                        except Exception: p=np.nan
                        row={"feature":feature,"group":group,"task_a":a,"task_b":b,"n_complete_pair":len(d),
                             "mean_paired_difference_a_minus_b":float(np.mean(d)),"median_paired_difference_a_minus_b":float(np.median(d)),
                             "median_diff_boot_ci95_low":float(np.percentile(boot,2.5)),"median_diff_boot_ci95_high":float(np.percentile(boot,97.5)),
                             "wilcoxon_p_unadjusted":p}
                        local.append(row)
            if local:
                adj=holm_adjust([r["wilcoxon_p_unadjusted"] for r in local])
                for row,padj in zip(local,adj): row["holm_p_within_feature_group_10_pairs"]=padj
                pair_rows.extend(local)
    pd.DataFrame(pair_rows).to_csv(output_dir/f"{prefix}paired_vowel_comparisons.csv",index=False)
    pd.DataFrame(omnibus).to_csv(output_dir/f"{prefix}vowel_friedman_omnibus.csv",index=False)
    # Explicit complete-case account for every feature and group.
    cc=[]
    for feature in candidate:
        if feature not in features: continue
        for group in ["ELDERLY_HC","PD"]:
            sub=features[features.group.eq(group)&features.task_code.isin(vowels)]
            w=sub.pivot_table(index="participant_folder",columns="task_code",values=feature,aggfunc="first").reindex(columns=vowels)
            full=w.dropna()
            cc.append({"feature":feature,"group":group,"participants_with_any_vowel":int(w.notna().any(axis=1).sum()),"complete_all_five_vowels":len(full),"incomplete":int(w.notna().any(axis=1).sum()-len(full))})
    pd.DataFrame(cc).to_csv(output_dir/f"{prefix}paired_vowel_complete_participants.csv",index=False)
    return pd.DataFrame(pair_rows),pd.DataFrame(omnibus),pd.DataFrame(cc)
