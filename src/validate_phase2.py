from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.italian_research_phase2 import OUT, ROOT
from src.italian_multitask.review_materials import refresh_review_summary


def main():
    checks=[]
    def add(name, passed, observed, expected):
        checks.append({"check":name,"status":"PASS" if passed else "FAIL","observed":str(observed),"expected":str(expected)})
    primary=pd.read_csv(ROOT/"results/italian_multitask_v1/features/primary_native16_first_vowels.csv")
    old=pd.read_csv(ROOT/"results/italian_multitask_v1/statistics/native16_group_comparisons_effect_sizes_ci.csv")
    new=pd.read_csv(OUT/"sensitivity/group_contrast_sensitivity.csv")
    d12=pd.read_csv(OUT/"d1d2/recording_characterization.csv")
    connected=pd.read_csv(OUT/"connected_speech/participant_recording_features.csv")
    connected_long=pd.read_csv(OUT/"connected_speech/vad_threshold_sensitivity_long.csv")
    variants=pd.read_csv(OUT/"second_variants/second_and_first_variant_signal_characterization.csv")
    q=pd.read_csv(OUT/"review/human_review_queue.csv")
    a=pd.read_csv(OUT/"review/manual_annotations.csv")
    qc=pd.read_csv(ROOT/"results/italian_multitask_v1/qc/per_recording_qc.csv")
    base_integrity=json.loads((ROOT/"results/italian_multitask_v1/reliability/baseline_integrity.json").read_text())
    audit=pd.read_csv(ROOT/"reports/italian_task_audit/validation_checks.csv")
    task_audit_pass=(audit.status.astype(str).str.upper()=="PASS").all()
    add("Historical Italian task-audit checks remain passing",task_audit_pass,audit.status.value_counts().to_dict(),"all PASS")
    add("Primary native first-vowel input has 145 participant-task rows",len(primary)==145,len(primary),145)
    add("Primary first-vowel group counts are 85 HC / 60 PD",primary.group.value_counts().to_dict()=={"ELDERLY_HC":85,"PD":60},primary.group.value_counts().to_dict(),{"ELDERLY_HC":85,"PD":60})
    add("Primary feature table has no duplicate participant-task rows",not primary.duplicated(["group","participant_folder","task_code"]).any(),int(primary.duplicated(["group","participant_folder","task_code"]).sum()),0)
    add("Sensitivity has 20 contrasts in each rate cohort",new.groupby("cohort").size().to_dict()=={"combined_resampled_sensitivity":20,"native16":20},new.groupby("cohort").size().to_dict(),{"native16":20,"combined_resampled_sensitivity":20})
    vi_old=old[(old.task_code=="VI1")&(old.feature=="shimmer_local")].iloc[0]
    vi_new=pd.read_csv(OUT/"sensitivity/vi1_shimmer_bootstrap_interval_audit.csv").iloc[0]
    add("VI1 shimmer mean difference verified against original CSV",np.isclose(vi_old.mean_difference_hc_minus_pd,vi_new.raw_mean_difference,rtol=0,atol=1e-12),(vi_old.mean_difference_hc_minus_pd,vi_new.raw_mean_difference),"equal within 1e-12")
    add("VI1 shimmer unadjusted Mann-Whitney p verified against original CSV",np.isclose(vi_old.mann_whitney_p_unadjusted,vi_new.raw_mann_whitney_p,rtol=0,atol=1e-12),(vi_old.mann_whitney_p_unadjusted,vi_new.raw_mann_whitney_p),"equal within 1e-12")
    vi_loo=pd.read_csv(OUT/"sensitivity/vi1_shimmer_leave_one_out.csv")
    add("VI1 shimmer direction remains positive after every single participant deletion",vi_loo.mean_diff_hc_minus_pd_loo.min()>0,float(vi_loo.mean_diff_hc_minus_pd_loo.min()),"> 0")
    intervals=pd.read_csv(OUT/"sensitivity/vi1_shimmer_bootstrap_interval_audit.csv").iloc[0]
    add("Independent percentile and BCa participant intervals exclude zero",bool(intervals.new_participant_bootstrap_percentile_ci_low>0 and intervals.new_participant_bootstrap_bca_ci_low>0),(intervals.new_participant_bootstrap_percentile_ci_low,intervals.new_participant_bootstrap_bca_ci_low),"both lower bounds > 0")
    add("D1/D2 characterization contains all 72 reconciled recordings",len(d12)==72,len(d12),72)
    plots=len(list((OUT/"d1d2/diagnostic_plots").glob("*.png")))
    add("D1/D2 plots generated for every characterized recording",plots==72,plots,72)
    add("Second/first vowel signal table contains 355 rows",len(variants)==355,len(variants),355)
    add("Connected speech has 104 default-threshold participant-task rows",len(connected)==104,len(connected),104)
    add("Connected speech threshold table has three thresholds for every recording",len(connected_long)==312 and set(connected_long.vad_threshold_db_below_p95.dropna())=={-40,-35,-30},(len(connected_long),sorted(connected_long.vad_threshold_db_below_p95.dropna().unique())),"312 rows; -40/-35/-30 dB")
    add("All connected speech primary candidates retain original sample rate",connected.sample_rate_hz.notna().all(),int(connected.sample_rate_hz.isna().sum()),"0 missing rates")
    add("Review queue includes all D1/D2 recordings",q[q.task_code.isin(["D1","D2"])].relative_path.nunique()==72,q[q.task_code.isin(["D1","D2"])].relative_path.nunique(),72)
    status=pd.read_csv(OUT/"review/review_status.csv").iloc[0]
    add("Review queue has no automated exclusions",int(status.automated_exclusions)==0,status.automated_exclusions,0)
    add("Review annotations remain separate and contain no auto-filled judgments",len(a)==0,len(a),0)
    flagged=int(qc.qc_flags.fillna("").astype(str).str.len().gt(0).sum())
    add("All 19 historical QC flags remain in source QC table",flagged==19,flagged,19)
    formant_ok=all(primary[c].fillna(False).astype(bool).all() for c in ["f1_range_valid","f2_range_valid","f3_range_valid"])
    add("First-vowel formant broad plausibility ranges have no automatic failures",formant_ok,{c:int((~primary[c].fillna(False).astype(bool)).sum()) for c in ["f1_range_valid","f2_range_valid","f3_range_valid"]},"0 false flags")
    add("Existing VA1/reference integrity record remains unchanged",base_integrity.get("unchanged") is True,base_integrity.get("changed_files"),"true; []")
    validation=pd.DataFrame(checks)
    validation.to_csv(OUT/"validation_checks.csv",index=False)
    meta={"experiment_id":"ITALIAN-PHASE2-REMAINING-20261030","computational_checks_passed":int(validation.status.eq("PASS").sum()),
          "computational_checks_failed":int(validation.status.eq("FAIL").sum()),"human_review_items_open":len(q),
          "manual_annotations_completed":len(a),"participant_exclusions_from_automated_qc":0,
          "classifier_trained":False,"validation_status":"PASS" if validation.status.eq("PASS").all() else "FAIL"}
    (OUT/"validation_summary.json").write_text(json.dumps(meta,indent=2))
    with (OUT/"logs/validation_report.md").open("w") as f:
        f.write("# Phase 2 validation report\n\n")
        f.write(f"Computational checks: {meta['computational_checks_passed']} passed, {meta['computational_checks_failed']} failed.\n\n")
        f.write("Human review is a separately tracked open requirement, not an automated validation failure. The 159-item queue has not been adjudicated. No participant exclusions were made.\n\n")
        cols=list(validation.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"]*len(cols)) + " |\n")
        for row in validation.itertuples(index=False,name=None):
            values=[str(v).replace("|","\\|").replace("\n"," ") for v in row]
            f.write("| " + " | ".join(values) + " |\n")
    print(json.dumps(meta,indent=2))
    if meta["computational_checks_failed"]: raise SystemExit(1)

if __name__=="__main__": main()
