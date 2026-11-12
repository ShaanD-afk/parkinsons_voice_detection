from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def create_review_materials(project_root: Path, out_dir: Path) -> pd.DataFrame:
    """Build a prioritized, non-exclusionary queue and blank manual annotation table."""
    qc = pd.read_csv(project_root / "results/italian_multitask_v1/qc/per_recording_qc.csv")
    rows=[]
    for r in qc[qc.qc_flags.fillna("").astype(str).str.len().gt(0)].itertuples(index=False):
        rows.append({"relative_path":r.relative_path,"task_code":r.task_code,"group":r.group,
                     "participant_folder":r.participant_folder,"sample_rate_hz":r.actual_sample_rate_hz,
                     "priority":"high","reason":r.qc_flags,"source":"automated_qc"})
    failures=pd.read_csv(project_root / "results/italian_multitask_v1/features/feature_extraction_failures.csv")
    for r in failures[failures.reason.astype(str).str.contains("plausibility|outside_broad",case=False,na=False)].itertuples(index=False):
        rows.append({"relative_path":r.relative_path,"task_code":r.task_code,"group":r.group,
                     "participant_folder":r.participant_folder,"sample_rate_hz":"","priority":"high",
                     "reason":r.reason+":"+str(r.feature),"source":"formant_or_feature_plausibility"})
    pitch=pd.read_csv(project_root / "data/pitch_sensitivity/pitch_track_review.csv")
    for r in pitch.to_dict("records"):
        key=str(r.get("participant_folder",r.get("participant",r.get("participant_id",""))))
        if not key: continue
        hit=qc[qc.participant_folder.astype(str).eq(key)]
        if hit.empty:
            hit=qc[qc.participant_folder.astype(str).str.endswith(key.split("/")[-1])]
        for q in hit[hit.task_code.eq("VA1")].itertuples(index=False):
            rows.append({"relative_path":q.relative_path,"task_code":q.task_code,"group":q.group,
                         "participant_folder":q.participant_folder,"sample_rate_hz":q.actual_sample_rate_hz,
                         "priority":"high","reason":"historical_pitch_tracking_review","source":"VA1_pitch_sensitivity"})
    sample=pd.read_csv(project_root / "results/italian_multitask_v1/segmentation/blind_pilot_sample.csv")
    for r in sample.itertuples(index=False):
        q=qc[qc.relative_path.eq(r.relative_path)]
        if q.empty: continue
        q=q.iloc[0]
        rows.append({"relative_path":r.relative_path,"task_code":r.task_code,"group":q.group,
                     "participant_folder":q.participant_folder,"sample_rate_hz":r.sample_rate_hz,
                     "priority":"medium","reason":"stratified_segmentation_mask_review","source":"blinded_pilot"})
    # Every D1/D2 recording remains reviewable because protocol structure is undecided.
    d12=qc[qc.task_code.isin(["D1","D2"])]
    for q in d12.itertuples(index=False):
        rows.append({"relative_path":q.relative_path,"task_code":q.task_code,"group":q.group,
                     "participant_folder":q.participant_folder,"sample_rate_hz":q.actual_sample_rate_hz,
                     "priority":"high","reason":"determine_repeated_syllable_vs_other_pattern; automated event counts are unvalidated",
                     "source":"D1_D2_protocol_review"})
    # A balanced threshold-mask review subset for connected speech, kept separate from the full feature table.
    read=qc[qc.task_code.isin(["B1","B2","FB1"])].copy()
    rng=np.random.default_rng(42026)
    for (task,group,rate),stratum in read.groupby(["task_code","group","actual_sample_rate_hz"],dropna=False):
        if stratum.empty: continue
        take=min(2,len(stratum))
        picks=stratum.iloc[np.sort(rng.choice(len(stratum),size=take,replace=False))]
        for q in picks.itertuples(index=False):
            rows.append({"relative_path":q.relative_path,"task_code":q.task_code,"group":q.group,
                         "participant_folder":q.participant_folder,"sample_rate_hz":q.actual_sample_rate_hz,
                         "priority":"medium","reason":"check_connected_speech_VAD_threshold_and_pause_masks",
                         "source":"connected_speech_stratified_threshold_review"})
    detail=pd.DataFrame(rows)
    if detail.empty:
        detail=pd.DataFrame(columns=["relative_path","task_code","group","participant_folder","sample_rate_hz","priority","reason","source"])
    # Deduplicate files while preserving every reason/source.
    q=detail.groupby(["relative_path","task_code","group","participant_folder","sample_rate_hz","priority"],dropna=False).agg(
        reason=("reason",lambda x:"; ".join(sorted(set(map(str,x))))),
        source=("source",lambda x:"; ".join(sorted(set(map(str,x)))))).reset_index()
    q=q.sort_values(["priority","task_code","group","participant_folder"],ascending=[True,True,True,True]).reset_index(drop=True)
    q.insert(0,"review_id",[f"R{i:04d}" for i in range(1,len(q)+1)])
    folder=out_dir/"review"; folder.mkdir(parents=True,exist_ok=True)
    q.to_csv(folder/"human_review_queue.csv",index=False)
    annotations=folder/"manual_annotations.csv"
    columns=["review_id","relative_path","acceptable_recording","suspected_recording_problem","segmentation_correct",
             "pitch_tracking_plausible","formants_plausible","uncertain","manual_event_onsets_s",
             "manual_syllable_count","reviewer_comments","reviewer_id","reviewed_at"]
    if not annotations.exists(): pd.DataFrame(columns=columns).to_csv(annotations,index=False)
    else:
        prior=pd.read_csv(annotations)
        for c in columns:
            if c not in prior: prior[c]=""
        prior[columns].to_csv(annotations,index=False)
    if annotations.exists():
        a=pd.read_csv(annotations)
        merged=q.merge(a,on=["review_id","relative_path"],how="left",suffixes=("","_annotation"))
    else: merged=q
    summary=q.merge(pd.read_csv(annotations),on=["review_id","relative_path"],how="left")
    completed=summary.reviewed_at.notna() & summary.reviewed_at.astype(str).str.len().gt(0)
    summary.groupby(["priority","task_code"],dropna=False).agg(items=("review_id","count")).reset_index().to_csv(folder/"review_queue_summary.csv",index=False)
    status=pd.DataFrame([{"queue_items":len(q),"completed_items":int(completed.sum()),"unresolved_items":int((~completed).sum()),
                         "automated_exclusions":0,"annotation_file":str(annotations)}])
    status.to_csv(folder/"review_status.csv",index=False)
    return q


def refresh_review_summary(out_dir: Path) -> pd.DataFrame:
    folder=out_dir/"review"
    q=pd.read_csv(folder/"human_review_queue.csv")
    a=pd.read_csv(folder/"manual_annotations.csv")
    merged=q.merge(a,on=["review_id","relative_path"],how="left")
    done=merged.reviewed_at.fillna("").astype(str).str.len().gt(0)
    summary=merged.groupby(["priority","task_code"],dropna=False).agg(queue_items=("review_id","count"))
    summary=summary.reset_index()
    per=merged.groupby(["priority","task_code"],dropna=False).apply(
        lambda g: pd.Series({"completed_items":int(g.reviewed_at.fillna("").astype(str).str.len().gt(0).sum()),
                             "unresolved_items":int((~g.reviewed_at.fillna("").astype(str).str.len().gt(0)).sum())}),
        include_groups=False).reset_index()
    per.to_csv(folder/"review_queue_summary.csv",index=False)
    pd.DataFrame([{"queue_items":len(q),"completed_items":int(done.sum()),"unresolved_items":int((~done).sum()),
                   "automated_exclusions":0,"annotation_file":str(folder/"manual_annotations.csv")}]).to_csv(folder/"review_status.csv",index=False)
    return per

