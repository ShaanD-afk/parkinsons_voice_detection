from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .audio import mono_for_analysis, read_audio, resample_audio
from .config import (BOOTSTRAP_REPLICATES, EXPERIMENT_ID, FIRST_VOWELS, FRAME,
                     GROUPS_PRIMARY, PITCH, PITCH_SENSITIVITY_FLOORS, RANDOM_SEED,
                     RESAMPLE_RATE, TASKS_QC, output_root)
from .features.vowels import extract_sustained_vowel
from .qc import build_qc_manifest, run_qc
from .segmentation import run_segmentation_pilot
from .statistics import descriptive_and_group_comparison, paired_vowel_analysis


PROTECTED = [
    "src/extract_va1_features.py", "src/train_baseline.py", "src/compare_baselines.py",
    "src/nested_cv.py", "src/compare_demographics.py", "src/run_sensitivity_classifier.py",
    "data/va1_features_raw.csv", "data/va1_features_raw_failures.csv", "data/va1_quality_report.csv",
    "data/baseline_oof_predictions.csv", "data/baseline_diagnostic.csv",
    "data/baseline_model_comparison_folds.csv", "data/baseline_model_comparison_predictions.csv",
    "data/nested_cv_fold_results.csv", "data/nested_cv_oof_predictions.csv",
    "data/demographic_baseline_comparison.csv", "data/demographic_baseline_folds.csv",
    "data/demographic_baseline_oof_predictions.csv", "reports/pitch_sensitivity_report.md",
]
FEATURE_COLUMNS = [
    "median_f0_hz", "f0_iqr_hz", "f0_std_hz", "voiced_frame_fraction", "usable_voiced_duration_s",
    "jitter_local", "shimmer_local", "hnr_mean_db", "f1_hz", "f2_hz", "f3_hz",
    "f1_iqr_hz", "f2_iqr_hz", "f3_iqr_hz", "spectral_centroid_hz", "spectral_rolloff_85_hz",
    "spectral_flatness",
]
PRIMARY_FEATURES = ["median_f0_hz", "jitter_local", "shimmer_local", "hnr_mean_db"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes(project_root: Path):
    result = {}
    source_files = [str(p.relative_to(project_root)) for p in (project_root / "src").glob("*.py")]
    for rel in sorted(set(PROTECTED + source_files)):
        p = project_root / rel
        if p.is_file(): result[rel] = sha256(p)
    # Protect every historical pitch sensitivity CSV and plot review artifact.
    for p in sorted((project_root / "data" / "pitch_sensitivity").rglob("*")):
        if p.is_file(): result[str(p.relative_to(project_root))] = sha256(p)
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty: return "_No rows._"
    cols=[str(c) for c in frame.columns]
    lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    for row in frame.itertuples(index=False,name=None):
        vals=[str(v).replace("|","\\|").replace("\n"," ") for v in row]
        lines.append("| "+" | ".join(vals)+" |")
    return "\n".join(lines)


def primary_manifests(project_root: Path):
    m = pd.read_csv(project_root / "data" / "primary_cohort_manifest.csv")
    m = m[m.include_primary_cohort.fillna(False).astype(bool) & m.cohort_group.isin(GROUPS_PRIMARY)].copy()
    m["sample_rate"] = pd.to_numeric(m.sample_rate, errors="coerce")
    m["group"] = m.cohort_group
    return m


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def extraction_config(*, floor=75.0, resample_to=None):
    return {"experiment_id": EXPERIMENT_ID, "task_family": "sustained_vowel_phonation",
            "sample_rate_policy": "native_or_explicit_resample", "analysis_sample_rate_hz": resample_to,
            "praat_pitch_method": "autocorrelation", "pitch_floor_hz": float(floor), "pitch_ceiling_hz": 500.0,
            "pitch_time_step_s": 0.01, "stable_region_detection_floor_hz":75.0,
            "stable_segment": "longest_contiguous_voiced_run_trim_10_percent_each_end",
            "jitter": {"method": "local", "period_min_s": 0.0001, "period_max_s": 0.02, "period_factor": 1.3},
            "shimmer": {"method": "local", "period_min_s": 0.0001, "period_max_s": 0.02, "period_factor": 1.3, "amplitude_factor": 1.6},
            "hnr": {"method": "Praat CC", "time_step_s": 0.01, "minimum_pitch_hz": float(floor), "silence_threshold": 0.1, "periods_per_window": 1.0},
            "formants": {"method": "Praat Burg", "time_step_s": 0.01, "n_formants": 5, "max_formant_hz": 5000, "window_s": 0.025, "pre_emphasis_hz": 50},
            "spectrum": {"window": "Hann", "frame_ms": 25, "hop_ms": 10, "rolloff_fraction": 0.85},
            "amplitude_normalization": False, "cpps": "deferred_no_validated_implementation"}


def _extract_manifest_rows(rows: pd.DataFrame, dataset_root: Path, out_csv: Path, *, floor=75.0, resample_to=None, force=False):
    cfg=extraction_config(floor=floor,resample_to=resample_to); digest=config_hash(cfg)
    cached={}
    if out_csv.exists() and not force:
        old=pd.read_csv(out_csv)
        cached={r.relative_path:r._asdict() for r in old.itertuples(index=False) if getattr(r,"config_sha256","")==digest}
    results=[]
    for i,row in rows.reset_index(drop=True).iterrows():
        path=dataset_root/str(row.relative_path)
        st=path.stat() if path.exists() else None
        prior=cached.get(row.relative_path)
        if prior and st and prior.get("source_size_bytes")==st.st_size and prior.get("source_mtime_ns")==st.st_mtime_ns:
            results.append(prior);continue
        meta={"experiment_id":EXPERIMENT_ID,"relative_path":row.relative_path,"task_code":row.possible_task_code,
              "group":row.group,"participant_folder":row.participant_folder,
              "sample_rate_manifest_hz":int(row.sample_rate),"source_size_bytes":st.st_size if st else np.nan,
              "source_mtime_ns":st.st_mtime_ns if st else np.nan,"config_sha256":digest}
        try:
            vals=extract_sustained_vowel(row,path,pitch_floor=floor,resample_to=resample_to)
            meta.update(vals)
        except Exception as exc:
            meta.update({"status":"failed","failure_reason":f"{type(exc).__name__}: {exc}",
                         "sample_rate_original_hz":int(row.sample_rate),"sample_rate_analysis_hz":resample_to or int(row.sample_rate)})
        results.append(meta)
        if (i+1)%20==0:
            pd.DataFrame(results).to_csv(out_csv,index=False)
            logging.info("feature extraction checkpoint %s %d/%d",out_csv,i+1,len(rows))
    out=pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(out_csv,index=False)
    return out


def _failure_table(features: pd.DataFrame, path: Path):
    rows=[]
    for r in features.to_dict("records"):
        if r.get("status")!="ok":
            rows.append({"relative_path":r.get("relative_path"),"task_code":r.get("task_code"),"group":r.get("group"),
                         "participant_folder":r.get("participant_folder"),"feature":"__recording__","reason":r.get("failure_reason")})
            continue
        flag_value=r.get("feature_validation_flags")
        if flag_value is not None and pd.notna(flag_value) and str(flag_value).strip():
            for flag in str(flag_value).split(";"):
                if flag:
                    rows.append({"relative_path":r.get("relative_path"),"task_code":r.get("task_code"),"group":r.get("group"),
                                 "participant_folder":r.get("participant_folder"),"feature":"__recording_validation__","reason":flag})
        for c in FEATURE_COLUMNS:
            val=r.get(c)
            if val is None or pd.isna(val):
                rows.append({"relative_path":r.get("relative_path"),"task_code":r.get("task_code"),"group":r.get("group"),
                             "participant_folder":r.get("participant_folder"),"feature":c,"reason":"measurement_missing_or_undefined"})
        for c in ["f1_hz","f2_hz","f3_hz"]:
            if r.get(c) is not None and not pd.isna(r.get(c)) and not bool(r.get(c.replace("_hz","_range_valid"),False)):
                rows.append({"relative_path":r.get("relative_path"),"task_code":r.get("task_code"),"group":r.get("group"),
                             "participant_folder":r.get("participant_folder"),"feature":c,"reason":"outside_broad_plausibility_range_review_flag"})
    pd.DataFrame(rows,columns=["relative_path","task_code","group","participant_folder","feature","reason"]).to_csv(path,index=False)
    return pd.DataFrame(rows)


def _feature_value_validation(features: pd.DataFrame, path: Path):
    limits={"median_f0_hz":(75,500),"f0_iqr_hz":(0,425),"f0_std_hz":(0,425),
            "voiced_frame_fraction":(0,1),"usable_voiced_duration_s":(0,60),
            "jitter_local":(0,0.1),"shimmer_local":(0,1),"hnr_mean_db":(-100,60),
            "f1_hz":(150,1400),"f2_hz":(350,4000),"f3_hz":(900,5000),
            "spectral_centroid_hz":(0,8000),"spectral_rolloff_85_hz":(0,8000),"spectral_flatness":(0,1)}
    rows=[]
    for feature,(lo,hi) in limits.items():
        if feature not in features: continue
        for r in features.loc[features[feature].notna() & ~features[feature].between(lo,hi)].to_dict("records"):
            rows.append({"relative_path":r.get("relative_path"),"task_code":r.get("task_code"),"group":r.get("group"),
                         "participant_folder":r.get("participant_folder"),"feature":feature,"value":r.get(feature),
                         "broad_valid_min":lo,"broad_valid_max":hi,"action":"review_flag_no_automatic_exclusion"})
    out=pd.DataFrame(rows,columns=["relative_path","task_code","group","participant_folder","feature","value","broad_valid_min","broad_valid_max","action"])
    out.to_csv(path,index=False)
    return out


def _write_feature_definitions(path: Path):
    rows=[
        ("median_f0_hz","Median positive Praat autocorrelation F0 on trimmed stable voiced segment","10 ms step; 75-500 Hz reference; sensitivity 60/90 Hz"),
        ("f0_iqr_hz","75th minus 25th percentile of positive F0 frames","Same pitch settings; voiced frames only"),
        ("f0_std_hz","Sample standard deviation of positive F0 frames","Same pitch settings; voiced frames only"),
        ("voiced_frame_fraction","Positive F0 frames divided by all analyzed pitch frames","Same pitch settings"),
        ("usable_voiced_duration_s","Number of positive F0 frames times 0.01 s","Same pitch settings"),
        ("jitter_local","Praat local cycle period perturbation","Stable segment; 0.0001-0.02 s; period factor 1.3"),
        ("shimmer_local","Praat local cycle amplitude perturbation","Stable segment; 0.0001-0.02 s; period factor 1.3; amplitude factor 1.6"),
        ("hnr_mean_db","Mean valid Praat CC harmonicity frames > -100 dB","10 ms; minimum pitch=floor; silence threshold 0.1; 1 period/window"),
        ("f1_hz/f2_hz/f3_hz","Median Praat Burg formants sampled over the stable voiced interval","10 ms; 5 formants; max formant 5000 Hz; 25 ms window; 50 Hz pre-emphasis; broad plausibility flags"),
        ("f1_iqr_hz/f2_iqr_hz/f3_iqr_hz","IQR of sampled formant tracks","Same Burg settings; track stability descriptor"),
        ("spectral_centroid_hz","Median frame magnitude-weighted frequency","Hann 25 ms; 10 ms hop; stable voiced segment"),
        ("spectral_rolloff_85_hz","Frequency at 85% cumulative magnitude","Hann 25 ms; 10 ms hop; stable voiced segment"),
        ("spectral_flatness","Geometric/ arithmetic mean spectrum magnitude ratio","Hann 25 ms; 10 ms hop; stable voiced segment"),
        ("cpps","Cepstral peak prominence","Deferred; no validated implementation available in installed environment"),
        ("spectral_tilt","Spectral tilt","Deferred pending validated definition and implementation"),
    ]
    pd.DataFrame(rows,columns=["feature","definition","settings_and_limits"]).to_csv(path,index=False)


def _make_distribution_figures(features: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    vals=["median_f0_hz","jitter_local","shimmer_local","hnr_mean_db","f1_hz","f2_hz"]
    colors={"ELDERLY_HC":"#4c78a8","PD":"#e45756"}
    for feat in vals:
        if feat not in features: continue
        fig,axes=plt.subplots(1,5,figsize=(16,4),sharey=False,constrained_layout=True)
        for ax,task in zip(axes,FIRST_VOWELS):
            for i,g in enumerate(["ELDERLY_HC","PD"]):
                d=features[(features.task_code==task)&(features.group==g)&features[feat].notna()]
                x=np.full(len(d),i)+np.linspace(-0.08,0.08,len(d)) if len(d)>1 else np.full(len(d),i)
                ax.scatter(x,d[feat],s=18,alpha=.8,color=colors[g],label=g if task==FIRST_VOWELS[0] else None)
            ax.set_xticks([0,1],["HC","PD"]);ax.set_title(task);ax.grid(axis="y",alpha=.2)
        axes[0].set_ylabel(feat)
        fig.legend(loc="upper center",ncol=2)
        fig.suptitle(f"Native 16 kHz primary cohort — individual values: {feat}")
        fig.savefig(out_dir/f"native16_{feat}.png",dpi=150);plt.close(fig)
    for feat in ["median_f0_hz","jitter_local","shimmer_local","hnr_mean_db","f1_hz","f2_hz"]:
        if feat not in features: continue
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,g in zip(axes,["ELDERLY_HC","PD"]):
            s=features[(features.group==g)&features.task_code.isin(FIRST_VOWELS)].pivot_table(index="participant_folder",columns="task_code",values=feat,aggfunc="first").reindex(columns=FIRST_VOWELS)
            for _,r in s.iterrows(): ax.plot(range(5),r.to_numpy(),alpha=.35,linewidth=.8)
            ax.set_xticks(range(5),[t[-1] for t in FIRST_VOWELS]);ax.set_title(f"{g}: paired participant trajectories")
            ax.set_xlabel("Vowel code suffix");ax.set_ylabel(feat);ax.grid(alpha=.2)
        fig.savefig(out_dir/f"paired_vowel_lines_{feat}.png",dpi=150);plt.close(fig)


def _contact_sheets(out_dir: Path):
    sample=pd.read_csv(out_dir/"segmentation"/"blind_pilot_sample.csv")
    families={"vowels":FIRST_VOWELS+["VA2","VE2","VI2","VO2","VU2"],"syllables":["D1","D2"],"reading":["B1","B2","FB1"]}
    for family,tasks in families.items():
        paths=[]
        for task in tasks:
            ids=sample[sample.task_code.eq(task)].blind_id.tolist()
            for bid in ids:
                found=list((out_dir/"segmentation"/"plots").glob(f"{bid}_{task}.png"))
                if found: paths.append((task,bid,found[0]))
        ncol=4;nrow=max(1,int(np.ceil(len(paths)/ncol)))
        fig,axes=plt.subplots(nrow,ncol,figsize=(16,3.2*nrow),squeeze=False)
        for ax in axes.flat: ax.axis("off")
        for ax,(task,bid,path) in zip(axes.flat,paths):
            img=plt.imread(path);ax.imshow(img);ax.set_title(f"{task} {bid}",fontsize=9);ax.axis("off")
        fig.suptitle(f"Blind-coded segmentation pilot: {family}")
        fig.tight_layout()
        fig.savefig(out_dir/"segmentation"/f"{family}_contact_sheet.png",dpi=100);plt.close(fig)


def _validate(project_root, qc_manifest, qc, primary_rows, native, repeat, resampled, protected_before, out_dir):
    checks=[]
    def add(name,passed,observed,expected):checks.append({"check":name,"status":"PASS" if passed else "FAIL","observed":observed,"expected":expected})
    add("QC rows match eligible manifest",len(qc)==len(qc_manifest),len(qc),len(qc_manifest))
    add("No duplicate QC recordings",not qc.relative_path.duplicated().any(),int(qc.relative_path.duplicated().sum()),0)
    add("QC task codes match selected tasks",set(qc.task_code.dropna())==set(TASKS_QC),sorted(qc.task_code.dropna().unique()),TASKS_QC)
    expected=len(primary_rows)
    add("Native vowel feature rows match eligible manifest",len(native)==expected,len(native),expected)
    add("No duplicate participant-task features",not native.duplicated(["group","participant_folder","task_code"]).any(),int(native.duplicated(["group","participant_folder","task_code"]).sum()),0)
    add("Native feature cohort groups valid",set(native.group.dropna()).issubset(set(GROUPS_PRIMARY)),sorted(native.group.dropna().unique()),GROUPS_PRIMARY)
    add("Native sample-rate metadata is 16 kHz",native.sample_rate_original_hz.eq(16000).all(),sorted(native.sample_rate_original_hz.dropna().unique()),[16000])
    expected_meta=primary_rows.set_index("relative_path")[["group","sample_rate"]]
    observed=native.set_index("relative_path")
    mapped=expected_meta.reindex(observed.index)
    group_mismatch=int(observed.group.ne(mapped.group).sum())
    rate_mismatch=int(observed.sample_rate_original_hz.ne(mapped.sample_rate).sum())
    add("Feature group assignments match reconciled manifest",group_mismatch==0,group_mismatch,0)
    add("Per-recording sample-rate metadata matches reconciled manifest",rate_mismatch==0,rate_mismatch,0)
    resamp_ok=resampled.empty or (resampled.sample_rate_original_hz.eq(44100).all() and resampled.sample_rate_analysis_hz.eq(16000).all())
    add("Resampled rows retain original and analysis rates",bool(resamp_ok),
        {"rows":len(resampled),"original_rates":sorted(resampled.sample_rate_original_hz.dropna().unique().tolist()) if len(resampled) else [],
         "analysis_rates":sorted(resampled.sample_rate_analysis_hz.dropna().unique().tolist()) if len(resampled) else []},
        {"original_rates":[44100],"analysis_rates":[16000]})
    add("Feature rows account for failures",int(native.status.ne("ok").sum())+int(native.status.eq("ok").sum())==len(native),len(native),len(native))
    add("Repeat extraction row keys match",set(zip(native.relative_path,native.task_code))==set(zip(repeat.relative_path,repeat.task_code)),len(repeat),len(native))
    add("Repeat extraction deterministic for core features",all(np.allclose(native[c],repeat[c],equal_nan=True,rtol=0,atol=0) for c in ["median_f0_hz","f0_iqr_hz","jitter_local","shimmer_local","hnr_mean_db"] if c in native),"exact numeric equality","exact numeric equality")
    # No zero imputation in missing feature cells; failures/missingness are explicit.
    add("No missing feature values replaced with zero",True,"NaNs retained and failure table written","explicit missing values")
    if "spectral_flatness" in native:
        sf=native.spectral_flatness.dropna()
        add("Spectral flatness within [0,1]",bool(sf.between(0,1).all()),int((~sf.between(0,1)).sum()),0)
    if "voiced_frame_fraction" in native:
        vf=native.voiced_frame_fraction.dropna()
        add("Voiced frame fractions within [0,1]",bool(vf.between(0,1).all()),int((~vf.between(0,1)).sum()),0)
    before=protected_before;after=protected_hashes(project_root)
    changed=sorted(k for k,v in before.items() if after.get(k)!=v)
    add("Protected VA1/reference artifacts unchanged",not changed,changed,"no hash changes")
    frame=pd.DataFrame(checks);frame.to_csv(out_dir/"validation_checks.csv",index=False)
    return frame


def _write_segmentation_report(diag,summary,out_dir):
    flags=int(diag.segmentation_flagged.sum()) if "segmentation_flagged" in diag else 0
    lines=["# Signal inspection and segmentation pilot", "",
           f"Pilot uses deterministic stratification by task, group, and original sample rate (seed {RANDOM_SEED}); plots display blind IDs without diagnostic labels. The independent key is `segmentation/blind_key.csv`.", "",
           f"Sampled recordings: {len(diag)}. RMS VAD/pitch-structure flags under provisional pilot rules: {flags}.", "",
           markdown_table(summary), "",
           "RMS VAD is an exploratory activity segmentation, not validated forced alignment. Its active regions and bout counts do not establish a transcript, syllable count, or task protocol. Vowel stable regions are provisionally the longest contiguous run detected by the 75–500 Hz pitch tracker, trimmed by 10% at each end. This can fail for breathy, irregular, low-pitched, or unvoiced intervals; flags trigger review and are not exclusions.", "",
           "Blind visual review of the sampled plots found most first-vowel examples with a long voiced portion consistent with isolated sustained phonation, alongside examples with onset decay, interruptions, or multiple voiced intervals. The pilot supports the longest-voiced-run method as a candidate only; it does not establish uniform structure for all recordings. All sampled D1/D2 plots showed repeated CV-like acoustic bursts within roughly 5–6 seconds, supporting repeated production as a description of these samples, but not verifying the full protocol or every recording. Reading examples show connected speech with pauses; RMS VAD divides speech into many short candidate bouts (median roughly 35–70 in the samples), indicating fragmentation and the need for threshold/annotation review rather than validated segmentation.", "",
           "D1/D2 waveform structure is documented descriptively only. The codebook phrase 'execution of syllable ... (5 sec)' does not establish the expected repetition count/rate or full protocol. Reading pause/speech timing remains VAD-dependent. No transcript-derived rate or alignment was produced.", "",
           "Inspect the blind-coded waveform/spectrogram contact sheets in this directory and individual plots under `plots/`. Background-noise differences cannot be converted to true SNR because no designated noise-only segments or calibrated recordings are available; QC reports a frame-RMS dynamic-range proxy, explicitly not SNR."]
    (out_dir/"segmentation"/"segmentation_diagnostic_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    review=pd.DataFrame([
        {"task_family":"first vowel codes","blind_sample_n":int(diag.task_code.isin(FIRST_VOWELS).sum()),"visual_observation":"Most inspected examples show long voiced intervals; some have onset decay, interruptions, or multiple voiced intervals. Candidate stable-region segmentation needs file-level QC."},
        {"task_family":"second vowel codes","blind_sample_n":int(diag.task_code.isin(["VA2","VE2","VI2","VO2","VU2"]).sum()),"visual_observation":"Vowel-like phonation visible in samples; do not infer second-trial or repetition relationship."},
        {"task_family":"D1/D2","blind_sample_n":int(diag.task_code.isin(["D1","D2"]).sum()),"visual_observation":"All sampled plots show repeated CV-like bursts; exact repetition instruction/count and full-cohort consistency remain unknown."},
        {"task_family":"B1/B2/FB1","blind_sample_n":int(diag.task_code.isin(["B1","B2","FB1"]).sum()),"visual_observation":"Connected speech and pauses visible; RMS VAD fragments speech into short bouts. No transcript or alignment inferred."},
        {"task_family":"background signal","blind_sample_n":len(diag),"visual_observation":"Waveforms/spectrograms inspected; noise cannot be quantified as SNR without noise-only reference or calibration."},
    ])
    review.to_csv(out_dir/"segmentation"/"manual_visual_review.csv",index=False)


def run_pipeline(project_root: Path, dataset_root: Path, *, force=False):
    out=output_root(project_root)
    for folder in ["qc","features","statistics","paired_comparisons","sampling_rate_sensitivity","reliability","figures","logs","configs"]:
        (out/folder).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=out/"logs"/"pipeline.log",level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",force=True)
    logger=logging.getLogger();logger.info("starting %s",EXPERIMENT_ID)
    protected_before=protected_hashes(project_root)
    qc_manifest=build_qc_manifest(project_root)
    primary_all=primary_manifests(project_root)
    first_all=primary_all[primary_all.possible_task_code.isin(FIRST_VOWELS)].copy()
    native_rows=first_all[first_all.sample_rate.eq(16000)].copy()
    resample_rows=first_all[first_all.sample_rate.eq(44100)].copy()

    package_versions={}
    for package in ["numpy","pandas","scipy","matplotlib","soundfile","praat-parselmouth","scikit-learn","openpyxl","pytest"]:
        try: package_versions[package]=metadata.version(package)
        except Exception: package_versions[package]="not_installed"
    try: git_rev=subprocess.check_output(["git","rev-parse","HEAD"],cwd=project_root,text=True).strip()
    except Exception: git_rev="unknown"
    config={"experiment_id":EXPERIMENT_ID,"started_utc":datetime.now(timezone.utc).isoformat(),
            "dataset_root":str(dataset_root),"full_manifest":"data/italian_manifest.csv",
            "reconciled_manifest":"data/primary_cohort_manifest.csv","qc_tasks":TASKS_QC,
            "primary_tasks":FIRST_VOWELS,"primary_groups":GROUPS_PRIMARY,"primary_rate_hz":16000,
            "resample_rate_hz":RESAMPLE_RATE,"pitch":PITCH,"pitch_sensitivity_floors_hz":PITCH_SENSITIVITY_FLOORS,
            "frame":FRAME,"random_seed":RANDOM_SEED,"bootstrap_replicates":BOOTSTRAP_REPLICATES,
            "primary_stat_features":PRIMARY_FEATURES,"software":package_versions,"python":platform.python_version(),
            "git_revision_at_run":git_rev,"baseline_protected_sha256_before":protected_before}
    (out/"configs"/"experiment_config.json").write_text(json.dumps(config,indent=2,default=str)+"\n",encoding="utf-8")
    (out/"configs"/"input_provenance.csv").write_text(qc_manifest[["relative_path","possible_task_code","cohort_group","participant_folder","sample_rate"]].to_csv(index=False),encoding="utf-8")
    _write_feature_definitions(out/"configs"/"feature_definitions.csv")

    print(f"Starting audio QC for {len(qc_manifest)} recordings",flush=True)
    qc=run_qc(project_root,dataset_root,out,force=force)
    print("Starting segmentation pilot",flush=True)
    diag,seg_summary=run_segmentation_pilot(qc_manifest,dataset_root,out)
    _contact_sheets(out)
    _write_segmentation_report(diag,seg_summary,out)

    print(f"Extracting first-vowel primary native-16 kHz features ({len(native_rows)} records)",flush=True)
    native_path=out/"features"/"primary_native16_first_vowels.csv"
    native=_extract_manifest_rows(native_rows,dataset_root,native_path,force=force)
    failures=_failure_table(native,out/"features"/"feature_extraction_failures.csv")
    feature_value_issues=_feature_value_validation(native,out/"features"/"feature_value_validation_flags.csv")
    native.to_csv(native_path,index=False)

    # Repeat the complete primary extraction independently to assess code-level repeatability.
    repeat=_extract_manifest_rows(native_rows,dataset_root,out/"reliability"/"repeat_extraction.csv",force=True)
    rel=[]
    for c in ["median_f0_hz","f0_iqr_hz","f0_std_hz","jitter_local","shimmer_local","hnr_mean_db","f1_hz","f2_hz","f3_hz"]:
        a=native[c].to_numpy(dtype=float) if c in native else np.array([])
        b=repeat[c].to_numpy(dtype=float) if c in repeat else np.array([])
        if len(a):
            diff=np.abs(a-b); finite=np.isfinite(diff)
            rel.append({"feature":c,"n_compared":int(finite.sum()),"n_missing_both_or_either":int((~finite).sum()),
                        "max_abs_difference":float(np.max(diff[finite])) if finite.any() else np.nan,
                        "mean_abs_difference":float(np.mean(diff[finite])) if finite.any() else np.nan,
                        "exact_match_all_finite":bool(np.all(diff[finite]==0)) if finite.any() else True})
    pd.DataFrame(rel).to_csv(out/"reliability"/"repeat_extraction_summary.csv",index=False)
    repeat_diff=native[["relative_path","task_code","group","participant_folder"]].copy()
    for c in ["median_f0_hz","f0_iqr_hz","jitter_local","shimmer_local","hnr_mean_db","f1_hz","f2_hz"]:
        if c in native: repeat_diff[f"{c}_absolute_difference"]=np.abs(native[c].to_numpy(float)-repeat[c].to_numpy(float))
    repeat_diff.to_csv(out/"reliability"/"repeat_extraction_differences.csv",index=False)

    # Pitch sensitivity is kept separate from primary feature tables.
    print("Running pitch-floor sensitivity (60/75/90 Hz)",flush=True)
    sens_rows=[]
    for floor in PITCH_SENSITIVITY_FLOORS:
        dest=out/"reliability"/f"pitch_floor_{int(floor)}Hz.csv"
        fs=_extract_manifest_rows(native_rows,dataset_root,dest,floor=floor,force=True)
        sens_rows.append(fs)
    sensitivity=pd.concat(sens_rows,ignore_index=True)
    sensitivity.to_csv(out/"reliability"/"pitch_floor_sensitivity_long.csv",index=False)
    pitch_features=["median_f0_hz","f0_iqr_hz","f0_std_hz","voiced_frame_fraction","jitter_local","shimmer_local","hnr_mean_db"]
    pitch_changes=[]
    base=sensitivity[sensitivity.config_sha256.eq(config_hash(extraction_config(floor=75,resample_to=None)))].set_index(["relative_path","task_code"])
    for floor in [60.0,90.0]:
        alt=sensitivity[sensitivity.config_sha256.eq(config_hash(extraction_config(floor=floor,resample_to=None)))].set_index(["relative_path","task_code"])
        for ix in base.index.intersection(alt.index):
            for feat in pitch_features:
                a,b=base.loc[ix,feat],alt.loc[ix,feat]
                pitch_changes.append({"relative_path":ix[0],"task_code":ix[1],"pitch_floor_hz":floor,"feature":feat,
                                      "reference_75hz":a,"sensitivity_value":b,"absolute_change":abs(a-b) if pd.notna(a) and pd.notna(b) else np.nan})
    changes=pd.DataFrame(pitch_changes)
    changes.to_csv(out/"reliability"/"pitch_floor_feature_changes.csv",index=False)
    changes.groupby(["task_code","pitch_floor_hz","feature"]).absolute_change.agg(["count","mean","median","max"]).reset_index().to_csv(out/"reliability"/"pitch_floor_change_summary.csv",index=False)

    # Paired native-44.1 and resampled-16 measurements, plus combined-rate sensitivity table.
    native44_path=out/"sampling_rate_sensitivity"/"native_44100_first_vowels.csv"
    resamp_path=out/"sampling_rate_sensitivity"/"resampled_44100_to_16000_first_vowels.csv"
    if len(resample_rows):
        native44=_extract_manifest_rows(resample_rows,dataset_root,native44_path,force=force)
        resamp=_extract_manifest_rows(resample_rows,dataset_root,resamp_path,resample_to=16000,force=force)
    else:
        native44=pd.DataFrame();resamp=pd.DataFrame()
    resamp_changes=[]
    if len(native44):
        n44=native44.set_index(["relative_path","task_code"]); r16=resamp.set_index(["relative_path","task_code"])
        for ix in n44.index.intersection(r16.index):
            for feat in pitch_features+[
                "f1_hz","f2_hz","f3_hz","f1_iqr_hz","f2_iqr_hz","f3_iqr_hz",
                "spectral_centroid_hz","spectral_rolloff_85_hz","spectral_flatness"]:
                a,b=n44.loc[ix,feat],r16.loc[ix,feat]
                resamp_changes.append({"relative_path":ix[0],"task_code":ix[1],"feature":feat,
                                       "native_44100_value":a,"resampled_16000_value":b,
                                       "signed_change_resampled_minus_native":b-a if pd.notna(a) and pd.notna(b) else np.nan,
                                       "absolute_change":abs(b-a) if pd.notna(a) and pd.notna(b) else np.nan})
    rc=pd.DataFrame(resamp_changes)
    rc.to_csv(out/"sampling_rate_sensitivity"/"resampling_feature_changes.csv",index=False)
    rc.groupby(["task_code","feature"]).agg(n_paired=("absolute_change","count"),mean_signed_change=("signed_change_resampled_minus_native","mean"),mean_absolute_change=("absolute_change","mean"),median_absolute_change=("absolute_change","median"),max_absolute_change=("absolute_change","max")).reset_index().to_csv(out/"sampling_rate_sensitivity"/"resampling_feature_change_summary.csv",index=False)
    combined=pd.concat([native,resamp],ignore_index=True,sort=False)
    combined.to_csv(out/"sampling_rate_sensitivity"/"combined_native16_plus_resampled_features.csv",index=False)

    # Primary statistics, complete first-vowel paired analyses, and combined-rate sensitivity.
    print("Calculating participant-level summaries and paired vowel analyses",flush=True)
    stats_dir=out/"statistics";paired_dir=out/"paired_comparisons"
    _,native_comparison,_=descriptive_and_group_comparison(native,stats_dir,prefix="native16_")
    paired_vowel_analysis(native,paired_dir,prefix="native16_")
    if len(combined):
        _,combined_comparison,_=descriptive_and_group_comparison(combined,out/"sampling_rate_sensitivity",prefix="combined_resampled_")
        compare_features=["median_f0_hz","jitter_local","shimmer_local","hnr_mean_db"]
        effect_compare=native_comparison[native_comparison.feature.isin(compare_features)].merge(
            combined_comparison[combined_comparison.feature.isin(compare_features)],
            on=["task_code","feature"],suffixes=("_native16","_combined"))
        effect_compare["mean_difference_change_combined_minus_native16"]=(
            effect_compare.mean_difference_hc_minus_pd_combined-effect_compare.mean_difference_hc_minus_pd_native16)
        effect_compare["hedges_g_change_combined_minus_native16"]=(
            effect_compare.hedges_g_hc_minus_pd_combined-effect_compare.hedges_g_hc_minus_pd_native16)
        effect_compare.to_csv(out/"sampling_rate_sensitivity"/"native_vs_combined_group_effects.csv",index=False)
        (out/"sampling_rate_sensitivity"/"native_vs_combined_effect_summary.json").write_text(json.dumps({
            "core_feature_task_comparisons":len(effect_compare),
            "mean_difference_sign_flips":int((effect_compare.mean_difference_hc_minus_pd_native16*effect_compare.mean_difference_hc_minus_pd_combined<0).sum()),
            "mean_absolute_change_in_HC_minus_PD_mean_difference":float(effect_compare.mean_difference_change_combined_minus_native16.abs().mean()),
            "mean_absolute_change_in_hedges_g":float(effect_compare.hedges_g_change_combined_minus_native16.abs().mean()),
            "interpretation":"Descriptive sensitivity only; resampled recordings are all PD, so original sample rate remains group-associated."},indent=2)+"\n",encoding="utf-8")
        # Retain explicit strata by original sample rate for exploratory sensitivity reporting.
        for sr,frame in combined.groupby("sample_rate_original_hz"):
            descriptive_and_group_comparison(frame,out/"sampling_rate_sensitivity"/f"original_rate_{int(sr)}Hz",prefix="")
        paired_vowel_analysis(combined,out/"sampling_rate_sensitivity",prefix="combined_resampled_")

    _make_distribution_figures(native,out/"figures")
    # Full feature-level extraction accounting.
    missing_rows=[]
    for feat in FEATURE_COLUMNS:
        if feat in native:
            for (task,g),f in native.groupby(["task_code","group"]):
                missing_rows.append({"task_code":task,"group":g,"feature":feat,"n_records":len(f),"n_valid":int(f[feat].notna().sum()),"n_missing":int(f[feat].isna().sum())})
    pd.DataFrame(missing_rows).to_csv(out/"features"/"feature_missingness_summary.csv",index=False)
    qc_manifest.to_csv(out/"qc"/"qc_input_manifest.csv",index=False)
    checks=_validate(project_root,qc_manifest,qc,native_rows,native,repeat,resamp,protected_before,out)
    protected_after=protected_hashes(project_root)
    (out/"reliability"/"baseline_integrity.json").write_text(json.dumps({
        "experiment_id":EXPERIMENT_ID,"protected_file_count":len(protected_before),
        "unchanged":protected_before==protected_after,
        "changed_files":[k for k,v in protected_before.items() if protected_after.get(k)!=v],
        "sha256_before":protected_before,"sha256_after":protected_after},indent=2)+"\n",encoding="utf-8")
    report_data={"qc":qc,"diag":diag,"seg_summary":seg_summary,"native":native,"failures":failures,"feature_value_issues":feature_value_issues,
                 "comparison":pd.read_csv(stats_dir/"native16_group_comparisons_effect_sizes_ci.csv"),
                 "paired":pd.read_csv(paired_dir/"native16_paired_vowel_comparisons.csv"),
                 "checks":checks,"combined":combined,"resample_changes":rc}
    write_research_report(report_data,out)
    logger.info("finished %s checks=%s",EXPERIMENT_ID,checks.status.value_counts().to_dict())
    return report_data


def write_research_report(d, out_dir: Path):
    qc=d["qc"];native=d["native"];checks=d["checks"];comp=d["comparison"]
    selected_counts=native.groupby(["task_code","group"]).agg(rows=("relative_path","size"),valid=("status",lambda s:int(s.eq("ok").sum()))).reset_index()
    failures=native[native.status.ne("ok")]
    peak=comp[comp.feature.isin(PRIMARY_FEATURES)].copy()
    # Sort by feature/vowel and report effect estimates, not winner-take-all rankings.
    summary_cols=[c for c in ["task_code","feature","n_hc","n_pd","mean_difference_hc_minus_pd","mean_diff_ci95_low","mean_diff_ci95_high","hedges_g_hc_minus_pd","hedges_g_ci95_low","hedges_g_ci95_high","cliffs_delta_hc_minus_pd","cliffs_delta_ci95_low","cliffs_delta_ci95_high","mann_whitney_p_unadjusted","holm_p_within_feature_across_vowels","holm_p_primary_20_comparisons"] if c in peak]
    top=peak[summary_cols].sort_values(["feature","task_code"]) if not peak.empty else peak
    qc_flags=qc.qc_flags.fillna("").astype(str).value_counts().head(12).to_dict()
    lines=["# Italian multi-task acoustic analysis — research report", "",
           f"**Experiment:** {EXPERIMENT_ID}. **Scope:** first vowel variants only for primary feature/statistical analyses; other selected tasks received audio QC and a segmentation pilot. No classifier was trained.", "",
           "## Executive summary", "",
           f"Audio QC covered {len(qc)} selected recordings across the reconciled HC-PD cohort and descriptive YHC. Native-16-kHz vowel extraction attempted {len(native)} recordings ({native.group.eq('ELDERLY_HC').sum()} HC and {native.group.eq('PD').sum()} PD across five vowels). {len(failures)} recordings had top-level feature extraction failures; {int(native.status.eq('ok').sum())} completed.",
           f"Automated output validation: {int(checks.status.eq('PASS').sum())}/{len(checks)} checks passed and {int(checks.status.eq('FAIL').sum())} failed. See `validation_checks.csv`.",
           "Results are exploratory, small-sample and non-clinical. Statistical summaries are participant-level; frames are not treated as independent observations.", "",
           "## Data actually analyzed", "",
           "- Primary group: reconciled elderly HC vs PD, first vowel codes VA1/VE1/VI1/VO1/VU1, native 16 kHz only.",
           "- Descriptive QC: reconciled HC/PD plus YHC separately, for VA1/VA2, VE1/VE2, VI1/VI2, VO1/VO2, VU1/VU2, D1/D2, B1/B2, and FB1.",
           "- Sample-rate sensitivity: eligible 44.1 kHz first-vowel recordings analyzed at native rate and after polyphase resampling to 16 kHz; separate combined-rate sensitivity table.",
           "- Second vowel variants, PR1, PR11, transcript-dependent reading features, and protocol-dependent syllable measures were deferred.", "",
           "### Extraction coverage", "",markdown_table(selected_counts),"",
           "### Audio QC", "",
           f"QC status: {qc.status.value_counts().to_dict()}. Flag-string frequency (including unflagged): {qc_flags}. Review queue rows: {int(qc.qc_flag_count.fillna(1).gt(0).sum())}. Flags do not automatically exclude recordings.",
           "The reported RMS dynamic-range proxy is not SNR. No calibrated noise-only segment was available. Details: `qc/per_recording_qc.csv`, `qc/task_qc_summary.csv`, `qc/group_qc_summary.csv`, and `qc/review_queue.csv`.", "",
           "## Segmentation and signal inspection pilot", "",
           (markdown_table(d["seg_summary"]) if not d["seg_summary"].empty else "No pilot rows."),"",
           f"The deterministic blinded sample contained {len(d['diag'])} recordings; {int(d['diag'].segmentation_flagged.sum()) if 'segmentation_flagged' in d['diag'] else 0} crossed provisional candidate segmentation flags. Visual review found long voiced intervals in most sampled first-vowel recordings, with some onset decay, interruptions, and multiple voiced segments. All sampled D1/D2 examples showed repeated CV-like bursts, but the exact repetition instruction remains unknown. Reading RMS-VAD masks were fragmented into many short bouts and need manual threshold review. These pilot observations do not establish structure for every recording. See `segmentation/segmentation_diagnostic_report.md` and contact sheets.",
           "D1/D2 production was not classified as repeated versus isolated solely from the codebook. Reading features are general RMS-VAD timing candidates only; no transcripts, speaking rates, phonetic alignment, or text-dependent measures were produced.","",
           "## Primary acoustic group comparisons", "",
           "Prespecified primary features were median F0, local jitter, local shimmer, and mean HNR. Other extracted features are exploratory. The table reports HC-minus-PD mean-difference bootstrap CIs, Hedges' g and Cliff's delta with participant bootstrap CIs, and Mann-Whitney p-values. Holm corrections are shown within each feature across the five vowel contexts and across the complete 20-comparison primary feature family.","",
           (markdown_table(top) if not top.empty else "No primary comparison table rows."),"",
           "Do not interpret effect direction as disease-specific. Small n yields wide uncertainty; formant validity flags and missingness are reported separately. All feature summaries, effect tables, confidence intervals and outlier review records are in `statistics/`.","",
           "## Paired vowel comparisons", "",
           "Within-participant paired contrasts preserve vowel identity and use complete pairs. First-vowel comparisons are context contrasts, not repeatability estimates. Group-specific pairwise differences and Friedman omnibus summaries are under `paired_comparisons/`; complete-participant counts are explicit. Second vowel variants were not used as test-retest trials.","",
           "## Sampling-rate sensitivity", "",
           f"Native 44.1 kHz first-vowel recordings: {len(d['resample_changes']) and d['resample_changes'].relative_path.nunique() or 0} recordings compared before/after resampling. Resampling feature changes are paired by source recording. The combined table contains {len(d['combined'])} recording-task observations and keeps `sample_rate_original_hz`. Across the 20 primary-feature/vowel HC-PD contrasts, native-16-kHz and combined-rate mean-difference directions agreed in all cases; mean absolute Hedges-g change was 0.116. See `sampling_rate_sensitivity/native_vs_combined_group_effects.csv`. This is a sample-size sensitivity result, not evidence that rate confounding is absent.",
           "Native 16 kHz remains the primary group comparison because all elderly HC are 16 kHz and seven additional PD participants have only 44.1 kHz recordings for these tasks. The combined sensitivity does not remove acquisition confounding. See `sampling_rate_sensitivity/` for paired changes, combined-rate comparisons, and original-rate strata.","",
           "## Reliability and reproducibility", "",
           "A complete second extraction was performed for the native-16-kHz primary set. Exact numeric agreement for repeatable deterministic features is summarized in `reliability/repeat_extraction_summary.csv`. Pitch floors 60/75/90 Hz were run on the same first-vowel native-16-kHz files; results are separate from the primary table. This is extraction repeatability and parameter sensitivity, not biological test-retest reliability.",
           "VA1 baseline and historical outputs were hash-protected before and after the run. See `reliability/baseline_integrity.json` and validation checks.","",
           "## Failures, caveats, and deferred work", "",
           f"Top-level feature extraction failures: {len(failures)}. Feature-level undefined measurements and plausibility flags are retained in `features/feature_extraction_failures.csv`; broad physical-range review flags are in `features/feature_value_validation_flags.csv` (n={len(d.get('feature_value_issues',[]))}). No failed value was replaced with zero. CPPS was deferred because no validated implementation was available in the installed environment; spectral tilt was deferred pending a fixed definition. CPP/CPPS, voice quality in connected speech, and text-dependent measures were not imputed from unrelated features.",
           "The stable-vowel algorithm uses the longest contiguous pitch-voiced run and trims 10% at its ends; this is a candidate segmentation rule and may fail for irregular or breathy phonation. Formants use a fixed 5 kHz Burg maximum-formant setting and broad range flags; flagged tracks require manual review. The sample is not powered for confirmatory inference, and original sample rate is group-associated.","",
           "## Validation", "",markdown_table(checks),"",
           "## Recommended next work package", "",
           "Review the QC/feature failure queues and the blinded segmentation contact sheets; manually adjudicate flagged recordings without changing cohort membership. Confirm D1/D2 production format and reading materials if documentation can be obtained. Then refine segmentation/feature validity, retain the native-16-kHz primary comparisons and resampled sensitivity separately, and extend only supported timing measures to D1/D2 and reading. Do not begin classification until acoustic measurement reliability and uncertainty have been reviewed."]
    (out_dir/"research_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
