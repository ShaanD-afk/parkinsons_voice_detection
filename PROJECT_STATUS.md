# Project status

**Last inspected:** 2026-10-30  
**Scope:** ItalianPVS voice and speech analysis for exploratory Parkinson's disease research. This is not a clinically validated diagnostic system.

## Research objective and guardrails

Study acoustic differences across speech tasks, measurement reliability, task coverage, and ultimately cross-language generalization. Establish coverage, acoustic differences, reliability, and participant-level validation before considering more complex models. The current VA1 cohort has 29 participants, so deep learning is not justified as the next step. Preserve the existing VA1 scripts, feature table, predictions, and results as the reference experiment. Any methodological change must be a separately named and documented experiment; never overwrite baseline outputs. Keep `PROJECT_STATUS.md`, `RESEARCH_DECISIONS.md`, and `EXPERIMENT_LOG.md` current.

## Verified project inventory

- `data/italian_manifest.csv`: 831 recordings; 437 PD, 349 elderly HC, 45 young HC; 65 distinct participant folders; 671 recordings at 16 kHz and 160 at 44.1 kHz.
- Observed task codes: `B1, B2, D1, D2, FB1, PR1, PR11, VA1, VA2, VE1, VE2, VI1, VI2, VO1, VO2, VU1, VU2`. `PR11` occurs once in YHC and remains an unresolved filename/task-code anomaly.
- `data/primary_cohort_manifest.csv` and related cohort files preserve the metadata reconciliation. Reconciliation summary reports 17 included elderly HC and 19 included PD participants before sample-rate filtering, with 455 primary-cohort 16 kHz recordings. Do not alter the reconciliation decisions without explicit approval.
- `data/va1_features_raw.csv`: 29 rows, 17 elderly HC and 12 PD, all 16 kHz, one participant per row. The six baseline model features are jitter, shimmer, HNR, spectral centroid, spectral rolloff, and spectral flatness.
- `data/va1_quality_report.csv`: 29 successful audio reads, no quality flags. `data/va1_features_raw_failures.csv` is empty (header only).
- Existing VA1 scripts include extraction, baseline training/comparison, nested CV, demographic comparison, pitch sensitivity, quality checks, pitch diagnostics, and `src/reference_reproduction_audit.py`. `tests/` exists; tests were not run during this inspection.
- Existing report: `reports/pitch_sensitivity_report.md`. Existing experiment outputs include baseline predictions, model-comparison fold/prediction tables, nested CV outputs, demographic baseline outputs, and pitch sensitivity tables/diagnostics under `data/`.
- Task audit: the exploratory precursor `src/audit_italian_tasks.py` remains preserved. The validated implementation is `src/italian_task_audit.py`, with deliverables in `reports/italian_task_audit/` (14/14 structural checks passed). Workbook and task-code questions remain open as documented below.
- Environment artifacts: `.venv/` exists. The visible `requirements.txt/` is a directory rather than a conventional requirements file; inspect its contents before relying on dependency documentation. README records Python 3.14.4 and package versions, but environment versions were not independently checked here.
- Working tree contains untracked `README.md`, `src/audit_italian_tasks.py`, `results/`, and `.DS_Store`. These were present before documentation changes and must be preserved.

## VA1 reference baseline

The six-feature logistic-regression setup in `src/train_baseline.py` uses median imputation and standardization inside a scikit-learn pipeline, balanced class weights, `max_iter=5000`, and shuffled stratified 5-fold CV with seed 42. The script writes `data/baseline_oof_predictions.csv`; preserve this output. The OOF table contains one prediction per participant (29 unique participant folders).

Historical results documented in the project report/master notes: pooled OOF balanced accuracy 0.900, ROC-AUC 0.917, sensitivity 0.917, specificity 0.882, confusion matrix TN=15, FP=2, FN=1, TP=11; fold balanced accuracies 0.833, 0.833, 1.000, 1.000, 0.833 (mean 0.900, SD 0.091). Treat as exploratory and uncertain. The demographic-only result is approximately BA 0.515 / ROC-AUC 0.512; this does not rule out confounding. Other existing analyses include RBF SVM, dummy classifier, and nested CV.

Pitch sensitivity report documents exact reproduction of the 75 Hz reference extraction across 29 recordings and compares 60, 75, and 90 Hz floors (500 Hz ceiling, 0.01 s step). Potential pitch tracking concerns remain for AGNESE P, Giovanni M, GILDA C, and GRAZIA G; participants remain included pending further investigation.

## Phase 2 progress

**Phase 2, remaining Italian multi-task analyses — computationally completed; human-dependent questions remain (2026-10-30).** Experiment `ITALIAN-PHASE2-REMAINING-20261030` is isolated in `results/italian_research_phase2/`. The new scripts perform first-vowel result sensitivity checks against historical CSVs, first/second vowel signal characterization, all-recording D1/D2 diagnostics, fixed-threshold connected-speech candidate measurements and VAD sensitivity, compatibility research for English datasets, and generation of review materials. The local macOS review GUI writes annotations separately from source data and automated QC. No features were extracted into or written over historical VA1 or `italian_multitask_v1` outputs; no classifier was trained; no participant was excluded.

The independent sensitivity audit confirms the native-16-kHz VI1 local-shimmer mean HC-minus-PD difference (0.04439; 17/12) and historical Mann–Whitney p-value (0.000600). Its direction remains positive under every leave-one-participant-out analysis (minimum difference 0.03335), median/trimmed/winsorized summaries, and independently constructed percentile and BCa bootstrap intervals. The inferential conclusion depends on method: original Holm-adjusted Mann–Whitney p≈0.012, while 20%-trimmed Yuen Holm p≈0.061. Treat as exploratory, not confirmatory. The group contrast is native-16-kHz only; adding PD-only resampled recordings cannot resolve rate/group confounding.

The 72 reconciled D1/D2 recordings show multiple CV-like energy events in most inspected plots, but candidate events are not validated syllable counts. D1/D2 timing analyses await manual onset review and task-protocol clarification. Connected-speech candidates cover 104 participant-task recordings; active-speech and pause summaries vary materially across −30/−35/−40 dB thresholds and require human mask review. B1/B2 changes are descriptive, with possible order/practice/fatigue effects. The 355 first/second vowel characterization rows show distinct signal-duration distributions, but codebook definitions for second variants remain blank; no repetition or repeatability claim is made.

Cross-dataset compatibility is documented without downloading or merging external data. Figshare licensing remains unverified; MDVR-KCL has broad reading-task similarity but uses different language/materials; other examined UCI resources do not offer comparable raw audio or control cohorts. The research manuscript is a structured draft, not a completed submission.

Validation: 21/21 computational checks passed; the full suite passed 51 tests. The 159-item deduplicated review queue remains open (0 annotations, 0 automatic exclusions). Therefore, computational deliverables are validated, but the Italian multi-task research phase is **not ready to be considered complete**. See `results/italian_research_phase2/integrated/integrated_evidence_report.md` and the review guide for the current evidence and handoff.

**Step 1 — ItalianPVS task inventory and participant coverage audit: structurally completed on 2026-10-30.** Reproducible implementation: `src/italian_task_audit.py`; outputs: `reports/italian_task_audit/`. The run passed 14/14 structural validation checks. Outputs include task/group counts for the full dataset, reconciled HC-PD cohort, and its 16 kHz subset; participant-task presence matrices for each; pairwise task overlap; roster and identity cross-checks; path, duplicate, and filename checks; workbook task-code reconciliation; and a readable Markdown report. It did not modify source manifests, reconciliation decisions, or VA1 outputs.

The audit found 831 full-manifest recordings across 65 group-qualified participants, 567 recordings across 36 reconciled elderly-HC/PD participants, and 455 16 kHz recordings across 29 participants (17 HC, 12 PD). All 831 audio paths matched files on disk. There were no duplicate paths or filename-prefix/task-code mismatches.

**Phase 2, Step 2 — Italian multi-task acoustic experiment completed 2026-10-30.** The user-authorized work package ran as `ITALIAN-MULTITASK-V1-20261030`; implementation is in `src/italian_multitask/`, outputs in `results/italian_multitask_v1/`. QC covered 561 selected recordings (reconciled HC/PD plus YHC, with YHC descriptive only). First vowel native-16-kHz feature extraction produced 145 recording-task rows: 85 HC and 60 PD, with 17 HC and 12 PD participants represented for each vowel. Thirty-five eligible PD 44.1-kHz recordings were also evaluated natively and after resampling for sensitivity. No classifier was trained.

The run passed all 17 output validation checks; the full test suite passed 43 tests. A second deterministic extraction matched the first for core features. VA1/reference artifacts were hash-verified unchanged. Native 16 kHz remains the primary comparison because the resampled additions are PD-only, leaving rate and group associated. In the 20 primary feature-vowel contrasts, native-versus-combined mean-difference directions agreed; mean absolute Hedges-g change was 0.116. This sensitivity does not remove rate confounding. Only VI1 shimmer survived Holm correction over the 20-comparison primary family (adjusted p=0.012); this is exploratory and not confirmatory evidence. Protocol-dependent interpretation of D1/D2, second vowel codes, and reading remains deferred.

Open audit findings: `FB2` is defined in each FILE CODES workbook but absent from observed codes; `PR1` and one `PR11` recording are undocumented; the protocol does not define expected task schedules; six workbook roster entries and five participant folders have unresolved name-key mismatches. In particular, the existing exclusion `17-28/Mario B` differs from workbook roster `Mario M`. Preserve the exclusion pending evidence. See the audit report and experiment log for detail.

## VA1 uncertainty follow-up

**`VA1-OOF-UNCERTAINTY-20261031` completed.** A separate participant-stratified bootstrap summarized uncertainty in the immutable 29-person VA1 OOF predictions (20,000 replicates; seed 20261031). Estimates (95% percentile intervals): balanced accuracy 0.900 [0.775, 1.000], ROC-AUC 0.917 [0.789, 1.000], sensitivity 0.917 [0.750, 1.000], specificity 0.882 [0.706, 1.000]. These intervals condition on fixed cross-validated predictions and do not capture model refitting/fold assignment or external-population uncertainty. No classifier was trained and no baseline, cohort, or prior output was changed. Reproduce in a fresh output directory with `.venv/bin/python src/va1_oof_uncertainty.py --output results/va1_oof_uncertainty_reproduction`; outputs and input hash are in `results/va1_oof_uncertainty_20261031/`.

## Known issues and unresolved questions

1. Clarify why documented FB2 is absent and what PR1/PR11 represent; retain literal codes pending documentation. FB1/FB2 protocol relationship remains unverified.
2. Resolve workbook-folder identity mismatches only with supporting evidence; do not alter the existing reconciliation or exclusions.
3. Establish the expected participant task schedule before labeling task-presence zeros as missing required recordings.
4. Complete pitch-track review, including validation that the 75 Hz reference exactly reproduces original extraction, and document participant-level errors and uncertainty for VA1 results.
5. Inspect the actual `requirements.txt/` directory and installed environment before proposing dependency changes.
6. The first reusable task-level pipeline is now implemented and has extracted first-vowel measures. Before extending syllable or connected-speech features, manually adjudicate QC/segmentation items and verify recording protocols; keep any extension in a separately configured experiment.
7. The remaining-analysis review queue has 159 distinct recordings with no annotations yet. Prioritize the 19 historical QC-flagged recordings, six pitch-tracking sensitivity cases, stratified B1/B2/FB1 masks, and all 72 D1/D2 recordings with manual onset/count fields. The queue de-duplicates overlapping reasons; consult `results/italian_research_phase2/review/human_review_queue.csv` for item-level rationale.
8. D1/D2 repeated-production pattern is supported by visual signal evidence in most recordings but is not a verified original protocol; no participant-level syllable-timing results are reported. Reading VAD masks are threshold-sensitive and not yet validated. CPP/CPPS, transcript-dependent timing, and syllable rate remain deferred.
9. Figshare license was unresolved at the prior handover; the final package verified the item page lists CC BY 4.0. Attribute the source and do not redistribute the audio in this package. Cross-language model training remains out of scope.

## Prioritized next steps

1. Resolve or explicitly retain as open the task-code and roster identity questions raised by the audit; keep the evidence trail.
2. Define expected task availability and decide which task pairs have adequate common-participant coverage by group and sample rate.
3. Finish VA1 reference/pitch-track and prediction review without changing baseline artifacts; participant-bootstrap uncertainty for the saved OOF metrics is now reported, while review and model-refitting uncertainty remain open.
4. Review flagged audio and first-vowel measurement validity, including pitch-floor sensitivity and formant flags; do not remove participants without evidence.
5. Resolve D1/D2 and reading protocol details, then propose a narrowly scoped, separately named follow-up for supported timing measures.
6. Complete the 159-item human review in the local interface, starting with the flagged and segmentation items noted above; reconcile disagreements and document reviewer identity/date.
7. Obtain authoritative D1/D2 production/count instructions and the second-vowel protocol; then decide whether validated event timing or isolated-CV measurements are supportable.
8. Review connected-speech masks across the stratified sample, select and document a defensible VAD setting, and retain B1/B2 paired analyses separate from FB1.
9. Figshare licensing is verified as CC BY 4.0 on the item page; cite and attribute. Before any follow-on audio reuse, verify the source task details and retain separate dataset analyses. Cross-language model training remains out of scope until measurement comparability is established.

## Change control

VA1 artifacts are immutable reference outputs. New extraction settings, cohorts, features, or evaluation methods require new files/directories and an entry in `EXPERIMENT_LOG.md`; document rationale in `RESEARCH_DECISIONS.md`. Never delete source data or modify reconciled participant decisions without explicit approval.

## Final research package update — 2026-10-31

- Built `results/final_research/` with Italian results and manuscript drafts, English dataset inventory and results, English manuscript draft, cross-dataset comparison, combined summary, reproducibility report, publication-readiness assessment, machine-readable tables, and 10 publication-oriented figures (PNG + PDF).
- Completed an independent English/related-dataset audit in `results/english_research_20261031/`. Figshare: 81 readable waveforms / 81 coded participants (41 HC, 40 PD). MDVR-KCL: 73 readable recordings / 37 coded participants (21 HC, 16 PD), with read/dialogue kept separate. UCI 174, 189, and 301 remain feature-table analyses; UCI 470 and mPower were not locally analyzed.
- All 28 package validation checks passed; full test suite: 59 passed. Source hash manifest covered 440 existing files and reports no changed paths. VA1 baseline, Italian task audit, reconciliation, and prior phase outputs were not modified.
- No classifier was trained. The final package remains exploratory and is not publication-ready for disease-specific confirmation or external/cross-language validation. Main remaining blockers: 159 Italian audio review items, undocumented task details, unvalidated connected-speech VAD and dialogue speaker attribution, and clarification of the UCI 174 participant-count mismatch.
- Recommended next work: human-adjudicate the Italian review queue and validate speech masks / speaker turns before any follow-on inferential analysis; then replicate prespecified sustained-vowel findings in a larger, task- and acquisition-matched cohort.


## Cross-language generalisation execution — 2026-10-31

**Experiment `CROSS-LANGUAGE-GENERALISATION-V1-20261031` completed.** New implementation is under `src/cross_language_generalisation/`; outputs are isolated in `results/cross_language_generalisation_v1/`. Italian VA1 (17 HC/12 PD) and Figshare sustained `/a/` (41 HC/40 PD, 8 kHz telephone) were re-extracted with a common feature implementation. Because Figshare language is not established, results are cross-dataset only. MDVR connected reading and UCI feature tables were not used for raw-audio external transfer; 4 task-mismatched directions are explicitly marked invalid.

- **Within-cohort:** best tested nested-CV balanced accuracy was 0.858 on Italian VA1 (Logistic Regression, core4; AUC 0.863; participant-bootstrap BAcc interval [0.716, 0.971]) and 0.679 on Figshare (Random Forest, core5; AUC 0.695; interval [0.579, 0.778]). The model/feature selection among tested variants makes these descriptive exploratory maxima. The existing six-feature VA1 OOF baseline remains unchanged and is not directly comparable.
- **Zero-shot:** native-rate Italian→Figshare BAcc ranged 0.382–0.551 (best ROC-AUC 0.501); Figshare→Italian BAcc ranged 0.277–0.569 (best ROC-AUC 0.618). Italian downsampling to 8 kHz did not restore useful transfer; rate-matched sensitivity remained low.
- **Dataset signal:** diagnosis-count-matched dataset identity reached median BAcc 0.672 (standardized Logistic), 0.707 (RBF SVM), and 0.655 (Random Forest), using 30 matched draws with 17 HC/12 PD per dataset. A fixed-logistic ablation showed the most domain signal in F0 IQR and jitter; this is not causal attribution.
- **Other baselines:** MDVR reading candidate features reached best internal BAcc 0.827, but VAD/speaker validity is unreviewed; it is provisional, not an externally comparable speech result. UCI #174/#301 feature-only results remain separate.
- **Validation:** 30/30 new integrity/metric/split/output checks passed. Full repository test suite: 66 passed. All 1,429 hashed pre-existing files matched before/after; no VA1, reconciliation, task inventory, or prior experiment outputs were changed. Six figures are provided in PNG/PDF. No environment dependencies were installed; statsmodels was unavailable, so HC3 regression used the equivalent matrix calculation with NumPy/SciPy.

**Interpretation:** The experiment found no useful cross-dataset transfer and no stable disease-related acoustic pattern replicated across the two raw-audio cohorts. Good within-dataset CV did not generalize. Figshare is not known to be English, so this phase cannot answer language transfer directly. The next high-value study is a documented, repeated `/a/` and `/i/` protocol in both language groups using a shared recording chain and a fully held-out cohort.
