from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.cross_language_generalisation.pipeline import OUT
T=OUT/'tables'

def md(df, cols=None, n=None, precision=3):
    z=df if cols is None else df[cols]
    if n is not None:z=z.head(n)
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.{precision}f}'
        return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,z.columns))+' |','| '+' | '.join(['---']*len(z.columns))+' |',*('| '+' | '.join(fmt(x) for x in row)+' |' for row in z.itertuples(index=False,name=None))])

def build():
    compat=pd.read_csv(T/'dataset_compatibility_matrix.csv')
    effects=pd.read_csv(T/'participant_level_group_effects.csv')
    adj=pd.read_csv(T/'age_sex_adjusted_domain_models.csv')
    within=pd.read_csv(T/'within_dataset_classifier_results.csv')
    ext=pd.read_csv(T/'external_classifier_results.csv')
    idc=pd.read_csv(T/'dataset_identity_matched_cv.csv')
    resamp=pd.read_csv(T/'italian_resampling_sensitivity.csv')
    floors=pd.read_csv(T/'feature_setting_sensitivity.csv')
    hashes=json.loads((OUT/'integrity_check.json').read_text())
    # Primary cross-domain subset: native-rate comparison; sensitivity is separately named.
    primary=ext.loc[~ext.training_dataset.str.contains('resampled',na=False)&~ext.test_dataset.str.contains('resampled',na=False)]
    best_within=within.loc[within.evaluation.eq('within_dataset_nested_cv')].sort_values('balanced_accuracy',ascending=False).groupby('training_dataset',as_index=False).head(1)
    id_summary=idc.groupby('model').balanced_accuracy.agg(median='median',minimum='min',maximum='max').reset_index()
    report=f'''# Cross-language generalisation experiment: README

## Experiment

`CROSS-LANGUAGE-GENERALISATION-V1-20261031` tests whether a small, pre-specified voice feature set transfers across independent datasets. The strongest defensible pair is ItalianPVS VA1 and Figshare sustained `/a/`. Figshare language is undocumented in the local source metadata, so this is **cross-dataset** testing, not a confirmed Italian-to-English language transfer experiment. MDVR reading/dialogue and UCI feature tables are kept out of raw-audio external testing.

All outputs are isolated in this versioned directory. The existing VA1 baseline, task audit, participant reconciliation, and prior result trees were treated as read-only inputs. Before/after hashes cover {hashes['source_hash_count_before']} pre-existing source files; status: **{'unchanged' if hashes['all_preexisting_sources_unchanged'] else 'CHANGED'}**.

## Main cohorts and execution

- ItalianPVS VA1: 29 participants (17 elderly HC, 12 PD), one native 16 kHz recording/person.
- Figshare sustained `/a/`: 81 local waveform/workbook-matched participants (41 HC, 40 PD), one native 8 kHz telephone recording/person.
- Shared acoustic features were re-extracted from raw files using the same implementation and settings. A held-out participant never entered its fold's imputer/scaler/model fit. Zero-shot models were tuned only within their source cohort; no target labels were used for tuning, preprocessing, or threshold selection.
- MDVR reading is a separately labelled provisional within-dataset baseline. UCI #174/#301 are feature-table-only supplementary internal analyses. They are not raw-audio or language validation.

## Reproduction

```bash
MPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.cross_language_generalisation.run_experiment
MPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.cross_language_generalisation.run_ablation
MPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.cross_language_generalisation.validate_experiment
MPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/ -q
```

The full run takes approximately 21 minutes on the recorded local environment. Configuration, selected fold parameters, participant split IDs, predictions, metrics, source hashes, QC, and derived 8 kHz Italian copies are retained here. No source audio is rewritten or copied except the explicitly derived resampling sensitivity files.

## Key result

Within-cohort nested CV performed materially better in Italian VA1 than the external zero-shot direction. Italian-trained models did not transfer reliably to Figshare: the best tested native-rate core-5 balanced accuracy was about {primary.loc[primary.training_dataset.str.startswith('Italian'),'balanced_accuracy'].max():.3f}, with ROC-AUC no higher than {primary.loc[primary.training_dataset.str.startswith('Italian'),'roc_auc'].max():.3f}. Reverse Figshare-to-Italian transfer also remained weak: best core-5 balanced accuracy {primary.loc[primary.training_dataset.str.startswith('Figshare'),'balanced_accuracy'].max():.3f}. Exact estimates and bootstrap intervals are in `tables/external_classifier_results.csv` and `tables/master_classifier_summary.csv`.

A diagnosis-count-matched dataset-origin classifier achieved median 5-fold balanced accuracy of {id_summary['median'].min():.3f}–{id_summary['median'].max():.3f} by model-specific medians (individual model medians: {', '.join(f'{r.model} {r.median:.3f}' for r in id_summary.itertuples())}). This means the common acoustic features retain dataset-domain information even with HC/PD proportions matched. It does not identify whether that signal comes from sample rate, device, language, cohort, or protocol.

## Directory guide

- `dataset_compatibility.md`, `feature_harmonisation.md`, `statistical_comparison.md`
- `classifier_results.md`, `generalisation_analysis.md`, `limitations_and_failed_experiments.md`, `experiment_decisions.md`
- `tables/` contains machine-readable feature, QC, statistical, model, prediction, fold-audit, and validation outputs.
- `figures/` contains six figures in 320 dpi PNG and PDF.
- `configs/experiment_config.json` and `logs/` contain reproducibility details.

This is an exploratory experiment. It establishes no clinical validity and does not support a language-transfer claim.
'''
    (OUT/'README.md').write_text(report)
    (OUT/'dataset_compatibility.md').write_text(f'''# Dataset and task compatibility

## Decision

ItalianPVS VA1 versus Figshare sustained `/a/` is **comparable with explicit limitations**. Both source descriptions identify phonation of /a/: the Italian FILE CODES workbook says “2 phonation of the vocal ‘a’”; the Figshare item describes prolonged /a/ by telephone. The exact Italian duration/repetition protocol is not documented in the files reviewed. Figshare language is not established. Thus the matched phonetic target supports an exploratory cross-dataset test, but it does not establish cross-language generalisation.

## Compatibility matrix

{md(compat)}

## Why MDVR is excluded from external transfer

MDVR ReadText and SpontaneousDialogue are connected speech, while Italian VA1 and Figshare are sustained phonation. Their speaking task, segmentation assumptions, and feature meanings differ. MDVR reading's candidate VAD features have not been human-reviewed; spontaneous dialogue may contain examiner speech. They can inform a provisional within-dataset exercise, not a direct raw-audio zero-shot test against sustained `/a/`.

## Why UCI is excluded from external raw-audio transfer

UCI #174, #189, and #301 are precomputed feature tables with no local waveforms. Their historical feature extraction settings cannot be rerun under the common raw-audio pipeline. UCI #174 and UCI #301 are analyzed only as separate within-table classifiers. UCI #189 is PD-only and cannot support a binary classifier. The local UCI #174 file resolves to 32 IDs (24 PD/8 HC), unlike the UCI source page narrative (31/23/8); this remains an unresolved source discrepancy.

## Cohort coverage

| Dataset/task | HC | PD | Participant unit |
|---|---:|---:|---|
| ItalianPVS VA1 primary native 16 kHz | 17 | 12 | One VA1 recording/person |
| Figshare prolonged `/a/`, native 8 kHz | 41 | 40 | One local waveform/workbook ID/person |
| MDVR reading text | 21 | 16 | One local reading file/person |
| MDVR spontaneous dialogue | 21 | 15 | One local dialogue file/person |
| UCI #174 | 8 | 24 | 195 rows collapsed by local participant ID to one median row/person |
| UCI #301 training vowels | 20 | 20 | One task row/person/vowel; `/a/`, `/o/`, `/u/` analyzed separately |

No cross-dataset identity linkage is available; IDs are dataset-scoped and pseudonymized. A zero overlap of scoped IDs is a split-integrity check, not proof that no person could appear in two separately sourced cohorts.
''')
    features=pd.DataFrame([
      ['median_f0_hz','Median positive F0 in candidate stable vowel segment','Praat autocorrelation; 10 ms step; pitch floor 75 Hz; ceiling 500 Hz','Meaningful for sustained phonation','Pitch tracking can octave-error; floor sensitivity evaluated at 60/90 Hz'],
      ['f0_iqr_hz','75th minus 25th percentile of positive F0 frames','Same stable segment and pitch track','Meaningful as within-segment pitch variability; short token limits reliability','IQR can be dominated by tracking failures and task duration'],
      ['jitter_local','Praat local period perturbation','PointProcess periodic cc; floor 75 Hz/ceiling 500; period bounds .0001–.02 s; period factor 1.3','Meaningful only on adequate, stationary voiced vowel segment','Sensitive to cycle detection, noise, bandwidth and duration'],
      ['shimmer_local','Praat local cycle-amplitude perturbation','Same pulse sequence; amplitude factor 1.6','Meaningful only on adequate, stationary voiced vowel segment','Sensitive to gain, microphone, compression and segmentation'],
      ['hnr_mean_db','Mean Praat CC harmonicity over stable interval','10 ms step; minimum pitch 75 Hz; silence threshold .1; one period/window','Voice periodicity summary for vowel phonation','Bandwidth and noise-floor dependent; 16→8 kHz check performed'],
    ],columns=['feature','definition','settings','task_meaning','failure_or_bias'])
    (OUT/'feature_harmonisation.md').write_text(f'''# Feature harmonisation and signal processing

## Shared raw-audio feature implementation

The same new experiment code re-extracted both cohorts from original waveforms. It uses `mono_for_analysis` and the previously established candidate stable-interval method: Praat autocorrelation, longest contiguous voiced run, and trim 10% from both ends. The primary reference pitch floor/ceiling are 75/500 Hz and 10 ms step. Jitter, shimmer, and HNR settings match the existing stable-vowel extractor. No normalization or amplitude gain correction was applied before jitter/shimmer. Participant records remain in the table even when an extraction fails; there were **zero failed extractions** among the 110 primary recordings. Automated QC flags and implausibility/outlier flags never cause exclusion.

| Feature | Definition and use |
|---|---|
{chr(10).join('| '+r[0]+' | '+r[1]+'; '+r[2]+'; '+r[3]+'; limitation: '+r[4]+' |' for r in features.itertuples(index=False,name=None))}

The `core4` classifier set is median F0, local jitter, local shimmer and HNR. `core5` adds F0 IQR. These limited feature sets were selected before looking at classifier results; no supervised feature selection was performed. Imputation and scaling, when used, are fitted inside each training fold. Models marked unscaled use median imputation only. Random Forest receives raw feature units by design.

## Sample-rate and bandwidth checks

- Native comparison preserves Italian 16 kHz and Figshare 8 kHz signals. No upsampling was applied to Figshare.
- A separate polyphase 16→8 kHz signal copy was generated for each Italian VA1 waveform under `analysis_copies/italian_va1_8khz/`. It tests processing-rate sensitivity on the same 29 people; it cannot recreate telephone microphone bandwidth, codec, noise, or channel effects.
- The Italian downsampling comparison preserves very high participant rank correlations, but HNR shifts systematically (Holm-adjusted paired Wilcoxon p≈0.00007). Rates being numerically equal is not evidence of acquisition equivalence.
- Pitch-floor setting sensitivity was evaluated at 60, 75, and 90 Hz in both raw cohorts. One Figshare record failed at the 90 Hz floor; it remains present in extraction/failure tables and is not excluded from the primary 75 Hz analysis.

Detailed output: `tables/harmonised_raw_audio_features.csv`, `tables/all_pitch_floor_extractions.csv`, `tables/feature_setting_sensitivity.csv`, `tables/italian_resampling_sensitivity.csv`, `tables/raw_audio_qc.csv`, and `tables/quality_outlier_flags.csv`.
''')
    core=effects.loc[effects.dataset.isin(['ItalianPVS_VA1','Figshare_sustained_a'])]
    keycols=['dataset','feature','n_hc','n_pd','mean_difference_hc_minus_pd','hedges_g_hc_minus_pd','hedges_g_boot_ci95_low','hedges_g_boot_ci95_high','mann_whitney_p','holm_p_within_dataset_task']
    (OUT/'statistical_comparison.md').write_text(f'''# Participant-level statistical comparison

Each group contrast uses one participant-level feature vector per dataset/task. Mean differences and Hedges g are oriented HC minus PD, with participant-bootstrap percentile 95% intervals (5,000 resamples). Two-sided Mann–Whitney tests are Holm-adjusted over the five common features within each dataset. No recording frames or repeated people were treated as independent. For UCI #174, repeated rows were first collapsed to participant medians. MDVR and UCI results are separately reported and do not imply feature harmonisation.

## Sustained `/a/` shared features

{md(core,keycols,precision=4)}

No common feature demonstrated a stable, statistically supported disease effect in both raw-audio cohorts. Median F0 had the same unadjusted direction (HC–PD < 0), but its size differed and Italian uncertainty was broad. F0 IQR and jitter had opposite directions. Shimmer had the same positive HC–PD direction but a near-zero Figshare estimate and wide intervals. HNR shared a negative direction but the Figshare estimate was close to zero. None of the five Italian effects survived its within-dataset Holm family; Figshare's F0-IQR result did, but its definition can be sensitive to short-token tracking and does not replicate an Italian signal.

## Age/sex adjustment

An HC3 linear model used 110 participants and included diagnosis, dataset, diagnosis×dataset, age (linear, centered) and sex. This is an exploratory covariate sensitivity check. Dataset is entangled with rate, device, cohort, and possibly language; covariate adjustment cannot disentangle those properties. Interaction p-values did not survive correction across features. See `tables/age_sex_adjusted_domain_models.csv` for all contrasts and intervals.

## Other dataset-specific effects

MDVR reading candidate measures and UCI #174/#301 precomputed-feature contrasts are reported in `tables/participant_level_group_effects.csv`, with Holm correction by dataset/task family. MDVR candidate VAD/prosody signals need human segmentation and speaker review. UCI #174 participant IDs conflict with its dataset-page count. These are not evidence of raw-acoustic cross-language replication.

## Visualisations

- `figures/fig1_common_vowel_distributions`: participant distributions by cohort/group
- `figures/fig2_common_vowel_effect_forest`: shared-feature effect sizes and intervals
- `figures/fig4_common_vowel_pca`: descriptive PCA, not evidence of biological separation
''')
    compact_cols=['training_dataset','test_dataset','task_compatibility','model','feature_set','training_participants','training_n_hc','training_n_pd','test_participants','test_n_hc','test_n_pd','balanced_accuracy','balanced_accuracy_ci95_low','balanced_accuracy_ci95_high','sensitivity','specificity','roc_auc','roc_auc_ci95_low','roc_auc_ci95_high','evaluation_status']
    internalcols=['training_dataset','test_dataset','task_compatibility','model','feature_set','participants','n_hc','n_pd','balanced_accuracy','balanced_accuracy_ci95_low','balanced_accuracy_ci95_high','sensitivity','specificity','roc_auc','roc_auc_ci95_low','roc_auc_ci95_high','evaluation_status']
    exshow=ext.loc[~ext.training_dataset.str.contains('resampled',na=False)&~ext.test_dataset.str.contains('resampled',na=False)]
    (OUT/'classifier_results.md').write_text(f'''# Classifier results

Classical models only: Logistic Regression with and without training-fold standardization, RBF SVM with training-fold standardization, and Random Forest. Gradient Boosting was not added: the smallest raw cohort has 29 people and only five pre-specified inputs; adding another tuned model family would increase selection variance without a clear scientific gain. No deep model was trained.

## Internal evaluation

Within-dataset estimates use outer stratified 5-fold CV and 3-fold inner grid search; imputation, scaling, and model selection occur only inside training folds. The OOF prediction table has one prediction per participant/model/feature set. Intervals resample participants while holding those OOF predictions fixed; they do not account for fold assignment or refitting uncertainty. `tables/within_dataset_classifier_results.csv` contains all 36 model/feature-set evaluations (including confusion counts and intervals). Its top observed score is not an unbiased selected-model estimate.

Highest balanced-accuracy run per dataset (descriptive selection from the tested grid):

{md(best_within,['training_dataset','model','feature_set','n_hc','n_pd','balanced_accuracy','balanced_accuracy_ci95_low','balanced_accuracy_ci95_high','sensitivity','specificity','roc_auc','roc_auc_ci95_low','roc_auc_ci95_high'],precision=3)}

The Italian nested estimates use four/five harmonized features, so they are not a reproduction of the historical six-feature VA1 pipeline or its fixed OOF result. Figshare internal performance is near chance for several configurations. MDVR's internal reading numbers are provisional because its VAD masks have not been human-validated. UCI internal numbers are based on historical feature tables and separate task-specific feature definitions.

## Genuine external zero-shot evaluation

The source cohort alone was used for hyperparameter search and final model fitting. External target labels did not enter imputation, scaling, feature selection, tuning, or threshold choice. The decision threshold is the model default. Bootstrap intervals are conditional on predictions for these fixed target participants.

Native-rate transfers, using both pre-specified feature sets:

{md(exshow,compact_cols,precision=3)}

The resampling-rate sensitivity runs and invalid directions are in the `external_classifier_results.csv`, `invalid_cross_dataset_directions.csv`, and `master_classifier_summary.csv` tables. Four direct sustained↔reading directions were marked invalid and not scored.

## Metrics and uncertainty

Balanced accuracy, sensitivity, specificity, ROC-AUC, precision, F1, and confusion counts are supplied for every scored run. Accuracy alone is not used. Small cohorts yield broad intervals; repeated model comparisons are exploratory. The external result, rather than the best internal CV result, is the key generalisation test.
''')
    (OUT/'generalisation_analysis.md').write_text(f'''# Cross-dataset generalisation and dataset effects

## Zero-shot transfer result

Italian VA1 (17 HC/12 PD) and Figshare sustained `/a/` (41 HC/40 PD) are comparable only with explicit limitations. The best tested 16 kHz Italian→8 kHz Figshare core-5 balanced accuracy was **{primary.loc[primary.training_dataset.str.startswith('Italian'),'balanced_accuracy'].max():.3f}**; ROC-AUC maximum was **{primary.loc[primary.training_dataset.str.startswith('Italian'),'roc_auc'].max():.3f}**. The reverse best native-rate core-5 balanced accuracy was **{primary.loc[primary.training_dataset.str.startswith('Figshare'),'balanced_accuracy'].max():.3f}**, with maximum ROC-AUC **{primary.loc[primary.training_dataset.str.startswith('Figshare'),'roc_auc'].max():.3f}**. For the same model, the internal score was generally higher than external performance, but all tested configurations and individual results are in the CSV, not just the maxima.

Thus ItalianPVS-trained models did not generalise usefully to Figshare in this zero-shot experiment, and Figshare-trained models did not generalise usefully to Italian VA1. This is **cross-dataset failure**, not proof that language transfer fails: Figshare language is unknown and dataset/device/rate/cohort are confounded.

## Sample-rate sensitivity

The Italian 16→8 kHz copies were tested separately. Aligning nominal sample rates did not make cross-dataset performance reliable: resampled transfer balanced accuracy remained between {ext.loc[ext.training_dataset.str.contains('resampled'),'balanced_accuracy'].min():.3f} and {ext.loc[ext.training_dataset.str.contains('resampled'),'balanced_accuracy'].max():.3f}. The reverse Figshare→Italian resampled sensitivity achieved its best balanced accuracy {ext.loc[ext.test_dataset.str.contains('resampled'),'balanced_accuracy'].max():.3f}. Resampling cannot restore telephone channel response or make recording chains equivalent.

## Dataset identity

To avoid using differing case proportions as an easy domain cue, each of 30 draws retained 17 HC and 12 PD from both datasets (58 participants total). A diagnosis-stratified five-fold CV classified dataset identity using only the five common acoustic features. Median balanced accuracy across matched draws was:

{md(id_summary,['model','median','minimum','maximum'],precision=3)}

Feature patterns can identify cohort origin above chance after observed group-count matching. In a fixed, fold-standardized logistic ablation over the same 30 matched draws, removing F0 IQR reduced median dataset-identification balanced accuracy from 0.672 to 0.603 (median change −0.095); removing jitter reduced it to 0.612 (−0.069). Removing median F0 slightly increased the median to 0.690 (+0.017), while removing shimmer or HNR left the median near 0.672. F0 IQR alone reached 0.621; no single feature reproduced the full score. Fold-standardized logistic coefficients were most consistently negative for F0 IQR and jitter in the dataset-origin model. These ablations indicate which features carry domain information in this sample; they do not establish why or a causal recording mechanism. See `tables/dataset_identity_feature_ablation_summary.csv`, coefficient table, OOF predictions, and Figure 6.

The origin classifier does not establish what causes that separability; plausible sources include recording rate/device/bandwidth, cohort demographics, phonation protocol, and residual diagnosis effects. No causality is claimed.

## Disease-feature direction and robustness

No common sustained `/a/` feature produced a convincing corrected group effect in both cohorts. F0 direction agreed but magnitude and uncertainty differed. F0 variability and jitter directions disagreed. Shimmer was directionally aligned but close to zero in Figshare. HNR was similarly directed but weak in Figshare. Adjusted age/sex models and feature tests are reported with their assumptions; none disentangles the dataset acquisition factors.

Native-to-8 kHz Italian paired feature rank correlations were high, but HNR changed systematically. This sensitivity comparison tests resampling in the same 29 waveforms only; it cannot estimate telephone-vs-microphone domain effects.

## VI1 shimmer

The historical Italian VI1 `/i/` shimmer observation remains exploratory and method-dependent. No local external cohort provides a matched sustained `/i/` task with comparable raw measurements, so this experiment does not replicate it. VA1 `/a/` comparisons are not a substitute for an `/i/` replication.

## Interpretation

The most informative negative finding is that good within-cohort CV does not imply transfer. Current data can detect dataset origin more consistently than they support stable diagnosis transfer. No language-general acoustic biomarker or clinical classifier is established.

## Next experiment that would improve the evidence

Collect a prospectively documented repeated-token sustained `/a/` and `/i/` cohort in both language groups using the same microphone, calibrated distance, native sampling rate, room instructions, and locked feature code. Balance age/sex where feasible, retain participant IDs, and reserve one full cohort for independent testing. This would separate language/task claims from device/rate effects and permit a true `/i/` shimmer replication.
''')
    (OUT/'limitations_and_failed_experiments.md').write_text('''# Limitations, invalid comparisons, and negative results

## Invalid or withheld

- Italian VA1 / Figshare prolonged `/a/` is the only raw-audio external candidate. It is a phonetic-task match with limitations, not a confirmed language pair: Figshare language is unknown; Italian exact task duration/repetition is not documented; 16 vs 8 kHz, devices, recording channel, cohort, and demographics differ.
- Italian VA1↔MDVR reading and Figshare↔MDVR reading are invalid direct transfers: sustained vowels are not connected reading. Spontaneous dialogue is even less compatible. Four sustained-vowel-to-reading directions are included with `invalid_not_comparable` status and no metrics.
- UCI feature tables cannot be used as raw-audio transfer tests because local waveforms and common extraction are absent. UCI #189 is PD-only and cannot support binary HC/PD discrimination.
- Figshare is not labeled English. No cross-language claim is made.

## Negative and inconclusive results

- Zero-shot Italian→Figshare and Figshare→Italian scores are weak and largely near chance despite higher within-cohort CV in Italian.
- Diagnosis-matched dataset classification remains above chance, showing domain-specific feature structure.
- Group-effect directions disagree for F0 variability and jitter. Shimmer's `/a/` direction is not stable enough to infer a general disease feature. No raw-audio result replicates Italian VI1 `/i/` shimmer.
- Matching sample rate by downsampling Italian waveforms produces no meaningful restoration of transfer; channel/device differences remain.
- High nested CV values on the small Italian and MDVR cohorts have wide participant-bootstrap intervals and are exploratory. Fixed-prediction intervals omit training/fold variability.

## Measurement and metadata limitations

- Figshare provides 8 kHz telephone audio and age/sex but not language certainty in the available item metadata. Recording frequency response and device metadata are not harmonized.
- Italian VA1 codebook phrase (“2 phonation of the vocal ‘a’”) identifies the phonation target but not duration/repetition details.
- MDVR connected-speech features use unreviewed candidate VAD. Dialogue may contain examiner speech. Its classifier is a provisional within-reading baseline only; not an external task match.
- UCI #174 local rows yield 32 IDs / 24 PD / 8 HC, unlike the UCI source page's 31 / 23 / 8. UCI #301 feature values are not re-extracted.
- Cross-dataset participant identity linkage is unavailable. Dataset-scoped hashed IDs ensure split bookkeeping; they cannot detect the same real individual appearing in distinct source corpora.
- No language-level causal contrast is identifiable because dataset and acquisition are entangled.

No participants were dropped for inconvenient values. Extraction failures, QC flags, implausible-value flags, and IQR outlier flags are retained. No external cohort was used for adaptation.
''')
    (OUT/'experiment_decisions.md').write_text('''# Experiment decisions

1. **Protect references:** VA1, reconciliation, audit, and prior outputs remain immutable. This work has a new directory and source package.
2. **Primary external task:** compare Italian VA1 with Figshare prolonged `/a/` because the available task descriptions share a sustained `/a/` target. Classify as comparable with explicit limitations; Figshare language and exact Italian repetition/duration are unknown.
3. **Feature set:** use four core features (median F0, local jitter, local shimmer, HNR) and one pre-specified extension (F0 IQR). No supervised feature selection was run; five features and n=29 do not support stable high-dimensional selection.
4. **Raw audio:** re-extract both cohorts from original waveforms with the same implementation. Keep Figshare native 8 kHz and Italian native 16 kHz for primary transfer. Store separate Italian polyphase 8 kHz copies for rate sensitivity; never overwrite originals.
5. **Other datasets:** MDVR reading is a provisional within-dataset exercise with candidate speech features; no connected-speech cross-transfer. MDVR dialogue classifier withheld due speaker attribution uncertainty. UCI #174/#301 receive separate feature-table-only within-dataset baselines; UCI #189 has no HC comparator.
6. **Models:** Logistic Regression, RBF SVM, and Random Forest. Logistic raw-unit variant contrasts training-set standardization with no scaling; RF uses raw units. Preprocessing remains inside each split. Gradient Boosting was omitted because cohorts are small and simple models suffice; deep learning and domain adaptation were not used.
7. **Validation:** nested stratified 5-fold outer / 3-fold inner participant CV for within-cohort performance; external source-only hyperparameter search and refit for zero-shot transfer. Class weights are balanced. Decision thresholds remain defaults; no external-label threshold tuning. Bootstrap CIs condition on saved predictions.
8. **Domain test:** match Figshare participants by HC/PD counts to the Italian sample and classify dataset identity. Repeated subsample spread is not treated as independent replications.
9. **Adjustments:** age/sex HC3 regression is a sensitivity model only. Sample rate, hardware, protocol, and cohort remain entangled with dataset identity; no harmonization algorithm or external adaptation is justified.
10. **Interpretation:** state cross-dataset results only. Do not call them cross-language results, clinical validation, or confirmation of VI1 shimmer.
''')
    (OUT/'logs/run_command.txt').write_text('MPLCONFIGDIR=results/cross_language_generalisation_v1/configs/mpl_cache PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.cross_language_generalisation.run_experiment\n')

if __name__=='__main__':build()
