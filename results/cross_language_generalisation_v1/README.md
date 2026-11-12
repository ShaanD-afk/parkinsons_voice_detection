# Cross-language generalisation experiment: README

## Experiment

`CROSS-LANGUAGE-GENERALISATION-V1-20261031` tests whether a small, pre-specified voice feature set transfers across independent datasets. The strongest defensible pair is ItalianPVS VA1 and Figshare sustained `/a/`. Figshare language is undocumented in the local source metadata, so this is **cross-dataset** testing, not a confirmed Italian-to-English language transfer experiment. MDVR reading/dialogue and UCI feature tables are kept out of raw-audio external testing.

All outputs are isolated in this versioned directory. The existing VA1 baseline, task audit, participant reconciliation, and prior result trees were treated as read-only inputs. Before/after hashes cover 1429 pre-existing source files; status: **unchanged**.

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

Within-cohort nested CV performed materially better in Italian VA1 than the external zero-shot direction. Italian-trained models did not transfer reliably to Figshare: the best tested native-rate core-5 balanced accuracy was about 0.551, with ROC-AUC no higher than 0.501. Reverse Figshare-to-Italian transfer also remained weak: best core-5 balanced accuracy 0.569. Exact estimates and bootstrap intervals are in `tables/external_classifier_results.csv` and `tables/master_classifier_summary.csv`.

A diagnosis-count-matched dataset-origin classifier achieved median 5-fold balanced accuracy of 0.655–0.707 by model-specific medians (individual model medians: Logistic_scaled 0.672, RandomForest 0.655, SVM_RBF_scaled 0.707). This means the common acoustic features retain dataset-domain information even with HC/PD proportions matched. It does not identify whether that signal comes from sample rate, device, language, cohort, or protocol.

## Directory guide

- `dataset_compatibility.md`, `feature_harmonisation.md`, `statistical_comparison.md`
- `classifier_results.md`, `generalisation_analysis.md`, `limitations_and_failed_experiments.md`, `experiment_decisions.md`
- `tables/` contains machine-readable feature, QC, statistical, model, prediction, fold-audit, and validation outputs.
- `figures/` contains six figures in 320 dpi PNG and PDF.
- `configs/experiment_config.json` and `logs/` contain reproducibility details.

This is an exploratory experiment. It establishes no clinical validity and does not support a language-transfer claim.
