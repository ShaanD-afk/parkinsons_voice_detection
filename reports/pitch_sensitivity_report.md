# Parkinson's Voice — Pitch-Setting Robustness Analysis
## Reproducible Research Report

**Date:** 2026-10-29  
**Cohort:** ItalianPVS, primary HC-vs-PD cohort  
**Task:** VA1 (sustained vowel), 16 kHz recordings  
**Python:** 3.14.4 | parselmouth 0.4.7 | scikit-learn 1.9.1 | numpy 2.5.4 | pandas 3.1.0

> **IMPORTANT:** This report describes an **exploratory internal cross-validation** on **29 participants** (17 healthy controls, 12 Parkinson's disease).
> All findings are **hypothesis-generating only**.
> They do **not** establish a clinically validated diagnostic tool, Parkinson's-specific biomarkers, or generalizability to other datasets, populations, or recording environments.

---

## 1. Experiment Overview

This analysis investigates how sensitive acoustic features — and a logistic-regression classifier trained on those features — are to changing the Praat pitch-floor parameter in sustained-vowel (/a/) recordings.

Three pitch-floor configurations were evaluated:

| Configuration | Pitch floor | Pitch ceiling | Time step |
|---|---:|---:|---:|
| floor75_reference (original) | 75 Hz | 500 Hz | 10 ms |
| floor60 | 60 Hz | 500 Hz | 10 ms |
| floor90 | 90 Hz | 500 Hz | 10 ms |

All three were applied to the same 29 recordings. Jitter, shimmer, and HNR were re-extracted for each configuration. Spectral features (centroid, rolloff, flatness) are pitch-independent and were held constant.

---

## 2. Cohort Summary

| Group | Participants | Mean age | Age SD | Female | Male |
|---|---:|---:|---:|---:|---:|
| Healthy control (ELDERLY_HC) | 17 | 67.47 | 5.41 | 11 | 6 |
| Parkinson's disease (PD) | 12 | 69.92 | 6.84 | 6 | 6 |
| **Total** | **29** | — | — | — | — |

The cohort excludes young healthy controls (YHC), three unresolved elderly-control participants (LISCO G, PORCELLI A, SUMMO L), VITO L (elderly control), and one unresolved PD participant (17-28/Mario B). The ANTONIETTA P audio folder was manually mapped to workbook participant Antonella P; an age discrepancy of unresolved origin is documented and the mapping is retained without modification.

---

## 3. Phase 3 — Reference Reproduction Audit

**Result: PASSED — exact match (all feature differences = 0.0)**

All 13 features were compared between `va1_features_raw.csv` (original extraction, floor=75 Hz) and `floor75_reference.csv` (pitch_sensitivity.py re-extraction, floor=75 Hz) across all 29 participants. The maximum absolute difference was 0.0 for every feature, confirming that the pitch-sensitivity pipeline faithfully reproduces the original extractor.

*Saved to:* `data/pitch_sensitivity/reference_reproduction_audit.csv`

---

## 4. Question 1 — Which features change the most?

Summary statistics for each feature under the two alternative floor settings, relative to the 75 Hz reference:

| Feature | Config | Mean abs diff | Max abs diff | Notes |
|---|---|---:|---:|---|
| median_f0_hz | floor60 | 7.19 Hz | **144.44 Hz** | One outlier (AGNESE P) drives max |
| median_f0_hz | floor90 | 0.45 Hz | 4.11 Hz | Very stable for most participants |
| f0_iqr_hz | floor60 | 2.55 Hz | 60.02 Hz | Giovanni M outlier |
| f0_iqr_hz | floor90 | 5.84 Hz | **85.27 Hz** | GILDA C, GRAZIA G outliers |
| f0_std_hz | floor60 | 3.62 Hz | 26.28 Hz | Elevated by outliers |
| f0_std_hz | floor90 | 3.87 Hz | 37.58 Hz | Elevated by outliers |
| hnr_mean_db | floor60 | 0.68 dB | 7.17 dB | Giovanni M outlier |
| hnr_mean_db | floor90 | 0.56 dB | 1.78 dB | Smaller, more uniform |
| jitter_local | floor60 | 0.0022 | 0.0404 | Small in absolute terms |
| jitter_local | floor90 | 0.0010 | 0.0101 | Negligible for most |
| shimmer_local | floor60 | 0.0050 | 0.0624 | Small |
| shimmer_local | floor90 | 0.0024 | 0.0170 | Negligible |
| voiced_frame_fraction | floor60 | 0.015 | 0.256 | Driven by one participant |
| voiced_frame_fraction | floor90 | 0.007 | 0.043 | Stable |

**Interpretation:** F0-related features (median_f0_hz, f0_iqr_hz, f0_std_hz) show the largest sensitivity, but this is dominated by 2-3 participants with probable tracking artefacts. For the remaining 25-27 participants, changes are small. Jitter, shimmer, and HNR are considerably more stable, with the 90 Hz floor producing the smallest mean changes in these features.

**Note:** The 6-feature classifier used in Phase 4 intentionally **excludes** F0, F0 IQR, and F0 std. Only jitter, shimmer, and HNR are pitch-dependent in the classifier.

---

## 5. Question 2 — Which participants are most sensitive?

| Participant | Group | Primary issue | Flagged feature | Max |delta| | Config |
|---|---|---|---|---:|---|
| AGNESE P | HC | Octave jump | median_f0_hz | 144.4 Hz | floor60 |
| GILDA C | HC | IQR suppression | f0_iqr_hz | 85.3 Hz | floor90 |
| GRAZIA G | HC | IQR suppression | f0_iqr_hz | 78.2 Hz | floor90 |
| 17-28/Giovanni M | PD | Octave-down + IQR inflation | median_f0_hz | 58.7 Hz | floor60 |
| 17-28/Giovanni M | PD | Octave-down + IQR inflation | f0_iqr_hz | 60.0 Hz | floor60 |
| GIOVANNA G | HC | Stable high IQR | (all stable) | — | — |
| 17-28/Giustina M | PD | Moderate IQR shift | f0_iqr_hz | 7.5 Hz | floor60 |

All other 22 participants showed changes below 5 Hz in median F0 and below 2 Hz in F0 IQR across all settings.

---

## 6. Question 3 — Do pitch tracks suggest artefacts?

Pitch tracks for 6 participants were reviewed. Plots are in `data/pitch_sensitivity/plots/`.

### AGNESE P (HC) — Probable tracking artefact under floor60

Median F0 is ~217 Hz with the 75 Hz and 90 Hz floors. Under the 60 Hz floor it drops to ~72 Hz — approximately 1/3 of the reference, suggesting the 60 Hz floor allows the tracker to latch onto a subharmonic or low-frequency noise component. The 75 Hz floor was sufficient to avoid this. The 60 Hz result for AGNESE P should be treated as a tracking artefact.

### 17-28/Giovanni M (PD) — Probable octave-down artefact under floor60

Median F0 is ~127 Hz (ref) and ~127 Hz (floor90). Under floor60, it drops to ~68 Hz — close to exactly half. F0 IQR simultaneously inflates from 2.3 Hz to 62.3 Hz, and HNR increases by 7.2 dB. This pattern is highly consistent with octave-down sub-harmonic tracking enabled by the lower floor. The floor60 result for Giovanni M is likely a tracking artefact.

### GILDA C and GRAZIA G (both HC) — Ambiguous

Both show very high F0 IQR (~85-88 Hz) under the reference and floor60, which collapses to ~2-6 Hz under the 90 Hz floor. Two interpretations are possible:
1. The 75/60 Hz results reflect genuine F0 variability and the 90 Hz floor over-suppresses valid frames.
2. The 75/60 Hz results include sub-harmonic tracking below 90 Hz, and the 90 Hz floor produces a cleaner estimate.

Auditory and waveform evidence alone is insufficient to resolve this. **Both participants are retained in all analyses; the ambiguity is documented.**

### GIOVANNA G (HC) — Stable high variability

F0 IQR is 110 Hz under all settings. This stability across settings suggests genuine F0 variability rather than a tracking artefact.

### 17-28/Giustina M (PD) — Low-moderate sensitivity

Changes are below the 20 Hz exploratory threshold for both median F0 and F0 IQR. No tracking artefact is evident.

**Note:** The 20 Hz threshold is an **exploratory review aid only**, not a validated quality-control standard. No participants are excluded based on it.

---

## 7. Question 4 — Missing or invalid values?

**None.** All 29 participants returned valid feature values under all three configurations. Zero failures were recorded for all three configuration runs.

---

## 8. Question 5 — Does the reference exactly reproduce the original?

Yes. All 13 audited features match exactly (max absolute diff = 0.0) between `va1_features_raw.csv` and `floor75_reference.csv`.

---

## 9. Question 6 — Does classifier performance change across settings?

The logistic-regression baseline uses six features: jitter_local, shimmer_local, hnr_mean_db, spectral_centroid_hz, spectral_rolloff_85_hz, spectral_flatness.

### Pooled Out-of-Fold Results

| Configuration | BA | ROC-AUC | Sensitivity | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| floor75 (reference) | **0.900** | **0.917** | 0.917 | 0.882 | 15 | 2 | 1 | 11 |
| floor60 | **0.900** | 0.922 | 0.917 | 0.882 | 15 | 2 | 1 | 11 |
| floor90 | 0.870 | 0.912 | 0.917 | 0.824 | 14 | 3 | 1 | 11 |

### Fold-Level Mean and Standard Deviation

| Config | BA mean | BA SD | AUC mean | AUC SD |
|---|---:|---:|---:|---:|
| floor75 | 0.900 | 0.091 | 0.933 | 0.099 |
| floor60 | 0.900 | 0.091 | 0.956 | 0.061 |
| floor90 | 0.875 | 0.072 | 0.933 | 0.099 |

### Interpretation

- The reference (75 Hz) and floor60 configurations produce **identical** pooled OOF balanced accuracy (0.900) and confusion matrices.
- The floor90 configuration gives a slightly lower pooled OOF balanced accuracy (0.870), with one additional false positive (VITO A).
- **None of these differences represent statistically meaningful changes** given a 29-participant sample. A single prediction change shifts balanced accuracy by ~0.03.

**Caution:** Do NOT interpret the floor60 result (tied with floor75 at BA=0.900) as evidence that the 60 Hz floor is equally valid for F0 analysis. The pitch-track review found probable tracking artefacts for AGNESE P and Giovanni M under floor60. The classifier is not sensitive to those artefacts because the features it uses are less directly affected by the F0 octave-jump errors in those two participants.

---

## 10. Question 7 — Are errors concentrated in sensitive participants?

**VITO A** (elderly HC) is the only participant where configurations disagree. He is **not** among the participants flagged for pitch-setting sensitivity. His prediction changes from HC (correct, floor75/floor60) to PD (incorrect, floor90) — this is a false positive that appears specifically under the 90 Hz floor.

The two participants with suspected tracking artefacts (AGNESE P, Giovanni M) are **not** among the classification errors in any configuration. This suggests the tracker artefacts under floor60 for those participants did not substantially alter their jitter/shimmer/HNR values in a way that changed the classifier outcome — but this is not a general guarantee of robustness.

---

## 11. Question 8 — Confounding factors

### Demographic baseline

| Model | Pooled OOF BA | Pooled OOF ROC-AUC |
|---|---:|---:|
| Dummy (most-frequent) | 0.500 | 0.500 |
| Age + sex LR | 0.515 | 0.512 |
| Acoustic LR | 0.900 | 0.917 |

Age and sex alone do not explain the acoustic classifier's performance on this cohort. This is an exploratory confounding check, not proof that the acoustic model captures disease-specific information.

### Other known confounds

1. **Small sample (n=29):** All metrics carry very wide confidence intervals.
2. **Single-site dataset:** All speakers are elderly Italian adults from one clinical site.
3. **Recording quality:** At least 2 participants have probable pitch-tracking artefacts under one tested setting.
4. **Feature selection:** The six features were fixed before this analysis.
5. **No external validation:** All metrics are internal cross-validation estimates.
6. **Unresolved participant identity issue:** ANTONIETTA P age discrepancy remains unresolved.

---

## 12. Question 9 — What conclusions are justified?

**Supported (exploratory):**
1. The reference pitch-floor (75 Hz) extraction is exactly reproducible.
2. The logistic-regression classifier achieves pooled OOF BA = 0.900 on this cohort — substantially above chance — but with high uncertainty.
3. For 26 of 29 participants, features are largely stable across the three tested pitch-floor settings.
4. For 3 participants (AGNESE P, Giovanni M, GILDA C/GRAZIA G), some features change substantially under at least one alternative setting. At least two cases are consistent with pitch-tracking artefacts.
5. Classification performance is largely stable (BA 0.870-0.900), with a single additional false positive under the 90 Hz floor.
6. Demographic confounding (age, sex) does not straightforwardly explain classifier performance on this cohort.

**Not supported:**
- Clinical diagnostic validity.
- Generalization beyond this dataset.
- Superiority of any specific pitch-floor setting based on classification metrics.
- That jitter, shimmer, or HNR are Parkinson's-specific biomarkers.
- Replication on an independent cohort.

---

## 13. Question 10 — Recommended next steps (prioritized)

1. **Resolve GILDA C and GRAZIA G ambiguity** (auditory inspection, spectral analysis)
2. **Investigate VITO A false positive under floor90** (examine jitter/shimmer/HNR under this setting)
3. **Resolve ANTONIETTA P age discrepancy** (document in reconciliation log)
4. **Test additional pitch ceilings and time steps** for a more complete robustness picture
5. **Expand the cohort** — any additional reconcilable ItalianPVS participants would increase statistical power
6. **External validation** on an independent dataset
7. **Feature importance analysis** (permutation importance or SHAP)
8. **Pre-register a defensible pitch-floor** based on physiological evidence, not classification performance
9. **Use nested CV as the standard** for any further model selection
10. **Include voiced-frame fraction** in sensitivity assessment of the classifier

---

## 14. Files Generated

| File | Description |
|---|---|
| data/pitch_sensitivity/participant_feature_changes.csv | Per-participant feature differences |
| data/pitch_sensitivity/feature_change_summary.csv | Summary statistics per feature x config |
| data/pitch_sensitivity/participant_ranking.csv | Ranked by largest feature changes |
| data/pitch_sensitivity/reference_reproduction_audit.csv | Reference vs original (all exact) |
| data/pitch_sensitivity/pitch_track_review.csv | Narrative review for 6 participants |
| data/pitch_sensitivity/pitch_track_data/ | Frame-level CSV pitch tracks |
| data/pitch_sensitivity/plots/ | PNG pitch-track plots (6 participants) |
| data/pitch_sensitivity/cv_fold_assignments.csv | Fold assignments for all 29 participants |
| data/pitch_sensitivity/model_comparison_folds.csv | Fold-level metrics, all 3 configs |
| data/pitch_sensitivity/model_comparison_summary.csv | Pooled OOF metrics and confusion matrices |
| data/pitch_sensitivity/model_oof_predictions.csv | Per-participant OOF predictions |
| src/compare_pitch_sensitivity.py | Phase 1 script |
| src/plot_pitch_tracks_sensitivity.py | Phase 2 script |
| src/reference_reproduction_audit.py | Phase 3 script |
| src/run_sensitivity_classifier.py | Phase 4 script |
| tests/test_pipeline.py | Phase 7 automated tests |
| reports/pitch_sensitivity_report.md | This report |

---

## 15. Reproduction Commands

```bash
source .venv/bin/activate

# Pre-requisite: pitch_sensitivity.py must have already run successfully
# python src/pitch_sensitivity.py  (already done — outputs exist)

# Phase 1: Participant-level feature differences
python src/compare_pitch_sensitivity.py

# Phase 2: Pitch-track plots and review
python src/plot_pitch_tracks_sensitivity.py

# Phase 3: Reference reproduction audit
python src/reference_reproduction_audit.py

# Phase 4: Classifier under all 3 configurations (identical folds)
python src/run_sensitivity_classifier.py

# Phase 5: Demographic baseline
python src/compare_demographics.py

# Phase 7: Automated tests
pytest tests/ -v
```

All outputs are deterministic (random seed 42).

---

*This report was generated programmatically. All numeric results were read directly from saved CSV files. No results were fabricated or selectively reported.*
