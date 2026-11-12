# ItalianPVS research synthesis

**Synthesis date:** 31 October 2026. **Analysis sources:** preserved experiments `ITALIAN-MULTITASK-V1-20261030` and `ITALIAN-PHASE2-REMAINING-20261030`; this synthesis did not rerun historical feature extraction or alter their outputs.

## What we investigated

The work examined the reconciled elderly healthy-control (HC) versus Parkinson’s disease (PD) cohort, task coverage and sample-rate restrictions; five sustained vowels (VA1/VE1/VI1/VO1/VU1); second-vowel signal structure; D1/D2 `/pa/` and `/ta/` recordings; B1/B2 reading and FB1 phrase recordings; a 29-person VA1 cross-validation baseline; and sensitivity of the main vowel results to participant influence, alternative summaries and resampling.

Primary group comparisons use native 16 kHz: **17 HC and 12 PD**. The first-vowel table has 145 task-participant rows (29 people per vowel). The mixed-rate reconciled cohort has 17 HC and 19 PD, but the additional seven PD participants with 44.1 kHz recordings do not have HC counterparts at that rate.

## Measurements and principal results

The sustained-vowel pipeline measured median F0, F0 IQR/SD, voiced fraction, local jitter, local shimmer, HNR, F1–F3 tracks and spectral centroid, 85% rolloff and flatness. The prespecified primary family was median F0, local jitter, local shimmer and HNR in each of five vowels (20 tests). Vowel contexts are separate phonetic conditions; they are not repeatability trials.

**VI1 local shimmer is the only primary-family contrast surviving the original Holm correction** (HC-minus-PD mean difference 0.04439, Hedges’ g 0.889, historical participant-bootstrap 95% interval [0.01853, 0.07745], Mann–Whitney p=0.000600, Holm p=0.01199; 17/12 participants). It suggests a group-associated difference in cycle-to-cycle amplitude perturbation during the Italian `/i/` sustained task in this sample. The direction is notable because it persists when any one participant is omitted (range 0.03335–0.04739) and under median/trimmed/winsorized summaries; independent percentile and BCa participant-bootstrap intervals are positive. However, the 20%-trimmed Yuen test’s Holm-adjusted p≈0.061 does not cross .05. It is an exploratory signal, not a definitive biological or diagnostic effect. Present participant dots, HC-minus-PD estimate, bootstrap intervals, leave-one-out estimates, and both adjusted inferential results together (Figure 5); avoid a single p-value headline.

Several other unadjusted vowel-specific contrasts are suggestive, including lower HNR in VI1 (mean HC-minus-PD ≈−7.18 dB; primary-family Holm p≈0.0815), and shimmer estimates in VA1, VO1 and VU1. These do not survive the 20-test correction. Median F0 group estimates are imprecise, and the other primary contrasts are inconclusive. The full table supplies means, participant-level effect sizes, intervals and adjusted p-values.

### Prespecified 20-comparison family

| task_code | feature | n_hc | n_pd | mean_difference_hc_minus_pd | mean_diff_ci95_low | mean_diff_ci95_high | hedges_g_hc_minus_pd | hedges_g_ci95_low | hedges_g_ci95_high | mann_whitney_p_unadjusted | holm_p_primary_20_comparisons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VA1 | hnr_mean_db | 17 | 12 | -3.9420 | -7.2129 | -0.3287 | -0.7556 | -1.7276 | -0.0566 | 0.0488 | 0.6342 |
| VA1 | jitter_local | 17 | 12 | -0.0052 | -0.0120 | -0.0005 | -0.7124 | -1.2787 | -0.1262 | 0.1501 | 1.0000 |
| VA1 | median_f0_hz | 17 | 12 | -7.3224 | -32.9447 | 20.3095 | -0.1940 | -0.9353 | 0.5280 | 0.6420 | 1.0000 |
| VA1 | shimmer_local | 17 | 12 | 0.0313 | 0.0083 | 0.0556 | 0.8125 | 0.2781 | 1.5130 | 0.0201 | 0.3214 |
| VE1 | hnr_mean_db | 17 | 12 | -2.5483 | -6.7707 | 2.3073 | -0.4038 | -1.5481 | 0.3524 | 0.0729 | 0.8021 |
| VE1 | jitter_local | 17 | 12 | -0.0078 | -0.0217 | 0.0005 | -0.5025 | -1.0023 | 0.3038 | 0.2588 | 1.0000 |
| VE1 | median_f0_hz | 17 | 12 | -4.4836 | -29.1772 | 20.1461 | -0.1212 | -0.8167 | 0.6419 | 0.8768 | 1.0000 |
| VE1 | shimmer_local | 17 | 12 | 0.0233 | -0.0046 | 0.0535 | 0.5163 | -0.1346 | 1.1509 | 0.0354 | 0.5315 |
| VI1 | hnr_mean_db | 17 | 12 | -7.1803 | -11.0352 | -3.5299 | -1.2164 | -2.1896 | -0.5596 | 0.0043 | 0.0815 |
| VI1 | jitter_local | 17 | 12 | 0.0047 | 0.0004 | 0.0095 | 0.5857 | 0.0939 | 1.0328 | 0.2233 | 1.0000 |
| VI1 | median_f0_hz | 17 | 12 | -21.6715 | -69.1940 | 20.6812 | -0.3663 | -1.0672 | 0.4036 | 0.4925 | 1.0000 |
| VI1 | shimmer_local | 17 | 12 | 0.0444 | 0.0185 | 0.0775 | 0.8892 | 0.6725 | 1.4361 | 0.0006 | 0.0120 |
| VO1 | hnr_mean_db | 17 | 12 | -3.1915 | -7.1768 | 0.7776 | -0.5199 | -1.2134 | 0.1257 | 0.2588 | 1.0000 |
| VO1 | jitter_local | 17 | 12 | -0.0009 | -0.0032 | 0.0011 | -0.2878 | -0.9791 | 0.4386 | 0.3411 | 1.0000 |
| VO1 | median_f0_hz | 17 | 12 | -8.1333 | -46.7536 | 28.6999 | -0.1566 | -0.8860 | 0.6031 | 0.9119 | 1.0000 |
| VO1 | shimmer_local | 17 | 12 | 0.0330 | 0.0105 | 0.0611 | 0.7880 | 0.4339 | 1.3076 | 0.0178 | 0.3032 |
| VU1 | hnr_mean_db | 17 | 12 | -5.2926 | -9.4476 | -1.0241 | -0.7901 | -1.5605 | -0.2186 | 0.0395 | 0.5528 |
| VU1 | jitter_local | 17 | 12 | 0.0039 | 0.0004 | 0.0082 | 0.5891 | 0.0753 | 1.0458 | 0.0661 | 0.7934 |
| VU1 | median_f0_hz | 17 | 12 | -15.4966 | -60.3321 | 21.9982 | -0.2838 | -0.9729 | 0.5183 | 0.8076 | 1.0000 |
| VU1 | shimmer_local | 17 | 12 | 0.0322 | 0.0026 | 0.0653 | 0.6395 | 0.0326 | 1.2585 | 0.0056 | 0.1017 |

### Evidence classification

**Stronger exploratory evidence:** VI1 shimmer has a large observed standardized contrast and stable positive direction across individual deletion and alternative summaries. Its adjusted inferential status is analysis-dependent.

**Suggestive:** selected HNR/shimmer vowel contrasts and the pattern that voice measures depend on vowel context. These estimates have wide uncertainty, multiple comparisons and small groups; they do not establish disease-specific effects.

**Inconclusive:** F0 contrasts and most primary tests; external validation of any Italian association; resampled group contrasts because extra observations are PD-only.

**Requires human validation:** formant overlays and the open 159-recording review queue; D1/D2 onset segmentation; connected-speech VAD masks.

**Do not include as main inferential results:** automated D1/D2 event counts/intervals, connected-speech pause or speech-activity estimates, second-vowel paired differences as repeatability, frame-level observations as independent n, or classifier metrics as clinical evidence.

## Tasks and scientific contribution

VA1/VE1/VI1/VO1/VU1 are the only tasks with participant-level acoustic group analysis and matched per-vowel coverage. The VA1 cross-validation reference used 29 participants and reported BA 0.900, AUC 0.917, sensitivity 0.917 and specificity 0.882. A later uncertainty analysis on the fixed saved OOF predictions gives BA 95% interval [0.775,1.000] and AUC [0.789,1.000]; it conditions on fitted fold predictions and omits model refitting/fold uncertainty. This is internal exploratory cross-validation, not external validation or clinical diagnostic performance.

Second vowels have markedly shorter duration distributions than first variants, but blank protocol descriptions prevent a repetition or reliability interpretation. Most D1/D2 plots show repeated CV-like bursts, not enough to validate syllable counts, voice-onset time or per-participant timing. B1/B2/FB1 generic connected-speech features are threshold-dependent candidates, not validated speech-activity measures; FB1 materials are unavailable, and no transcript-based rate/alignment was attempted.

The contribution is a transparent, participant-level, multi-task characterization that establishes coverage and where the data do and do not support inference; it identifies an exploratory VI1 shimmer association and quantifies its sensitivity instead of treating model CV as the only result.

## What the Results Actually Tell Us

Cycle amplitude perturbation in Italian `/i/` deserves a targeted independent replication. It is not yet a general PD voice marker: the effect may depend on vowel, sample composition, age/sex, recording device, or analysis choice. HNR in `/i/` is also worth checking, but it did not pass the prespecified family correction. More vowel contexts did not produce more validated evidence. The second variants currently add protocol uncertainty rather than test-retest information.

Connected speech may eventually add timing and prosody information, but current VAD settings materially change pauses and active proportions; there is no validated segmentation or speaker attribution. Until reviewed, it adds diagnostic plots and candidate variables, not a defensible inferential claim. Avoid developing classifiers or large feature sets on these cohorts. Deprioritize text-specific analyses until verified materials and alignments are available.

## Recommended Next Research Experiments

1. **Independent sustained `/i/` shimmer replication** (highest scientific value; moderate feasibility): collect a larger, demographically matched cohort using controlled microphone distance, fixed sample rate, repeat tokens and blinded quality review; prespecify one primary shimmer definition and interval. Include `/a/` as a planned contrast, not a fishing family.
2. **Italian audio adjudication and task validity** (high feasibility): complete the open review queue; annotate D1/D2 onsets and connected-speech masks; estimate inter-rater agreement and detector error before any participant summaries.
3. **Protocol recovery for vowel variants and reading** (low cost): obtain the collection sheet, exact texts and sample-rate/hardware details; only then decide which paired or text-dependent analyses are meaningful.
4. **External data access for mPower or an explicitly English task dataset** (moderate effort, access-dependent): run a preregistered protocol-matched sustained-phonation study only after access, language/task, label and demographic eligibility are verified.
5. Reconsider model evaluation only after validated measures and adequate independent participant counts exist; participant-level nested/external validation should precede any deep model.

## Limitations

n=17/12 at native 16 kHz; no power-based confirmatory claim is established. Age/sex and device/session differences may confound group contrasts. All additional 44.1 kHz task recordings are PD. Repeated extraction verifies software determinism, not biological reliability. Formant broad ranges do not constitute expert validation. 159 manual reviews are open. D1/D2 protocol/count, second vowel definitions, FB1 text, FB2 absence, PR1/PR11 and some roster identities remain unresolved. The resampling sensitivity does not remove sample-rate/group confounding. No clinical claims are supported.

## Data and figure notes

The participant-flow and task-coverage figures use the frozen manifests and reconciliation. Acoustic distributions and per-vowel effects use the saved native-16-kHz table and historical comparison table. The VI1 robustness figure joins the new sensitivity audit. The Italian connected-speech figure uses unadjudicated −30/−35/−40 dB RMS-VAD summaries; the MDVR figure uses −20/−25/−30/−35 dB sensitivity summaries with a provisional −25 dB feature mask. Both remain explicitly exploratory.

References and protocol sources are listed in the [manuscript draft](italian_manuscript_draft.md). The original published dataset/protocol sources include Dimauro et al. (2016, 2017) and the 2022 task paper; see the cited DOI links in the manuscript.
