# Italian multi-task acoustic analysis — research report

**Experiment:** ITALIAN-MULTITASK-V1-20261030. **Scope:** first vowel variants only for primary feature/statistical analyses; other selected tasks received audio QC and a segmentation pilot. No classifier was trained.

## Executive summary

Audio QC covered 561 selected recordings across the reconciled HC-PD cohort and descriptive YHC. Native-16-kHz vowel extraction attempted 145 recordings (85 HC and 60 PD across five vowels). 0 recordings had top-level feature extraction failures; 145 completed.
Automated output validation: 17/17 checks passed and 0 failed. See `validation_checks.csv`.
Results are exploratory, small-sample and non-clinical. Statistical summaries are participant-level; frames are not treated as independent observations.

## Data actually analyzed

- Primary group: reconciled elderly HC vs PD, first vowel codes VA1/VE1/VI1/VO1/VU1, native 16 kHz only.
- Descriptive QC: reconciled HC/PD plus YHC separately, for VA1/VA2, VE1/VE2, VI1/VI2, VO1/VO2, VU1/VU2, D1/D2, B1/B2, and FB1.
- Sample-rate sensitivity: eligible 44.1 kHz first-vowel recordings analyzed at native rate and after polyphase resampling to 16 kHz; separate combined-rate sensitivity table.
- Second vowel variants, PR1, PR11, transcript-dependent reading features, and protocol-dependent syllable measures were deferred.

### Extraction coverage

| task_code | group | rows | valid |
| --- | --- | --- | --- |
| VA1 | ELDERLY_HC | 17 | 17 |
| VA1 | PD | 12 | 12 |
| VE1 | ELDERLY_HC | 17 | 17 |
| VE1 | PD | 12 | 12 |
| VI1 | ELDERLY_HC | 17 | 17 |
| VI1 | PD | 12 | 12 |
| VO1 | ELDERLY_HC | 17 | 17 |
| VO1 | PD | 12 | 12 |
| VU1 | ELDERLY_HC | 17 | 17 |
| VU1 | PD | 12 | 12 |

### Audio QC

QC status: {'ok': 561}. Flag-string frequency (including unflagged): {'': 542, 'possible_clipping': 17, 'high_silence_fraction_for_vowel_task': 2}. Review queue rows: 19. Flags do not automatically exclude recordings.
The reported RMS dynamic-range proxy is not SNR. No calibrated noise-only segment was available. Details: `qc/per_recording_qc.csv`, `qc/task_qc_summary.csv`, `qc/group_qc_summary.csv`, and `qc/review_queue.csv`.

## Segmentation and signal inspection pilot

| task_family | task_code | pilot_recordings | mean_duration_s | median_active_speech_s | median_vowel_pitch_run_s | median_rms_vad_bouts | segmentation_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reading | B1 | 4 | 83.070859375 | 43.145 | 0.79 | 69.5 | 0 |
| reading | B2 | 4 | 65.88354173752835 | 41.345 | 0.8600000000000001 | 66.5 | 0 |
| reading | FB1 | 3 | 28.614095804988665 | 21.13 | 1.01 | 35.0 | 0 |
| sustained_vowel_candidate | VA1 | 3 | 11.454187499999998 | 9.290000000000001 | 7.8 | 1.0 | 0 |
| sustained_vowel_candidate | VA2 | 3 | 8.928262707860922 | 5.67 | 3.62 | 1.0 | 0 |
| sustained_vowel_candidate | VE1 | 3 | 14.425708333333333 | 6.57 | 6.43 | 2.0 | 0 |
| sustained_vowel_candidate | VE2 | 3 | 8.81 | 5.86 | 5.84 | 1.0 | 0 |
| sustained_vowel_candidate | VI1 | 3 | 11.451846041194257 | 10.18 | 8.77 | 1.0 | 0 |
| sustained_vowel_candidate | VI2 | 3 | 6.1542708333333325 | 5.51 | 5.5 | 1.0 | 0 |
| sustained_vowel_candidate | VO1 | 3 | 14.7104375 | 16.830000000000002 | 10.55 | 1.0 | 0 |
| sustained_vowel_candidate | VO2 | 3 | 8.14989947089947 | 6.11 | 6.09 | 1.0 | 0 |
| sustained_vowel_candidate | VU1 | 3 | 20.873527966742255 | 17.6 | 17.52 | 1.0 | 0 |
| sustained_vowel_candidate | VU2 | 3 | 8.594208333333333 | 5.86 | 5.84 | 1.0 | 0 |
| syllable | D1 | 3 | 6.200666666666667 | 3.11 | 0.24 | 11.0 | 0 |
| syllable | D2 | 3 | 6.255166666666667 | 1.7 | 0.19 | 7.0 | 0 |

The deterministic blinded sample contained 47 recordings; 0 crossed provisional candidate segmentation flags. Visual review found long voiced intervals in most sampled first-vowel recordings, with some onset decay, interruptions, and multiple voiced segments. All sampled D1/D2 examples showed repeated CV-like bursts, but the exact repetition instruction remains unknown. Reading RMS-VAD masks were fragmented into many short bouts and need manual threshold review. These pilot observations do not establish structure for every recording. See `segmentation/segmentation_diagnostic_report.md` and contact sheets.
D1/D2 production was not classified as repeated versus isolated solely from the codebook. Reading features are general RMS-VAD timing candidates only; no transcripts, speaking rates, phonetic alignment, or text-dependent measures were produced.

## Primary acoustic group comparisons

Prespecified primary features were median F0, local jitter, local shimmer, and mean HNR. Other extracted features are exploratory. The table reports HC-minus-PD mean-difference bootstrap CIs, Hedges' g and Cliff's delta with participant bootstrap CIs, and Mann-Whitney p-values. Holm corrections are shown within each feature across the five vowel contexts and across the complete 20-comparison primary feature family.

| task_code | feature | n_hc | n_pd | mean_difference_hc_minus_pd | mean_diff_ci95_low | mean_diff_ci95_high | hedges_g_hc_minus_pd | hedges_g_ci95_low | hedges_g_ci95_high | cliffs_delta_hc_minus_pd | cliffs_delta_ci95_low | cliffs_delta_ci95_high | mann_whitney_p_unadjusted | holm_p_within_feature_across_vowels | holm_p_primary_20_comparisons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VA1 | hnr_mean_db | 17 | 12 | -3.9420496880659215 | -7.212949130974315 | -0.3286816424377842 | -0.7556105356431689 | -1.727566909848425 | -0.05661951286955533 | -0.4411764705882353 | -0.803921568627451 | -0.0196078431372549 | 0.04878188267844301 | 0.15795602982413912 | 0.6341644748197591 |
| VE1 | hnr_mean_db | 17 | 12 | -2.5482626278555713 | -6.770743468999981 | 2.307343980499074 | -0.40383728751598896 | -1.5480500604365515 | 0.3524463919299966 | -0.4019607843137255 | -0.8235294117647058 | 0.049019607843137254 | 0.07291375835251934 | 0.15795602982413912 | 0.8020513418777128 |
| VI1 | hnr_mean_db | 17 | 12 | -7.180288650927164 | -11.035187770905168 | -3.5299109624538216 | -1.2163680364936964 | -2.18962628474348 | -0.5595992746487507 | -0.6372549019607843 | -0.9019607843137255 | -0.28431372549019607 | 0.004288692165622441 | 0.02144346082811221 | 0.08148515114682639 |
| VO1 | hnr_mean_db | 17 | 12 | -3.1915151135776085 | -7.176793197141213 | 0.7775657652137703 | -0.5198542500764002 | -1.2133991570976679 | 0.12566232966122437 | -0.2549019607843137 | -0.6372549019607843 | 0.17647058823529413 | 0.2588307748056683 | 0.2588307748056683 | 1.0 |
| VU1 | hnr_mean_db | 17 | 12 | -5.292599345533773 | -9.447571738732341 | -1.0241346165591239 | -0.790142778939529 | -1.5605249794826035 | -0.21855724920327388 | -0.4607843137254902 | -0.7941176470588235 | -0.029411764705882353 | 0.03948900745603478 | 0.15795602982413912 | 0.5528461043844869 |
| VA1 | jitter_local | 17 | 12 | -0.005162013171354163 | -0.011993720205598765 | -0.00045120982173399763 | -0.7124203555446317 | -1.2787359295426006 | -0.12615661786763271 | -0.32352941176470584 | -0.7352941176470589 | 0.09803921568627451 | 0.15011533973795838 | 0.6004613589518335 | 1.0 |
| VE1 | jitter_local | 17 | 12 | -0.00778917816011675 | -0.021667731592522357 | 0.0005242634252505486 | -0.5025129090916148 | -1.0022831968745094 | 0.30378165214333647 | -0.2549019607843137 | -0.6372549019607843 | 0.17647058823529413 | 0.2588307748056683 | 0.6699906491778462 | 1.0 |
| VI1 | jitter_local | 17 | 12 | 0.004715712558425359 | 0.0003747019826109998 | 0.009498462320350258 | 0.5857118104531529 | 0.0939042034480996 | 1.032759138887255 | 0.27450980392156854 | -0.14730392156862746 | 0.6666666666666666 | 0.2233302163926154 | 0.6699906491778462 | 1.0 |
| VO1 | jitter_local | 17 | 12 | -0.0008968640018637737 | -0.0031588480572441132 | 0.0011023290164039348 | -0.28778418721723387 | -0.979051823323104 | 0.4385872134104767 | -0.21568627450980393 | -0.6078431372549019 | 0.21568627450980393 | 0.3410787237087195 | 0.6699906491778462 | 1.0 |
| VU1 | jitter_local | 17 | 12 | 0.003929171748453586 | 0.00035388633047367954 | 0.008203773971239184 | 0.5891223324575063 | 0.07533121717097918 | 1.045837744280806 | 0.41176470588235303 | 0.0 | 0.7647058823529411 | 0.06611386515052313 | 0.33056932575261566 | 0.7933663818062775 |
| VA1 | median_f0_hz | 17 | 12 | -7.3223589667027795 | -32.9447432612353 | 20.3094988826028 | -0.19397320274532434 | -0.9353149249584533 | 0.5279844099478214 | -0.10784313725490191 | -0.5098039215686274 | 0.30392156862745096 | 0.6419688279727049 | 1.0 | 1.0 |
| VE1 | median_f0_hz | 17 | 12 | -4.483574934342954 | -29.177171290783185 | 20.146098443471647 | -0.12124474394099635 | -0.8167238829306463 | 0.641915867101651 | -0.039215686274509776 | -0.47058823529411764 | 0.39215686274509803 | 0.8768350308234345 | 1.0 | 1.0 |
| VI1 | median_f0_hz | 17 | 12 | -21.671481109672186 | -69.19402805821446 | 20.68120378212849 | -0.3662763818339034 | -1.0672382629460178 | 0.40364780735089606 | -0.1568627450980392 | -0.5686274509803921 | 0.29411764705882354 | 0.49249145763962077 | 1.0 | 1.0 |
| VO1 | median_f0_hz | 17 | 12 | -8.13329230017348 | -46.75359379926836 | 28.699885269887496 | -0.15660512169111143 | -0.8860104419240894 | 0.6030879208069884 | -0.02941176470588236 | -0.47058823529411764 | 0.4019607843137255 | 0.9118527704321069 | 1.0 | 1.0 |
| VU1 | median_f0_hz | 17 | 12 | -15.496619325347268 | -60.33210575796559 | 21.99815022187431 | -0.28378468863665823 | -0.9729002757686942 | 0.5182605790520246 | -0.05882352941176472 | -0.47058823529411764 | 0.35294117647058826 | 0.8075839314378953 | 1.0 | 1.0 |
| VA1 | shimmer_local | 17 | 12 | 0.031283786015975486 | 0.008333977593376325 | 0.05556561678921053 | 0.812541101496086 | 0.2780544561759614 | 1.5129972370942752 | 0.5196078431372548 | 0.14705882352941177 | 0.8529411764705882 | 0.020085916545488653 | 0.05350589313627957 | 0.32137466472781845 |
| VE1 | shimmer_local | 17 | 12 | 0.023317913141458292 | -0.004642948287286276 | 0.053472684632841085 | 0.5163128484025545 | -0.13459325942540595 | 1.1508922405298516 | 0.47058823529411775 | 0.0392156862745098 | 0.8237745098039202 | 0.03543646514850231 | 0.05350589313627957 | 0.5315469772275346 |
| VI1 | shimmer_local | 17 | 12 | 0.04438941439941426 | 0.01852534710530665 | 0.07745115264834075 | 0.8892474497835906 | 0.6725272595827528 | 1.4360902953047585 | 0.7647058823529411 | 0.49019607843137253 | 0.9607843137254902 | 0.0005996831225510919 | 0.0029984156127554596 | 0.011993662451021838 |
| VO1 | shimmer_local | 17 | 12 | 0.03304036617745526 | 0.01047061699415249 | 0.06109891161707419 | 0.7879664041095256 | 0.43391453501596006 | 1.3075768153326381 | 0.5294117647058822 | 0.13725490196078433 | 0.8333333333333334 | 0.01783529771209319 | 0.05350589313627957 | 0.30320006110558423 |
| VU1 | shimmer_local | 17 | 12 | 0.032153940519956375 | 0.002646439882780263 | 0.0653105704008571 | 0.6394765228730488 | 0.0326416251946687 | 1.258514595320713 | 0.6176470588235294 | 0.21568627450980393 | 0.9411764705882353 | 0.005647998455206972 | 0.02259199382082789 | 0.1016639721937255 |

Do not interpret effect direction as disease-specific. Small n yields wide uncertainty; formant validity flags and missingness are reported separately. All feature summaries, effect tables, confidence intervals and outlier review records are in `statistics/`.

## Paired vowel comparisons

Within-participant paired contrasts preserve vowel identity and use complete pairs. First-vowel comparisons are context contrasts, not repeatability estimates. Group-specific pairwise differences and Friedman omnibus summaries are under `paired_comparisons/`; complete-participant counts are explicit. Second vowel variants were not used as test-retest trials.

## Sampling-rate sensitivity

Native 44.1 kHz first-vowel recordings: 35 recordings compared before/after resampling. Resampling feature changes are paired by source recording. The combined table contains 180 recording-task observations and keeps `sample_rate_original_hz`. Across the 20 primary-feature/vowel HC-PD contrasts, native-16-kHz and combined-rate mean-difference directions agreed in all cases; mean absolute Hedges-g change was 0.116. See `sampling_rate_sensitivity/native_vs_combined_group_effects.csv`. This is a sample-size sensitivity result, not evidence that rate confounding is absent.
Native 16 kHz remains the primary group comparison because all elderly HC are 16 kHz and seven additional PD participants have only 44.1 kHz recordings for these tasks. The combined sensitivity does not remove acquisition confounding. See `sampling_rate_sensitivity/` for paired changes, combined-rate comparisons, and original-rate strata.

## Reliability and reproducibility

A complete second extraction was performed for the native-16-kHz primary set. Exact numeric agreement for repeatable deterministic features is summarized in `reliability/repeat_extraction_summary.csv`. Pitch floors 60/75/90 Hz were run on the same first-vowel native-16-kHz files; results are separate from the primary table. This is extraction repeatability and parameter sensitivity, not biological test-retest reliability.
VA1 baseline and historical outputs were hash-protected before and after the run. See `reliability/baseline_integrity.json` and validation checks.

## Failures, caveats, and deferred work

Top-level feature extraction failures: 0. Feature-level undefined measurements and plausibility flags are retained in `features/feature_extraction_failures.csv`; broad physical-range review flags are in `features/feature_value_validation_flags.csv` (n=0). No failed value was replaced with zero. CPPS was deferred because no validated implementation was available in the installed environment; spectral tilt was deferred pending a fixed definition. CPP/CPPS, voice quality in connected speech, and text-dependent measures were not imputed from unrelated features.
The stable-vowel algorithm uses the longest contiguous pitch-voiced run and trims 10% at its ends; this is a candidate segmentation rule and may fail for irregular or breathy phonation. Formants use a fixed 5 kHz Burg maximum-formant setting and broad range flags; flagged tracks require manual review. The sample is not powered for confirmatory inference, and original sample rate is group-associated.

## Validation

| check | status | observed | expected |
| --- | --- | --- | --- |
| QC rows match eligible manifest | PASS | 561 | 561 |
| No duplicate QC recordings | PASS | 0 | 0 |
| QC task codes match selected tasks | PASS | ['B1', 'B2', 'D1', 'D2', 'FB1', 'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2'] | ['VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2', 'D1', 'D2', 'B1', 'B2', 'FB1'] |
| Native vowel feature rows match eligible manifest | PASS | 145 | 145 |
| No duplicate participant-task features | PASS | 0 | 0 |
| Native feature cohort groups valid | PASS | ['ELDERLY_HC', 'PD'] | ['ELDERLY_HC', 'PD'] |
| Native sample-rate metadata is 16 kHz | PASS | [np.int64(16000)] | [16000] |
| Feature group assignments match reconciled manifest | PASS | 0 | 0 |
| Per-recording sample-rate metadata matches reconciled manifest | PASS | 0 | 0 |
| Resampled rows retain original and analysis rates | PASS | {'rows': 35, 'original_rates': [44100], 'analysis_rates': [16000]} | {'original_rates': [44100], 'analysis_rates': [16000]} |
| Feature rows account for failures | PASS | 145 | 145 |
| Repeat extraction row keys match | PASS | 145 | 145 |
| Repeat extraction deterministic for core features | PASS | exact numeric equality | exact numeric equality |
| No missing feature values replaced with zero | PASS | NaNs retained and failure table written | explicit missing values |
| Spectral flatness within [0,1] | PASS | 0 | 0 |
| Voiced frame fractions within [0,1] | PASS | 0 | 0 |
| Protected VA1/reference artifacts unchanged | PASS | [] | no hash changes |

## Recommended next work package

Review the QC/feature failure queues and the blinded segmentation contact sheets; manually adjudicate flagged recordings without changing cohort membership. Confirm D1/D2 production format and reading materials if documentation can be obtained. Then refine segmentation/feature validity, retain the native-16-kHz primary comparisons and resampled sensitivity separately, and extend only supported timing measures to D1/D2 and reading. Do not begin classification until acoustic measurement reliability and uncertainty have been reviewed.
