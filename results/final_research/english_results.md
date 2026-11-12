# Independent English and related-dataset analyses

Experiment ID: `ENGLISH-VOICE-DATASET-AUDIT-20261031`. The experiment is isolated in `results/english_research_20261031/`; local source data were read only. Raw-audio analysis covered Figshare sustained `/a/` and MDVR-KCL reading/dialogue. UCI tables and archive data were analyzed as a separate precomputed-feature investigation. No classifier was fitted.

## Acoustic findings from raw audio

### Figshare sustained `/a/`

The workbook-to-file join verified 81 unique Sample IDs (41 HC/40 PD), one waveform/person, with group labels consistent between workbook and directory. We extracted median F0, F0 IQR, voiced fraction, Praat local jitter, local shimmer and mean HNR in the longest candidate voiced segment (75–500 Hz pitch range; 10 ms step; 10% trimmed at both ends). No participant was excluded based on QC. Group difference estimates, participant bootstrap intervals and four-feature Holm correction are in the table below; age/sex-adjusted HC3 estimates are reported separately. Given 8 kHz telephone waveforms, group-age/sex imbalance risk, small n, and unverified protocol duration, any difference is dataset-specific exploratory evidence.

### MDVR-KCL reading and spontaneous dialogue

All 73 files were readable: 21 HC/16 PD reading and 21 HC/15 PD dialogue; one PD reading file is absent and one dialogue filename has a separator anomaly. Original 44.1 kHz waveforms were polyphase-resampled to 16 kHz for analysis. Candidate connected-speech measurements used 25 ms RMS frames, 10 ms hop, provisional −25 dB threshold below each file’s p95 RMS, and a 10 ms gap closure. Sensitivity at −20/−25/−30/−35 dB is reported; thresholds substantially alter candidate activity and pause estimates, and none of the masks has been human-validated. Features were active-speech fraction, pauses >=250 ms, pause fraction, active-window median F0/IQR and voiced fraction, active-window HNR, relative RMS IQR, spectral centroid and 85% rolloff. Group tests use one participant-task row per person and Holm-adjust each task’s feature family. There is no transcript or human VAD/speaker diarization. The examiner’s voice may appear in dialogue, so its F0/HNR/activity summaries are not confidently participant-specific. Read and dialogue results are not combined.

**Figure E2 reports candidate effect sizes, not validated speech biomarkers.** The analysis is not powered to distinguish subtle effects; p-values are exploratory, and wide effect intervals should guide any interpretation. Jitter/shimmer and formants were not extended to full connected-speech recordings.

### Participant-level group effects

| dataset | task | feature | n_hc | n_pd | mean_difference_hc_minus_pd | mean_diff_boot_ci95_low | mean_diff_boot_ci95_high | hedges_g_hc_minus_pd | hedges_g_boot_ci95_low | hedges_g_boot_ci95_high | mann_whitney_p | holm_adjusted_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figshare | sustained_a | median_f0_hz | 41 | 40 | -28.3781 | -50.4707 | -6.2238 | -0.5535 | -0.9981 | -0.1522 | 0.0220 | 0.0879 |
| Figshare | sustained_a | jitter_local | 41 | 40 | 0.0002 | -0.0009 | 0.0014 | 0.0759 | -0.3670 | 0.5006 | 0.6401 | 1.0000 |
| Figshare | sustained_a | shimmer_local | 41 | 40 | 0.0030 | -0.0105 | 0.0160 | 0.0986 | -0.3389 | 0.5173 | 0.8465 | 1.0000 |
| Figshare | sustained_a | hnr_mean_db | 41 | 40 | -0.2789 | -1.8399 | 1.2615 | -0.0756 | -0.5136 | 0.3708 | 0.8761 | 1.0000 |
| MDVR-KCL | read_text | active_speech_fraction | 21 | 16 | 0.0551 | 0.0081 | 0.1056 | 0.7406 | 0.1349 | 1.4559 | 0.0948 | 0.3330 |
| MDVR-KCL | read_text | pause_count_candidate | 21 | 16 | 3.3304 | -7.2092 | 14.0450 | 0.1923 | -0.4681 | 0.8509 | 0.6674 | 1.0000 |
| MDVR-KCL | read_text | pause_fraction_candidate | 21 | 16 | 0.0894 | 0.0356 | 0.1451 | 1.0629 | 0.4749 | 1.7740 | 0.0061 | 0.0387 |
| MDVR-KCL | read_text | median_f0_active_hz | 21 | 16 | 24.6873 | 4.3947 | 44.2589 | 0.8104 | 0.1379 | 1.7630 | 0.0385 | 0.1926 |
| MDVR-KCL | read_text | f0_iqr_active_hz | 21 | 16 | 8.6794 | 1.6127 | 15.4469 | 0.8111 | 0.1494 | 1.8195 | 0.0055 | 0.0387 |
| MDVR-KCL | read_text | voiced_fraction_active | 21 | 16 | -0.0679 | -0.1245 | 0.0021 | -0.7164 | -2.0614 | 0.0035 | 0.0019 | 0.0149 |
| MDVR-KCL | read_text | hnr_median_active_db | 21 | 16 | 0.4788 | -0.6877 | 1.7052 | 0.2574 | -0.4504 | 0.9201 | 0.8181 | 1.0000 |
| MDVR-KCL | read_text | relative_intensity_iqr_db | 21 | 16 | 0.9168 | -0.3435 | 2.0841 | 0.4901 | -0.1674 | 1.3597 | 0.0833 | 0.3330 |
| MDVR-KCL | read_text | spectral_centroid_active_hz | 21 | 16 | 226.6330 | 122.7851 | 334.5748 | 1.3987 | 0.7704 | 2.2706 | 0.0002 | 0.0025 |
| MDVR-KCL | read_text | spectral_rolloff_85_active_hz | 21 | 16 | 624.8810 | 333.2083 | 934.1935 | 1.3523 | 0.7852 | 2.1130 | 0.0005 | 0.0048 |
| MDVR-KCL | spontaneous_dialogue | active_speech_fraction | 21 | 15 | 0.0482 | -0.0451 | 0.1354 | 0.3603 | -0.3022 | 1.1747 | 0.2351 | 0.7122 |
| MDVR-KCL | spontaneous_dialogue | pause_count_candidate | 21 | 15 | 11.1143 | -2.7245 | 24.9057 | 0.5278 | -0.1512 | 1.3248 | 0.1314 | 0.7122 |
| MDVR-KCL | spontaneous_dialogue | pause_fraction_candidate | 21 | 15 | 0.0669 | -0.0422 | 0.1672 | 0.4330 | -0.2226 | 1.3324 | 0.1488 | 0.7122 |
| MDVR-KCL | spontaneous_dialogue | median_f0_active_hz | 21 | 15 | 23.4244 | 1.7273 | 44.3319 | 0.7471 | 0.0424 | 1.7137 | 0.1017 | 0.7122 |
| MDVR-KCL | spontaneous_dialogue | f0_iqr_active_hz | 21 | 15 | 9.5051 | 0.4170 | 17.8657 | 0.7294 | 0.0405 | 1.8852 | 0.0103 | 0.1026 |
| MDVR-KCL | spontaneous_dialogue | voiced_fraction_active | 21 | 15 | -0.0392 | -0.1420 | 0.0707 | -0.2475 | -1.1131 | 0.4455 | 0.2614 | 0.7122 |
| MDVR-KCL | spontaneous_dialogue | hnr_median_active_db | 21 | 15 | 1.7597 | -0.0852 | 3.9882 | 0.5786 | -0.0058 | 1.1417 | 0.1086 | 0.7122 |
| MDVR-KCL | spontaneous_dialogue | relative_intensity_iqr_db | 21 | 15 | 1.8082 | -0.2148 | 3.7687 | 0.5951 | -0.0676 | 1.3533 | 0.1235 | 0.7122 |
| MDVR-KCL | spontaneous_dialogue | spectral_centroid_active_hz | 21 | 15 | 178.8728 | 76.0777 | 291.4913 | 1.1014 | 0.5571 | 1.7053 | 0.0103 | 0.1026 |
| MDVR-KCL | spontaneous_dialogue | spectral_rolloff_85_active_hz | 21 | 15 | 370.0952 | 78.0810 | 686.2952 | 0.8092 | 0.2343 | 1.4510 | 0.0539 | 0.4310 |
| UCI174_feature_table | sustained_vowel_a | median_f0_hz | 8 | 24 | 38.9519 | 0.1184 | 75.3000 | 0.9775 | 0.0048 | 2.3500 | 0.0777 | 0.1407 |
| UCI174_feature_table | sustained_vowel_a | jitter_local | 8 | 24 | -0.0028 | -0.0046 | -0.0012 | -0.8006 | -1.3380 | -0.4624 | 0.0088 | 0.0352 |
| UCI174_feature_table | sustained_vowel_a | shimmer_local | 8 | 24 | -0.0142 | -0.0212 | -0.0071 | -0.9428 | -1.4185 | -0.5918 | 0.0102 | 0.0352 |
| UCI174_feature_table | sustained_vowel_a | hnr_mean_db | 8 | 24 | 3.6477 | 0.8272 | 6.5856 | 0.8906 | 0.2525 | 1.6580 | 0.0704 | 0.1407 |
| UCI301_feature_table | sustained_a | jitter_local | 20 | 20 | -0.0558 | -0.3719 | 0.2632 | -0.1041 | -0.7940 | 0.4937 | 0.8604 | 1.0000 |
| UCI301_feature_table | sustained_a | shimmer_local | 20 | 20 | -0.8706 | -3.5237 | 2.1872 | -0.1758 | -1.1261 | 0.3644 | 0.0810 | 0.4052 |
| UCI301_feature_table | sustained_a | htn | 20 | 20 | 1.8347 | -1.1635 | 4.6275 | 0.3696 | -0.2205 | 1.2173 | 0.0810 | 0.4052 |
| UCI301_feature_table | sustained_a | median_pitch_hz | 20 | 20 | 1.2917 | -27.3209 | 29.3632 | 0.0271 | -0.5997 | 0.6422 | 0.8604 | 1.0000 |
| UCI301_feature_table | sustained_a | unvoiced_fraction | 20 | 20 | 3.2577 | -0.0215 | 7.5075 | 0.5069 | 0.0115 | 0.8968 | 0.3315 | 0.9944 |
| UCI301_feature_table | sustained_o | jitter_local | 20 | 20 | 0.0141 | -0.5071 | 0.5647 | 0.0155 | -0.6449 | 0.5820 | 0.6849 | 1.0000 |
| UCI301_feature_table | sustained_o | shimmer_local | 20 | 20 | 0.2069 | -2.7121 | 3.3022 | 0.0415 | -0.6243 | 0.6291 | 0.7557 | 1.0000 |
| UCI301_feature_table | sustained_o | htn | 20 | 20 | 0.4148 | -3.2344 | 4.0236 | 0.0675 | -0.5597 | 0.7061 | 0.8392 | 1.0000 |
| UCI301_feature_table | sustained_o | median_pitch_hz | 20 | 20 | -0.1774 | -27.6646 | 28.2573 | -0.0037 | -0.6566 | 0.6050 | 0.9461 | 1.0000 |
| UCI301_feature_table | sustained_o | unvoiced_fraction | 20 | 20 | 1.5969 | 0.2785 | 3.0201 | 0.6779 | 0.1019 | 1.3702 | 0.0044 | 0.0218 |
| UCI301_feature_table | sustained_u | jitter_local | 20 | 20 | -0.0239 | -0.4223 | 0.3683 | -0.0360 | -0.6271 | 0.6008 | 0.8604 | 1.0000 |
| UCI301_feature_table | sustained_u | shimmer_local | 20 | 20 | 1.2834 | -1.9645 | 4.6624 | 0.2293 | -0.4204 | 0.7966 | 0.9031 | 1.0000 |
| UCI301_feature_table | sustained_u | htn | 20 | 20 | -0.4318 | -3.9508 | 2.9405 | -0.0733 | -0.6662 | 0.6094 | 0.8817 | 1.0000 |
| UCI301_feature_table | sustained_u | median_pitch_hz | 20 | 20 | 7.2031 | -22.9637 | 38.5640 | 0.1368 | -0.4970 | 0.7331 | 0.7557 | 1.0000 |
| UCI301_feature_table | sustained_u | unvoiced_fraction | 20 | 20 | 3.2770 | -0.9539 | 7.5199 | 0.4664 | -0.1103 | 1.0664 | 0.0783 | 0.3917 |

Task-specific Holm-corrected findings: Corrected signals: MDVR-KCL/read_text/pause_fraction_candidate (g=1.06, adjusted p=0.039), MDVR-KCL/read_text/f0_iqr_active_hz (g=0.81, adjusted p=0.039), MDVR-KCL/read_text/voiced_fraction_active (g=-0.72, adjusted p=0.015), MDVR-KCL/read_text/spectral_centroid_active_hz (g=1.40, adjusted p=0.002), MDVR-KCL/read_text/spectral_rolloff_85_active_hz (g=1.35, adjusted p=0.005), UCI174_feature_table/sustained_vowel_a/jitter_local (g=-0.80, adjusted p=0.035), UCI174_feature_table/sustained_vowel_a/shimmer_local (g=-0.94, adjusted p=0.035), UCI301_feature_table/sustained_o/unvoiced_fraction (g=0.68, adjusted p=0.022). These remain exploratory.

### Figshare age/sex-adjusted association checks

| feature | n | adjusted_pd_minus_hc | hc3_ci95_low | hc3_ci95_high | p_value_unadjusted | holm_p_within_task_4_features |
| --- | --- | --- | --- | --- | --- | --- |
| median_f0_hz | 81 | 46.4454 | 26.6057 | 66.2851 | 0.0000 | 0.0001 |
| jitter_local | 81 | -0.0006 | -0.0019 | 0.0008 | 0.3980 | 0.5141 |
| shimmer_local | 81 | -0.0089 | -0.0244 | 0.0066 | 0.2571 | 0.5141 |
| hnr_mean_db | 81 | 1.3457 | -0.4498 | 3.1412 | 0.1397 | 0.4191 |

## Supplementary precomputed-feature analyses

UCI 174’s local 195 rows resolve to 32 subject IDs (24 PD/8 HC), not the repository text count of 31/23/8. We summarized repeated rows by participant median before group tests. This retains participant as the unit, but the discrepancy needs upstream clarification before publication claims or re-use of labels. UCI 301 train data support participant-level contrasts across `/a/`, `/o/`, `/u/` (20 PD/20 HC); the independent test file contains 28 PD participants and three repetitions each for `/a/` and `/o/`, allowing only a within-PD repeat-spread summary. These Turkish-clinic/no-explicit-language, precomputed-feature results are not English replication and are not merged with raw-audio features.

UCI 189 provides no HC group and no audio. For each of 42 PD participants, we computed within-person Spearman correlations between four voice features and motor/total UPDRS over repeated visits, then summarized correlations across people. These are supplementary severity association summaries; they do not estimate diagnostic differences or raw-audio effects.

## What the English evidence does and does not show

The strongest data assets are the one-to-one Figshare sustained-vowel set and the two-task MDVR set. They support independent group-level descriptive effect estimation at the participant level. The local analysis identifies candidate group directions, but the cohorts are small, recordings are observational and protocol/channel differences are large. The provided precomputed UCI datasets add independent participant structures and task-specific features, not repeatable common measurements. We found no dataset adequate for externally validating Italian VI1 `/i/` shimmer: Figshare and UCI 174 provide `/a/`, UCI 301 provides `/a,o,u/`, and MDVR has no sustained `/i/` task.

No frame-, task-row-, or repeated recording-level inferential pseudo-replication was used. No pooled cross-dataset classifier was trained. See the inventory, effect tables, audio QC and reproducibility report for complete counts and methods.
