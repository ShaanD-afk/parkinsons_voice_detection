# Participant-level statistical comparison

Each group contrast uses one participant-level feature vector per dataset/task. Mean differences and Hedges g are oriented HC minus PD, with participant-bootstrap percentile 95% intervals (5,000 resamples). Two-sided Mann–Whitney tests are Holm-adjusted over the five common features within each dataset. No recording frames or repeated people were treated as independent. For UCI #174, repeated rows were first collapsed to participant medians. MDVR and UCI results are separately reported and do not imply feature harmonisation.

## Sustained `/a/` shared features

| dataset | feature | n_hc | n_pd | mean_difference_hc_minus_pd | hedges_g_hc_minus_pd | hedges_g_boot_ci95_low | hedges_g_boot_ci95_high | mann_whitney_p | holm_p_within_dataset_task |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figshare_sustained_a | median_f0_hz | 41 | 40 | -28.3781 | -0.5535 | -1.0138 | -0.1217 | 0.0220 | 0.0879 |
| Figshare_sustained_a | f0_iqr_hz | 41 | 40 | -0.8287 | -0.3338 | -1.1782 | 0.0956 | 0.0002 | 0.0008 |
| Figshare_sustained_a | jitter_local | 41 | 40 | 0.0002 | 0.0759 | -0.3703 | 0.4975 | 0.6401 | 1.0000 |
| Figshare_sustained_a | shimmer_local | 41 | 40 | 0.0030 | 0.0986 | -0.3415 | 0.5201 | 0.8465 | 1.0000 |
| Figshare_sustained_a | hnr_mean_db | 41 | 40 | -0.2789 | -0.0756 | -0.4934 | 0.3476 | 0.8761 | 1.0000 |
| ItalianPVS_VA1 | median_f0_hz | 17 | 12 | -7.3224 | -0.1940 | -0.9279 | 0.5432 | 0.6420 | 1.0000 |
| ItalianPVS_VA1 | f0_iqr_hz | 17 | 12 | 5.5123 | 0.2466 | -1.0774 | 0.6669 | 0.5208 | 1.0000 |
| ItalianPVS_VA1 | jitter_local | 17 | 12 | -0.0052 | -0.7124 | -1.3116 | -0.1434 | 0.1501 | 0.4503 |
| ItalianPVS_VA1 | shimmer_local | 17 | 12 | 0.0313 | 0.8125 | 0.2810 | 1.5072 | 0.0201 | 0.1004 |
| ItalianPVS_VA1 | hnr_mean_db | 17 | 12 | -3.9420 | -0.7556 | -1.7556 | -0.0763 | 0.0488 | 0.1951 |

No common feature demonstrated a stable, statistically supported disease effect in both raw-audio cohorts. Median F0 had the same unadjusted direction (HC–PD < 0), but its size differed and Italian uncertainty was broad. F0 IQR and jitter had opposite directions. Shimmer had the same positive HC–PD direction but a near-zero Figshare estimate and wide intervals. HNR shared a negative direction but the Figshare estimate was close to zero. None of the five Italian effects survived its within-dataset Holm family; Figshare's F0-IQR result did, but its definition can be sensitive to short-token tracking and does not replicate an Italian signal.

## Age/sex adjustment

An HC3 linear model used 110 participants and included diagnosis, dataset, diagnosis×dataset, age (linear, centered) and sex. This is an exploratory covariate sensitivity check. Dataset is entangled with rate, device, cohort, and possibly language; covariate adjustment cannot disentangle those properties. Interaction p-values did not survive correction across features. See `tables/age_sex_adjusted_domain_models.csv` for all contrasts and intervals.

## Other dataset-specific effects

MDVR reading candidate measures and UCI #174/#301 precomputed-feature contrasts are reported in `tables/participant_level_group_effects.csv`, with Holm correction by dataset/task family. MDVR candidate VAD/prosody signals need human segmentation and speaker review. UCI #174 participant IDs conflict with its dataset-page count. These are not evidence of raw-acoustic cross-language replication.

## Visualisations

- `figures/fig1_common_vowel_distributions`: participant distributions by cohort/group
- `figures/fig2_common_vowel_effect_forest`: shared-feature effect sizes and intervals
- `figures/fig4_common_vowel_pca`: descriptive PCA, not evidence of biological separation
