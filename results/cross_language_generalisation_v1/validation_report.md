# Cross-language generalisation validation

**Status:** PASS — 30/30 checks passed.

| Check | Result | Observed | Expected |
|---|---|---|---|
| Common raw audio feature rows | PASS | 110 | 110 |
| Italian VA1 participants and groups | PASS | {'HC': 17, 'PD': 12} | 29 unique; HC 17 / PD 12 |
| Figshare participants and groups | PASS | {'HC': 41, 'PD': 40} | 81 unique; HC 41 / PD 40 |
| One raw feature row per participant/task | PASS | 0 | zero duplicates |
| No raw acoustic extraction failures | PASS | 0 | 0 |
| Figshare workbook labels match folders | PASS | -162 | all labels agree |
| Native sample rates retained | PASS | Italian=[np.int64(16000)], Figshare=[np.int64(8000)] | Italian 16 kHz; Figshare 8 kHz |
| Italian 8 kHz analysis copies | PASS | rows=29, rates=[np.int64(8000)], IDs=29 | 29 separate derived 8 kHz copies |
| Pitch-floor extraction sensitivity settings | PASS | [np.float64(60.0), np.float64(75.0), np.float64(90.0)] | [60, 75, 90] |
| No automatic outlier exclusions | PASS | all outlier flags retained, no exclusions | no automatic exclusions |
| Within-dataset model count | PASS | 36 | 36 |
| External and rate-sensitivity model count | PASS | 24 | 24 |
| External classifier targets fixed | PASS | {False: 24} | False |
| External participant-key overlap | PASS | 0 | 0 |
| Nested CV participant split disjointness | PASS | False | zero participant overlap each fold |
| Nested CV test folds cover each participant once | PASS | True | each participant exactly once per model/feature set |
| Metrics independently recomputed from saved predictions | PASS | 60/60 match | all confusion counts and six metrics match |
| External source/test participant manifests disjoint | PASS | False | zero overlap by dataset-scoped participant key |
| Dataset identity matching and repetitions | PASS | 90 rows, [np.int64(58)] each; exact diagnosis counts retained | 30 draws × 3 models; n=58 matched per draw |
| Dataset identity CV participant split disjointness | PASS | True | zero train/test participant overlap |
| Dataset identity CV covers each matched participant once per draw | PASS | True | 58 unique test participants per model/draw |
| Dataset identity feature ablation coverage | PASS | 330 repeated scores, 11 feature sets, 19140 OOF predictions | 30 draws × 11 pre-specified feature sets × 58 people |
| Dataset identity ablation split disjointness | PASS | True | zero train/test overlap |
| Dataset identity ablation covers matched participants | PASS | True | 58 unique test participants per draw/feature set |
| Invalid raw-audio task transfers documented | PASS | 4 | 4 |
| Age/sex adjusted domain model cohort | PASS | [np.int64(110)] | 110 |
| Pre-existing source hashes unchanged | PASS | 1429 | 1429 hashes identical |
| Integrity manifest reports no modified source | PASS | [] | [] |
| Figure export pairs | PASS | 6 PNG / 6 PDF | 6 paired figures |
| Required experiment documents exist | PASS | [] | ['README.md', 'dataset_compatibility.md', 'feature_harmonisation.md', 'statistical_comparison.md', 'classifier_results.md', 'generalisation_analysis.md', 'limitations_and_failed_experiments.md', 'experiment_decisions.md'] |
