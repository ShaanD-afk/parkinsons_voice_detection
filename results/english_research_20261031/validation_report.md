# Final research package validation

**Status:** PASS — 28/28 checks passed.

| Check | Result | Observed | Expected |
|---|---|---|---|
| Figshare audio rows | PASS | 81 | 81 |
| Figshare participant count | PASS | 81 | 81 |
| Figshare group counts | PASS | {'HC': 41, 'PD': 40} | {'HC': 41, 'PD': 40} |
| Figshare all readable | PASS | 81 | 81 |
| Figshare workbook match | PASS | {1.0: 81} | all ids match exactly once |
| MDVR audio rows | PASS | 73 | 73 |
| MDVR participant count | PASS | 37 | 37 |
| MDVR group counts | PASS | {'HC': 21, 'PD': 16} | {'HC': 21, 'PD': 16} |
| MDVR task coverage | PASS | {'read_text': 37, 'spontaneous_dialogue': 36} | {'read_text': 37, 'spontaneous_dialogue': 36} |
| MDVR all readable | PASS | 73 | 73 |
| MDVR separator anomaly retained | PASS | 1 | >=1 |
| UCI174 local file structure | PASS | rows=195, ids=32, groups={'HC': 8, 'PD': 24} | 195 rows; 32 ids; HC8/PD24 |
| UCI174 status consistency | PASS | 0 | 0 conflicting participants |
| UCI189 local file structure | PASS | rows=5875, ids=42 | 5875 rows; 42 IDs |
| UCI189 duplicate source copies | PASS | True | identical SHA-256 |
| UCI301 train structure | PASS | rows=1040, ids=40, task rows/person=[np.int64(26)] | 1040 rows; 40 IDs; 26 task rows each |
| UCI301 test structure | PASS | rows=168, ids=28, repeats/person=[np.int64(6)] | 168 rows; 28 IDs; 6 repeats each |
| Raw audio QC row totals | PASS | 154 | 154 |
| Raw audio feature row totals | PASS | 154 | 154 |
| Raw audio participant-task uniqueness | PASS | 0 | 0 duplicates |
| Effect comparisons have independent group participants | PASS | ['dataset', 'task', 'feature', 'n_hc', 'n_pd', 'mean_hc', 'mean_pd', 'median_hc', 'median_pd', 'mean_difference_hc_minus_pd', 'mean_diff_boot_ci95_low', 'mean_diff_boot_ci95_high', 'hedges_g_hc_minus_pd', 'hedges_g_boot_ci95_low', 'hedges_g_boot_ci95_high', 'cliffs_delta_hc_minus_pd', 'mann_whitney_p', 'participant_unit_verified', 'holm_p_within_task_4_features', 'holm_p_within_task_10_features', 'holm_p_within_task_5_features', 'holm_adjusted_p'] | positive n_hc and n_pd columns |
| Effect table contains no duplicate dataset-task-feature tests | PASS | 0 | 0 duplicates |
| MDVR VAD sensitivity rows | PASS | 292 | 292 |
| MDVR VAD thresholds | PASS | [np.int64(-35), np.int64(-30), np.int64(-25), np.int64(-20)] | [-20, -25, -30, -35] |
| Protected source hashes unchanged | PASS | 440 | 440 source hashes identical |
| Required research reports exist | PASS | [] | ['italian_results.md', 'italian_manuscript_draft.md', 'english_dataset_inventory.md', 'english_results.md', 'english_manuscript_draft.md', 'cross_language_comparison.md', 'combined_research_summary.md', 'reproducibility_report.md', 'publication_readiness.md'] |
| Figures exported as PNG and PDF | PASS | 10 PNG / 10 PDF | >=10 of each, paired |
| Classifier not trained | PASS | reports state classifier disabled | no classifier training |
