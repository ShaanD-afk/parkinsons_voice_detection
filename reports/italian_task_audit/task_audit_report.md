# ItalianPVS task inventory and participant coverage audit

**Status: VALIDATION PASSED.** This is a manifest and workbook coverage audit; it does not extract features, test group differences, or train models.

## Inputs and method

Primary task field: `possible_task_code` from `data/italian_manifest.csv`. The full and reconciled manifests were compared by relative path. Participant matrices use a group-qualified participant key to prevent same-named people in different groups from being merged. `FILE CODES.xlsx` is read separately for all three groups; roster cross-checks are descriptive and do not recalculate reconciliation decisions. Audio files are checked against the supplied ItalianPVS root.

## Population totals

| population | group | recordings | unique_participants | sample_rates |
| --- | --- | --- | --- | --- |
| full_dataset | YOUNG_HC | 45 | 15 | 16000 Hz: 45 |
| full_dataset | ELDERLY_HC | 349 | 22 | 16000 Hz: 349 |
| full_dataset | PD | 437 | 28 | 16000 Hz: 277;44100 Hz: 160 |
| reconciled_hc_pd | YOUNG_HC | 0 | 0 |  |
| reconciled_hc_pd | ELDERLY_HC | 272 | 17 | 16000 Hz: 272 |
| reconciled_hc_pd | PD | 295 | 19 | 16000 Hz: 183;44100 Hz: 112 |
| reconciled_hc_pd_16khz | YOUNG_HC | 0 | 0 |  |
| reconciled_hc_pd_16khz | ELDERLY_HC | 272 | 17 | 16000 Hz: 272 |
| reconciled_hc_pd_16khz | PD | 183 | 12 | 16000 Hz: 183 |

## Observed task codes

| task_code | recordings | participants |
| --- | --- | --- |
| B1 | 64 | 64 |
| B2 | 60 | 60 |
| D1 | 50 | 50 |
| D2 | 50 | 50 |
| FB1 | 47 | 47 |
| PR1 | 64 | 64 |
| PR11 | 1 | 1 |
| VA1 | 50 | 50 |
| VA2 | 49 | 49 |
| VE1 | 50 | 50 |
| VE2 | 49 | 49 |
| VI1 | 50 | 50 |
| VI2 | 49 | 49 |
| VO1 | 50 | 50 |
| VO2 | 49 | 49 |
| VU1 | 50 | 50 |
| VU2 | 49 | 49 |

Observed codes: B1, B2, D1, D2, FB1, PR1, PR11, VA1, VA2, VE1, VE2, VI1, VI2, VO1, VO2, VU1, VU2. Full data contain 831 manifest rows, 65 group-qualified participants, and 831 audio files.

## Task-code documentation findings

- `FB2` is documented in all three codebooks as reading phonemically balanced words, but is absent from the manifest and filename-derived task-code inventory. This audit cannot determine whether the task was not recorded, omitted from the supplied dataset, or represented under another code.
- Observed but undocumented codes: PR1, PR11. `PR1` and `PR11` match their filename prefixes; the codebooks do not define either. `PR11` occurs 1 time. No meaning is assigned.
- `VA1`, `VE1`, `VI1`, `VO1`, and `VU1` codebook descriptions say `2 phonation of the vocal ...`; the literal wording is retained in the codebook output and not interpreted.
- Documentation codes absent from observed files: FB2.

## Files, names, and coverage gaps

- Manifest paths missing on disk: 0; audio files absent from manifest: 0.
- Exact duplicate manifest paths: 0; filename/task-code prefix mismatches: 0.
- Workbook roster names not found by normalized name key in the corresponding group folder: 6. See `workbook_roster_audio_crosscheck.csv`; this conservative comparison may flag spelling/alias differences and does not override reconciliation decisions.
  - Names: ELDERLY_HC: Giuseppe L; ELDERLY_HC: Antonella P; ELDERLY_HC: Antonio P; ELDERLY_HC: Luigia S; ELDERLY_HC: Laforgia V; PD: Mario M.
- Existing participant folders not found in the corresponding workbook roster: 17-28/Mario B; LISCO G; PORCELLI A; SUMMO L; VITO L. The known `17-28/Mario B` exclusion does not match the workbook roster entry `Mario M`; preserve the existing exclusion pending identity evidence.
- Binary participant/task coverage gaps across the three defined populations: 357. These are absences from the manifest only. Since the supplied materials do not establish an expected task schedule per participant, they cannot be labeled missing required recordings.
- Existing reconciliation cross-check statuses: {'matched': 53, 'ambiguous_workbook_match': 7, 'not_found_in_roster': 5}. See `participant_identity_reconciliation.csv` for participant-level detail; no cohort decision was changed.
- Identity key uses group plus the original participant folder. Similar terminal names in different groups or numbered PD folders are not automatically merged.

## Pairwise coverage

`pairwise_task_overlap.csv` reports participant counts for each task pair in each population and group, including common participants and participants exclusive to either task. The three `participant_task_presence_*.csv` files are binary matrices; recording counts remain available in `task_recording_participant_summary.csv`.

## Automated validation

| check | status | observed | expected | detail |
| --- | --- | --- | --- | --- |
| Full manifest recording total | PASS | 831 | 831 |  |
| Full manifest unique participants | PASS | 65 | 65 |  |
| Participant identity keys unique by group and folder | PASS | 0 | 0 |  |
| Participant folder IDs are not reused across groups | PASS | 0 | 0 | Repeated terminal names are separately reported and are not treated as duplicate IDs. |
| Full manifest group recording totals | PASS | {'ELDERLY_HC': 349, 'PD': 437, 'YOUNG_HC': 45} | {'PD': 437, 'ELDERLY_HC': 349, 'YOUNG_HC': 45} |  |
| Full manifest group participant totals | PASS | {'ELDERLY_HC': 22, 'PD': 28, 'YOUNG_HC': 15} | {'PD': 28, 'ELDERLY_HC': 22, 'YOUNG_HC': 15} |  |
| Reconciled cohort participant counts | PASS | {'ELDERLY_HC': 17, 'PD': 19} | {'ELDERLY_HC': 17, 'PD': 19} |  |
| Reconciled 16 kHz recording total | PASS | 455 | 455 |  |
| Manifest relative paths unique | PASS | 0 | 0 |  |
| Manifest paths match audio files on disk | PASS | {'missing_on_disk': 0, 'unlisted_audio': 0} | {'missing_on_disk': 0, 'unlisted_audio': 0} |  |
| Task-code prefix consistency | PASS | 0 | 0 |  |
| Primary cohort manifest row inventory matches full manifest | PASS | 0 | 0 |  |
| Primary manifest decisions match participant decision table | PASS | 0 | 0 |  |
| Three FILE CODES workbooks agree | PASS | [['B1', 'B2', 'D1', 'D2', 'FB1', 'FB2', 'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2'], ['B1', 'B2', 'D1', 'D2', 'FB1', 'FB2', 'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2'], ['B1', 'B2', 'D1', 'D2', 'FB1', 'FB2', 'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2']] | same task-code set |  |

**Passed:** 14 / 14. **Failed:** 0.

## Unresolved questions

1. Why is documented FB2 absent from the observed recording set? Consult dataset custodians/protocol records; do not relabel another task without evidence.
2. What do PR1 and PR11 represent, and is PR11 a filename anomaly or a distinct undocumented code? Retain both literal codes pending source documentation.
3. Which task absences are genuinely missing recordings? A participant-level expected task schedule is not provided in the checked materials.
4. Several workbook-to-folder identity comparisons can be ambiguous or unmatched; the pre-existing reconciliation decisions remain authoritative and require targeted review only if a future analysis depends on those participants.

## Output files

- `population_totals.csv` — recording and participant totals for each population/group.
- `task_recording_participant_summary.csv` — recording and unique-participant counts by task and group for all populations.
- `participant_task_presence_full_dataset.csv`, `participant_task_presence_reconciled_hc_pd.csv`, `participant_task_presence_reconciled_hc_pd_16khz.csv` — participant-by-task presence matrices.
- `pairwise_task_overlap.csv` — pairwise overlap/common participants by population/group.
- `participant_task_coverage_gaps.csv` — observed participant/task absences, not assertions of protocol omissions.
- `workbook_task_code_definitions.csv`, `task_code_reconciliation.csv` — codebook definitions and observed/documented status.
- `participant_identity_reconciliation.csv`, `workbook_roster_audio_crosscheck.csv` — identity/reconciliation cross-checks.
- `participant_identity_collisions.csv` — reused folder IDs and repeated terminal names, clearly distinguished.
- `duplicate_manifest_entries.csv`, `audio_manifest_file_discrepancies.csv`, `filename_task_code_mismatches.csv`, `pr1_pr11_review.csv` — integrity and anomaly detail.
- `validation_checks.csv` — machine-readable validation outcomes.

## Scope conclusion

All implemented structural validation checks passed. The inventory is reproducible for the supplied manifest, cohort decisions, workbooks, and audio tree. Research questions about code meaning, FB2 absence, and expected participant task schedules remain unresolved; these do not permit inference about undocumented task meanings or missing protocol-required recordings.
