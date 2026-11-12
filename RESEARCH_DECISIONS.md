# Research decisions and open questions

This file records methodological choices that should persist across sessions. Changes to an agreed decision require explicit rationale and must not silently alter the VA1 reference experiment.

## Agreed decisions

| Decision | Rationale / scope | Status |
|---|---|---|
| Primary comparison is elderly healthy controls versus Parkinson's participants; exclude YHC from primary HC-PD analyses. | Avoid conflating age-group and disease-group contrasts. | Current |
| Participant-level separation is mandatory for any model evaluation. | Recordings from one person must not leak across train and test folds. | Current |
| VA1 is the completed reference baseline, not general speech analysis. | VA1 represents sustained /a/ phonation; preserve its scripts, tables, predictions, and results. | Current |
| Keep VA1 baseline immutable. | Improvements belong to a separately documented experiment with distinct outputs. | Current |
| Do not jump to deep learning on the 29-person VA1 cohort. | First establish task coverage, acoustic differences, measurement reliability, and participant-level validation. | Current |
| Treat all current classifier results as exploratory; make no clinical claims. | Small sample and internal validation only. | Current |
| Keep original filenames and task codes intact; do not infer undocumented task definitions. | `PR1`, `PR11`, second variants, and FB1/FB2 relationship require source verification. | Current |
| Preserve existing participant reconciliation. | Includes ANTONIETTA P -> Antonella P mapping, age discrepancy note, and previously excluded unresolved identities. | Current |
| Do not exclude participants solely due to pitch sensitivity. | Investigate waveform, pitch tracks, voicing, and spectra first. | Current |
| Use `possible_task_code` as the primary task inventory field; report workbook descriptions verbatim. | Preserve source labeling and avoid inferring undocumented codes. | Current |
| Treat matrix zeros as observed coverage absence, not missing protocol-required recordings. | No participant-level expected task schedule has been established. | Current |

## Recorded cohort decisions

- Historical reconciliation: 17 included elderly HC and 19 included PD participants before sample-rate filtering; 455 included recordings at 16 kHz. Current VA1 baseline is narrower: 17 elderly HC and 12 PD, one 16 kHz VA1 recording each.
- Manual mapping: audio folder `ANTONIETTA P` maps to workbook identity `Antonella P`; filename age 56 versus workbook age 61 remains documented and unresolved.
- Unresolved identities excluded: LISCO G, PORCELLI A, SUMMO L, elderly-control VITO L, and PD 17-28 / Mario B. PD 17-28 / Vito L is distinct from the excluded elderly-control VITO L.

| Report uncertainty for existing VA1 pooled OOF metrics by resampling participants while holding saved OOF predictions fixed. | Provides a transparent sampling-uncertainty summary without retraining or changing the reference baseline; does not estimate model-refitting uncertainty. | Completed in `VA1-OOF-UNCERTAINTY-20261031`; intervals are conditional and exploratory |

## Open questions

| Question | Why it matters | Next evidence/action |
|---|---|---|
| What are the exact meanings of PR1 and the one-off PR11? | Prevent incorrect task grouping. | Check original code workbook and dataset documentation; retain literal codes until resolved. |
| What do the `2` variants mean, and is FB1 related to documented FB2? | Determines valid task pairing and repeated-measure comparisons. | Inspect workbook cells and source protocol. |
| Are task participants comparable across HC/PD and sample-rate subsets? | Different cohorts can confound task comparisons. | Produce coverage, overlap, common-participant, and 16 kHz summaries. |
| Are pitch anomalies genuine voice traits or tracking errors? | Affects feature reliability and interpretation. | Continue participant-level waveform, spectrum, and pitch-track review; retain participants meanwhile. |
| How much uncertainty surrounds VA1 performance? | n=29, internal CV estimates may be unstable. | Conditional participant-bootstrap intervals for saved OOF predictions completed; prediction review and broader model uncertainty remain open. |
| Which acoustic features are appropriate and repeatable for each task? | Sustained vowels and connected speech need different measurement choices. | Establish task-specific definitions and reliability before large-scale extraction. |
| Why is documented FB2 absent, and what do PR1 / PR11 mean? | Task-code interpretation affects task grouping and protocol claims. | Seek original protocol/catalog; keep codes literal in the meantime. |
| Is workbook `Mario M` the person in audio folder `17-28/Mario B`? | The spelling discrepancy affects the already excluded unresolved identity. | Preserve current exclusion; seek identity evidence before any reconciliation change. |

## Multi-task analysis decisions — authorized and executed 2026-10-30

This section records the design choices that were implemented in the authorized follow-up. It does not resolve protocol questions that remain open.

| Proposal | Rationale | Status |
|---|---|---|
| Use native 16 kHz recordings for primary group comparisons; include resampled 44.1 kHz files only in sensitivity analyses. | All reconciled HC recordings are 16 kHz, while 7 PD participants have eligible task recordings only at 44.1 kHz. Resampling cannot remove acquisition/channel confounding and can alter bandwidth-dependent measures. | Executed in `ITALIAN-MULTITASK-V1-20261030`; retain as current strategy |
| Prioritize first vowel codes for initial participant-level feature/statistical analysis; QC D1/D2 and reading without claiming unsupported protocol meanings. | Strongest coverage and common-participant overlap; native-16-kHz comparison remains 17 HC/12 PD. | Executed; task-specific timing extraction deferred pending protocol/QC review |
| Keep second vowel variants separate until their relationship is verified. | Codebook descriptions are blank for VA2/VE2/VI2/VO2/VU2; do not assume repeats or reliability pairs. | Maintained; no second-variant features extracted |
| Treat B1/B2 as documented first/second readings of the same text, and FB1 as a separate task with unspecified materials; do not compute text-dependent measures. | FILE CODES wording gives limited task labels but not full materials/procedure. | QC/segmentation pilot only; no reading feature extraction |
| Choose task-specific features; avoid whole-recording jitter/shimmer for connected speech. | Sustained phonation, syllable timing, and connected speech have different valid measurement assumptions. | Executed for first vowels; D1/D2 and reading feature extraction deferred |
| Keep PR1/PR11 and FB2 unresolved; preserve literal codes and existing reconciliation. | Definitions or observed files do not support further interpretation. | Maintained; no PR features extracted |

### Executed experiment notes

`ITALIAN-MULTITASK-V1-20261030` used participant-level first-vowel summaries and native 16 kHz for the primary group contrasts. Primary features were median F0, local jitter, local shimmer, and mean HNR, with other extracted acoustic measures treated as exploratory. Holm correction covered the 20 prespecified primary feature-by-vowel contrasts. VI1 shimmer alone remained below 0.05 after this family correction (adjusted p≈0.012); small samples and exploratory context preclude confirmatory or clinical interpretation. Thirty-five PD-only 44.1 kHz recordings were analyzed before and after polyphase resampling. The mean absolute Hedges-g change across 20 native-versus-combined contrasts was ≈0.116, with no direction flips; this does not address rate/group confounding. Deterministic repeat extraction is not biological test-retest reliability. See the experiment report and its config for exact settings.

## Remaining-analysis decisions — 2026-10-30

| Decision | Rationale | Status |
|---|---|---|
| Treat VI1 shimmer as a robust exploratory direction, not a confirmed group finding. | Native-16-kHz HC-minus-PD mean difference is 0.04439 (n=17/12); all leave-one-person-out contrasts and several robust summaries remain positive. Inference varies by method: Holm Mann–Whitney p≈0.012 versus Holm Yuen trimmed-mean p≈0.061. | Sensitivity audit completed; no exclusions |
| Keep D1/D2 candidate event detections out of participant-level timing inference until human segmentation review. | Most diagnostic plots show multiple CV-like events, but amplitude peaks are not validated syllable onsets and the codebook does not establish production count/pattern. | 72 recordings queued; manual annotations pending |
| Treat B1/B2/FB1 speech activity, pitch, intensity and HNR values as candidate descriptive measurements pending manual VAD review. | Threshold sensitivity is material, FB1 text is unavailable, and B1/B2 order can mix practice/fatigue effects. | 104 recording-task rows generated; review pending |
| Keep all second vowel codes separate and descriptive. | The second-variant codebook descriptions are blank; observed duration/morphology differences cannot establish repeat or test-retest status. | 355 signal-characterization rows; protocol unresolved |
| Maintain native 16 kHz as primary group-comparison rate; keep resampled PD-only files as sensitivity analysis. | Resampling does not resolve group/rate confounding and changes spectral measures. | Retained |
| Do not pool external datasets or label results language transfer without evidence. Independently sourced cohorts may be tested in a locked zero-shot design. | Dataset, task, device, rate, cohort and language certainty remain distinct; no mixed-dataset random split or target adaptation. | Cross-dataset V1 completed; no confirmed language transfer |
| Keep the 159-item review queue and manual annotations separate from automated QC and source data. | Preserve auditability and prevent automated flags from excluding participants. | Local review GUI and blank annotations created; no reviews entered |

### Remaining unresolved protocol and review items

- 159 unique audio items are awaiting review. The queue merges overlapping reasons and records task, participant, sample rate, automated QC context, and required judgment. All 72 reconciled D1/D2 recordings are included; review forms support manual onset times and syllable counts.
- D1/D2 codebook duration (/pa/ or /ta/, five seconds) does not specify whether to repeat syllables or how many. Signal morphology suggests repeated CV-like bursts in most files but cannot establish the intended protocol.
- B1/B2 VAD segmentations have not been manually adjudicated. Threshold comparison (−30/−35/−40 dB relative to per-recording p95 frame RMS) shows active-speech/pause summaries change with threshold. No transcript-based rate or alignment is reported; CPP/CPPS is deferred.
- VA2/VE2/VI2/VO2/VU2 codebook descriptions are blank. No biological repeatability is claimed.
- Figshare item access/license must be confirmed before redistribution or integration. Other reviewed English datasets are only broadly compatible or lack raw audio/control participants.
- PR1/PR11, absent FB2, and roster identity mismatches remain open from the original audit and were not reinterpreted.

The work package passed 21/21 computational validation checks and the full suite passed 51 tests. Human review is not a failed automated check, but it is a prerequisite for advancing D1/D2 timing or connected-speech claims. Do not declare the Italian research phase complete until this review and/or final documented deferrals are resolved.


## Final research package decisions — 2026-10-31

- **Preservation:** The VA1 baseline, completed Italian task inventory, reconciliation decisions, and earlier experiment outputs remain read-only reference inputs. A 440-file SHA-256 before/after audit reported no changed paths.
- **English datasets:** Analyze Figshare and MDVR-KCL independently at participant level; keep MDVR reading and dialogue separate. Analyze UCI #174/#189/#301 only within their respective precomputed-feature structures. Do not describe Figshare or UCI #301 as English-language evidence without source confirmation; neither is merged with the MDVR English tasks. UCI #470 and mPower remain inventory-only because no local data were present.
- **MDVR sample-rate and VAD:** Preserve native 44.1 kHz originals and use a polyphase-resampled 16 kHz analysis copy for the current candidate feature pass. The 16 kHz conversion does not recover missing bandwidth. Use −25 dB below per-recording p95 RMS as a provisional feature mask only, and publish the −20/−25/−30/−35 dB threshold sensitivity; no speech/pause result is treated as validated until human mask and speaker review.
- **Participant unit:** Each group contrast uses one row per participant-task. UCI 174 repeats are collapsed to participant medians; UCI 301 training comparisons use each participant's task row; its PD-only repeated-vowel partition is descriptive within-person spread; UCI 189 associations are computed within person before summarizing across participants.
- **Interpretation gate:** Report candidate group effects and uncertainty without clinical or language-transfer claims. MDVR reading contrasts that pass within-task Holm adjustment remain provisional because segmentation and connected-speech feature validity have not been reviewed. For the later authorized cross-dataset V1, no pooled datasets or target adaptation were used; classifiers were trained only on source cohorts and tested zero-shot.
- **Data license:** The Figshare item page lists CC BY 4.0; attribution is required. No raw audio is redistributed in this package.


## Cross-dataset generalisation decision — 2026-10-31

- **Scope:** Execute raw-audio external testing only for Italian VA1 and Figshare prolonged `/a/`. This is a task-matched **cross-dataset** comparison with limitations, not cross-language validation because Figshare language is not established and acquisition/task metadata are not fully aligned.
- **Feature plan:** Use identical raw-waveform extraction and pre-specified core4/core5 features. Keep native 16 kHz vs native 8 kHz primary; downsample Italian in separate copies only as a sensitivity analysis. Do not impute/scale/tune/threshold on the external cohort.
- **Model plan:** Classical LR, SVM, and RF only; nested participant CV internally; source-only tuning externally; no ComBat/domain adaptation, pooled split, deep model, or target label use.
- **Outcome:** Transfer was weak in both directions. Matched diagnosis-count dataset identity remained predictable, with F0 IQR and jitter carrying much of the logistic domain signal. This argues for a prospective matched acquisition cohort before further model claims; feature effects are not causal.
- **Preservation:** Existing VA1 and prior results stayed unchanged under the 1,429-file source hash comparison. Results and split audits live only in `results/cross_language_generalisation_v1/`.
