# Task-specific acoustic characterization of Italian Parkinsonian voice recordings: an exploratory participant-level study

**Manuscript status:** working draft based on completed analyses in this workspace. It has not been peer reviewed and contains no clinical validation claim. All results are exploratory.

## Abstract

**Background:** ItalianPVS provides sustained vowel, syllable, and reading recordings, but task-level analyses require participant-aware coverage, task-appropriate measurements, and explicit handling of sample-rate differences. **Objective:** characterize coverage and acoustic measurement for the reconciled elderly healthy-control (HC) versus Parkinson's disease (PD) cohort, while preserving uncertain protocol definitions. **Methods:** The supplied 831-file corpus was reconciled using existing project decisions. Audio QC covered 561 selected recordings across primary HC/PD and descriptive young HC. First-vowel analyses used five sustained vowel contexts and native 16 kHz files (17 HC, 12 PD; 145 recording-task rows). Participant-level group contrasts used four prespecified features, bootstrap intervals, and Holm correction across 20 contrasts. Leave-one-participant-out and robust-summary sensitivity analyses assessed the VI1 shimmer observation. Resampling sensitivity added 35 PD-only 44.1 kHz recordings. D1/D2 received all-file waveform/spectrogram characterization; B1/B2/FB1 received generic connected-speech feature extraction and VAD threshold sensitivity. Second vowel variants were characterized separately. **Results:** The 17 QC flags for possible clipping and two high-silence vowel flags prompted review but no exclusions. No automatic first-vowel formant-range flags occurred. VI1 local shimmer showed a positive HC-minus-PD contrast (mean difference 0.044, n=17/12; historical Holm-adjusted p=0.012). The direction remained positive under all single-participant deletions, median and trimmed summaries, and two bootstrap interval constructions. The multiplicity-adjusted trimmed-mean test did not cross 0.05 in the native-16-kHz cohort (p=0.061), showing inference depends on summary/test choice. Rate sensitivity did not remove group-rate confounding. D1/D2 signals mostly showed multiple separated CV-like bursts, supporting a repeated-production segmentation pilot but not a validated syllable count. Reading VAD pause counts changed with threshold. Second-code duration distributions were shorter than first-code distributions, with variant meanings unresolved. **Conclusions:** The VI1 shimmer contrast is a robust descriptive pattern in this cohort but not confirmatory disease evidence. First-vowel measurements are reproducible under the same software/configuration; biological repeatability is unknown. Syllable and reading measurements remain pending human review. No classifier was trained in this phase.

## 1. Introduction

Parkinson's disease can affect voice and speech production, but repeated recordings from a small number of people do not create an equally large independent sample. ItalianPVS includes multiple task types and group folders, with 16 kHz and 44.1 kHz audio in the supplied manifest. Its codebooks provide incomplete descriptions for some variants. These features make the corpus useful for task-specific measurement work while requiring explicit identity, task, and acquisition checks.

The motivating problem here is methodological: determine which task measurements can be interpreted from available files and documentation, how stable their extraction is to software repetition and analysis choices, and which task comparisons share participants. This report does not develop a diagnostic system. It keeps measurements at the participant level and treats frames or candidate speech events as within-recording observations.

## 2. Research motivation and scope

The project first established a reproducible inventory and participant reconciliation, then analyzed the five first vowel codes in a separately versioned experiment. The remaining phase evaluates the limits of syllable and connected-speech measures, describes the second vowel codes without calling them repeated trials, and prepares an English-dataset compatibility map. A central design consideration is that sample-rate availability differs by group: all reconciled HC recordings used in the current candidate tasks are 16 kHz, while seven additional PD participants have relevant recordings only at 44.1 kHz.

## 3. Dataset and participant reconciliation

The full manifest contains 831 files: 437 PD, 349 elderly-HC, and 45 young-HC recordings, from 65 group-qualified participant folders. The corpus contains 671 files at 16 kHz and 160 at 44.1 kHz. Existing reconciliation decisions define 17 included elderly-HC and 19 included PD participants across mixed-rate task analyses. The native-16-kHz comparison retains 17 HC and 12 PD. Young HC remain separate and descriptive; no participant identity or reconciliation decision was changed.

The five first vowel codes have 17 HC/19 PD participants at mixed rates and 17 HC/12 PD at 16 kHz. Second vowel codes have 17 HC/18 PD at mixed rates (17/11 native 16 kHz). D1/D2 each have 17 HC/19 PD and full paired overlap. B1 has 17 HC/19 PD, B2 has 17 HC/16 PD, and FB1 has 17 HC/18 PD. At native 16 kHz, B1/B2/FB1 counts are 17/12, 17/9, and 17/11 respectively. The existing task inventory documents these counts and preserves unresolved `FB2`, `PR1`, and `PR11` labels.

## 4. Audio preprocessing and QC

The original audio was read without modifying files. The first multi-task experiment inspected 561 selected recordings across the reconciled HC/PD cohort and descriptive young-HC recordings. Files were mono PCM-16 in that selected set. Automated flags marked 17 possible-clipping files and two high-silence vowel recordings. Flags led to human review prompts only; no participant was excluded. All 145 primary first-vowel rows passed broad F1/F2/F3 validity ranges, which does not substitute for manual formant inspection.

Native 16 kHz was used for primary group contrasts because including resampled 44.1 kHz observations adds PD participants only. Resampling can alter spectral measures and cannot remove microphone/session differences. First-vowel rate sensitivity compared eligible PD files at original 44.1 kHz and after polyphase resampling to 16 kHz; 35 task-recordings were evaluated in this sensitivity analysis.

## 5. Task-specific methods

### 5.1 Sustained vowels

Primary feature extraction used VA1, VE1, VI1, VO1 and VU1. Candidate stable phonation was defined as the longest contiguous Praat-voiced run at a fixed 75 Hz reference floor, with 10% trimmed at both ends. Pitch used autocorrelation, 75–500 Hz, 10 ms step. Local jitter and shimmer used fixed cycle settings; HNR used Praat CC. Formants used Praat Burg with fixed maximum formant 5 kHz, 25 ms window, 10 ms step, and 50 Hz pre-emphasis. Spectral centroid, 85% rolloff and flatness were exploratory. CPP/CPPS was not computed because no validated implementation was available in the installed environment. This segmentation is a candidate rule and may fail for irregular/breathy voicing.

### 5.2 Syllables

The supplied codebook labels D1 `/pa/` for five seconds and D2 `/ta/` for five seconds but does not say how productions are made. A published task description additionally notes a 20-second gap between the two task prompts but still does not specify repeated syllable count. All 72 reconciled D1/D2 recordings received waveform, spectrogram and RMS-envelope diagnostics. An exploratory peak detector generated candidate event times, intervals, event counts and thresholded pause fractions. These do not equal validated syllable onsets or counts and were not used for participant-level D1/D2 inferential summaries.

### 5.3 Connected speech

B1 and B2 are documented as first and second readings of “Il ramarro della zia”; the same participants are compared only in matched B1/B2 pairs, retaining group and original-rate strata. FB1 is analyzed separately because exact phrase content is missing. A recording-specific RMS VAD used 25 ms frames, 10 ms hop, a threshold −35 dB below each recording's 95th-percentile frame RMS, and single-frame gap closing. Sensitivity at −30 and −40 dB was calculated. Candidate measures include active speech and internal pause timing, voiced-frame/F0 summaries, relative RMS dynamics, and HNR in voiced-active windows. No transcript-dependent measures, syllable rate, or whole-recording jitter/shimmer were computed. The VAD masks are pending human validation.

## 6. Statistical analysis

The participant is the unit for first-vowel group comparisons. Primary features were median F0, local jitter, local shimmer and mean HNR across five vowels, yielding 20 prespecified contrasts. The historical analysis used participant bootstrap intervals and Mann-Whitney tests with Holm correction across the 20-comparison family. Independent sensitivity analyses reproduced the raw VI1 shimmer mean difference and unadjusted Mann-Whitney p-value directly from the feature table. They added 10,000-replicate independent participant bootstrap percentile and BCa intervals, median differences, 20% trimmed mean differences with Yuen tests, 10% winsorized differences, and leave-one-participant-out contrasts. No participant was automatically removed or winsorized out of the primary data; winsorization appears only as a robustness summary.

Paired vowel-context analyses are in the existing first-vowel report; distinct vowels are not treated as repeats. No inferential tests were run on D1/D2 candidate events or connected-speech features because segmentation validity has not been established. No cross-task effect-size ranking was made across different tasks, participant coverage, or feature definitions.

## 7. Results

### 7.1 First-vowel comparisons

The VI1 shimmer estimate was HC-minus-PD mean difference 0.0444, historical participant bootstrap 95% interval [0.0185, 0.0775], Hedges' g 0.889, and primary-family Holm-adjusted Mann-Whitney p=0.01199. It was the only primary feature-by-vowel contrast below 0.05 after the historical Holm correction. VI1 HNR had a substantial negative mean difference (approximately −7.18 dB), but its primary-family adjusted p-value was 0.0815. Median F0 contrasts were imprecise. Paired first-vowel results indicate that acoustic values depend on vowel context, particularly formants and HNR; these contrasts are phonetic-context differences, not reliability trials.

### 7.2 Influence and statistical-choice sensitivity

Every leave-one-participant-out VI1 shimmer contrast remained positive. The mean-difference range was 0.0335–0.0474 and Hedges' g range was approximately 0.82–0.95 for the native-16-kHz cohort. The median difference was 0.0217 (10,000-resample percentile interval [0.0085, 0.0345]); the 20% trimmed-mean difference was 0.0226; the 10% winsorized mean difference was 0.0405. Thus, the direction and nonzero interval are not attributable to a single participant or to the arithmetic mean alone. However, the Holm-adjusted Yuen 20% trimmed-mean p-value was 0.061 for the native-16-kHz family, versus 0.012 for the historical Mann-Whitney family. Alternative reasonable inferential summaries therefore change whether the result crosses a conventional threshold.

The new independent bootstrap yielded a percentile 95% CI [0.0182, 0.0756] and BCa CI [0.0225, 0.0862]. Both remain positive, although BCa's upper bound is wider. The historical interval construction was verified against source values: group counts, mean difference and unadjusted Mann-Whitney p match exactly when recomputed from the underlying participant-level CSV.

### 7.3 Sample-rate sensitivity

Adding 35 PD-only resampled task recordings left the VI1 shimmer mean difference similar (0.0442) and the direction agreed across all 20 primary contrasts; mean absolute Hedges-g change across contrasts was 0.116. The combined sensitivity has 17 HC/19 PD, but the seven additional participants are PD-only. The original 44.1 kHz stratum has no HC comparison. This analysis cannot separate rate from group or recording-condition effects. In paired same-recording rate changes, median F0 and voiced fraction were little changed; mean absolute HNR shift was about 0.43 dB for VA1. Spectral centroid and rolloff showed larger changes (VA1 mean absolute changes about 317 Hz and 446 Hz respectively; n=7 PD recordings), indicating that spectral measures are more rate-sensitive in this small paired subset.

### 7.4 D1/D2 and reading

Most D1/D2 recordings show repeated, separated CV-like bursts. Median automated candidate energy peaks ranged 7–9 across task/group strata; these are diagnostic outputs, not validated syllable counts. No participant-level D1/D2 timing contrast was computed. For reading, active-speech proportion and pause count varied materially across the three VAD thresholds. For example, B1 median active proportion ranged 0.695 at −30 dB to 0.774 at −40 dB, while median pause count changed from 85 to 65.5. This threshold sensitivity and the fragmented masks prevent valid pause-based group inference. No connected-speech significance tests or transcript-based measures were produced.

### 7.5 Second vowel variants

Across vowel contexts, native-16-kHz first-code duration medians were generally around 11–15 s, versus approximately 5.8–6.4 s for second-code medians. The paired code contrasts preserve 17 HC and 18 PD participants but do not establish repeatability. Their shorter duration distribution is a structural difference requiring protocol resolution before any combined-trial interpretation.

## 8. Measurement reliability and sensitivity

Exact repeated extraction of the native-16-kHz first-vowel set reproduced core measurements numerically for all 145 rows. This verifies deterministic extraction with the same software and configuration. It is not biological test-retest or inter-annotator reliability; the codebook does not establish which second variants are repeats. Pitch-floor sensitivity was separately measured at 60/75/90 Hz using a fixed 75 Hz segment-detection reference. The review queue retains six historical VA1 pitch cases for listening and track inspection.

## 9. Discussion

The main robust descriptive observation is higher VI1 local shimmer in the HC group than PD group in this small native-16-kHz reconciled subset. Participant deletion, median/trimmed summaries and percentile/BCa bootstrap intervals preserve its direction. The corrected inferential conclusion is less stable: an alternate trimmed-mean test does not meet the same Holm threshold. The combined-rate estimate adds only PD participants and cannot establish rate-independent disease differences. The signal should therefore be reported as exploratory, not confirmatory or clinical.

The different task families do not yet supply comparable outcome evidence. D1/D2 waveforms support repeated-event characterization, but machine event candidates require manual onset labels before count/interval reliability can be estimated. Reading masks show substantial threshold dependence and require human review. Second vowel variants have different duration distributions but unspecified relation to first codes. These are concrete reasons to continue review and protocol retrieval rather than infer missing definitions.

## 10. Limitations and unresolved questions

The primary vowel cohort is only 17 HC and 12 PD at native 16 kHz; no arbitrary minimum sample size was imposed, and the study is not adequately established for confirmatory inference. All additional 44.1 kHz records are PD. Age/sex distributions and microphone/session conditions may confound group contrasts. Formant validity is assessed with broad fixed ranges and requires listening/visual review. VAD thresholds are adaptive to each file and not yet human-adjudicated. The six historical pitch cases are open. D1/D2 production count remains unspecified even though repeated bursts are visible. The second vowel variants are undefined. B1/B2 order confounds practice, fatigue and task sequence; FB1 materials are unavailable. `FB2` is documented but absent. `PR1` and `PR11` meanings remain unknown. English external data access/licensing must be confirmed; the Figshare item license was not verified during this phase.

The existing VA1 baseline is retained as a historical reference: its previously reported 29-participant pooled OOF balanced accuracy was 0.900. It was not rerun or modified here and is not evidence of clinical performance. No classifiers were trained in this work package.

## 11. Future work

1. Use the review interface to annotate the 159-item deduplicated queue, prioritizing clipping/silence and six pitch cases, then D1/D2 onset times and the stratified reading masks.
2. Validate an event detector against manual D1/D2 onset annotations; only then produce participant-level syllable count, interval and pause summaries and paired `/pa/` vs `/ta/` contrasts.
3. Revise connected-speech VAD using manual segmentation decisions and report reliability by threshold, group and rate before group inference.
4. Obtain source protocol material for second vowel variants, D1/D2 production, B1/B2 and FB1; keep codes literal until verified.
5. For English integration, first confirm item-level Figshare license and compare raw task/channel metadata; seek authorized mPower access only if a sustained-vowel analysis is planned. Do not merge feature tables as if their measurements were interchangeable.
6. Only after acoustic measure validity and matched cohort planning, design participant-level cross-task/cross-language validation. Deep learning and clinical claims are out of scope.

## 12. References and dataset sources

1. Dimauro G, Caivano D, Bevilacqua V, Girardi F, Napoletano V. VoxTester, software for digital evaluation of speech changes in Parkinson disease. IEEE MeMeA, 2016. [doi:10.1109/MeMeA.2016.7533761](https://doi.org/10.1109/MeMeA.2016.7533761).
2. Dimauro G, Di Nicola V, Bevilacqua V, Caivano D, Girardi F. Assessment of Speech Intelligibility in Parkinson’s Disease Using a Speech-To-Text System. IEEE Access, 2017. [doi:10.1109/ACCESS.2017.2762475](https://doi.org/10.1109/ACCESS.2017.2762475).
3. Frontiers in Physics. Exploring robust computer-aided diagnosis of Parkinson’s disease based on various voice signals. 2022. [doi:10.3389/fphy.2022.1048833](https://doi.org/10.3389/fphy.2022.1048833).
4. Jaeger H, Trivedi D, Stadtschnitzer M. Mobile Device Voice Recordings at King’s College London (MDVR-KCL) [Data set]. Zenodo, 2019. [doi:10.5281/zenodo.2867216](https://doi.org/10.5281/zenodo.2867216).
5. Tsanas A, Little MA, McSharry PE, Ramig LO. Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests. IEEE Transactions on Biomedical Engineering, 2010;57(4):884–893. [doi:10.1109/TBME.2009.2036000](https://doi.org/10.1109/TBME.2009.2036000).
6. Sakar C et al. Parkinson’s Disease Classification [Data set]. UCI Machine Learning Repository, 2018. [doi:10.24432/C5MS4X](https://doi.org/10.24432/C5MS4X).
7. Little MA. Parkinsons [Data set]. UCI Machine Learning Repository, 2007. [doi:10.24432/C59C74](https://doi.org/10.24432/C59C74).
8. Tsanas A, Little M. Parkinsons Telemonitoring [Data set]. UCI Machine Learning Repository, 2009. [doi:10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N).
9. Figshare. Voice Samples for Patients with Parkinson’s Disease and Healthy Controls [Data set], item 23849127. [Dataset record](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127). Item-level license still requires confirmation.
10. mPower study investigators. The mPower study, Parkinson disease mobile data collected using ResearchKit. [Study/data documentation](https://pmc.ncbi.nlm.nih.gov/articles/4776701/).
