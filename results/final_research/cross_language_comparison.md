# Italian–English and cross-dataset comparison

## Evidence boundary

This is a comparison of independent, task-labelled cohort estimates, not a pooled model and not proof of language transfer. Only ItalianPVS, Figshare and MDVR-KCL have raw local audio; Figshare is a sustained `/a/` telephone task with language not explicitly identified in its metadata, while MDVR-KCL includes English reading and spontaneous dialogue. UCI 174, 189 and 301 are precomputed feature tables; UCI 301 is from an Istanbul clinic and its exact task language is not identified in the UCI record. UCI 470 and mPower were documented but are not locally analyzed.

## Feature compatibility matrix

| feature_family | ItalianPVS | Figshare | MDVR_read_text | MDVR_spontaneous | UCI174 | UCI301 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Median F0 | Yes: Praat pitch on stable vowel; 75–500 Hz | Yes: same Praat candidate stable-vowel definition | Yes: active voiced frames | Yes: active voiced frames | Precomputed MDVP:Fo; definition differs | Precomputed median pitch; source task conditions differ | Task and sample rate comparable only for sustained phonation; utterance-level meanings differ |
| F0 variability | Yes: IQR/SD positive-pitch frames | Yes: IQR/SD positive-pitch frames | Yes: voiced active-frame IQR | Yes: voiced active-frame IQR | Range/variability columns; definition differs | Pitch SD/range features; precomputed | Feature family compatible; numerical harmonization requires exact definitions |
| Jitter | Praat local jitter | Same Praat local jitter | Not extracted: connected speech | Not extracted: speaker diarization absent | MDVP jitter, repeated-record median | Dataset local jitter feature; original algorithm/protocol | Candidate compatible within sustained vowel only; not a numeric replication without source methods |
| Shimmer | Praat local shimmer | Same Praat local shimmer | Not extracted | Not extracted | MDVP shimmer; participant median | Local shimmer feature in precomputed archive | Potential sustained-vowel association comparison; differing hardware/extractor confounds |
| HNR | Praat CC HNR | Praat CC HNR | Praat CC in active voiced frames | Praat CC; mixed voices possible | MDVP HNR | Historical HTN/NTH not identical to HNR | Same concept, but bandwidth/window/implementation differ for feature-table sources |
| Formants | Praat Burg F1–F3 | Not extracted; 8 kHz bandwidth limits upper formants | Not extracted | Not extracted | Not available in local raw audio | Not available as verified raw measurements | No cross-dataset formant comparison |
| Spectral centroid/rolloff/flatness | Frame spectra on stable vowel, 16 kHz source | Not used in primary cross-dataset comparison | Candidate frame spectra at 44.1 kHz | Candidate frame spectra at 44.1 kHz | Not raw/reproducible | No exact equivalent specified | Acquisition bandwidth and task strongly limit comparison |
| Speech activity / pauses | Candidate RMS-VAD; threshold sensitivity; review open | Not applicable to sustained vowel | Candidate RMS-VAD, −25 dB provisional; no manual validation | Candidate RMS-VAD, examiner voice possible | No audio | No task-resolved raw measure available | Not comparable across different prompts, segmentation and speaker content |
| Jitter/shimmer variability reliability | No test-retest; deterministic extraction only | One file/person; no repeat task | Repeated task recordings absent | No clean participant-only turns | Repeated records but no raw audio | UCI301 PD test has 3 same-vowel feature rows/person | UCI301 repeat spread is within-PD feature-table stability only; not biological day-to-day reliability |

## Sustained-vowel effect directions

Figure C1 compares standardized HC-minus-PD effects for `/a/`-labelled tasks across ItalianPVS VA1, Figshare, UCI 174 and UCI 301. This figure visualizes direction and uncertainty only. ItalianPVS / Figshare were extracted with matching local Praat stable-vowel settings, but differ in language/phonetic realization, 16 vs 8 kHz bandwidth, phone/hardware and demographics. UCI features use historical software definitions and participant aggregation, so their intervals are not a raw-audio harmonization.

The Italian VI1 `/i/` shimmer observation has no direct task-matched replication in the local English datasets. The Figshare corpus and UCI 174 provide `/a/`; UCI 301’s training data provide `/a,o,u/`; MDVR has no sustained `/i/`. A same-name shimmer feature for a different vowel is not a replication of the VI1 contrast.

## What is and is not consistent

**Acoustic feature compatibility:** median F0, local jitter, local shimmer and HNR can be defined consistently on sustained vowels in raw audio, but absolute values remain channel, bandwidth and implementation dependent. For the two raw-audio `/a/` corpora, the same Praat definition is applied here. UCI table definitions are not identical enough for pooled absolute values. Pause/active-speech features share a broad concept but are not interchangeable across task materials and VAD validity.

**Directional consistency:** inspect the plotted signs and table values per feature. A shared sign across independent tables is only directional agreement, especially where language/task, age and recording methods differ. The precomputed UCI sets have independent participants but historical feature extraction and task metadata limitations.

**Effect-size consistency:** standardized effects and intervals are displayed by dataset; overlap/non-overlap of small-sample intervals is descriptive and does not establish equivalence. We do not rank unrelated tasks.

**Generalization:** no feature set can validate cross-language transfer in this work package. The small and task-mismatched corpora do not support a pooled classifier or dataset-held-out participant experiment with common task, labels and sufficient covariate balance. mPower may provide a future controlled-access sustained phonation cohort after task and consent filtering.

## Recommended cross-language next step

Pre-register a sustained-vowel protocol that captures `/a/` and `/i/` (plus repeated tokens) in Italian and English speakers using the same microphone, calibrated distance, sample rate, quiet-room instructions and feature code. Balance or model age/sex, retain participant identifiers, and reserve a completely independent cohort for replication. Treat `/i/` shimmer as a targeted hypothesis and `/a/` as the bridge task. Do not attempt domain adaptation until raw measurement comparability and cohort labels are verified.
