# ItalianPVS multi-task acoustic analysis plan

**Status:** Design report; initial authorized QC pilot and first-vowel experiment executed as `ITALIAN-MULTITASK-V1-20261030`. Results and validation are in `results/italian_multitask_v1/research_report.md`. This plan remains the rationale and protocol boundary for subsequent task work.  
**Primary population:** reconciled elderly healthy controls (HC) versus Parkinson's disease participants (PD).  
**Descriptive population:** full ItalianPVS dataset, with YHC retained as a separate descriptive group.  
**Design inputs:** completed inventory in `reports/italian_task_audit/` and the three group-specific `FILE CODES.xlsx` workbooks.

## 1. Task selection rationale

Begin with participant-comparable tasks whose labels have at least a documented description. The five first vowel codes and D1/D2 have the broadest primary-cohort coverage. B1/B2 and FB1 are useful connected-speech tasks, but B2 has lower PD coverage and its second-reading order may introduce practice or fatigue effects. Keep `PR1` and `PR11` out of the primary analysis because the supplied codebooks do not define them. `FB2` is documented but no FB2 recording is present; do not substitute FB1 or another code.

### Codebook wording and protocol limits

The three FILE CODES workbooks agree. Their literal relevant descriptions are:

| Code | Wording in FILE CODES.xlsx | Design implication |
|---|---|---|
| VA1 / VE1 / VI1 / VO1 / VU1 | “2 phonation of the vocal” followed by a/e/i/o/u | Wording is ambiguous; confirm with protocol/source before interpreting the leading “2”. |
| VA2 / VE2 / VI2 / VO2 / VU2 | Blank | Do not assume these are repetitions, second trials, or equivalent variants. |
| D1 | Execution of syllable “pa” (5 sec) | Five-second duration is documented; whether this is repeated syllables or another production pattern is not stated. |
| D2 | Execution of syllable “ta” (5 sec) | As for D1; do not assume repetition rate task without waveform/protocol confirmation. |
| B1 | First reading of “Il ramarro della zia” | First reading is explicit. |
| B2 | Second reading of the same text | Second reading is explicit; likely paired with B1, with order/practice effects retained in interpretation. |
| FB1 | Reading of some phonemically balanced phrases | Phrase content/transcript and relation to B1/B2 are not supplied. |
| FB2 | Reading of some phonemically balanced words | Documented, but absent from the observed code inventory and audio manifest. |

The audit confirms filename/task-code consistency, not the production protocol or content of the recordings. Confirm the ambiguous details before finalizing feature definitions.

## 2. Participant coverage and comparability

Counts below are unique participants in the reconciled HC-PD cohort. Task recording counts equal participant counts for these listed tasks. “16 kHz” means recordings already at native 16 kHz, not resampled recordings.

| Candidate task | HC, any rate | PD, any rate | HC, native 16 kHz | PD, native 16 kHz | Shared HC / PD participants across indicated pair |
|---|---:|---:|---:|---:|---|
| VA1 | 17 | 19 | 17 | 12 | — |
| VA2 | 17 | 18 | 17 | 11 | VA1–VA2: 17 / 18; at 16 kHz: 17 / 11 |
| VE1 | 17 | 19 | 17 | 12 | — |
| VE2 | 17 | 18 | 17 | 11 | VE1–VE2: 17 / 18; at 16 kHz: 17 / 11 |
| VI1 | 17 | 19 | 17 | 12 | — |
| VI2 | 17 | 18 | 17 | 11 | VI1–VI2: 17 / 18; at 16 kHz: 17 / 11 |
| VO1 | 17 | 19 | 17 | 12 | — |
| VO2 | 17 | 18 | 17 | 11 | VO1–VO2: 17 / 18; at 16 kHz: 17 / 11 |
| VU1 | 17 | 19 | 17 | 12 | — |
| VU2 | 17 | 18 | 17 | 11 | VU1–VU2: 17 / 18; at 16 kHz: 17 / 11 |
| D1 | 17 | 19 | 17 | 12 | D1–D2: 17 / 19; at 16 kHz: 17 / 12 |
| D2 | 17 | 19 | 17 | 12 | D1–D2: 17 / 19; at 16 kHz: 17 / 12 |
| B1 | 17 | 19 | 17 | 12 | B1–B2: 17 / 16; at 16 kHz: 17 / 9 |
| B2 | 17 | 16 | 17 | 9 | B1–FB1: 17 / 18; at 16 kHz: 17 / 11 |
| FB1 | 17 | 18 | 17 | 11 | B2–FB1: 17 / 16; at 16 kHz: 17 / 9 |

For descriptive analysis using the full, unreconciled dataset, the audit shows these task-level participant counts (YHC / elderly HC / PD; all available sample rates):

| Task family/code | YHC | Elderly HC | PD | PD recordings at 16 kHz / 44.1 kHz |
|---|---:|---:|---:|---:|
| VA1, VE1, VI1, VO1, VU1 (each) | 0 | 22 | 28 | 18 / 10 |
| VA2, VE2, VI2, VO2, VU2 (each) | 0 | 22 | 27 | 17 / 10 |
| D1 and D2 (each) | 0 | 22 | 28 | 18 / 10 |
| B1 | 15 | 21 | 28 | 18 / 10 |
| B2 | 15 | 21 | 24 | 14 / 10 |
| FB1 | 0 | 21 | 26 | 16 / 10 |

These are descriptive inventory denominators, not the primary analysis cohort. The audit task summary is the source for each task/group's recording and participant counts.

For vowel first variants (VA1/VE1/VI1/VO1/VU1), the same 17 HC and 19 PD participants are present across all five codes; at native 16 kHz this is 17 HC and 12 PD. Across all ten vowel codes, common coverage is 17 HC and 18 PD, dropping to 17 HC and 11 PD at 16 kHz. These are participant overlaps, not proof that variants are repeats or that the tasks followed the same protocol.

For B1/B2/FB1 together, 17 HC and 16 PD have all three tasks; at native 16 kHz the common set is 17 HC and 9 PD. Among all-rate B1/B2 paired participants, 7 PD are present only in 44.1 kHz recordings; the native-rate paired set retains 9 PD. B2 additionally lacks a B2 file for 3 participants from the 12-person 16 kHz PD subset.

### Sample-rate loss and participant overlap

All 17 reconciled elderly HC participants have their candidate recordings at 16 kHz. Of 19 reconciled PD participants, 7 have the relevant candidate recordings only at 44.1 kHz and 12 have them at 16 kHz. Thus rate restriction removes 7 PD participants across the first vowel variants, D1/D2, B1, and the second vowel variants’ available 18-person sets; it removes 7 from B2 and FB1 as well, with additional task-specific recording gaps described above. The common 16 kHz group cohort is 17 HC / 12 PD.

The seven rate-excluded PD folders for the candidate tasks are `1-5/Domenico C`, `1-5/Felicetta C`, `1-5/Leonarda L`, `1-5/Lucia R`, `6-10/Luigi B`, `6-10/Saverio S`, and `6-10/Ugo B`. This exclusion is a consequence of recording availability, not a change to reconciled participant identity decisions.

### Suitability and power limitations

- **Strongest coverage candidates:** first vowel codes and D1/D2. Their 17/19 group coverage and full paired overlap make task-specific descriptive summaries and within-participant contrasts feasible.
- **Conditional candidates:** second vowel codes have 17/18 coverage, but the meaning of the `2` variants is unknown. B1/B2/FB1 have useful overlap, although B2 is less complete for PD and is an ordered second reading.
- **No candidate is established as adequately powered for confirmatory group inference.** The available n is 17/19 at mixed rates or 17/12 at native 16 kHz. These samples can support exploratory estimates, but confidence intervals may be wide and multiple acoustic outcomes further reduce precision. No minimum sample-size threshold is imposed here. Prospective power/precision planning needs defensible effect-size and variance estimates from a justified pilot or external evidence.
- The full dataset remains available for descriptive summaries by YHC, elderly HC, and PD. It must not replace the reconciled HC-PD population for primary comparisons or silently merge YHC with elderly HC.

## 3. Candidate acoustic measurements by task

The measures below are proposals. Their implementation depends on task-protocol confirmation and a small QC/segmentation pilot. Preserve original audio and write all derived files to a new experiment-specific output directory. Do not apply the VA1 six-feature classifier set wholesale.

### Sustained vowel phonation: VA1/VA2, VE1/VE2, VI1/VI2, VO1/VO2, VU1/VU2

Apply only if audio inspection/protocol confirms isolated sustained vowel phonation. Use the vowel label to keep phonetic context explicit.

| Feature family | Proposed measure and definition | Method/settings proposal | Meaning and limits |
|---|---|---|---|
| Pitch | Median F0; within-recording F0 IQR and SD; voiced-frame fraction | Praat autocorrelation; 10 ms time step. For continuity with VA1, retain 75–500 Hz as a reference extraction setting and manually inspect tracks. Include predeclared 60/90 Hz sensitivity checks because existing VA1 sensitivity results show participant-specific tracking shifts. | Relevant to phonation, but pitch floor/ceiling can cause octave errors; do not treat estimator failures as true low/high F0. |
| Period perturbation | Local jitter from period-to-period variation | Praat point-process measure on stable, periodic voiced middle region; matching floor/ceiling and pulse limits must be logged. | Meaningful only for sufficiently periodic, stable phonation; undefined values are QC outcomes, not zero. |
| Amplitude perturbation | Local shimmer from cycle amplitude variation | Same stable voiced segment and Praat pulse settings as jitter. Avoid gain normalization before this measure. | Sensitive to recording gain, noise, and segmentation; interpret cautiously across sessions/sites. |
| Noise/periodicity | Mean HNR over valid frames; optionally CPP as a distinct cepstral measure | Praat harmonicity settings fixed and recorded; CPP implementation/version and frame settings fixed separately. Report valid-frame coverage. | Relevant to voice quality, but sensitive to pitch tracking and background noise. |
| Vowel articulation | F1/F2 (optionally F3) at a defined stable segment; formant bandwidth/QC | Praat Burg or validated LPC implementation, fixed frame step/window and predeclared speaker-appropriate formant ceiling; visually inspect spectra/formant tracks for a stratified sample and all flagged cases. | Vowel-dependent articulation/resonance. Do not compare raw formants across different vowels as if phonetic context were identical. |
| Spectral shape | Harmonic spectral tilt or spectral slope over a predeclared band; optional centroid/flatness as secondary descriptive metrics | Windowed spectra with same frame length, window, pre-emphasis policy, and frequency band at common sampling rate. | Centroid/flatness alone are not disease-specific or directly comparable across different vowel spectra; avoid treating them as primary biomarkers. |
| Duration / stability | Usable voiced duration, onset/offset transition duration, within-vowel variability across windows | Waveform plus voicing/signal-quality segmentation; summarize valid central region and report duration. | Useful for recording/QC and stability. The protocol does not establish target duration for these non-VA1 codes. |

For compatibility only, the existing VA1 extractor's explicit Praat reference values are pitch floor 75 Hz, ceiling 500 Hz, 10 ms step; local jitter period bounds 0.0001–0.02 s and period factor 1.3; local shimmer with amplitude factor 1.6; and CC harmonicity at 10 ms step, 75 Hz minimum pitch, silence threshold 0.1, one period/window. These are candidate reference settings to log and validate for each new task, not automatic defaults for every speaker or task. The pitch-sensitivity results make manual track review and 60/90 Hz sensitivity checks especially important.

**Not proposed as primary sustained-vowel outcomes:** speech rate, pause rate, articulation rate, consonant VOT, lexical/syntactic measures, or whole-recording MFCC summaries. These do not match sustained isolated phonation.

### Syllable production: D1 and D2

The workbook specifies “pa” and “ta” and “5 sec,” but not whether participants repeat the syllable, sustain an element, or use another timing pattern. Do not choose a single feature protocol until waveform review and source clarification establish the production pattern.

| If protocol/waveform confirms… | Candidate measures | Extraction and interpretation |
|---|---|---|
| Repeated syllables | Syllable count/rate; inter-onset interval median and variability; pause fraction/duration; cycle-to-cycle duration; voiced-vowel F0/CPP/HNR | Segment syllable nuclei and boundaries, then have a human reviewer verify a subset. Report missed/extra detection rates. Distinguish speech rate from articulation rate. |
| One isolated CV syllable per recording | Total syllable duration; stop closure/release-to-voicing timing (VOT) if boundaries are visible; vowel-nucleus duration and vowel-only F0/voice measures | Mark burst, voicing onset, and vowel offset manually or with a validated aligner. Report boundary uncertainty. Do not call the five seconds “syllable duration” without evidence. |
| Sustained/reiterated vowel-like production | Stable voiced interval, F0, CPP/HNR, and vowel quality | Use the vowel protocol only for the confirmed voiced segments; do not compute jitter/shimmer across unvoiced consonant closures. |

`pa` and `ta` have different consonant places and articulatory demands, so D1-D2 is a within-person task contrast, not a repeated identical stimulus. VOT and consonant timing are meaningful only after verifying actual syllable realization and boundary conventions.

### Reading: B1, B2, FB1

B1/B2 are documented as first and second readings of the same text. FB1 is a different, unspecified set of phonemically balanced phrases. Do not treat FB1 as the same text or assume its relationship to B1/B2.

| Feature family | Proposed measure and definition | Method/settings proposal | Meaning and limits |
|---|---|---|---|
| Global timing | Total active-speech duration; pause count/rate; pause duration distribution; speech-to-pause ratio | Voice activity detection (VAD) with fixed frame/hop and thresholds, manually checked on a stratified subset; conduct threshold sensitivity review. Define pause threshold before analysis. | Relevant to speech fluency/organization, but VAD may mislabel breath, low intensity, or frication as silence. |
| Rate/rhythm | Speaking rate (syllables per total time) and articulation rate (syllables per active speech time); inter-utterance timing | Requires verified transcript/text and syllable counts; align/annotate with manual QC. Do not infer syllable counts from audio duration alone. | Potentially relevant, but text familiarity, reading errors, and task content affect it. |
| Prosody | F0 median/range/IQR over voiced speech; phrase-final F0 only if phrase boundaries are available; intensity contour only as relative within-recording dynamics | Praat/validated pitch tracking at 10 ms step; same candidate 75–500 Hz reference bounds, with voicing-track QC. Use robust summaries over voiced frames, not one global pitch mark. | Meaningful for connected speech, but depends on language, intonation, content, and tracker errors. Absolute intensity requires calibrated recording gain. |
| Voice quality | CPP/CPPS over voiced frames; optionally HNR in adequately long voiced windows | Fixed frame length, hop, pre-emphasis and pitch bounds; report valid voiced coverage. | CPP can summarize dysphonia in speech; jitter/shimmer are not proposed on whole connected-speech recordings because nonstationarity and consonants violate assumptions. |
| Articulation | Segment/syllable duration variability; consonant-vowel transitions; selected VOTs only for prespecified phonemes | Requires transcript/forced alignment and human verification; B1/B2 same text potentially permit aligned paired phoneme summaries. | Appropriate if transcripts and timing accuracy are available. Raw formants/MFCCs pooled over speech confound phonetic composition. |
| Spectral descriptors | MFCCs or spectral tilt only as secondary, clearly defined summaries | Fixed frame, window, filterbank/bandwidth and normalization; identical implementation across files. | Not directly interpretable alone; distribution reflects text/phonetic content and recording channel. Avoid as first-line hypotheses. |

**Not proposed for connected speech:** sustained-phonation local jitter/shimmer applied to the full file, sustained-vowel formant summaries pooled across all phonemes, or absolute loudness without calibration.

## 4. Provisional signal-processing and QC protocol

These settings are a design to validate in a pilot, not a completed extraction configuration.

1. **Input preservation:** read source WAVs read-only; record path, manifest and actual sample rate, bit depth, channels, duration, and software versions. Write no files into source dataset or baseline locations.
2. **Primary sampling rate:** use native 16 kHz audio only for primary HC-PD comparisons. Keep signals unnormalized for amplitude-sensitive features. Convert multichannel audio to mono only after checking channel correlation/phase cancellation; retain channel-level QC.
3. **Resampling sensitivity:** for secondary sensitivity only, resample 44.1 kHz recordings to 16 kHz with a documented anti-aliasing polyphase resampler. Record original rate and resampling ratio. Do not upsample 16 kHz signals to 44.1 kHz. Resampling does not equalize microphones, rooms, compression, or acquisition pipelines.
4. **Filtering:** no undocumented denoising, noise gate, or band-pass filter in the primary pipeline. Remove DC offset only in a derived analysis signal where required by a specific measurement; record the operation. Any mains-noise filtering must be a separate sensitivity condition.
5. **Frame settings:** for connected speech, start pilot with 25 ms analysis windows and 10 ms hop for short-time measures; freeze after manual review. For Praat pitch, use 10 ms step and the VA1 75–500 Hz bounds as the compatibility reference, with 60/90 Hz sensitivity checks. Do not change bounds selectively by diagnosis.
6. **Amplitude:** do not normalize recordings before jitter/shimmer or intensity analysis. Relative intensity contours can be summarized within file; absolute sound pressure level is unavailable without calibration.
7. **Automatic failure:** unreadable/empty file, manifest/actual sample-rate mismatch, no usable signal, or no voiced frames for a requested voiced-only measure. Preserve row and failure reason; do not silently drop it.
8. **Quality flags:** clipping, long silence/low energy, excessive noise, channel mismatch, implausible duration, pitch octave jumps, unstable formant tracks, VAD/transcript mismatch, or inadequate voiced coverage. Flags trigger review, not automatic participant exclusion. Determine any hard thresholds on blinded QC review before outcome comparisons.
9. **Manual QC:** waveform/spectrogram review on a documented participant sample spanning both groups, tasks, rates, and signal-quality range; inspect all algorithmic failures/flags. Record reviewer, segmentation decisions, and inter-rater agreement where annotation is subjective.
10. **Reliability:** distinguish extraction repeatability from biological test-retest reliability. The `2` vowel variants cannot support a test-retest claim until their protocol relationship is known.

## 5. Sample-rate strategy

| Strategy | Advantages | Disadvantages / bias risk | Proposed role |
|---|---|---|---|
| Restrict to native 16 kHz | Sampling frequency is aligned across groups; avoids interpolation and some rate-linked measurement differences; all elderly HC remain; uses the current 12 PD participants with 16 kHz candidate recordings. | Drops 7 of 19 reconciled PD participants. The group becomes 17 HC / 12 PD and may be less representative. Rate restriction does not remove microphone/site or recording-condition confounding. | **Recommended primary strategy** for group comparisons because HC are all 16 kHz and PD contain mixed rates. State the reduced sample and uncertainty. |
| Resample native 44.1 kHz to 16 kHz and include all eligible recordings | Common nominal rate and retains up to 17 HC / 19 PD for tasks with coverage; avoids upsampling. | All HC are native 16 kHz while 7 PD are native 44.1 kHz; native rate is therefore associated with group. Anti-aliasing removes >8 kHz content, interpolation/filtering can alter spectral features, and rate does not account for recording-chain differences. | **Sensitivity analysis only.** Report results by original rate and compare estimates with the native-16-kHz analysis; do not claim resampling removes rate/channel confounding. |

For features below the 8 kHz Nyquist limit, anti-aliased conversion is technically feasible, but formant/spectral choices and high-frequency breath/noise measures may respond differently to bandwidth. Compare feature-level distributions by original sampling rate within PD where overlap permits. Because rate is group-associated, rate-adjusted modeling cannot reliably identify separate rate and disease effects from these data alone.

## 6. Proposed statistical analyses

No models/classifiers are in scope for this stage.

1. **QC and missingness first:** publish task-by-group recording counts, participant counts, valid feature counts, extraction failures, and exclusion/flag reasons. Compare attrition by group and original sample rate.
2. **Participant is the unit:** reduce frames to a small prespecified set of participant-level summaries per task/feature. Frames and syllables are nested measurements, not independent participants.
3. **Group contrasts:** for each prespecified task-feature, report raw distributions, individual participant values, HC-PD effect estimate (e.g., Hedges g for roughly symmetric continuous outcomes or a rank-based effect such as Cliff's delta otherwise), and participant-level bootstrap confidence interval. Report group medians/IQR alongside means/SD where useful. Emphasize interval width over p-value thresholds.
4. **Paired task contrasts:** calculate within-participant differences for VA1-vs-VA2 etc. only after variant relationship is verified; for D1-vs-D2 compare distinct syllable contexts; for B1-vs-B2 analyze paired change with reading order explicitly. Report paired difference and interval by group. Do not label distinct vowels/syllables as repeatability trials.
5. **Across task models:** only after pilot/QC, consider a simple repeated-measures model with participant random intercept and task, group, and task-by-group terms. With these sample sizes, convergence and overfitting are risks; direct paired contrasts may be easier to audit. Any model must be prespecified and report participant-level uncertainty.
6. **Multiplicity:** choose a small primary feature set per task before looking at group effects. For exploratory secondary features, control family-wise error or false discovery rate within clearly declared feature/task families and report unadjusted estimates with adjusted p-values as supplementary context.
7. **No diagnostic interpretation:** findings concern exploratory acoustic differences in this cohort; they do not validate screening or diagnosis.

## 7. Known limitations

- Reconciled sample is only 17 HC and 19 PD before native-rate restriction, or 17 HC and 12 PD at 16 kHz.
- The 7 rate-excluded participants are all PD; sample-rate availability is not balanced across groups.
- Workbook-to-audio identity mismatches remain, including the unresolved audio `17-28/Mario B` versus workbook `Mario M`. Existing reconciliation decisions must remain unchanged.
- No participant-level task expectation schedule is available. Manifest absences cannot be called protocol failures.
- Vowel `2` variants have blank descriptions. Do not infer repetitions, trials, or longitudinal pairs.
- D1/D2 phrase “execution ... (5 sec)” does not state whether syllables are repeated, sustained, or individually produced.
- FB1 phrase set/transcripts are unavailable; FB2 is documented but absent.
- Text familiarity, reading errors, speech content, and microphone conditions may affect connected-speech measurements.
- Existing VA1 pitch sensitivity analysis documents tracker instability for some participants under alternate floors; new tasks require their own QC, not automatic transfer of conclusions.
- With 17/12 participants under the proposed primary rate restriction, effect estimates will be imprecise. No defensible power threshold can be set without external or pilot variance/effect assumptions.

## 8. Unresolved questions before extraction

1. Does the leading “2” in first-vowel codebook descriptions mean anything, and what do VA2/VE2/VI2/VO2/VU2 encode?
2. Are D1 and D2 repeated syllables over five seconds, single CV productions, or another protocol?
3. What are the exact text/phrase materials and available transcripts for FB1? What explains documented FB2's absence?
4. What do PR1/PR11 mean? Keep them excluded from primary analyses until verified.
5. Does original-rate variation correspond to separate recording equipment/session conditions beyond sampling frequency?
6. Can identity documentation resolve `17-28/Mario B` versus workbook `Mario M`? Do not alter the current exclusion pending evidence.
7. What are the actual duration, channel, bit-depth, and quality distributions by task? These must be inspected during the approved QC pilot before freezing thresholds.

## 9. Recommended experiment sequence and execution status

1. **Protocol resolution and audit lock — open.** The inventory remains preserved. Obtain task instructions/transcripts if possible; clarify vowel variants, D1/D2 production pattern, FB1 materials, and the FB2 gap. Carry unresolved items explicitly if source documentation is unavailable.
2. **QC-only pilot — completed.** Inspected 561 task recordings and a blinded 47-recording segmentation sample. QC flags and diagnostics are in the experiment output; flags did not trigger automatic exclusions.
3. **Feature-definition pilot — completed for sustained vowels; deferred for syllable/reading.** Modular task-specific code was added. First-vowel measurements underwent deterministic repeat extraction and pitch-floor sensitivity. D1/D2 and reading have QC/segmentation diagnostics only; no unsupported task measures were extracted.
4. **Primary native-16-kHz vowel analysis — completed for VA1/VE1/VI1/VO1/VU1.** Participant-level analysis retained 17 HC and 12 PD; second variants remain separate and unanalysed pending definition.
5. **D1/D2 syllable analysis — deferred.** Confirm whether productions are repeated or isolated before choosing timing measures; retain separate `/pa/` and `/ta/` labels.
6. **Reading analysis — deferred.** B1/B2/FB1 received QC and segmentation inspection only. Preserve order for B1/B2; obtain text/phrase material before text-dependent measures.
7. **Sampling-rate sensitivity — completed for first vowels.** Thirty-five eligible PD-only 44.1 kHz recordings were compared native and resampled to 16 kHz, with original-rate strata and combined sensitivity outputs. This does not remove acquisition confounding.
8. **Integrated summary — partially completed.** The report compares native and combined first-vowel estimates. Cross-task informativeness comparison is deferred until supported measurements exist for D1/D2 and reading.

**Execution boundary:** The user authorized the work package that produced the first-vowel analysis. Further large-scale extraction for second vowel variants, syllables, or reading needs protocol clarification and a separately documented scope. PR1/PR11 remain excluded until definitions are verified.
