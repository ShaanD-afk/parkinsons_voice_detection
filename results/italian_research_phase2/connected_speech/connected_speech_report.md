# B1/B2/FB1 connected-speech characterization

## Population and implementation

This is an exploratory, participant-recording-level analysis of the reconciled elderly-HC/PD cohort. The same participant appears at most once per task. Sample-rate strata are retained in every summary. Generic VAD uses 25 ms RMS frames and 10 ms hop, with a per-recording threshold of −35 dB relative to the 95th percentile of frame RMS; a single isolated inactive frame is closed. Threshold sensitivity is reported at −30, −35, and −40 dB. Pitch uses Praat autocorrelation at 75–500 Hz and 10 ms step. Relative intensity is measured from log RMS. HNR is calculated in active segments of at least 150 ms and summarized over valid Praat frames. CPP/CPPS is deferred because no validated implementation was available. Whole-recording connected speech jitter/shimmer were not applied.

| Task | HC native 16 kHz | PD native 16 kHz | PD native 44.1 kHz | Common participants with B1/B2 (mixed rate) |
|---|---:|---:|---:|---:|
| B1 | 17 | 12 | 7 | 17 HC / 16 PD |
| B2 | 17 | 9 | 7 | 17 HC / 16 PD |
| FB1 | 17 | 11 | 7 | Separate; not treated as same content |

The exact text for B1/B2 is documented as “Il ramarro della zia,” with B1 first and B2 second reading. FB1 is kept separate: its exact phrases and transcript are unavailable. Seven PD participants contribute only 44.1 kHz recordings; all HC recordings are native 16 kHz. Native-rate strata are never combined invisibly.

## Candidate feature coverage

`participant_recording_features.csv` contains one candidate row per participant-task-rate at −35 dB. `vad_threshold_sensitivity_long.csv` contains three thresholds per recording. The output includes active speech duration/proportion; internal pause count, duration, median and rate; speech-to-pause ratio; median F0, F0 IQR/range, pitch SD and voiced-frame proportion; relative-intensity IQR/SD; and median HNR in suitable voiced-active windows. B1/B2 matched within-person differences are in `b1_b2_paired_descriptives.csv`, stratified by group and original sample rate. They are descriptive order contrasts; practice, fatigue, learning, and task order effects cannot be separated.

## VAD threshold sensitivity

Task-level medians demonstrate meaningful threshold dependence. The selected threshold changes both active proportion and the number of pause boundaries:

| Task | Active proportion at −30 dB | at −35 dB | at −40 dB | Pause count at −30 dB | at −35 dB | at −40 dB |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.695 | 0.742 | 0.774 | 85 | 73.5 | 65.5 |
| B2 | 0.730 | 0.767 | 0.806 | 79 | 61 | 56 |
| FB1 | 0.650 | 0.689 | 0.723 | 46 | 36 | 30 |

These are medians across recording-level values, pooling the displayed task's HC/PD and rate strata. Pitch summaries are computed on Praat-voiced frames across the recording and are therefore not a validation of the VAD mask. Relative intensity dynamics also shift with threshold because the active-frame set changes.

## QC and limitations

A stratified set of 22 B1/B2/FB1 masks (task × group × original-rate strata, up to two recordings per stratum) has waveform, spectrogram, and three-threshold RMS plots in `stratified_vad_review_plots/`; it is also represented in the human review queue. Existing pilot plots show that read-speech masks can fragment into many short bouts. Automated pause counts consequently cannot be interpreted as linguistic pauses until a reviewer checks masks against audio.

No transcript or verified text alignment was used. Speaking rate, syllable rate, errors, phonetic alignment, and lexical/phonetic content measures are not reported. HNR in voiced windows may still be sensitive to the segment and pitch tracker. No connected-speech group tests are treated as supported until manual mask review and reliability checks are completed. CPP/CPPS remains deferred. This is not a diagnostic or clinical analysis.
