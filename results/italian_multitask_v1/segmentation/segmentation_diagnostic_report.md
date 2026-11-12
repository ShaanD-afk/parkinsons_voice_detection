# Signal inspection and segmentation pilot

Pilot uses deterministic stratification by task, group, and original sample rate (seed 42); plots display blind IDs without diagnostic labels. The independent key is `segmentation/blind_key.csv`.

Sampled recordings: 47. RMS VAD/pitch-structure flags under provisional pilot rules: 0.

| task_family | task_code | pilot_recordings | mean_duration_s | median_active_speech_s | median_vowel_pitch_run_s | median_rms_vad_bouts | segmentation_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reading | B1 | 4 | 83.070859375 | 43.145 | 0.79 | 69.5 | 0 |
| reading | B2 | 4 | 65.88354173752835 | 41.345 | 0.8600000000000001 | 66.5 | 0 |
| reading | FB1 | 3 | 28.614095804988665 | 21.13 | 1.01 | 35.0 | 0 |
| sustained_vowel_candidate | VA1 | 3 | 11.454187499999998 | 9.290000000000001 | 7.8 | 1.0 | 0 |
| sustained_vowel_candidate | VA2 | 3 | 8.928262707860922 | 5.67 | 3.62 | 1.0 | 0 |
| sustained_vowel_candidate | VE1 | 3 | 14.425708333333333 | 6.57 | 6.43 | 2.0 | 0 |
| sustained_vowel_candidate | VE2 | 3 | 8.81 | 5.86 | 5.84 | 1.0 | 0 |
| sustained_vowel_candidate | VI1 | 3 | 11.451846041194257 | 10.18 | 8.77 | 1.0 | 0 |
| sustained_vowel_candidate | VI2 | 3 | 6.1542708333333325 | 5.51 | 5.5 | 1.0 | 0 |
| sustained_vowel_candidate | VO1 | 3 | 14.7104375 | 16.830000000000002 | 10.55 | 1.0 | 0 |
| sustained_vowel_candidate | VO2 | 3 | 8.14989947089947 | 6.11 | 6.09 | 1.0 | 0 |
| sustained_vowel_candidate | VU1 | 3 | 20.873527966742255 | 17.6 | 17.52 | 1.0 | 0 |
| sustained_vowel_candidate | VU2 | 3 | 8.594208333333333 | 5.86 | 5.84 | 1.0 | 0 |
| syllable | D1 | 3 | 6.200666666666667 | 3.11 | 0.24 | 11.0 | 0 |
| syllable | D2 | 3 | 6.255166666666667 | 1.7 | 0.19 | 7.0 | 0 |

RMS VAD is an exploratory activity segmentation, not validated forced alignment. Its active regions and bout counts do not establish a transcript, syllable count, or task protocol. Vowel stable regions are provisionally the longest contiguous run detected by the 75–500 Hz pitch tracker, trimmed by 10% at each end. This can fail for breathy, irregular, low-pitched, or unvoiced intervals; flags trigger review and are not exclusions.

Blind visual review of the sampled plots found most first-vowel examples with a long voiced portion consistent with isolated sustained phonation, alongside examples with onset decay, interruptions, or multiple voiced intervals. The pilot supports the longest-voiced-run method as a candidate only; it does not establish uniform structure for all recordings. All sampled D1/D2 plots showed repeated CV-like acoustic bursts within roughly 5–6 seconds, supporting repeated production as a description of these samples, but not verifying the full protocol or every recording. Reading examples show connected speech with pauses; RMS VAD divides speech into many short candidate bouts (median roughly 35–70 in the samples), indicating fragmentation and the need for threshold/annotation review rather than validated segmentation.

D1/D2 waveform structure is documented descriptively only. The codebook phrase 'execution of syllable ... (5 sec)' does not establish the expected repetition count/rate or full protocol. Reading pause/speech timing remains VAD-dependent. No transcript-derived rate or alignment was produced.

Inspect the blind-coded waveform/spectrogram contact sheets in this directory and individual plots under `plots/`. Background-noise differences cannot be converted to true SNR because no designated noise-only segments or calibrated recordings are available; QC reports a frame-RMS dynamic-range proxy, explicitly not SNR.
