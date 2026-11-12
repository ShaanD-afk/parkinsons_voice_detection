# D1/D2 acoustic task characterization

**Experiment:** `ITALIAN-PHASE2-REMAINING-20261030`. **Population:** reconciled elderly HC/PD, 17 HC and 19 PD across mixed original rates; 17 HC and 12 PD at native 16 kHz. **Files:** 36 D1 plus 36 D2, with all 36 participants having both task codes. No participants were excluded.

## Evidence from documentation and recordings

The supplied FILE CODES workbooks label D1 as execution of /pa/ for five seconds and D2 as execution of /ta/ for five seconds; the wording does not say how many syllables to produce. A published description of ItalianPVS lists /pa/ for five seconds, a 20-second pause, and /ta/ for five seconds, but likewise does not specify the production count or onset/offset rules ([Frontiers in Physics, 2022](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.1048833/full)). This supports the separation of the `/pa/` and `/ta/` codes but does not resolve the exact repetition instruction.

Waveform and spectrogram diagnostics were generated for all 72 recordings in `diagnostic_plots/` and summarized in `d1_all_recordings_contact_sheet.png` and `d2_all_recordings_contact_sheet.png`. Across the set, the signals show multiple temporally separated, vowel-like energy/spectral bursts. The visual pattern is consistent with repeated CV-like production in most recordings, rather than one isolated CV token. There is variability in burst count, spacing, loudness decay, pauses, and recording duration. This is acoustic evidence only; it does not prove participants followed a particular instruction or that every detected burst corresponds to a syllable.

## Automated segmentation pilot

`recording_characterization.csv` includes every D1/D2 recording. Its exploratory detector smooths a 10 ms RMS envelope over 50 ms, estimates an adaptive floor from the lower envelope and signal maximum, then finds local energy peaks with 120 ms minimum spacing and an RMS/prominence requirement. The output includes candidate event times/counts, candidate inter-event intervals and coefficient of variation, thresholded active fraction (and its complement as a pause fraction), recording duration, and pitch-voiced fraction. These are expressly named candidate events; the detector has not been shown to count syllables.

| Task | Group | n | Median duration (s) | Median candidate peaks (range) | Median thresholded pause fraction |
|---|---|---:|---:|---:|---:|
| D1 | elderly HC | 17 | 6.84 | 9 (6–14) | 0.581 |
| D1 | PD | 19 | 5.88 | 7 (6–12) | 0.578 |
| D2 | elderly HC | 17 | 6.57 | 9 (5–11) | 0.580 |
| D2 | PD | 19 | 5.75 | 7 (6–14) | 0.608 |

The candidate peaks and pause fractions depend on the amplitude threshold and should not be used as validated timing measures. The observed durations cluster near or above the five-second codebook duration, with extra leading/trailing silence and variation. The 44.1 kHz recordings are PD only, so they are not used for HC-PD inference here.

## Review and decision

The repeated-production interpretation is supported strongly enough by the signal morphology to justify an event-segmentation pilot, but the pilot is not validated. All 72 reconciled D1/D2 recordings are in the human review queue so a reviewer can mark recording acceptability, segmentation correctness, uncertainty, and comments. No manually labeled examples are yet available; therefore precision/recall, timing error, participant-level syllable count, interval variability, and paired D1/D2 statistics are deferred. After review, the event definition should be adjudicated against manually located syllable onsets and re-evaluated without tuning to diagnosis labels.

The candidate features for a later validated repeated-production analysis are syllable count, median inter-onset interval, interval variability, pause fraction, and timing variability, with `/pa/` and `/ta/` retained as distinct contexts. If review instead finds isolated CV tokens in a subset, those files require a separate burst/VOT/vowel-duration protocol; the two protocols must not be mixed. No classifier was trained.
