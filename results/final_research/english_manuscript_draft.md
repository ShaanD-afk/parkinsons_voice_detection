# Independent voice-feature analyses across English-language and related Parkinson datasets

## Abstract

This exploratory work audited local Parkinson voice datasets and analyzed participant-level acoustic differences where raw audio and labels were available. Figshare supplied 81 sustained `/a/` phone recordings linked one-to-one with demographic workbook IDs. MDVR-KCL supplied 73 mono 44.1 kHz recordings from 37 people (21 HC, 16 PD), in reading and spontaneous dialogue. Praat-derived sustained-vowel F0, jitter, shimmer and HNR were measured in Figshare; candidate activity, pause, F0, voicing, HNR and spectral summaries were measured separately for MDVR tasks. UCI 174, 189 and 301 were audited as precomputed feature datasets; their repeated records were aggregated or modeled within participants. The MDVR dialogue and pause measures remain unvalidated, UCI 174 local participant counts differ from the source description, and UCI feature tables do not permit raw-audio harmonization. These data support independent exploratory estimates, not external clinical validation or a pooled cross-language model.

## Introduction and objectives

The purpose was to determine whether locally available non-Italian voice resources can supply independent participant-level acoustic evidence and which are suitable for comparison with ItalianPVS. Analyses were limited to task-appropriate measurements and did not treat multiple recordings as independent participants.

## Dataset and participants

Figshare: 81 WAVs, 41 HC/40 PD, one workbook-linked Sample ID per file, 8 kHz mono FLOAT. The source describes prolonged `/a/` by telephone. MDVR-KCL: 73 WAVs, 37 unique speakers, 21 HC/16 PD, 44.1 kHz mono PCM_24; 37 reading and 36 dialogue files, with one PD absent from the local reading task. The dialogue filename ID22 separator anomaly is documented. UCI 174: 195 feature rows, 32 local participant IDs (24 PD/8 HC); source discrepancy versus the UCI page is retained. UCI 189: 5,875 longitudinal PD voice-feature rows and 42 people. UCI 301: 1,040 training task rows, 40 people (20/20), plus a separate 168-row PD-only repeated vowel table. No raw audio exists locally for these UCI files. UCI 470 and mPower were not locally analyzed.

## Methods

Figshare sustained vowel measurements used the same candidate stable-voiced interval and Praat definitions as the Italian vowel pipeline: 75 Hz floor/500 Hz ceiling, 10 ms pitch step, longest voiced run with 10% end trimming; local jitter and shimmer; Praat CC HNR. The 8 kHz native signal was not upsampled. Age/sex-adjusted linear associations used HC3 standard errors as a confounding sensitivity check.

MDVR read and spontaneous tasks were analyzed separately. Original 44.1 kHz audio was polyphase-resampled to 16 kHz for the analysis copy. RMS VAD used 25 ms frames, 10 ms hop, a provisional −25 dB threshold relative to the recording p95 frame RMS, with 10 ms hole closure; sensitivity at −20/−25/−30/−35 dB is reported. Pauses were gaps of at least 250 ms. F0 and HNR were summarized only in candidate active/voiced windows; intensity IQR and spectral centroid/rolloff were computed on active windows. We did not extract sustained-phonation jitter/shimmer from dialogue or reading.

UCI 174 repeated feature rows were reduced to participant medians. UCI 301 training contrasts used one subject-task row for sustained `/a/`, `/o/`, `/u/`; tasks remained separate. The independent UCI 301 test partition was PD-only and used only for descriptive within-person repetition spread. UCI 189 used each participant’s repeated measurements to compute within-person Spearman feature/UPDRS association; the distribution across people, not 5,875 rows, was summarized. Group tests report effect sizes and participant-bootstrap intervals with within-dataset/task Holm correction. No models were trained.

## Results

See the participant-level effect table in `english_results.md` and full CSV outputs under `english_tables/`. The Figshare and MDVR group contrasts are task-specific exploratory estimates with wide intervals and potential demographic/channel confounding. Dialogue results may mix participant and examiner speech. UCI 174 can inform historical feature-table direction only, and its local count conflict (32 IDs vs source page 31) requires resolution. UCI 301 training data offer separate `/a,o,u/` precomputed-feature contrasts; the repeated PD-only test file cannot estimate group differences. UCI 189 adds severity-association context but no HC comparison. No local UCI 470 data or mPower data were available.

## Discussion and limitations

Raw-audio Figshare and MDVR are suitable for careful exploratory independent analysis. Figshare is not a confirmed language-specific English reading/speech corpus; it is a sustained phone `/a/` task. MDVR supports English reading and dialogue but is small and lacks detailed demographics locally; dialogue voice attribution is not solved. The UCI sets add sample structures but use precomputed features and cannot reproduce a common extraction method. The UCI 174 participant-count discrepancy, UCI 301 language/protocol metadata gaps, incomplete task availability, age/sex/device effects and no clinical holdout preclude definitive inference. The local analysis did not externally validate Italian `/i/` shimmer; no comparable `/i/` task exists.

## Conclusion

The current English-language evidence is useful for separate, low-dimensional participant-level effect estimation and method planning. It is not yet adequate to claim replication or generalization. A common controlled protocol and fully documented independent cohort are required for cross-language validation.

## References

1. Figshare. Voice Samples for Patients with Parkinson’s Disease and Healthy Controls. [Dataset item 23849127](https://doi.org/10.6084/m9.figshare.23849127); license CC BY 4.0.
2. Jaeger H, Trivedi D, Stadtschnitzer M. Mobile Device Voice Recordings at King’s College London (MDVR-KCL). Zenodo, 2019. [doi:10.5281/zenodo.2867216](https://doi.org/10.5281/zenodo.2867216).
3. Little MA. Parkinsons [Dataset]. UCI Machine Learning Repository, 2007. [doi:10.24432/C59C74](https://doi.org/10.24432/C59C74).
4. Tsanas A, Little MA, McSharry PE, Ramig LO. Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests. IEEE TBME, 2010. [UCI 189 dataset](https://doi.org/10.24432/C5ZS3N).
5. Erdogdu Sakar B et al. Collection and Analysis of a Parkinson Speech Dataset With Multiple Types of Sound Recordings. IEEE JBHI, 2013. [doi:10.1109/JBHI.2013.2245674](https://doi.org/10.1109/JBHI.2013.2245674); [UCI 301](https://doi.org/10.24432/C5NC8M).
6. Sakar C et al. Parkinson’s Disease Classification [Dataset]. UCI, 2018. [doi:10.24432/C5MS4X](https://doi.org/10.24432/C5MS4X).
