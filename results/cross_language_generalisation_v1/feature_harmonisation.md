# Feature harmonisation and signal processing

## Shared raw-audio feature implementation

The same new experiment code re-extracted both cohorts from original waveforms. It uses `mono_for_analysis` and the previously established candidate stable-interval method: Praat autocorrelation, longest contiguous voiced run, and trim 10% from both ends. The primary reference pitch floor/ceiling are 75/500 Hz and 10 ms step. Jitter, shimmer, and HNR settings match the existing stable-vowel extractor. No normalization or amplitude gain correction was applied before jitter/shimmer. Participant records remain in the table even when an extraction fails; there were **zero failed extractions** among the 110 primary recordings. Automated QC flags and implausibility/outlier flags never cause exclusion.

| Feature | Definition and use |
|---|---|
| median_f0_hz | Median positive F0 in candidate stable vowel segment; Praat autocorrelation; 10 ms step; pitch floor 75 Hz; ceiling 500 Hz; Meaningful for sustained phonation; limitation: Pitch tracking can octave-error; floor sensitivity evaluated at 60/90 Hz |
| f0_iqr_hz | 75th minus 25th percentile of positive F0 frames; Same stable segment and pitch track; Meaningful as within-segment pitch variability; short token limits reliability; limitation: IQR can be dominated by tracking failures and task duration |
| jitter_local | Praat local period perturbation; PointProcess periodic cc; floor 75 Hz/ceiling 500; period bounds .0001–.02 s; period factor 1.3; Meaningful only on adequate, stationary voiced vowel segment; limitation: Sensitive to cycle detection, noise, bandwidth and duration |
| shimmer_local | Praat local cycle-amplitude perturbation; Same pulse sequence; amplitude factor 1.6; Meaningful only on adequate, stationary voiced vowel segment; limitation: Sensitive to gain, microphone, compression and segmentation |
| hnr_mean_db | Mean Praat CC harmonicity over stable interval; 10 ms step; minimum pitch 75 Hz; silence threshold .1; one period/window; Voice periodicity summary for vowel phonation; limitation: Bandwidth and noise-floor dependent; 16→8 kHz check performed |

The `core4` classifier set is median F0, local jitter, local shimmer and HNR. `core5` adds F0 IQR. These limited feature sets were selected before looking at classifier results; no supervised feature selection was performed. Imputation and scaling, when used, are fitted inside each training fold. Models marked unscaled use median imputation only. Random Forest receives raw feature units by design.

## Sample-rate and bandwidth checks

- Native comparison preserves Italian 16 kHz and Figshare 8 kHz signals. No upsampling was applied to Figshare.
- A separate polyphase 16→8 kHz signal copy was generated for each Italian VA1 waveform under `analysis_copies/italian_va1_8khz/`. It tests processing-rate sensitivity on the same 29 people; it cannot recreate telephone microphone bandwidth, codec, noise, or channel effects.
- The Italian downsampling comparison preserves very high participant rank correlations, but HNR shifts systematically (Holm-adjusted paired Wilcoxon p≈0.00007). Rates being numerically equal is not evidence of acquisition equivalence.
- Pitch-floor setting sensitivity was evaluated at 60, 75, and 90 Hz in both raw cohorts. One Figshare record failed at the 90 Hz floor; it remains present in extraction/failure tables and is not excluded from the primary 75 Hz analysis.

Detailed output: `tables/harmonised_raw_audio_features.csv`, `tables/all_pitch_floor_extractions.csv`, `tables/feature_setting_sensitivity.csv`, `tables/italian_resampling_sensitivity.csv`, `tables/raw_audio_qc.csv`, and `tables/quality_outlier_flags.csv`.
