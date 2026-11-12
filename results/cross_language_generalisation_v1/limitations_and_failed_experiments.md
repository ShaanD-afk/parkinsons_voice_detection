# Limitations, invalid comparisons, and negative results

## Invalid or withheld

- Italian VA1 / Figshare prolonged `/a/` is the only raw-audio external candidate. It is a phonetic-task match with limitations, not a confirmed language pair: Figshare language is unknown; Italian exact task duration/repetition is not documented; 16 vs 8 kHz, devices, recording channel, cohort, and demographics differ.
- Italian VA1↔MDVR reading and Figshare↔MDVR reading are invalid direct transfers: sustained vowels are not connected reading. Spontaneous dialogue is even less compatible. Four sustained-vowel-to-reading directions are included with `invalid_not_comparable` status and no metrics.
- UCI feature tables cannot be used as raw-audio transfer tests because local waveforms and common extraction are absent. UCI #189 is PD-only and cannot support binary HC/PD discrimination.
- Figshare is not labeled English. No cross-language claim is made.

## Negative and inconclusive results

- Zero-shot Italian→Figshare and Figshare→Italian scores are weak and largely near chance despite higher within-cohort CV in Italian.
- Diagnosis-matched dataset classification remains above chance, showing domain-specific feature structure.
- Group-effect directions disagree for F0 variability and jitter. Shimmer's `/a/` direction is not stable enough to infer a general disease feature. No raw-audio result replicates Italian VI1 `/i/` shimmer.
- Matching sample rate by downsampling Italian waveforms produces no meaningful restoration of transfer; channel/device differences remain.
- High nested CV values on the small Italian and MDVR cohorts have wide participant-bootstrap intervals and are exploratory. Fixed-prediction intervals omit training/fold variability.

## Measurement and metadata limitations

- Figshare provides 8 kHz telephone audio and age/sex but not language certainty in the available item metadata. Recording frequency response and device metadata are not harmonized.
- Italian VA1 codebook phrase (“2 phonation of the vocal ‘a’”) identifies the phonation target but not duration/repetition details.
- MDVR connected-speech features use unreviewed candidate VAD. Dialogue may contain examiner speech. Its classifier is a provisional within-reading baseline only; not an external task match.
- UCI #174 local rows yield 32 IDs / 24 PD / 8 HC, unlike the UCI source page's 31 / 23 / 8. UCI #301 feature values are not re-extracted.
- Cross-dataset participant identity linkage is unavailable. Dataset-scoped hashed IDs ensure split bookkeeping; they cannot detect the same real individual appearing in distinct source corpora.
- No language-level causal contrast is identifiable because dataset and acquisition are entangled.

No participants were dropped for inconvenient values. Extraction failures, QC flags, implausible-value flags, and IQR outlier flags are retained. No external cohort was used for adaptation.
