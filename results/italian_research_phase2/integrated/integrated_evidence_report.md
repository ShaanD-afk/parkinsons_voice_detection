# Integrated Italian multi-task evidence report

This synthesis combines only measurements with explicit task labels and preserves their separate measurement definitions. It does not rank tasks by p-value or treat unlike effect sizes as directly comparable.

| Task family | Current participant coverage (HC / PD) | Measurement status | Integrated conclusion |
|---|---|---|---|
| First sustained vowels (VA1/VE1/VI1/VO1/VU1) | Native 16 kHz: 17 / 12 per vowel; all rates: 17 / 19 | Praat extraction, deterministic repeat extraction and participant-level analysis completed | VI1 shimmer is the only primary contrast surviving original 20-test Holm correction; direction robust to individual influence and robust summaries, but adjusted inference varies with test. |
| Second vowel variants (VA2/VE2/VI2/VO2/VU2) | 17 / 11 native 16 kHz; 17 / 18 mixed rate per code | Duration, voicing, F0 and morphology described separately | Second-code duration medians are shorter; relationship to first variants is unknown. Do not treat as repetitions. |
| D1 /pa/, D2 /ta/ | Native 16 kHz: 17 / 12 each; all rates: 17 / 19; all participants paired | All-recording diagnostics and candidate energy-event pilot; human onset validation pending | Repeated CV-like bursts are apparent in most recordings. Candidate events are not validated syllable counts, so no participant-level timing contrasts yet. |
| B1/B2 reading | B1: 17 / 19; B2: 17 / 16 (mixed rate); native 16 kHz B2 PD n=9 | Generic speech-activity, pitch, intensity and HNR features calculated at fixed threshold plus threshold sensitivity | Mask counts depend on VAD threshold; paired B1/B2 changes remain descriptive until masks are reviewed. Practice/fatigue/order effects cannot be separated. |
| FB1 phrases | 17 / 18 mixed rate; 17 / 11 native 16 kHz | Generic connected-speech candidates only | Kept separate; no transcript or materials for phonetic/syllabic measures. |

## Features and reliability

First-vowel measurements reproduce exactly under a second extraction with the same software and configuration. That is extraction repeatability, not test-retest reliability. No task currently has biological or inter-rater reliability established. Median F0, voiced fraction, and HNR can be computed in connected speech, but current segment validity is unresolved. Sustained-vowel jitter/shimmer are not extended to B1/B2/FB1. Formant range flags were zero of 145 first-vowel rows, but manual overlays remain useful for targeted review.

## Sample-rate sensitivity

Native-16-kHz group contrasts are primary because all HC are native 16 kHz and all extra 44.1 kHz recordings are PD. Adding resampled PD recordings left the VI1 shimmer mean contrast nearly unchanged, but changed Hedges g modestly and cannot eliminate acquisition confounding. Paired-rate measures indicate larger shifts in spectral centroid/rolloff than F0/voicing for seven PD same-recording pairs per vowel. Cross-task comparisons must therefore retain rate and matched-participant constraints.

## Participant influence and multiplicity

For VI1 shimmer, all leave-one-person-out mean contrasts remain positive; the median, 20% trimmed mean, 10% winsorized contrast and percentile/BCa bootstrap intervals also remain positive. Holm-adjusted Mann-Whitney p is 0.012 in the native cohort, while the adjusted 20% trimmed-mean/Yuen p is 0.061. In the combined-resampled sensitivity, those values are about 0.003 and 0.037, respectively, but that analysis adds PD-only records. The substantive direction is more stable than the precise inferential threshold. No participant was removed.

## Complementarity and decision

At present, the sustained vowels provide the only validated participant-level group comparison. D1/D2 may add articulation timing after manual segmentation validation. Reading tasks may add pause/prosodic information after VAD mask validation, but their text and recording conditions are not fully harmonized with English corpora. Second-code morphology adds a protocol question rather than an established complementary feature set. No integrated multi-task model, task ranking, or classifier was built. The Italian multi-task phase is **not ready to be called complete** until human audio review and protocol resolution permit valid D1/D2 and reading measures or document their final deferral.
