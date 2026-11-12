# Second vowel variant signal characterization

The codebook description is blank for VA2/VE2/VI2/VO2/VU2. This report does not call them repetitions, trials, or test-retest recordings.

## Coverage and descriptive comparison

The analysis characterized all 355 reconciled first- and second-code recordings (five first variants at 17 HC/19 PD each, with 16 kHz and 44.1 kHz retained; five second variants at 17 HC/18 PD each, also with original rates retained). For each recording it reports duration, RMS and peak amplitude, pitch-voiced fraction, voiced-run count, longest voiced run, median F0, and F0 IQR. The output includes 875 matched first-versus-second code differences (five vowel contexts × five descriptive measures × 17 HC/18 PD participants), with sample rate retained in the recording-level rows. Summaries are under `signal_summary_by_code.csv` and `paired_variant_descriptive_summary.csv`.

Across all five vowel contexts, native-16-kHz first-code duration medians are generally around 11–15 seconds, while second-code medians are around 5.8–6.4 seconds. This structural difference is visible in the recordings and makes it especially unsafe to combine the codes as equivalent trials. It could reflect different protocol content, segmentation, or file boundaries; the codebook does not resolve which explanation applies. Pitch and voicing descriptors also vary by vowel, group and sample-rate stratum. No test-retest reliability coefficient, repeatability claim, or inferential “second versus first” effect is reported.

All 17 elderly-HC participants with first variants have the corresponding second variant. For PD, 18 participants have a second variant (11 native 16 kHz plus 7 at 44.1 kHz); 19 have a first variant (12 native 16 kHz plus 7 at 44.1 kHz). The 44.1 kHz recordings are PD-only. The descriptive paired table preserves original rates and code identities.

## Interpretation boundary

The similarity of long voiced portions in many first and second code recordings is not proof of identical task instructions. The difference in duration distributions suggests the code pairs should remain separate pending protocol documentation. PR1/PR11 remain outside the analysis. No participant was excluded and no classifier was trained.
