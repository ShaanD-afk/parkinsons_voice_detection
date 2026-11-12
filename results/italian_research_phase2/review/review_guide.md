# Human review guide and current status

Launch the interface using the command in the result-directory README. The queue contains **159 distinct recordings** and currently has **0 completed / 159 unresolved** annotations. Automated QC and prior pilot flags remain unchanged; no participant was excluded.

## Why each file is queued

- **19 automated QC flags:** 17 possible-clipping recordings and 2 vowel recordings with high silence fraction (PD Antonia G / VA1 and PD Giustina M / VO1). Most clipping flags involve HC AGNESE P and GILDA C; low clipping fractions can arise from a small number of full-scale samples, while several GILDA C reading/syllable recordings have materially larger clipping fractions. Inspect the actual signal and listen; flags alone are not exclusion grounds. The exact file paths are in `human_review_queue.csv`.
- **6 existing VA1 pitch-sensitivity cases:** AGNESE P, GILDA C, GRAZIA G, Giovanni M, GIOVANNA G, and Giustina M. Their previous setting comparisons range from probable subharmonic tracking to ambiguous F0 spread. These require listening with F0 overlay; they are not automatic exclusions.
- **72 reconciled D1/D2 recordings:** repeated CV-like bursts appear in the diagnostics, but event peaks have not been matched to manually marked syllable onsets. Please record whether the segmentation is correct, the manual event onsets (seconds), a manual syllable count, whether F0 tracking looks plausible, uncertainty, and comments. Retain /pa/ and /ta/ task labels.
- **47 prior blinded segmentation-pilot recordings:** these ensure the existing vowel, syllable, and reading masks receive review. Pilot samples overlap other reasons in the deduplicated queue.
- **22 connected-speech threshold checks:** a stratified subset across B1/B2/FB1, group, and original rate (including YHC only for descriptive mask review) has −30/−35/−40 dB threshold plots. Mark whether the candidate activity mask is acceptable and record what fails in comments.

The queue merges reasons when one recording has more than one; counts above therefore sum to more than 159. The formant plausibility flags in the existing first-vowel feature table are all true (0 automatic range flags of 145 rows). The interface still plots formants to support manual inspection. Every annotation is stored separately in `manual_annotations.csv`; `review_status.csv` and `review_queue_summary.csv` report completion state. Do not infer a required task schedule or remove participants based on the flags.
