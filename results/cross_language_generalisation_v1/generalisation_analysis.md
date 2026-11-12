# Cross-dataset generalisation and dataset effects

## Zero-shot transfer result

Italian VA1 (17 HC/12 PD) and Figshare sustained `/a/` (41 HC/40 PD) are comparable only with explicit limitations. The best tested 16 kHz Italian→8 kHz Figshare core-5 balanced accuracy was **0.551**; ROC-AUC maximum was **0.501**. The reverse best native-rate core-5 balanced accuracy was **0.569**, with maximum ROC-AUC **0.618**. For the same model, the internal score was generally higher than external performance, but all tested configurations and individual results are in the CSV, not just the maxima.

Thus ItalianPVS-trained models did not generalise usefully to Figshare in this zero-shot experiment, and Figshare-trained models did not generalise usefully to Italian VA1. This is **cross-dataset failure**, not proof that language transfer fails: Figshare language is unknown and dataset/device/rate/cohort are confounded.

## Sample-rate sensitivity

The Italian 16→8 kHz copies were tested separately. Aligning nominal sample rates did not make cross-dataset performance reliable: resampled transfer balanced accuracy remained between 0.382 and 0.537. The reverse Figshare→Italian resampled sensitivity achieved its best balanced accuracy 0.598. Resampling cannot restore telephone channel response or make recording chains equivalent.

## Dataset identity

To avoid using differing case proportions as an easy domain cue, each of 30 draws retained 17 HC and 12 PD from both datasets (58 participants total). A diagnosis-stratified five-fold CV classified dataset identity using only the five common acoustic features. Median balanced accuracy across matched draws was:

| model | median | minimum | maximum |
| --- | --- | --- | --- |
| Logistic_scaled | 0.672 | 0.603 | 0.741 |
| RandomForest | 0.655 | 0.534 | 0.810 |
| SVM_RBF_scaled | 0.707 | 0.638 | 0.793 |

Feature patterns can identify cohort origin above chance after observed group-count matching. In a fixed, fold-standardized logistic ablation over the same 30 matched draws, removing F0 IQR reduced median dataset-identification balanced accuracy from 0.672 to 0.603 (median change −0.095); removing jitter reduced it to 0.612 (−0.069). Removing median F0 slightly increased the median to 0.690 (+0.017), while removing shimmer or HNR left the median near 0.672. F0 IQR alone reached 0.621; no single feature reproduced the full score. Fold-standardized logistic coefficients were most consistently negative for F0 IQR and jitter in the dataset-origin model. These ablations indicate which features carry domain information in this sample; they do not establish why or a causal recording mechanism. See `tables/dataset_identity_feature_ablation_summary.csv`, coefficient table, OOF predictions, and Figure 6.

The origin classifier does not establish what causes that separability; plausible sources include recording rate/device/bandwidth, cohort demographics, phonation protocol, and residual diagnosis effects. No causality is claimed.

## Disease-feature direction and robustness

No common sustained `/a/` feature produced a convincing corrected group effect in both cohorts. F0 direction agreed but magnitude and uncertainty differed. F0 variability and jitter directions disagreed. Shimmer was directionally aligned but close to zero in Figshare. HNR was similarly directed but weak in Figshare. Adjusted age/sex models and feature tests are reported with their assumptions; none disentangles the dataset acquisition factors.

Native-to-8 kHz Italian paired feature rank correlations were high, but HNR changed systematically. This sensitivity comparison tests resampling in the same 29 waveforms only; it cannot estimate telephone-vs-microphone domain effects.

## VI1 shimmer

The historical Italian VI1 `/i/` shimmer observation remains exploratory and method-dependent. No local external cohort provides a matched sustained `/i/` task with comparable raw measurements, so this experiment does not replicate it. VA1 `/a/` comparisons are not a substitute for an `/i/` replication.

## Interpretation

The most informative negative finding is that good within-cohort CV does not imply transfer. Current data can detect dataset origin more consistently than they support stable diagnosis transfer. No language-general acoustic biomarker or clinical classifier is established.

## Next experiment that would improve the evidence

Collect a prospectively documented repeated-token sustained `/a/` and `/i/` cohort in both language groups using the same microphone, calibrated distance, native sampling rate, room instructions, and locked feature code. Balance age/sex where feasible, retain participant IDs, and reserve one full cohort for independent testing. This would separate language/task claims from device/rate effects and permit a true `/i/` shimmer replication.
