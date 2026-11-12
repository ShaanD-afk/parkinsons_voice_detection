# VA1 OOF participant uncertainty

This experiment summarizes uncertainty in the existing VA1 out-of-fold predictions. It does not modify or rerun the VA1 baseline.

- Participants: 29 (17 HC, 12 PD)
- Bootstrap: 20,000 class-stratified participant resamples; seed 20261031
- Predictions were held fixed; no classifier was fitted.

## Results

| Metric | Estimate | 95% interval |
|---|---:|---:|
| balanced_accuracy | 0.900 | [0.775, 1.000] |
| roc_auc | 0.917 | [0.789, 1.000] |
| sensitivity | 0.917 | [0.750, 1.000] |
| specificity | 0.882 | [0.706, 1.000] |

## Interpretation and limits

The intervals describe participant-sampling variability conditional on the saved cross-validated predictions. Participants' predictions share fitted fold models, so this bootstrap does not capture model refitting or fold-assignment variability. It is not an external validation interval and does not support diagnostic claims. The small cohort and internal cross-validation remain substantial limits.

## Reproduction

Run `.venv/bin/python src/va1_oof_uncertainty.py --output results/va1_oof_uncertainty_reproduction` from the project root to reproduce in a fresh directory. Inputs are read-only; the runner refuses to overwrite a non-empty output directory.
