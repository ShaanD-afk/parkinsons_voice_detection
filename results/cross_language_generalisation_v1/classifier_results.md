# Classifier results

Classical models only: Logistic Regression with and without training-fold standardization, RBF SVM with training-fold standardization, and Random Forest. Gradient Boosting was not added: the smallest raw cohort has 29 people and only five pre-specified inputs; adding another tuned model family would increase selection variance without a clear scientific gain. No deep model was trained.

## Internal evaluation

Within-dataset estimates use outer stratified 5-fold CV and 3-fold inner grid search; imputation, scaling, and model selection occur only inside training folds. The OOF prediction table has one prediction per participant/model/feature set. Intervals resample participants while holding those OOF predictions fixed; they do not account for fold assignment or refitting uncertainty. `tables/within_dataset_classifier_results.csv` contains all 36 model/feature-set evaluations (including confusion counts and intervals). Its top observed score is not an unbiased selected-model estimate.

Highest balanced-accuracy run per dataset (descriptive selection from the tested grid):

| training_dataset | model | feature_set | n_hc | n_pd | balanced_accuracy | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | sensitivity | specificity | roc_auc | roc_auc_ci95_low | roc_auc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ItalianPVS_VA1 | Logistic_scaled | core4 | 17 | 12 | 0.858 | 0.716 | 0.971 | 0.833 | 0.882 | 0.863 | 0.706 | 0.980 |
| MDVR-KCL_read_text | Logistic_scaled | task_specific | 21 | 16 | 0.827 | 0.695 | 0.945 | 0.750 | 0.905 | 0.845 | 0.690 | 0.973 |
| UCI174_feature_table | Logistic_scaled | task_specific | 8 | 24 | 0.729 | 0.562 | 0.854 | 0.583 | 0.875 | 0.745 | 0.562 | 0.896 |
| Figshare_sustained_a | RandomForest | core5 | 41 | 40 | 0.679 | 0.579 | 0.778 | 0.650 | 0.707 | 0.695 | 0.571 | 0.810 |
| UCI301_u_feature_table | SVM_RBF_scaled | task_specific | 20 | 20 | 0.625 | 0.475 | 0.775 | 0.750 | 0.500 | 0.640 | 0.458 | 0.810 |
| UCI301_a_feature_table | Logistic_scaled | task_specific | 20 | 20 | 0.600 | 0.450 | 0.750 | 0.650 | 0.550 | 0.640 | 0.463 | 0.812 |
| UCI301_o_feature_table | Logistic_unscaled | task_specific | 20 | 20 | 0.525 | 0.375 | 0.675 | 0.600 | 0.450 | 0.518 | 0.330 | 0.710 |

The Italian nested estimates use four/five harmonized features, so they are not a reproduction of the historical six-feature VA1 pipeline or its fixed OOF result. Figshare internal performance is near chance for several configurations. MDVR's internal reading numbers are provisional because its VAD masks have not been human-validated. UCI internal numbers are based on historical feature tables and separate task-specific feature definitions.

## Genuine external zero-shot evaluation

The source cohort alone was used for hyperparameter search and final model fitting. External target labels did not enter imputation, scaling, feature selection, tuning, or threshold choice. The decision threshold is the model default. Bootstrap intervals are conditional on predictions for these fixed target participants.

Native-rate transfers, using both pre-specified feature sets:

| training_dataset | test_dataset | task_compatibility | model | feature_set | training_participants | training_n_hc | training_n_pd | test_participants | test_n_hc | test_n_pd | balanced_accuracy | balanced_accuracy_ci95_low | balanced_accuracy_ci95_high | sensitivity | specificity | roc_auc | roc_auc_ci95_low | roc_auc_ci95_high | evaluation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | Logistic_scaled | core4 | 29 | 17 | 12 | 81 | 41 | 40 | 0.525 | 0.476 | 0.575 | 0.075 | 0.976 | 0.389 | 0.266 | 0.513 | external_zero_shot |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | Logistic_unscaled | core4 | 29 | 17 | 12 | 81 | 41 | 40 | 0.468 | 0.368 | 0.578 | 0.350 | 0.585 | 0.468 | 0.343 | 0.592 | external_zero_shot |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | SVM_RBF_scaled | core4 | 29 | 17 | 12 | 81 | 41 | 40 | 0.551 | 0.477 | 0.626 | 0.200 | 0.902 | 0.491 | 0.363 | 0.617 | external_zero_shot |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | RandomForest | core4 | 29 | 17 | 12 | 81 | 41 | 40 | 0.526 | 0.464 | 0.588 | 0.125 | 0.927 | 0.484 | 0.359 | 0.613 | external_zero_shot |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | Logistic_scaled | core5 | 29 | 17 | 12 | 81 | 41 | 40 | 0.525 | 0.476 | 0.575 | 0.075 | 0.976 | 0.396 | 0.273 | 0.523 | external_zero_shot |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | Logistic_unscaled | core5 | 29 | 17 | 12 | 81 | 41 | 40 | 0.382 | 0.283 | 0.493 | 0.325 | 0.439 | 0.330 | 0.218 | 0.454 | external_zero_shot |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | SVM_RBF_scaled | core5 | 29 | 17 | 12 | 81 | 41 | 40 | 0.501 | 0.440 | 0.551 | 0.075 | 0.927 | 0.501 | 0.372 | 0.633 | external_zero_shot |
| ItalianPVS_VA1 16k | Figshare_sustained_a 8k | matched sustained /a/ with rate/device/cohort differences | RandomForest | core5 | 29 | 17 | 12 | 81 | 41 | 40 | 0.537 | 0.500 | 0.588 | 0.075 | 1.000 | 0.465 | 0.337 | 0.602 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | Logistic_scaled | core4 | 81 | 41 | 40 | 29 | 17 | 12 | 0.277 | 0.147 | 0.424 | 0.083 | 0.471 | 0.176 | 0.039 | 0.353 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | Logistic_unscaled | core4 | 81 | 41 | 40 | 29 | 17 | 12 | 0.407 | 0.260 | 0.561 | 0.167 | 0.647 | 0.328 | 0.127 | 0.544 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | SVM_RBF_scaled | core4 | 81 | 41 | 40 | 29 | 17 | 12 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 | 0.480 | 0.265 | 0.706 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | RandomForest | core4 | 81 | 41 | 40 | 29 | 17 | 12 | 0.485 | 0.301 | 0.669 | 0.500 | 0.471 | 0.522 | 0.301 | 0.730 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | Logistic_scaled | core5 | 81 | 41 | 40 | 29 | 17 | 12 | 0.569 | 0.385 | 0.740 | 0.667 | 0.471 | 0.618 | 0.377 | 0.858 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | Logistic_unscaled | core5 | 81 | 41 | 40 | 29 | 17 | 12 | 0.402 | 0.230 | 0.591 | 0.333 | 0.471 | 0.363 | 0.162 | 0.583 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | SVM_RBF_scaled | core5 | 81 | 41 | 40 | 29 | 17 | 12 | 0.478 | 0.323 | 0.645 | 0.250 | 0.706 | 0.392 | 0.186 | 0.623 | external_zero_shot |
| Figshare_sustained_a 8k | ItalianPVS_VA1 16k | matched sustained /a/ with rate/device/cohort differences | RandomForest | core5 | 81 | 41 | 40 | 29 | 17 | 12 | 0.569 | 0.385 | 0.740 | 0.667 | 0.471 | 0.618 | 0.373 | 0.843 | external_zero_shot |

The resampling-rate sensitivity runs and invalid directions are in the `external_classifier_results.csv`, `invalid_cross_dataset_directions.csv`, and `master_classifier_summary.csv` tables. Four direct sustained↔reading directions were marked invalid and not scored.

## Metrics and uncertainty

Balanced accuracy, sensitivity, specificity, ROC-AUC, precision, F1, and confusion counts are supplied for every scored run. Accuracy alone is not used. Small cohorts yield broad intervals; repeated model comparisons are exploratory. The external result, rather than the best internal CV result, is the key generalisation test.
