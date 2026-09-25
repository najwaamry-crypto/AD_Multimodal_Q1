# Primary Model Selection Lock v1

## Purpose

This document freezes the primary model-selection decision before any
evaluation on the held-out test set.

## Data split

The fixed subject-level split is:

- Train: 576 subjects
- Validation: 82 subjects
- Test: 164 subjects

The test set has not been used for model training, hyperparameter
selection, architecture selection, model selection, or performance
inspection.

## Primary clinical block

The leakage-aware primary clinical block contains:

- Age at index
- Sex
- Years of education
- APOE4 allele count

Diagnosis-linked cognitive variables are excluded from the primary
clinical block.

Clinical preprocessing is estimated using the training set only.

## Primary model-selection metric

The pre-specified primary model-selection metric is:

- Macro one-vs-rest ROC-AUC on the validation set

Macro F1 and balanced accuracy are retained as complementary metrics.

## Validation model comparison

The main validation experiments include:

- Clinical primary
- MRI-only
- PET-only
- MRI + Clinical
- PET + Clinical
- MRI + PET simple fusion
- MRI + PET fixed late fusion
- MRI-PET cross-modal attention
- MRI + PET + Clinical

The complete reproducible summary is stored in:

- `results/summary/validation_primary_models_v1.csv`
- `results/summary/validation_all_experiments_v1.csv`
- `results/summary/validation_summary_v1.json`

## Selected primary model

The selected primary model is:

**MRI + PET + Primary Clinical Fusion v1**

Validation performance at the selected checkpoint:

- Macro OvR ROC-AUC: 0.7837
- Balanced accuracy: 0.6534
- Macro F1: 0.6108
- Accuracy: 0.6220

The corresponding training implementation is:

`src/training/10_train_mri_pet_clinical_primary_fusion_v1.py`

## Selection lock

From this point onward, the selected primary architecture is considered
locked.

The held-out test set must not be used to:

- select another primary architecture;
- tune learning rates, loss functions, class weights, or fusion design;
- choose epochs or checkpoints;
- choose modality combinations;
- modify preprocessing based on test performance.

Any secondary analyses performed before final test evaluation must be
defined independently of test-set outcomes.

## Test-set policy

The primary test set will be evaluated only after the remaining
pre-specified secondary analyses and evaluation procedures have been
finalized.

Any final test-set result will be reported as a held-out evaluation and
will not be used for further model tuning.

## Status

PRIMARY MODEL SELECTION: LOCKED

TEST SET ACCESSED: NO
