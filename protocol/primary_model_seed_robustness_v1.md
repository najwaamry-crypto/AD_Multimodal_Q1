# Primary Model Seed Robustness v1

## Purpose

This analysis quantifies optimization variability of the already locked
primary architecture:

MRI + PET + Primary Clinical Fusion v1

The analysis is a sensitivity analysis only and must not be used to
replace the locked primary model or select a more favorable seed.

## Predefined seeds

The predefined random seeds are:

- 20260921
- 11
- 42
- 1337
- 31415

Seed 20260921 corresponds to the original locked primary run.

The remaining four seeds will be trained using exactly the same:

- training cohort;
- validation cohort;
- imaging inputs;
- clinical variables;
- clinical preprocessing policy;
- architecture;
- class weights;
- optimizer;
- learning rate;
- weight decay;
- learning-rate scheduler;
- maximum number of epochs;
- early-stopping patience;
- validation model-selection metric.

## Reporting

The following validation metrics will be summarized across all five
seeds:

- Macro OvR ROC-AUC
- Balanced accuracy
- Macro F1
- Accuracy
- Log loss

Results will be reported using mean and standard deviation across seeds.

The original seed 20260921 remains the locked primary model regardless
of the performance of the additional seeds.

No seed will be selected because it performs better on validation.

## Test policy

The held-out test set must not be accessed during this analysis.

TEST SET ACCESSED: NO
