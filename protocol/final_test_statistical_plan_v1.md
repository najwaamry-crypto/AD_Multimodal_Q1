# Final Held-Out Test Statistical Plan v1

## Purpose

This document freezes the final statistical evaluation procedure before
the held-out test set is accessed.

The primary architecture, secondary analyses, and random-seed
sensitivity analysis have already been locked.

No test-set result has been inspected at the time this plan is frozen.

## Held-out test population

The primary held-out test set contains:

- 164 subjects

The fixed test split was defined before model development and has not
been used for:

- model training;
- architecture selection;
- hyperparameter selection;
- checkpoint selection;
- modality selection;
- preprocessing modification;
- random-seed selection.

All primary model comparisons will use the same held-out subjects.

## Locked primary model

The primary model remains:

**MRI + PET + Primary Clinical Fusion v1**

Primary seed:

- 20260921

The checkpoint selected on validation remains fixed.

No seed-robustness run may replace the primary seed.

## Primary outcome

The primary test metric is:

- Macro one-vs-rest ROC-AUC

The three classes are:

- CN
- MCI
- Dementia

## Secondary performance metrics

The following secondary metrics will be reported:

- Accuracy
- Balanced accuracy
- Macro F1
- Multiclass log loss
- Multiclass Brier score

Class-specific descriptive results will include:

- CN one-vs-rest ROC-AUC
- MCI one-vs-rest ROC-AUC
- Dementia one-vs-rest ROC-AUC
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix

Calibration will additionally be described using:

- Mean confidence
- Confidence minus accuracy
- 10-bin expected calibration error

Calibration metrics are descriptive secondary analyses.

## Confidence intervals

Uncertainty will be quantified using paired stratified bootstrap
resampling of the held-out subjects.

Predefined bootstrap settings:

- Number of bootstrap replicates: 10,000
- Confidence level: 95%
- Bootstrap random seed: 20260925
- Resampling unit: subject
- Sampling with replacement
- Stratification: diagnosis class

For each replicate, subjects will be resampled within CN, MCI, and
Dementia separately so that the original class counts are preserved.

Percentile 95% confidence intervals will be reported.

No bootstrap replicate will be used for model selection or tuning.

## Predefined paired model comparisons

The same bootstrap resamples will be used for both models in every
comparison.

### Primary comparison

- MRI + PET + Clinical
  versus
  PET + Clinical

The primary comparison statistic is the paired difference in:

- Macro OvR ROC-AUC

### Secondary predefined comparisons

- Clinical primary
  versus
  MRI + PET + Clinical

- MRI-only
  versus
  MRI + Clinical

- PET-only
  versus
  PET + Clinical

- PET-only
  versus
  MRI + PET + Clinical

For each comparison, the paired bootstrap distribution of the
difference in Macro OvR ROC-AUC will be reported with a 95% confidence
interval.

These comparisons are inferentially descriptive and will not trigger
additional model development.

## Primary model seed robustness

The primary seed remains 20260921.

The previously defined five-seed validation sensitivity analysis is
reported separately as an optimization-robustness analysis.

The best-performing alternative seed will not be substituted for the
primary model.

## Cognitive benchmark

The cognitive-only benchmark remains a secondary analysis.

Because complete cognitive data are unavailable for every subject, its
held-out evaluation will use the predefined complete-case cognitive
subset only.

Its sample size will therefore be reported explicitly and its
performance will not be treated as directly interchangeable with the
primary 164-subject analysis.

## Test opening rule

After this statistical plan is committed to version control, the test
set may be accessed for final evaluation.

The test set will be evaluated using the already locked checkpoints and
preprocessing procedures.

After test results are generated, the following are prohibited:

- retraining based on test performance;
- changing the primary checkpoint;
- changing architecture;
- changing fusion strategy;
- changing class weights;
- modifying preprocessing;
- tuning thresholds to improve reported performance;
- selecting another seed;
- selecting another primary model because it performs better on test.

Unexpected findings may be discussed, but they must not trigger
test-driven optimization.

## Reporting rule

The manuscript will clearly distinguish:

1. validation model-selection results;
2. seed-robustness results;
3. secondary cognitive benchmark results;
4. final held-out test performance.

Validation results will not be presented as final generalization
performance.

## Status before test opening

PRIMARY MODEL: LOCKED

PRIMARY SEED: LOCKED

STATISTICAL PLAN: LOCKED

BOOTSTRAP PROCEDURE: LOCKED

TEST SET ACCESSED: NO
