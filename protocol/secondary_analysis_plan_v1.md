# Secondary Analysis Plan v1

## Purpose

This document defines the secondary analyses that are permitted after
primary model selection and before the held-out test set is accessed.

The primary model architecture has already been locked in:

`protocol/primary_model_selection_lock_v1.md`

No new primary architecture will be selected after this point.

## 1. Cognitive benchmark analysis

A secondary cognitive benchmark will evaluate the diagnostic signal
available from cognitive assessments that were deliberately excluded
from the leakage-aware primary clinical block.

The predefined cognitive variables are:

- MMSE
- ADAS13
- CDR-SB
- Logical Memory delayed recall
- RAVLT Trials 1-5 sum
- RAVLT 30-minute delayed recall

These variables are not used to define the diagnostic labels.

Because cognitive assessments may be closely related to the diagnostic
decision, results from this block will be explicitly reported as a
secondary benchmark and will not replace the primary leakage-aware
clinical analysis.

Missing-data handling and the analysis population must be fixed before
any test-set evaluation.

## 2. Class-specific validation analysis

For the already selected models, validation predictions may be used to
report:

- CN one-vs-rest ROC-AUC
- MCI one-vs-rest ROC-AUC
- Dementia one-vs-rest ROC-AUC
- confusion matrices
- per-class precision
- per-class recall
- per-class F1 score

These analyses are descriptive and must not be used to select a new
primary architecture.

## 3. Random-seed robustness

The locked primary architecture may be retrained using a predefined set
of random seeds to characterize optimization variability.

Seed robustness is a sensitivity analysis only.

The original locked model and original seed remain the primary model.
No alternative seed may be selected because it gives superior
validation performance.

## 4. Calibration analysis

Calibration procedures and metrics will be specified before the test
set is opened.

Planned descriptive metrics include:

- multiclass log loss
- multiclass Brier score
- reliability/calibration assessment

Calibration results must not be used to redesign the primary model
after the test set is accessed.

## 5. Final statistical evaluation

Once all analysis procedures are frozen, the held-out test set may be
opened once for final evaluation.

The final evaluation will include predefined performance metrics and
confidence intervals.

Primary comparisons will focus on clinically meaningful model pairs,
including:

- Clinical primary vs MRI + PET + Clinical
- MRI-only vs MRI + Clinical
- PET-only vs PET + Clinical
- PET + Clinical vs MRI + PET + Clinical
- PET-only vs MRI + PET + Clinical

Confidence intervals and paired model comparisons will be computed on
the same held-out subjects using a predefined resampling procedure.

The test set must not be used for subsequent hyperparameter tuning,
architecture selection, checkpoint selection, modality selection, or
preprocessing modification.

## Prohibited after this lock

The following are not permitted based on future test performance:

- testing additional primary architectures;
- tuning modality weights;
- tuning fusion mechanisms;
- selecting a different checkpoint;
- changing class weights;
- modifying imaging preprocessing;
- modifying the primary clinical feature set;
- selecting a seed because it performs better on test;
- reopening the test set for iterative optimization.

## Current status

PRIMARY MODEL: LOCKED

SECONDARY ANALYSIS PLAN: LOCKED

TEST SET ACCESSED: NO
