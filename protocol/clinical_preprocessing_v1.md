# Clinical preprocessing protocol v1

## Purpose

This protocol defines the leakage-aware preprocessing strategy for the
structured clinical variables used in the primary modelling experiments.

All preprocessing parameters must be estimated exclusively from the training
partition and then applied unchanged to validation and test data.

The fixed subject-level split is:

- training: 576 subjects
- validation: 82 subjects
- test: 164 subjects

The test set must remain untouched until final model evaluation.

## Primary clinical block

The primary leakage-aware clinical block contains:

- age at imaging index date (`age_at_index`)
- sex (`sex`)
- years of education (`education_years`)
- APOE epsilon-4 allele count (`apoe4_count`)

Cognitive scores are not included in this primary clinical block.

## APOE representation

`apoe4_count` is used as the primary APOE representation:

- 0 = no APOE epsilon-4 allele
- 1 = one APOE epsilon-4 allele
- 2 = two APOE epsilon-4 alleles

`apoe4_carrier` is not simultaneously included because it is directly
derived from `apoe4_count` and would introduce redundant information.

Two subjects have missing APOE information.

Both missing APOE observations occur exclusively in the training partition.

Training APOE4-count distribution before imputation:

- 0 alleles: 304
- 1 allele: 207
- 2 alleles: 63
- missing: 2

The training-set median is 0.

Missing APOE4-count values will therefore be imputed using a median imputer
fitted exclusively on the training partition.

The imputation value must not be estimated using validation or test data.

## Continuous variables

The continuous variables are:

- `age_at_index`
- `education_years`

For models requiring scaled input, standardization parameters will be fitted
on the training partition only:

z = (x - training_mean) / training_std

The fitted transformation will then be applied unchanged to validation and
test subjects.

## Sex

Sex is treated as a categorical variable.

The encoding scheme must be fitted or defined without using outcome
information.

No sex values are missing in the analytic cohort.

## Missingness

Primary clinical completeness:

- complete: 820 / 822 subjects
- incomplete: 2 / 822 subjects

The only missing primary variable is APOE4 count.

No subject will be excluded because of the two missing APOE values.

## Cognitive variables

The secondary cognitive block is kept separate from the primary clinical
analysis.

The secondary cognitive variables are:

- MMSE
- ADAS13
- CDR-SB
- Logical Memory delayed recall
- RAVLT trials 1-5 sum
- RAVLT 30-minute delayed recall

Complete cognitive data are available for 708 / 822 subjects.

Cognitive variables will be evaluated in explicitly labelled secondary or
ablation experiments and must not be mixed into the primary leakage-aware
clinical baseline.

## Leakage prevention

The following operations must be fitted using training data only:

- missing-value imputation
- feature scaling
- any feature transformation learned from data
- model fitting
- hyperparameter selection

Validation data may be used for model selection.

Test data must not be used for preprocessing estimation, feature selection,
hyperparameter tuning, or model selection.

## Status

Primary clinical feature definition:
LOCKED.

APOE missing-data policy:
LOCKED.

Train-only preprocessing rule:
LOCKED.

Ready for baseline modelling.
