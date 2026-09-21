# Fixed train/validation/test split v1

## Population

The split is applied to the locked analytic multimodal cohort:

N = 822 unique subjects.

Each RID appears exactly once.

## Split sizes

The fixed subject-level partition is:

- Training: 576 subjects
- Validation: 82 subjects
- Test: 164 subjects

This corresponds approximately to a 70/10/20 partition.

## Random seed

The fixed random seed is:

20260921

The seed was specified before model training.

It must not be changed based on downstream model performance.

## Stratification

The split is stratified on diagnosis only.

Diagnosis categories are:

- CN
- MCI
- Dementia

ADNI phase was not included in the stratification because some
diagnosis-by-phase cells are extremely small:

- CN / ADNIGO: 3 subjects
- Dementia / ADNIGO: 2 subjects
- CN / ADNI3: 0 subjects

Forcing diagnosis-by-phase stratification would therefore create unstable or
artificial partitions.

Sex, APOE, MRI field strength, cognitive completeness, and other covariates
were not used to determine assignment.

## Diagnosis distribution

Training:

- CN: 157
- MCI: 307
- Dementia: 112

Validation:

- CN: 22
- MCI: 44
- Dementia: 16

Test:

- CN: 45
- MCI: 87
- Dementia: 32

The diagnostic proportions are approximately preserved across all three
partitions.

## ADNI phase audit

Phase distribution was audited after assignment but did not influence the
split.

Training:

- ADNI1: 109
- ADNI2: 340
- ADNI3: 78
- ADNIGO: 49

Validation:

- ADNI1: 21
- ADNI2: 44
- ADNI3: 11
- ADNIGO: 6

Test:

- ADNI1: 40
- ADNI2: 96
- ADNI3: 16
- ADNIGO: 12

## Sex audit

Training:

- Female: 262
- Male: 314

Validation:

- Female: 39
- Male: 43

Test:

- Female: 66
- Male: 98

Sex was not used for stratification.

## Continuous covariate audit

Age at index:

- Train mean: 73.485 years
- Validation mean: 73.475 years
- Test mean: 73.237 years

Education:

- Train mean: 16.023 years
- Validation mean: 16.476 years
- Test mean: 16.256 years

No substantial imbalance was identified.

## APOE4 audit

APOE4 count distribution:

Training:

- 0 alleles: 304
- 1 allele: 207
- 2 alleles: 63
- missing: 2

Validation:

- 0 alleles: 41
- 1 allele: 32
- 2 alleles: 9
- missing: 0

Test:

- 0 alleles: 97
- 1 allele: 55
- 2 alleles: 12
- missing: 0

The two subjects with missing APOE remain in the training partition and remain
part of the primary cohort.

## MRI field-strength audit

Training:

- 1.5T: 113
- 3T: 463

Validation:

- 1.5T: 21
- 3T: 61

Test:

- 1.5T: 40
- 3T: 124

MRI field strength was not used to construct the split.

## Secondary cognitive completeness audit

Complete six-variable secondary cognitive panel:

- Train: 496 / 576 (86.11%)
- Validation: 67 / 82 (81.71%)
- Test: 145 / 164 (88.41%)

Cognitive completeness was not used to determine split assignment.

## Leakage control

The split was created before:

- image preprocessing;
- tabular preprocessing;
- imputation;
- scaling;
- feature normalization;
- model selection;
- hyperparameter optimization;
- model training;
- evaluation.

Any preprocessing parameters must be estimated using training data only.

Validation data may be used for model and hyperparameter selection.

The test partition must remain untouched until final evaluation.

## Locked rule

The RID assignments in:

splits/split_assignments_v1.csv

are fixed and must not be regenerated or modified based on model results.

## Reproducibility

Split-generation script:

src/cohort/23_create_fixed_splits.py

Fixed assignments:

splits/split_assignments_v1.csv

Aggregate summary:

tables/split_assignments_v1_summary.json
