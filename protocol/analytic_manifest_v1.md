# Final analytic multimodal manifest v1

## Purpose

This manifest combines the locked post-MRI-QC cohort with demographic,
genetic, and cognitive information before any train/validation/test split,
image preprocessing, or model training.

## Population

Final primary analytic cohort:

N = 822 unique subjects.

Diagnosis distribution:

- CN: 224
- MCI: 438
- Dementia: 160

Diagnosis phase distribution:

- ADNI1: 170
- ADNIGO: 67
- ADNI2: 480
- ADNI3: 105

Each RID appears exactly once.

## Imaging identifiers

All 822 subjects have the required imaging identifiers:

- PET image ID
- selected MRI LONI series
- selected MRI image ID(s)
- selected MRI SeriesInstanceUID

MRI selection and QC rules were locked before construction of this manifest.

## Primary non-cognitive clinical block

The primary non-cognitive clinical covariates are:

- age_at_index
- sex
- education_years
- apoe4_count

Availability:

- age_at_index: 822 / 822
- sex: 822 / 822
- education_years: 822 / 822
- apoe4_count: 820 / 822

The two subjects with unavailable APOE remain in the primary cohort.

No imputation rule is fixed at this stage.

## Secondary cognitive block

The pre-specified secondary cognitive variables are:

- MMSE
- ADAS-13
- CDR-SB
- Logical Memory delayed recall
- RAVLT trials 1-5 sum
- RAVLT 30-minute delayed recall

Availability:

- MMSE: 790 / 822
- ADAS-13: 810 / 822
- CDR-SB: 787 / 822
- Logical Memory delayed: 724 / 822
- RAVLT trials 1-5 sum: 819 / 822
- RAVLT 30-minute delayed recall: 817 / 822

Complete data for all six secondary cognitive variables:

708 / 822 subjects (86.13%).

Missing cognitive values do not remove subjects from the primary cohort.

## Analysis roles

Primary non-cognitive clinical block:

Age + Sex + Education + APOE4

Secondary cognitive block:

MMSE + ADAS-13 + CDR-SB + Logical Memory delayed +
RAVLT trials 1-5 sum + RAVLT 30-minute delayed recall

The cognitive block is kept separate because these measures are closely
related to the clinical diagnostic process and may otherwise dominate
classification performance.

## Leakage control

No model performance was used to:

- select subjects;
- select MRI series;
- choose the cognitive alignment window;
- choose demographic/genetic variables;
- choose cognitive variables.

No train/validation/test split exists at this stage.

No preprocessing statistics have been computed.

No model training has been performed.

Any later scaling, normalization, categorical encoding, or imputation must be
fit using training data only.

## Reproducibility

Final analytic manifest construction script:

src/cohort/22_build_final_analytic_manifest.py

Aggregate summary:

tables/analytic_manifest_v1_summary.json

Subject-level analytic manifest:

data/interim/cohort/analytic_manifest_v1.csv

The subject-level manifest remains local and is not committed to Git.
