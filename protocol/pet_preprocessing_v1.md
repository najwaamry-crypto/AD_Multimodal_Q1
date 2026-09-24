# FDG-PET Preprocessing Protocol v1

## Scope

This protocol defines the current FDG-PET preprocessing pipeline for the
multimodal Alzheimer study.

The spatial preprocessing strategy has been validated on a six-subject pilot
panel spanning CN, MCI, and Dementia, with representation from ADNI1 and
ADNIGO/ADNI2.

The intensity normalization strategy described below is the current v1
reference approach and remains subject to final confirmation before cohort-wide
locking.

## Input cohort

Final analytic cohort:

- subjects: 822
- unique RIDs: 822
- unique FDG-PET image IDs: 822
- one PET image per subject

Input PET directory:

`data/interim/pet_nifti_v1`

All PET NIfTI volumes were previously converted from the selected ADNI FDG-PET
DICOM series.

## PET source preprocessing

Selected ADNI PET description:

`Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res`

The downloaded PET volumes have:

- shape: 160 x 160 x 96
- voxel size: 1.5 x 1.5 x 1.5 mm
- modality: FDG-PET
- finite voxel values
- harmonized spatial resolution

These images are standardized ADNI PET products but are not assumed to be
already registered to the final MNI reference space used in this study.

## Spatial preprocessing rationale

PET and MRI are aligned subject-wise before projection into the common MNI
space.

The final spatial chain is:

FDG-PET native
→ rigid PET-to-MRI registration
→ MRI-to-MNI affine transform
→ final PET in MNI152NLin2009cAsym 1 mm space.

No nonlinear PET-specific deformation is introduced.

The subject-specific MRI serves as the anatomical bridge between PET and the
common template.

## Step 1 — PET-to-MRI rigid registration

For each subject, the native PET is rigidly registered to the corresponding
N4-corrected T1 MRI.

Tool:

`antsRegistrationSyNQuick.sh`

Transform type:

`-t r`

This corresponds to rigid 6-DOF alignment.

Fixed image:

subject-specific N4-corrected T1 MRI.

Moving image:

subject-specific FDG-PET.

Mutual information is used by ANTs for multimodal registration.

## Step 2 — Composition with MRI-to-MNI transform

The PET-to-MRI rigid transform is combined with the previously computed
subject-specific MRI-to-MNI affine transform.

The PET is resampled directly into the final template space in a single
application step using:

`antsApplyTransforms`

Interpolation:

`Linear`

Reference template:

`tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz`

Final PET geometry:

- shape: 193 x 229 x 193
- voxel size: 1 x 1 x 1 mm
- orientation: RAS
- affine matched to the MNI reference

## Step 3 — PET acquisition support mask

A binary support image is created in native PET space by assigning value 1 to
all voxels in the native PET grid.

This support image is transformed to MNI space using exactly the same composed
spatial transforms as the PET.

Interpolation:

`NearestNeighbor`

The transformed support mask identifies voxels that are genuinely covered by
the original PET acquisition.

## Step 4 — Subject-specific brain mask

The final MRI-derived subject-specific brain mask in MNI space is reused for
PET masking.

The valid PET brain region is defined as:

`MRI brain mask ∩ PET support mask`

This prevents voxels outside the original PET field of view from being treated
as valid brain signal.

## Step 5 — FDG intensity normalization

The current v1 reference approach uses cerebellar vermis normalization.

The vermis mask is derived from the AAL SPM12 atlas using the following labels:

- 9100 Vermis_1_2
- 9110 Vermis_3
- 9120 Vermis_4_5
- 9130 Vermis_6
- 9140 Vermis_7
- 9150 Vermis_8
- 9160 Vermis_9
- 9170 Vermis_10

The AAL vermis mask is resampled from its original 2 mm space into the study
MNI152NLin2009cAsym 1 mm reference geometry using nearest-neighbor
interpolation.

Reference voxels are restricted to:

`vermis mask ∩ MRI brain mask ∩ PET support mask ∩ finite PET voxels ∩ PET > 0`

For each subject:

1. compute the mean PET intensity inside the valid vermis reference region;
2. divide all valid brain PET voxels by this mean;
3. set voxels outside the valid brain region to zero;
4. save the final PET as float32.

No global z-score normalization is applied to FDG-PET in v1.

## Important methodological note

The vermis mask originates from AAL SPM12 and is resampled into the
MNI152NLin2009cAsym study template.

The pilot QC showed anatomically plausible vermis placement, but this reference
choice remains subject to final confirmation before definitive cohort-wide
locking.

A pons-plus-vermis reference may be considered as a sensitivity analysis if a
reliable pons ROI compatible with the target MNI space is identified.

## Pilot panel

The pipeline was tested on six subjects:

### CN

- RID 5
  - PET: I1594687
  - MRI: I7070
  - MRI family: ADNI1
  - MRI-PET interval: 27 days

- RID 89
  - PET: I1599526
  - MRI: I202166
  - MRI family: ADNIGO/ADNI2
  - MRI-PET interval: 11 days

### MCI

- RID 33
  - PET: I1597335
  - MRI: I84235
  - MRI family: ADNI1
  - MRI-PET interval: 6 days

- RID 1155
  - PET: I1593107
  - MRI: I352948
  - MRI family: ADNIGO/ADNI2
  - MRI-PET interval: 14 days

### Dementia

- RID 3
  - PET: I1594605
  - MRI: I16547
  - MRI family: ADNI1
  - MRI-PET interval: 0 days

- RID 994
  - PET: I1602405
  - MRI: I346547
  - MRI family: ADNIGO/ADNI2
  - MRI-PET interval: 2 days

## Pilot quantitative QC

Across the six pilot subjects:

- brain coverage by transformed PET support:
  approximately 99.97% to 100%
- valid vermis reference voxels:
  16,320 for all six subjects
- vermis reference mean:
  approximately 0.862 to 1.145
- vermis reference coefficient of variation:
  approximately 0.219 to 0.268
- normalized brain mean:
  approximately 0.893 to 1.149

No pilot subject showed an empty reference region or grossly abnormal
normalization statistics.

## Pilot visual QC

Visual inspection was performed for:

- PET-to-native-MRI rigid alignment
- PET-to-MNI alignment
- PET-to-MRI overlay in MNI space
- vermis reference mask location
- final normalized PET appearance

Across the six-subject pilot panel, no gross spatial misregistration,
rotation error, major field-of-view mismatch, or obvious normalization artifact
was identified.

## Cohort-wide QC plan

Before final model training, the following must be audited across all 822 PET
volumes:

- successful preprocessing count
- final image shape
- final voxel size
- affine consistency with the MNI template
- finite voxel values
- PET support coverage
- vermis reference voxel count
- vermis reference mean
- vermis reference coefficient of variation
- normalized brain intensity statistics
- registration QC
- visual review of flagged outliers

Outliers must be flagged for review rather than automatically excluded.

Any exclusion criterion must be based on predefined imaging or registration QC
rules and must not depend on downstream classification performance.

## Status

Spatial preprocessing:
pilot validated.

Vermis-based intensity normalization:
pilot validated, pending final cohort-wide lock.

Cohort-wide execution:
not yet launched.
