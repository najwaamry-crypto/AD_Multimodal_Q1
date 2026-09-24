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

## Cohort-wide spatial QC

The spatial preprocessing pipeline was executed on the complete analytic
cohort of 822 FDG-PET scans.

Initial cohort-wide QC identified a small number of cases with reduced
brain coverage after PET-to-MRI registration. Cases with brain coverage
below 99% were flagged for targeted review rather than automatically
excluded.

Ten scans required registration recovery:

- one case was initially identified with approximately 42.4% brain coverage
- one case showed approximately 81.8% brain coverage
- eight additional cases showed approximately 97.9% to 98.9% coverage

Visual inspection demonstrated that the most severe failures arose from
suboptimal rigid PET-to-MRI registration rather than unusable PET data.

A robust rigid-registration recovery procedure was therefore applied using:

- the brain-extracted subject MRI as the fixed image
- the native PET image as the moving image
- geometric-center initialization
- rigid 6-degree-of-freedom transformation
- Mattes mutual information
- multiresolution optimization
- no nonlinear PET deformation

The recovered PET-to-MRI transform was then concatenated with the previously
locked subject-specific MRI-to-MNI affine transform, and the PET volume was
resampled once into MNI152NLin2009cAsym 1-mm space.

The same transform chain was applied to the PET support mask using
nearest-neighbor interpolation.

All ten registration outliers were successfully recovered and retained in the
analytic cohort.

After registration recovery, cohort-wide spatial QC showed:

- total subjects: 822
- mean brain coverage: 99.9705%
- minimum brain coverage: 99.0468%
- maximum brain coverage: 100%
- subjects below 99% brain coverage: 0
- subjects below 98% brain coverage: 0
- subjects below 95% brain coverage: 0

Fourteen residual flagged cases, defined by brain coverage below 99.5% or
the presence of negative brain voxels, underwent additional visual inspection.

No gross residual PET-to-MRI or PET-to-MNI misregistration was identified in
these cases.

Three subjects retained a very small fraction of negative brain voxels after
spatial preprocessing. The maximum observed negative fraction was
approximately 0.244%, and these cases did not show spatial registration
failure.

No subject was excluded based on the final PET spatial QC.

## Reference-region intensity normalization

Following successful spatial preprocessing, FDG-PET intensities were normalized
using a cerebellar vermis reference region.

The reference mask was derived from the AAL SPM12 vermis labels and resampled
to the study target space:

MNI152NLin2009cAsym, 1-mm isotropic resolution.

The final reference mask contained 16,320 voxels.

For each subject, the reference denominator was defined as the arithmetic mean
of all finite PET voxel intensities within the vermis mask.

The normalized PET image was defined as:

PET_norm(v) = PET_MNI(v) / mean(PET_MNI within AAL vermis)

This normalization strategy is an internal, reproducible study reference and
must not be described as equivalent to the UC Berkeley Top50PonsVermis method.

## Concordance with UC Berkeley FDG reference values

The internal AAL-vermis reference was compared with the official UC Berkeley
ADNI Top50PonsVermis values.

Exact RID and PET examination-date matching provided an official UC Berkeley
reference value for 821 of the 822 analytic subjects.

Across these 821 matched scans:

- Pearson correlation: r = 0.8244
- Spearman correlation: rho = 0.8096
- mean AAL-vermis / UC Berkeley ratio: 0.9151
- median AAL-vermis / UC Berkeley ratio: 0.9176
- mean absolute difference between reference values: 0.0902

The mean AAL-vermis / UC Berkeley ratio was:

- CN: 0.9154
- MCI: 0.9164
- Dementia: 0.9108

A Kruskal-Wallis comparison of the ratio across diagnostic groups showed:

- H = 1.2896
- p = 0.5248

These results indicate substantial inter-individual concordance between the
two reference measures, without evidence of a diagnosis-dependent difference
in their ratio.

The AAL-vermis and UC Berkeley Top50PonsVermis measures nevertheless represent
different anatomical reference definitions and are not treated as
interchangeable.

The UC Berkeley analysis is therefore used as a reference-method concordance
analysis rather than as a reproduction of the UC Berkeley normalization
procedure.

## Cohort-wide normalization QC

The final AAL-vermis normalization was applied to all 822 PET scans.

The raw vermis reference denominator showed:

- subjects with valid denominator: 822 / 822
- missing denominators: 0
- non-positive denominators: 0
- mean: 0.9534
- standard deviation: 0.0836
- minimum: 0.7124
- median: 0.9502
- maximum: 1.1974

After normalization:

- successfully normalized PET volumes: 822 / 822
- finite-value failures: 0
- post-normalization vermis mean range:
  approximately 0.9999998 to 1.0000002

The original spatially processed PET volumes are preserved separately from the
normalized volumes.

## Final PET preprocessing pipeline

The locked FDG-PET preprocessing pipeline is therefore:

1. native FDG-PET input
2. native-space PET support-mask generation
3. rigid PET-to-subject-MRI registration
4. composition with the locked MRI-to-MNI affine transform
5. single PET resampling into MNI152NLin2009cAsym 1-mm space
6. support-mask transformation using nearest-neighbor interpolation
7. cohort-wide spatial QC
8. targeted robust registration recovery for flagged cases
9. visual QC of residual outliers
10. AAL-vermis reference-mean estimation
11. subject-wise FDG-PET intensity normalization by the vermis mean
12. cohort-wide normalization QC

No nonlinear deformation is applied directly to PET.

No subject was excluded on the basis of downstream model performance.

## Status

Spatial preprocessing:
locked for 822 / 822 subjects.

Registration QC and recovery:
completed and locked.

AAL-vermis intensity normalization:
completed and locked for 822 / 822 subjects.

UC Berkeley reference-method concordance analysis:
completed for 821 / 822 subjects.

Final PET preprocessing status:
LOCKED FOR DOWNSTREAM MODELLING.
