# MRI Preprocessing Protocol v1

## Scope

This protocol defines the locked T1 MRI preprocessing pipeline applied to the
822 final MRI selections before model training.

All preprocessing decisions were fixed before downstream classification
experiments and must not be modified in response to model performance.

## Input cohort

Input manifest:

`data/interim/imaging_download_manifests/mri_final_image_selection_v1.csv`

Final cohort:

- MRI volumes: 822
- unique RIDs: 822
- unique selected LONI series: 822

## Reference space

Template:

MNI152NLin2009cAsym, resolution 1 mm, obtained through TemplateFlow.

Reference T1:

`tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz`

Reference brain mask:

`tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz`

Reference geometry:

- shape: 193 x 229 x 193
- voxel size: 1 x 1 x 1 mm
- orientation: RAS

## Step 1 — N4 bias-field correction

N4BiasFieldCorrection is applied in native MRI space.

Fixed parameters:

- dimension: 3
- shrink factor: 2
- B-spline parameter: 200
- convergence: 50x50x30x20
- tolerance: 1e-6

N4 correction must preserve the original image geometry.

## Step 2 — Subject-specific brain extraction

Brain extraction is performed in native space with:

`antsBrainExtraction.sh`

using the MNI152NLin2009cAsym T1 template and associated brain mask as priors.

The resulting subject-specific brain mask is retained for subsequent masking.

The fixed MNI mask is not directly applied to individual subjects.

## Step 3 — Affine registration to MNI

The N4-corrected whole-head MRI is registered to the MNI reference using:

`antsRegistrationSyNQuick.sh -t a`

Only rigid/affine spatial normalization is used.

No nonlinear deformation is applied in the primary pipeline in order to avoid
nonlinear removal or alteration of disease-related morphological variation.

## Step 4 — Subject-mask transformation

The native subject-specific brain mask is transformed to MNI space using the
same affine transform as the MRI.

Interpolation:

`NearestNeighbor`

The transformed mask must remain binary.

## Step 5 — Brain masking

The affine-registered N4 MRI is multiplied by its transformed subject-specific
brain mask.

Voxels outside the subject-specific brain mask are set to zero.

## Step 6 — Intensity normalization

For each subject independently:

1. use only nonzero voxels inside the transformed subject-specific brain mask;
2. compute the mean and standard deviation from those brain voxels;
3. apply z-score normalization within the brain;
4. retain zero outside the mask;
5. save the final volume as float32.

No diagnosis or split information is used during subject-level MRI intensity
normalization.

## Output geometry

All final MRI volumes must have:

- shape: 193 x 229 x 193
- voxel size: 1 x 1 x 1 mm
- affine equal to the MNI reference
- finite voxel values
- subject-specific brain mask in the same geometry

## Pilot validation

The pipeline was tested before cohort-wide execution on six representative
scans:

- I7070 — ADNI1, 1.5 T
- I7679 — ADNI1, 1.5 T
- I179555 — ADNIGO/ADNI2, 3 T
- I180305 — ADNIGO/ADNI2, 3 T
- I844181 — ADNI3, 3 T
- I872948 — ADNI3, 3 T

Visual QC confirmed acceptable:

- N4 output,
- affine MNI registration,
- subject-specific brain extraction,
- transformed brain-mask coverage.

No major truncation, gross registration failure, severe brain-mask leakage, or
major cortical removal was identified in the pilot cases.

## Cohort-wide QC

After processing all 822 subjects, the following will be audited before model
training:

- successful output count,
- final geometry,
- affine consistency,
- finite voxel values,
- binary transformed masks,
- subject brain-mask volumes,
- normalized intensity statistics,
- registration QC,
- brain-extraction QC.

Outliers will be flagged for review rather than automatically excluded.

Any exclusion must be based on predefined image-quality failure criteria and
must not depend on classification performance.
