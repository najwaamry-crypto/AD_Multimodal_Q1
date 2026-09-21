# Imaging download definition v1

## Population

The imaging download manifests correspond to the locked analytic cohort of
822 unique subjects.

No cohort, split, or imaging-selection rule is modified at this stage.

## FDG-PET

Each subject has exactly one selected PET image identifier.

Total:

- subjects: 822
- unique PET Image IDs: 822
- PET Image IDs shared across subjects: 0

All selected PET images correspond to:

Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res

PET download unit:

Image ID.

## T1-weighted MRI

MRI selection was previously performed at the series level using the locked
phase-specific QC rules.

Total:

- subjects: 822
- unique selected LONI series: 822
- LONI series shared across subjects: 0

Field strength:

- 3.0T: 648
- 1.5T: 174

Selection families:

- ADNIGO/ADNI2: 545
- ADNI1: 174
- ADNI3: 103

## MRI download unit

The preferred MRI download unit is the selected LONI series.

The selected series must remain the unit of truth during download and local
file reconciliation.

The 822 selected MRI series correspond to 825 Key MRI Image IDs.

For 819 subjects there is one selected Image ID.

Three subjects have two Key MRI Image IDs belonging to the same selected
LONI series:

- RID 4138, series 200478: Image IDs 389652 and 389660
- RID 4468, series 213644: Image IDs 416046 and 416053
- RID 4571, series 216492: Image IDs 420552 and 420554

No arbitrary selection between these paired Image IDs is permitted.

If IDA supports series-level retrieval, all 822 selected series should be
downloaded directly.

If Image-ID-level retrieval is required instead, all 825 selected MRI Image
IDs should be retrieved and the three multi-image series resolved only after
local file-level QC.

## Leakage control

Download decisions are based exclusively on the previously locked imaging
selection.

Train/validation/test membership does not alter which images are downloaded.

No model result, preprocessing result, or downstream performance criterion is
used to alter image selection.

## Reproducibility

Identifier audit:

src/cohort/24_audit_imaging_download_ids.py

Download-manifest preparation:

src/cohort/25_prepare_imaging_download_manifests.py

Aggregate summary:

tables/imaging_download_manifests_v1_summary.json

Subject-level download manifests and ID lists are stored under:

data/interim/imaging_download_manifests/

and remain local.
