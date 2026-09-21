# MRI Selection Notes v1

## Snapshot
MRI metadata inventory:
- 20,848 T1w images
- 3,207 participants
- ADNI1, ADNIGO/ADNI2, ADNI3, ADNI4

## ADNI1 ranking

MRIMPRANK defines MPRAGE quality rankings:
- 1 = high quality
- 2 = medium quality
- 3 = low quality but passing
- >3 / other failure codes are not acceptable as preferred passing scans.

RANK is a quality rating and must NOT be interpreted as a unique
selected-series indicator.

Multiple scans can share the same best rank within the same visit.

## MRIMPPRO discrepancy

The ADNI data dictionary downloaded in September 2026 documents:
MRIMPPRO.LONIUID = LONI Unique Series ID selected for processing.

However:

1. ADNIMERGE2 MRIMPPRO.rda does not contain LONIUID.
2. The raw MRIMPPRO_21Sep2026.csv downloaded directly from IDA also
   does not contain LONIUID.

Therefore MRIMPPRO will NOT be used to identify the preferred series
unless a future ADNI source provides the missing selected-series field.

No selected image will be inferred from COMMENTS.

## Phase-specific provisional strategy

ADNI1:
Use MRIMPRANK quality ratings. Final tie handling will be resolved only
after construction of the multimodal MRI-FDG-PET candidate cohort.

ADNIGO/ADNI2:
Use Mayo MRI QC selected/recommended series when available.

ADNI3:
Use MAYOADIRL_MRI_QUALITY_ADNI3 and SERIES_SELECTED/QC information.

ADNI4:
ARC Builder T1w series are treated according to current ADNI MRI Core
guidance that redundant series are hidden unless they pass QC and are chosen.

## Important

The final one-MRI-per-index-observation selection rule must be frozen
before model training and documented in a new protocol version.
