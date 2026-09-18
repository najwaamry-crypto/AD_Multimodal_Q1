# Cohort Definition v1

## 1. Study design
Cross-sectional multimodal study for three-class classification:

- CN: cognitively normal / cognitively unimpaired
- MCI: mild cognitive impairment
- DEM: dementia / Alzheimer's disease dementia

Primary development dataset:
Alzheimer's Disease Neuroimaging Initiative (ADNI).

Each participant may contribute only ONE index observation to the primary analysis.

---

## 2. Core scientific objective

The primary objective is not only to maximize classification performance.

The study will quantify the independent and complementary value of:

1. structural MRI,
2. FDG-PET,
3. demographic/genetic variables,
4. cognitive/clinical variables,

for CN/MCI/DEM classification.

Particular attention will be given to clinical variables that overlap with or are
strongly related to ADNI diagnostic criteria.

---

## 3. Unit of analysis

Unit of analysis: participant.

Unique participant identifier:
RID.

No participant may appear in more than one train/validation/test partition.

Repeated visits from the same RID will not be treated as independent samples
in the primary cross-sectional analysis.

---

## 4. Modalities required for the primary multimodal cohort

Required:

- T1-weighted structural MRI
- FDG-PET
- diagnosis corresponding to the index period
- required demographic variables

Clinical/cognitive variables will be analyzed in separate predefined groups.

---

## 5. MRI modality

Target modality:
T1-weighted structural MRI.

Preferred acquisitions:
- MP-RAGE
- Accelerated MPRAGE
- equivalent ADNI T1-weighted structural sequences when phase-specific naming differs

Final sequence-selection rules will be determined from ADNI metadata before
image downloading.

MRI quality-control information will be retained whenever available.

---

## 6. FDG-PET modality

Tracer:
18F-FDG.

Preference:
ADNI preprocessed PET images rather than raw PET.

Target processing family:
Coregistered + Averaged + Standardized Image/Voxel Size + Uniform Resolution.

Exact resolution/version will be selected after inventory of FDG availability
to avoid mixing incompatible preprocessing versions without documentation.

---

## 7. Temporal matching

MRI, FDG-PET, diagnosis, and clinical information must represent the same
clinical period.

Candidate maximum MRI-PET interval:
90 days.

The final matching rule will be frozen before model training.

For every included participant, actual time differences between modalities
must be stored in the cohort manifest.

---

## 8. Index observation

Primary principle:
one index observation per participant.

The exact rule will be selected before analysis between:

A. baseline-only multimodal observation;

or

B. earliest eligible multimodal observation satisfying the temporal-matching rule.

This decision must be made from methodological considerations and documented
before model training.

---

## 9. Diagnosis

The label must correspond to the clinical period of the selected index
observation.

Target classes:

CN
MCI
DEM

Phase-specific ADNI diagnostic variables must be harmonized using documented
ADNI diagnosis information.

No diagnosis will be reconstructed manually from MMSE, CDR, or cognitive
scores.

---

## 10. Clinical variable groups

### Group A: demographic / genetic variables

Candidate variables:
- Age
- Sex
- Education
- APOE4

### Group B: cognitive / clinical variables

Candidate variables:
- MMSE
- ADAS-Cog
- CDR-SB
- Logical Memory delayed recall
- RAVLT immediate recall

Variables in Group B must not be treated as ordinary independent predictors
without acknowledging their relationship to diagnostic assignment.

A leakage-aware analysis will compare models with and without diagnosis-linked
cognitive variables.

---

## 11. Inclusion criteria

A participant is eligible for the primary multimodal cohort only if:

1. a valid diagnosis is available for the index period;
2. an eligible T1-weighted MRI is available;
3. an eligible FDG-PET is available;
4. MRI and PET satisfy the predefined temporal window;
5. imaging passes required quality control;
6. participant identity can be matched consistently across modalities;
7. no duplicated participant observation is present in the primary dataset.

---

## 12. Exclusion criteria

Exclude observations with:

- missing diagnosis;
- incompatible or uncertain diagnosis mapping;
- missing required imaging modality;
- modality interval outside the predefined matching window;
- unusable/corrupted imaging;
- failed preprocessing or registration after documented QC;
- duplicated RID after index-observation selection;
- unresolved metadata inconsistencies.

Every exclusion must receive an explicit exclusion_reason in the manifest.

---

## 13. Leakage prevention

All preprocessing steps requiring learned parameters must be fitted using
training data only.

Examples:
- feature scaling;
- imputation;
- feature selection;
- learned harmonization.

Subject-level partitioning is mandatory.

Clinical variables related to diagnostic criteria will be analyzed separately.

No final test information may influence:
- preprocessing choices,
- architecture selection,
- hyperparameters,
- threshold selection.

---

## 14. Reproducibility requirements

The final cohort must be represented by a single master manifest.

Minimum fields will include:

RID
PTID
VISCODE
ADNI_PHASE
INDEX_DATE
DIAGNOSIS
MRI_IMAGE_ID
MRI_DATE
MRI_PATH
MRI_QC
PET_IMAGE_ID
PET_DATE
PET_PATH
PET_QC
MRI_PET_DAYS
AGE
SEX
EDUCATION
APOE4
MMSE
ADAS11
CDRSB
LDELTOTL
RAVLTIMM
INCLUDE
EXCLUSION_REASON

The master manifest will be the single source of truth for all cohort statistics,
splits, tables, and experiments.

---

## 15. Current locked decisions

LOCKED:
- three-class CN/MCI/DEM task;
- subject-level analysis;
- one primary observation per RID;
- T1 MRI + FDG-PET;
- explicit leakage-aware clinical analysis;
- fixed reusable partitions;
- single master cohort manifest;
- all exclusions documented.

NOT YET LOCKED:
- ADNI phases included;
- baseline-only versus earliest eligible visit;
- exact MRI sequence filters;
- exact FDG preprocessing version;
- final MRI-PET temporal window;
- final handling of missing clinical values.

These items must be resolved before model training.

---

## 16. Locked diagnostic definition - 18 September 2026

Diagnostic source for the primary classification task:

DXSUM.DIAGNOSIS

The mapping was verified against the ADNI Data Dictionary downloaded on
18 September 2026:

- 1 = CN
- 2 = MCI
- 3 = Dementia
- 10 = TEAM_NODX

Primary CN/MCI/DEM analyses will include only DIAGNOSIS values 1, 2, and 3.

TEAM_NODX (10) represents the absence of a formal TEAM diagnosis assessment
and will not be mapped to any of the three target classes.

DXNORM, DXMCI, and DXAD will not be used to reconstruct the primary label
when DIAGNOSIS is available.

The diagnostic label will not be reconstructed from MMSE, CDR, Logical Memory,
ADAS-Cog, or any other cognitive score.
