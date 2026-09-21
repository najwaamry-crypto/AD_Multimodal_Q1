# MRI Series Selection Protocol v2

## Scope

This document records the phase-specific T1-weighted MRI series-selection
rules used for the primary cross-sectional ADNI multimodal cohort.

MRI selection rules are defined before model training and must not be changed
based on downstream classification performance.

---

## ADNIGO / ADNI2 protocol family

### Candidate population

The primary multimodal candidate manifest contained 574 subjects whose selected
MRI examination belonged to the ADNIGO/ADNI2 MRI protocol family.

A total of 1,184 candidate T1-weighted Key MRI rows were associated with these
subjects before QC-based series selection.

### QC sources

The primary QC source is the archived official Mayo ADIR Lab GO/2 MRI QC table:

MAYOADIRL_MRI_IMAGEQC_05_07_15_21Sep2026.csv

The archive was downloaded from the official ADNI/IDA archived Study Files
collection.

A more recent Mayo GO/2/3 QC table distributed in ADNIMERGE2 was additionally
used as a fallback for candidate series absent from the historical archive:

MAYOADIRL_MRI_QUALITY_ADNI3.rda

The historical GO/2 table remains the primary QC source because the current
GO/2/3 table does not retain direct coverage of most historical GO/2 candidate
series.

### Unit of selection

Selection is performed at the distinct LONI series level rather than at the
Key MRI image_id level.

This is necessary because multiple Key MRI image_id records can occasionally
refer to the same LONI MRI series.

### Primary GO/2 rule

For the primary analysis, the retained T1 series must satisfy all of the
following:

1. The acquisition is classified as Accelerated in Key MRI.
2. Mayo SERIES_SELECTED = 1.
3. Mayo SERIES_QUALITY is one of {1, 2, 3}.
4. Exactly one distinct eligible LONI series remains for the subject.

The accelerated acquisition strategy was fixed before model training and is
not selected according to model performance.

### Final GO/2 selection

Total candidate subjects: 574

Selected subjects: 545

Selected diagnostic distribution:

- CN: 178
- MCI: 278
- Dementia: 89

Excluded subjects: 29

Exclusion reasons:

- accelerated candidate available but not selected by Mayo QC: 13
- no accelerated T1 candidate: 11
- accelerated candidate lacking usable official QC information: 5

No subject had more than one distinct eligible accelerated LONI series after
series-level deduplication.

Three selected subjects contained more than one Key MRI image_id referring to
the same selected LONI series. These are not considered competing MRI
acquisitions. Exact downloadable image identifiers will be resolved later
without changing the selected LONI series.

### Reproducibility

Primary selection output:

data/interim/cohort/go2_accelerated_selection_v1.csv

Aggregate summary:

tables/go2_accelerated_selection_v1_summary.json

Selection script:

src/cohort/10_finalize_go2_accelerated_selection.py

---

## Remaining phase-specific MRI selection

ADNI3 uses the Mayo ADIR Lab MRI QC table and SERIES_SELECTED.

ADNI1 remains unresolved for subjects with tied best MPRAGE RANK values.
No arbitrary image-id, acquisition-order, or series-description tie-break will
be introduced. The preferred-series identifier must be recovered when
possible from official ADNI1 QC/process metadata; otherwise a predefined
exclusion rule will be used.

---

## ADNI1 MRI selection

### Candidate population

The primary multimodal candidate manifest contained 204 subjects whose selected
MRI examination belonged to the ADNI1 protocol family.

All 204 subjects were scanned at 1.5 T.

The selected observation dates contained 419 distinct candidate raw T1-weighted
LONI series.

### Primary QC source

ADNI1 raw T1 series were evaluated using the official MRI MPRAGE Ranking table:

MRIMPRANK_21Sep2026.csv

Passing ranks were defined as RANK values 1, 2, or 3.

For each subject, the lowest passing RANK was considered the best rank.

### Unique best-rank selection

A subject was selected directly when exactly one candidate LONI series had the
best passing RANK.

This resolved 73 subjects.

### Resolution of tied best ranks

A total of 127 subjects had more than one candidate series sharing the best
passing RANK.

No arbitrary tie-break based on image identifier, acquisition order, SCAN,
series description, or model performance was used.

The official ADNI 1.5 T MRI Standardized Lists were used as an independent
selection source.

For a tied subject, a tie was resolved only when exactly one of the tied
best-ranked raw LONI series appeared as the standardized Series.ID for the same
RID and exact primary MRI acquisition date.

This rule resolved 101 of the 127 tied subjects.

No conflicts were observed between the standardized-list selection and the set
of best-ranked MRIMPRANK candidate series.

The remaining 26 tied subjects had no corresponding best-ranked series in the
standardized lists, including when the same RID was searched across all
available standardized-list dates. These subjects were therefore excluded
rather than resolved using an arbitrary tie-break.

### Subjects without a passing rank

Four subjects had no candidate series with a passing MRIMPRANK value and were
excluded.

### Final ADNI1 selection

Total candidate subjects: 204

Selected subjects: 174

Selection source:

- MRIMPRANK unique best rank: 73
- ADNI 1.5 T standardized list resolving a tied best rank: 101

Excluded subjects: 30

Exclusion reasons:

- unresolved tied best rank with no usable standardized-series match: 26
- no passing MRIMPRANK candidate series: 4

Selected diagnostic distribution:

- CN: 46
- MCI: 87
- Dementia: 41

Excluded diagnostic distribution:

- CN: 12
- MCI: 12
- Dementia: 6

MRI selection was fixed before model training and was not determined using
classification performance.

### Reproducibility

Primary selection output:

data/interim/cohort/adni1_mri_selection_v1.csv

Aggregate summary:

tables/adni1_mri_selection_v1_summary.json

Relevant scripts:

- src/cohort/07_audit_adni1_mri_ranking.py
- src/cohort/11_audit_adni1_standardized_lists.py
- src/cohort/12_resolve_adni1_ties_with_standardized_lists.py
- src/cohort/13_audit_adni1_unresolved_ties_global_series.py
- src/cohort/14_finalize_adni1_mri_selection.py

---

## ADNI3 MRI selection

### Candidate population

The primary multimodal candidate manifest contained 104 subjects whose selected
MRI examination belonged to the ADNI3 protocol family.

These observations contained 119 distinct candidate T1-weighted LONI series.

### QC source

ADNI3 MRI selection used the official Mayo ADIR Lab GO/2/3 MRI QC table:

MAYOADIRL_MRI_QUALITY_ADNI3.rda

Candidate Key MRI series were matched to Mayo QC using the LONI series
identifier, with RID consistency additionally verified.

### Primary ADNI3 rule

A series was eligible when all of the following were satisfied:

1. An exact Mayo QC match existed for the candidate LONI series.
2. The Mayo QC RID matched the cohort RID.
3. QC records for the series were internally consistent.
4. SERIES_SELECTED = True.
5. SERIES_QUALITY was one of {1, 2, 3}.
6. STUDY_OVERALLPASS = 1.

PROTOCOL_STATUS was retained as descriptive QC metadata but was not used as an
exclusion criterion because the available ADNI data dictionary did not define
the meaning of its numerical 1/2/3 coding.

Among the 103 selected T1 series, PROTOCOL_STATUS values were:

- 1: 101 series
- 2: 1 series
- 3: 1 series

All 103 had STUDY_OVERALLPASS = 1 and SERIES_QUALITY in {1, 2, 3}.

### Final ADNI3 selection

Total candidate subjects: 104

Selected subjects: 103

Excluded subjects: 1

The single excluded subject was RID 6679.

Its primary MRI acquisition was dated 2019-01-28, but no matching Mayo QC entry
was found by LONI series or SeriesInstanceUID. The available Mayo QC records for
this RID corresponded to a different examination dated 2021-05-07 and were not
used to validate the 2019 acquisition.

Exclusion reason:

- missing QC match for the primary candidate series: 1

Selected diagnostic distribution:

- CN: 0
- MCI: 73
- Dementia: 30

Excluded diagnostic distribution:

- CN: 0
- MCI: 1
- Dementia: 0

MRI selection was fixed independently of downstream model performance.

### Reproducibility

Primary selection output:

data/interim/cohort/adni3_mri_selection_v1.csv

Aggregate summary:

tables/adni3_mri_selection_v1_summary.json

Selection script:

src/cohort/15_finalize_adni3_mri_selection.py

---

## MRI selection status

ADNI1 selection is locked.

ADNIGO/ADNI2 accelerated T1 selection is locked.

ADNI3 selection is locked.

The three phase-specific selections must next be merged into one validated MRI
selection manifest before any primary-cohort MRI image volumes are downloaded.
