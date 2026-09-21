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
