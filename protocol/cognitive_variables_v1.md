# Cognitive variables definition v1

## Population

The cognitive audit applies to the 822 subjects retained after the locked
phase-specific MRI QC and selection procedure.

No subject is removed from the primary cohort because of missing cognitive
data.

The primary multimodal cohort therefore remains N = 822.

## Index date

The primary multimodal observation is PET-centered.

PET acquisition date is therefore used as the temporal reference for cognitive
assessment alignment.

## Cognitive alignment window

Candidate cognitive observations were audited using exact-date, same-visit,
±30-day, ±60-day, and ±90-day alignment relative to PET acquisition.

A ±60-day window is fixed before model training.

This choice reflects the observed ADNI acquisition structure:

- ADAS and RAVLT were generally acquired very close to PET, with median nearest
  absolute gaps of approximately 4 days.
- MMSE and CDR were frequently collected at screening while PET was acquired
  around baseline, producing a median nearest gap of approximately 29 days.
- A ±30-day rule would discard a large proportion of otherwise temporally
  related MMSE/CDR observations.
- Extending the window to ±90 days provides only a limited increase in
  coverage while allowing substantially larger temporal separation.

No temporal threshold was selected using downstream classification
performance.

## Selection rule

For each cognitive variable:

1. identify all available observations for the subject;
2. compute the absolute difference between cognitive assessment date and PET
   acquisition date;
3. retain the nearest observation if its absolute difference is <= 60 days;
4. if equally near observations contain the same score, the score may be
   retained;
5. if equally near observations contain conflicting scores, the variable is
   treated as missing;
6. observations outside ±60 days are treated as unavailable for the indexed
   multimodal observation.

No cognitive value is imputed at this stage.

## QC fields

HAS_QC_ERROR was examined in MMSE, CDR, ADAS, and NEUROBAT.

Within the ±60-day window, no row had an explicit HAS_QC_ERROR=True value.

The field was missing for all relevant rows and is therefore not interpreted
as proof of formal QC passage.

For MMSE, 790 scored rows were available within ±60 days:

- DONE = Yes: 102
- DONE missing: 688

The historical absence of DONE was not used to discard an otherwise available
MMSCORE.

## Variable definitions

### MMSE

Source:

MMSE.MMSCORE

Range documented in DATADIC:

0-30.

Availability within ±60 days:

790 / 822 (96.11%).

### ADAS-11

Source:

ADAS.TOTSCORE

Definition:

Classic 70-point ADAS-Cog total excluding delayed word recall and number
cancellation.

Availability within ±60 days:

818 / 822 (99.51%).

ADAS-11 is retained as an auxiliary cognitive variable.

### ADAS-13

Source:

ADAS.TOTAL13

Definition:

85-point ADAS-Cog total including delayed word recall and number cancellation.

Availability within ±60 days:

810 / 822 (98.54%).

ADAS-13 is included in the pre-specified secondary cognitive panel.

### CDR Global

Source:

CDR.CDGLOBAL

Availability within ±60 days:

787 / 822 (95.74%).

CDR Global is retained as an auxiliary cognitive variable.

### CDR Sum of Boxes

Source:

CDR.CDRSB

Documented range:

0-18.

Availability within ±60 days:

787 / 822 (95.74%).

CDR-SB is included in the pre-specified secondary cognitive panel.

CDR variables are never used to define or reconstruct the classification
target.

### Logical Memory immediate recall

Source:

NEUROBAT.LIMMTOTAL

Documented range:

0-25.

Availability within ±60 days:

725 / 822 (88.20%).

This variable is retained as auxiliary metadata.

RID 292 had conflicting equally distant nearest immediate-recall observations,
both outside the ±60-day window. The indexed value is therefore missing.

### Logical Memory delayed recall

Source:

NEUROBAT.LDELTOTAL

Documented range:

0-25.

Availability within ±60 days:

724 / 822 (88.08%).

Logical Memory delayed recall is included in the pre-specified secondary
cognitive panel.

### RAVLT learning sum

A provisional RAVLT learning score is defined as:

AVTOT1 + AVTOT2 + AVTOT3 + AVTOT4 + AVTOT5

The score is computed only when all five trials are available.

Availability within ±60 days:

819 / 822 (99.64%).

RID 4001 has incomplete baseline RAVLT learning trials and remains missing for
this variable. No missing trial is imputed.

### RAVLT trial 6

Source:

NEUROBAT.AVTOT6

Availability within ±60 days:

819 / 822 (99.64%).

This variable is retained as auxiliary metadata.

### RAVLT 30-minute delayed recall

Source:

NEUROBAT.AVDEL30MIN

Documented range:

0-15.

Availability within ±60 days:

817 / 822 (99.39%).

This variable is included in the pre-specified secondary cognitive panel.

### RAVLT recognition

Source:

NEUROBAT.AVDELTOT

Availability within ±60 days:

816 / 822 (99.27%).

This variable is retained as auxiliary metadata.

## Primary versus cognitive clinical information

The primary non-cognitive clinical block is:

- age_at_index
- sex
- education_years
- apoe4_count

Cognitive assessments are deliberately separated from this block because they
are closely related to the clinical diagnostic process and can dominate
classification performance.

The pre-specified secondary cognitive panel is:

- MMSE
- ADAS-13
- CDR-SB
- Logical Memory delayed recall
- RAVLT trials 1-5 sum
- RAVLT 30-minute delayed recall

Complete data for all six secondary cognitive variables are available for:

708 / 822 subjects (86.13%).

These cognitive variables are intended for secondary, benchmark, incremental,
and ablation analyses rather than being silently merged into the primary
non-cognitive clinical baseline.

## Auxiliary cognitive variables

The following variables are retained for descriptive or additional analyses:

- ADAS-11
- CDR Global
- Logical Memory immediate recall
- RAVLT trial 6
- RAVLT recognition

## Missing data

Missing cognitive values do not remove subjects from the primary N=822 cohort.

No cognitive imputation rule is fixed at this stage.

Any later imputation procedure must be performed within training data only and
must not use test-set information.

## Reproducibility

R export script:

src/cohort/export_cognitive_tables.R

Temporal alignment audit:

src/cohort/19_audit_cognitive_temporal_alignment.py

Window and missingness-bias audit:

src/cohort/20_audit_cognitive_window_bias.py

Final cognitive manifest script:

src/cohort/21_finalize_cognitive_manifest.py

Aggregate temporal audit:

tables/cognitive_temporal_audit_v1_summary.json

Aggregate final manifest summary:

tables/cognitive_manifest_v1_summary.json

Subject-level files under data/interim remain local and are not committed to
Git.
