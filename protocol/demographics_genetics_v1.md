# Demographics and genetics definition v1

## Population

This definition applies to the 822 subjects retained after the locked
phase-specific MRI QC and selection procedure.

No subject was excluded during the demographics/genetics audit.

## Index date

The primary multimodal observation is PET-centered.

Therefore, PET acquisition date is used as the index date for age alignment.

## Age

ADSL AGE is documented as age at baseline and is not used directly as age at
the primary multimodal observation.

For every one of the 822 subjects, exactly one original-protocol baseline visit
was identified in REGISTRY using:

- RID equality
- REGISTRY.PHASE = ADSL.ORIGPROT
- REGISTRY.VISCODE2 = "bl"

All 822 subjects had exactly one such baseline.

The resulting baseline date was identical to ADSL.ENRLDT for all 822 subjects.

Age at the PET-centered index observation is defined as:

age_at_index =
    age_at_baseline
    + (pet_date - baseline_date) / 365.25

Age availability: 822 / 822.

Fifty-three PET scans occurred before the formal baseline date, with the
largest negative offset equal to 31 days. These observations were retained;
the signed date difference is used in the age calculation.

## Sex

Sex is taken from ADSL.SEX.

Availability: 822 / 822.

Observed distribution:

- Male: 455
- Female: 367

## Education

Years of education are taken from ADSL.EDUC.

Availability: 822 / 822.

Observed distribution:

- mean: 16.11 years
- median: 16 years
- minimum: 7 years
- maximum: 20 years

## APOE genotype

The primary integrated genotype variable is taken from ADSL.APOE and normalized
to unordered allele pairs:

- 2/2
- 2/3
- 2/4
- 3/3
- 3/4
- 4/4

APOERES was used as an independent validation source.

Among the 822 post-MRI-QC subjects:

- ADSL APOE available: 820
- raw APOERES genotype available: 820
- both sources available: 820
- concordant: 820
- discordant: 0

APUSABLE was not used as a universal inclusion criterion because it is populated
as "Yes" for ADNI1 but is systematically missing for later ADNI phases despite
the presence of genotype results.

Two subjects have no genotype in either source:

- RID 4362
- RID 7079

They remain in the cohort with APOE encoded as missing.

## APOE4 variables

APOE4 count is defined as the number of epsilon-4 alleles:

- 0: no epsilon-4 allele
- 1: one epsilon-4 allele
- 2: two epsilon-4 alleles

Distribution:

- APOE4 count 0: 442
- APOE4 count 1: 294
- APOE4 count 2: 84
- missing: 2

APOE4 carrier status is additionally defined as:

- 0 if APOE4 count = 0
- 1 if APOE4 count = 1 or 2
- missing if genotype is unavailable

## Locked variables

The demographic/genetic covariates for downstream modeling are:

- age_at_index
- sex
- education_years
- apoe4_count

APOE genotype and APOE4 carrier status are retained as additional metadata.

No demographic or genetic threshold was selected based on downstream model
performance.

## Reproducibility

Audit script:

src/cohort/17_audit_demographics_genetics.py

Finalization script:

src/cohort/18_finalize_demographics_genetics.py

Aggregate audit summary:

tables/demographics_genetics_audit_v1_summary.json

Final aggregate summary:

tables/demographics_genetics_manifest_v1_summary.json

Subject-level outputs under data/interim are retained locally and are not
committed to Git.
