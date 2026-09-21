from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

COHORT_PATH = (
    ROOT / "data/interim/cohort/"
    "primary_post_mri_qc_manifest_v1.csv"
)

ADSL_PATH = (
    ROOT / "data/interim/adnimerge2_exports/"
    "ADSL_demographics_v1.csv"
)

APOE_PATH = (
    ROOT / "data/interim/adnimerge2_exports/"
    "APOERES_genotype_v1.csv"
)

OUT_AUDIT = (
    ROOT / "data/interim/cohort/"
    "demographics_genetics_audit_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables/"
    "demographics_genetics_audit_v1_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_genotype(value):
    """
    Normalize representations such as:
      ε3/ε4
      3/4
      e3/e4
      E4/E3
    to canonical ordered strings:
      3/4
    """

    if pd.isna(value):
        return pd.NA

    s = str(value).strip()

    if not s:
        return pd.NA

    s = (
        s.replace("ε", "")
        .replace("Ε", "")
        .replace("e", "")
        .replace("E", "")
        .replace(" ", "")
    )

    alleles = re.findall(r"[234]", s)

    if len(alleles) != 2:
        return pd.NA

    alleles = sorted(
        int(x)
        for x in alleles
    )

    return f"{alleles[0]}/{alleles[1]}"


def apoe4_count(genotype):
    if pd.isna(genotype):
        return pd.NA

    parts = str(genotype).split("/")

    return sum(
        int(x == "4")
        for x in parts
    )


def dict_int(series):
    return {
        str(k): int(v)
        for k, v in
        series.value_counts(
            dropna=False
        ).to_dict().items()
    }


# ============================================================
# LOAD
# ============================================================

cohort = pd.read_csv(
    COHORT_PATH,
    low_memory=False
)

adsl = pd.read_csv(
    ADSL_PATH,
    low_memory=False
)

apoe = pd.read_csv(
    APOE_PATH,
    low_memory=False
)


# ============================================================
# PRIMARY COHORT VALIDATION
# ============================================================

if len(cohort) != 822:
    raise RuntimeError(
        f"Expected 822 post-MRI-QC subjects, got {len(cohort)}"
    )

if cohort["RID"].duplicated().any():
    raise RuntimeError(
        "Post-MRI-QC cohort contains duplicate RIDs."
    )

cohort["RID"] = pd.to_numeric(
    cohort["RID"],
    errors="raise"
).astype(int)

cohort["pet_date"] = pd.to_datetime(
    cohort["pet_date"],
    errors="coerce"
)

if cohort["pet_date"].isna().any():
    raise RuntimeError(
        "Primary cohort contains missing PET index dates."
    )


# ============================================================
# ADSL
# ============================================================

adsl["RID"] = pd.to_numeric(
    adsl["SUBJID"],
    errors="coerce"
)

adsl = adsl[
    adsl["RID"].notna()
].copy()

adsl["RID"] = (
    adsl["RID"]
    .astype(int)
)

# ADSL should be subject level.
adsl_target = adsl[
    adsl["RID"].isin(
        cohort["RID"]
    )
].copy()

duplicate_adsl = adsl_target[
    adsl_target["RID"].duplicated(
        keep=False
    )
].copy()

print("=" * 88)
print("DEMOGRAPHICS / GENETICS AUDIT - 822 POST-MRI-QC SUBJECTS")
print("=" * 88)

print(
    "\nADSL rows matching target RIDs:",
    len(adsl_target)
)

print(
    "Unique target RIDs in ADSL:",
    adsl_target["RID"].nunique()
)

print(
    "Target RIDs with duplicate ADSL rows:",
    duplicate_adsl["RID"].nunique()
)

if duplicate_adsl["RID"].nunique() > 0:
    raise RuntimeError(
        "ADSL is not one-row-per-RID for target subjects."
    )

adsl_target["AGE"] = pd.to_numeric(
    adsl_target["AGE"],
    errors="coerce"
)

adsl_target["EDUC"] = pd.to_numeric(
    adsl_target["EDUC"],
    errors="coerce"
)

adsl_target["ENRLDT"] = pd.to_datetime(
    adsl_target["ENRLDT"],
    errors="coerce"
)

adsl_target["adsl_apoe_norm"] = (
    adsl_target["APOE"]
    .apply(normalize_genotype)
)

adsl_keep = adsl_target[
    [
        "RID",
        "ORIGPROT",
        "AGE",
        "AGEU",
        "SEX",
        "EDUC",
        "ENRLDT",
        "APOE",
        "adsl_apoe_norm",
    ]
].copy()

adsl_keep = adsl_keep.rename(
    columns={
        "ORIGPROT": "adsl_origprot",
        "AGE": "adsl_age",
        "AGEU": "adsl_age_unit",
        "SEX": "sex",
        "EDUC": "education_years",
        "ENRLDT": "adsl_enrollment_date",
        "APOE": "adsl_apoe_raw",
    }
)


# ============================================================
# APOERES
# ============================================================

apoe["RID"] = pd.to_numeric(
    apoe["RID"],
    errors="coerce"
)

apoe = apoe[
    apoe["RID"].notna()
].copy()

apoe["RID"] = (
    apoe["RID"]
    .astype(int)
)

apoe = apoe[
    apoe["RID"].isin(
        cohort["RID"]
    )
].copy()

apoe["APTESTDT"] = pd.to_datetime(
    apoe["APTESTDT"],
    errors="coerce"
)

apoe["genotype_norm"] = (
    apoe["GENOTYPE"]
    .apply(normalize_genotype)
)

apoe["usable"] = (
    apoe["APUSABLE"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("yes")
)

print(
    "\nAPOERES rows for target RIDs:",
    len(apoe)
)

print(
    "Target RIDs appearing in APOERES:",
    apoe["RID"].nunique()
)

print(
    "Target RIDs with >=1 APUSABLE=Yes:",
    apoe.loc[
        apoe["usable"],
        "RID"
    ].nunique()
)


# ============================================================
# SUBJECT-LEVEL APOERES AUDIT
# ============================================================

apoe_records = []

for rid in cohort["RID"]:

    grp = apoe[
        apoe["RID"].eq(rid)
    ].copy()

    usable = grp[
        grp["usable"]
        &
        grp["genotype_norm"].notna()
    ].copy()

    all_genotypes = sorted(
        grp["genotype_norm"]
        .dropna()
        .unique()
        .tolist()
    )

    usable_genotypes = sorted(
        usable["genotype_norm"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(usable_genotypes) == 1:
        final_genotype = usable_genotypes[0]
        genotype_status = "one_unique_usable_genotype"

    elif len(usable_genotypes) > 1:
        final_genotype = pd.NA
        genotype_status = "conflicting_usable_genotypes"

    elif len(grp) == 0:
        final_genotype = pd.NA
        genotype_status = "no_apoeres_record"

    elif len(all_genotypes) > 0:
        final_genotype = pd.NA
        genotype_status = "genotype_present_but_not_usable"

    else:
        final_genotype = pd.NA
        genotype_status = "apoeres_without_valid_genotype"

    apoe_records.append(
        {
            "RID": int(rid),

            "apoeres_row_count":
                int(len(grp)),

            "apoeres_usable_row_count":
                int(len(usable)),

            "apoeres_all_genotypes":
                "|".join(all_genotypes),

            "apoeres_usable_genotypes":
                "|".join(usable_genotypes),

            "apoeres_genotype":
                final_genotype,

            "apoeres_status":
                genotype_status,

            "apoeres_first_test_date":
                usable["APTESTDT"].min()
                if len(usable)
                else pd.NaT,

            "apoeres_last_test_date":
                usable["APTESTDT"].max()
                if len(usable)
                else pd.NaT,
        }
    )

apoe_subject = pd.DataFrame(
    apoe_records
)


# ============================================================
# MERGE TO 822-SUBJECT COHORT
# ============================================================

audit = cohort[
    [
        "RID",
        "PTID",
        "diagnosis",
        "dx_phase",
        "pet_visit",
        "pet_date",
        "mri_visit",
        "mri_date",
        "dx_visit",
        "dx_date",
        "mri_selection_family",
    ]
].copy()

audit = audit.merge(
    adsl_keep,
    how="left",
    on="RID",
    validate="one_to_one"
)

audit = audit.merge(
    apoe_subject,
    how="left",
    on="RID",
    validate="one_to_one"
)


# ============================================================
# AGE TEMPORAL AUDIT
# ============================================================

audit["years_enrollment_to_index"] = (
    (
        audit["pet_date"]
        -
        audit["adsl_enrollment_date"]
    )
    .dt.days
    /
    365.25
)

audit["age_at_index_candidate"] = (
    audit["adsl_age"]
    +
    audit["years_enrollment_to_index"]
)

# This is only an audit variable for now.
# It is NOT yet declared to be the final modeling age variable.

audit["index_before_enrollment"] = (
    audit["years_enrollment_to_index"] < 0
)


# ============================================================
# APOE CONCORDANCE
# ============================================================

audit["apoe_sources_both_available"] = (
    audit["adsl_apoe_norm"].notna()
    &
    audit["apoeres_genotype"].notna()
)

audit["apoe_sources_agree"] = pd.NA

both = audit[
    "apoe_sources_both_available"
]

audit.loc[
    both,
    "apoe_sources_agree"
] = (
    audit.loc[
        both,
        "adsl_apoe_norm"
    ]
    ==
    audit.loc[
        both,
        "apoeres_genotype"
    ]
)

# APOERES remains the explicit laboratory-result source.
# No final source hierarchy is locked yet.

audit["apoeres_apoe4_count"] = (
    audit["apoeres_genotype"]
    .apply(apoe4_count)
)

audit["adsl_apoe4_count"] = (
    audit["adsl_apoe_norm"]
    .apply(apoe4_count)
)


# ============================================================
# COVERAGE
# ============================================================

def coverage(col):
    present = int(
        audit[col]
        .notna()
        .sum()
    )

    return {
        "available": present,
        "missing": int(
            len(audit) - present
        ),
        "coverage_pct": round(
            100.0
            * present
            / len(audit),
            2
        ),
    }


coverage_summary = {
    "adsl_age":
        coverage("adsl_age"),

    "sex":
        coverage("sex"),

    "education_years":
        coverage("education_years"),

    "adsl_enrollment_date":
        coverage("adsl_enrollment_date"),

    "adsl_apoe":
        coverage("adsl_apoe_norm"),

    "apoeres_genotype":
        coverage("apoeres_genotype"),

    "age_at_index_candidate":
        coverage("age_at_index_candidate"),
}


# ============================================================
# SUMMARY
# ============================================================

both_available = audit[
    audit["apoe_sources_both_available"]
]

apoe_agree_n = int(
    both_available[
        "apoe_sources_agree"
    ]
    .eq(True)
    .sum()
)

apoe_disagree_n = int(
    both_available[
        "apoe_sources_agree"
    ]
    .eq(False)
    .sum()
)

age_nonmissing = audit[
    "age_at_index_candidate"
].dropna()

elapsed_nonmissing = audit[
    "years_enrollment_to_index"
].dropna()

summary = {
    "subjects": int(len(audit)),

    "coverage": coverage_summary,

    "sex_counts":
        dict_int(
            audit["sex"]
        ),

    "education": {
        "min": (
            float(
                audit[
                    "education_years"
                ].min()
            )
            if audit[
                "education_years"
            ].notna().any()
            else None
        ),

        "median": (
            float(
                audit[
                    "education_years"
                ].median()
            )
            if audit[
                "education_years"
            ].notna().any()
            else None
        ),

        "max": (
            float(
                audit[
                    "education_years"
                ].max()
            )
            if audit[
                "education_years"
            ].notna().any()
            else None
        ),
    },

    "apoeres_status_counts":
        dict_int(
            audit["apoeres_status"]
        ),

    "apoeres_genotype_counts":
        dict_int(
            audit["apoeres_genotype"]
        ),

    "adsl_apoe_genotype_counts":
        dict_int(
            audit["adsl_apoe_norm"]
        ),

    "apoe_sources_both_available":
        int(len(both_available)),

    "apoe_sources_agree":
        apoe_agree_n,

    "apoe_sources_disagree":
        apoe_disagree_n,

    "index_before_enrollment":
        int(
            audit[
                "index_before_enrollment"
            ].fillna(False).sum()
        ),

    "years_enrollment_to_index": {
        "min": (
            round(
                float(
                    elapsed_nonmissing.min()
                ),
                3
            )
            if len(elapsed_nonmissing)
            else None
        ),

        "median": (
            round(
                float(
                    elapsed_nonmissing.median()
                ),
                3
            )
            if len(elapsed_nonmissing)
            else None
        ),

        "max": (
            round(
                float(
                    elapsed_nonmissing.max()
                ),
                3
            )
            if len(elapsed_nonmissing)
            else None
        ),
    },

    "age_at_index_candidate": {
        "min": (
            round(
                float(
                    age_nonmissing.min()
                ),
                2
            )
            if len(age_nonmissing)
            else None
        ),

        "median": (
            round(
                float(
                    age_nonmissing.median()
                ),
                2
            )
            if len(age_nonmissing)
            else None
        ),

        "max": (
            round(
                float(
                    age_nonmissing.max()
                ),
                2
            )
            if len(age_nonmissing)
            else None
        ),
    },
}


# ============================================================
# SAVE
# ============================================================

audit.to_csv(
    OUT_AUDIT,
    index=False
)

with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 88)
print("COVERAGE")
print("=" * 88)

for variable, info in coverage_summary.items():
    print(
        f"{variable:25s} "
        f"{info['available']:3d}/822 "
        f"({info['coverage_pct']:6.2f}%) "
        f"missing={info['missing']}"
    )

print("\nSEX:")
print(
    audit["sex"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nEDUCATION:")
print(
    audit["education_years"]
    .describe()
    .to_string()
)

print("\n" + "=" * 88)
print("APOERES SUBJECT STATUS")
print("=" * 88)

print(
    audit["apoeres_status"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nAPOERES GENOTYPES:")
print(
    audit["apoeres_genotype"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nADSL APOE GENOTYPES:")
print(
    audit["adsl_apoe_norm"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nAPOE SOURCE CONCORDANCE:")
print(
    "Both available:",
    len(both_available)
)

print(
    "Agree:",
    apoe_agree_n
)

print(
    "Disagree:",
    apoe_disagree_n
)

if apoe_disagree_n:
    print("\nDISAGREEMENTS:")
    print(
        audit.loc[
            audit[
                "apoe_sources_agree"
            ].eq(False),
            [
                "RID",
                "PTID",
                "adsl_apoe_raw",
                "adsl_apoe_norm",
                "apoeres_genotype",
            ],
        ].to_string(
            index=False
        )
    )

print("\n" + "=" * 88)
print("AGE TEMPORAL AUDIT")
print("=" * 88)

print(
    "Index date definition: PET date"
)

print(
    "Subjects with index before ADSL enrollment date:",
    summary["index_before_enrollment"]
)

print(
    "\nYears enrollment -> PET index:"
)

print(
    audit[
        "years_enrollment_to_index"
    ]
    .describe()
    .to_string()
)

print(
    "\nProvisional AGE + elapsed years:"
)

print(
    audit[
        "age_at_index_candidate"
    ]
    .describe()
    .to_string()
)

print(
    "\nNOTE: age_at_index_candidate is AUDIT-ONLY "
    "and is not yet a locked modeling variable."
)

print(
    "\nSaved audit:\n",
    OUT_AUDIT
)

print(
    "\nSaved summary:\n",
    OUT_SUMMARY
)
