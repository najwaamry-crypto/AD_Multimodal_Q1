from pathlib import Path
import json
import re
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

REGISTRY_PATH = (
    ROOT / "data/raw/metadata/extracted/"
    "REGISTRY_18Sep2026.csv"
)

OUT_PATH = (
    ROOT / "data/interim/cohort/"
    "demographics_genetics_manifest_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables/"
    "demographics_genetics_manifest_v1_summary.json"
)


def normalize_genotype(value):
    if pd.isna(value):
        return pd.NA

    s = (
        str(value)
        .strip()
        .replace("ε", "")
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

    return int(
        sum(
            x == "4"
            for x in str(genotype).split("/")
        )
    )


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

apoeres = pd.read_csv(
    APOE_PATH,
    low_memory=False
)

registry = pd.read_csv(
    REGISTRY_PATH,
    low_memory=False
)


# ============================================================
# COHORT
# ============================================================

if len(cohort) != 822:
    raise RuntimeError(
        f"Expected 822 subjects, got {len(cohort)}"
    )

if cohort["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate RID in post-MRI-QC cohort."
    )

cohort["RID"] = pd.to_numeric(
    cohort["RID"],
    errors="raise"
).astype(int)

cohort["pet_date"] = pd.to_datetime(
    cohort["pet_date"],
    errors="raise"
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

adsl = adsl[
    adsl["RID"].isin(
        cohort["RID"]
    )
].copy()

if len(adsl) != 822:
    raise RuntimeError(
        f"Expected 822 matching ADSL rows, got {len(adsl)}"
    )

if adsl["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate ADSL rows for target RID."
    )

adsl["AGE"] = pd.to_numeric(
    adsl["AGE"],
    errors="coerce"
)

adsl["EDUC"] = pd.to_numeric(
    adsl["EDUC"],
    errors="coerce"
)

adsl["apoe_genotype"] = (
    adsl["APOE"]
    .apply(normalize_genotype)
)


# ============================================================
# ORIGINAL-PROTOCOL BASELINE
# ============================================================

registry["RID"] = pd.to_numeric(
    registry["RID"],
    errors="coerce"
)

registry["EXAMDATE"] = pd.to_datetime(
    registry["EXAMDATE"],
    errors="coerce"
)

baseline = registry[
    registry["VISCODE2"]
    .astype(str)
    .str.lower()
    .eq("bl")
].merge(
    adsl[
        [
            "RID",
            "ORIGPROT",
        ]
    ],
    how="inner",
    on="RID"
)

baseline = baseline[
    baseline["PHASE"]
    .astype(str)
    .eq(
        baseline["ORIGPROT"]
        .astype(str)
    )
].copy()

counts = (
    baseline
    .groupby("RID")
    .size()
)

if len(counts) != 822:
    raise RuntimeError(
        "Not every target RID has an original-protocol baseline."
    )

if not counts.eq(1).all():
    raise RuntimeError(
        "Expected exactly one original-protocol baseline per RID."
    )

baseline = baseline[
    [
        "RID",
        "EXAMDATE",
    ]
].rename(
    columns={
        "EXAMDATE": "baseline_date"
    }
)


# ============================================================
# FINAL SUBJECT-LEVEL VARIABLES
# ============================================================

final = cohort[
    [
        "RID",
        "PTID",
        "diagnosis",
        "dx_phase",
        "pet_visit",
        "pet_date",
        "mri_visit",
        "mri_date",
    ]
].merge(
    adsl[
        [
            "RID",
            "ORIGPROT",
            "AGE",
            "SEX",
            "EDUC",
            "apoe_genotype",
        ]
    ],
    how="left",
    on="RID",
    validate="one_to_one"
).merge(
    baseline,
    how="left",
    on="RID",
    validate="one_to_one"
)

final = final.rename(
    columns={
        "ORIGPROT": "orig_protocol",
        "AGE": "age_at_baseline",
        "SEX": "sex",
        "EDUC": "education_years",
    }
)

final["baseline_to_pet_days"] = (
    final["pet_date"]
    -
    final["baseline_date"]
).dt.days

final["age_at_index"] = (
    final["age_at_baseline"]
    +
    final["baseline_to_pet_days"]
    / 365.25
)

final["apoe4_count"] = (
    final["apoe_genotype"]
    .apply(apoe4_count)
)

final["apoe4_carrier"] = (
    final["apoe4_count"]
    .map({
        0: 0,
        1: 1,
        2: 1,
    })
)


# ============================================================
# VALIDATE APOERES CONCORDANCE
# ============================================================

apoeres["RID"] = pd.to_numeric(
    apoeres["RID"],
    errors="coerce"
)

apoeres = apoeres[
    apoeres["RID"].isin(
        cohort["RID"]
    )
].copy()

apoeres["genotype_norm"] = (
    apoeres["GENOTYPE"]
    .apply(normalize_genotype)
)

raw = (
    apoeres.dropna(
        subset=["genotype_norm"]
    )
    .groupby("RID")
    .agg(
        raw_genotype=("genotype_norm", "first"),
        genotype_nunique=("genotype_norm", "nunique"),
    )
    .reset_index()
)

if (
    raw["genotype_nunique"] > 1
).any():
    raise RuntimeError(
        "Conflicting APOERES genotypes detected."
    )

check = final.merge(
    raw[
        [
            "RID",
            "raw_genotype",
        ]
    ],
    how="left",
    on="RID",
    validate="one_to_one"
)

both = (
    check["apoe_genotype"].notna()
    &
    check["raw_genotype"].notna()
)

if not (
    check.loc[
        both,
        "apoe_genotype"
    ].values
    ==
    check.loc[
        both,
        "raw_genotype"
    ].values
).all():
    raise RuntimeError(
        "ADSL and APOERES genotype disagreement detected."
    )

concordant_n = int(
    both.sum()
)


# ============================================================
# VALIDATION
# ============================================================

for col in [
    "age_at_baseline",
    "sex",
    "education_years",
    "baseline_date",
    "age_at_index",
]:
    if final[col].isna().any():
        raise RuntimeError(
            f"Unexpected missing values in {col}."
        )

if final["RID"].duplicated().any():
    raise RuntimeError(
        "Final demographic/genetic manifest has duplicate RID."
    )

if len(final) != 822:
    raise RuntimeError(
        f"Expected 822 final rows, got {len(final)}"
    )


# ============================================================
# SUMMARY
# ============================================================

apoe_counts = (
    final["apoe4_count"]
    .value_counts(dropna=False)
    .sort_index()
)

summary = {
    "subjects": 822,

    "age_available": int(
        final["age_at_index"]
        .notna()
        .sum()
    ),

    "sex_available": int(
        final["sex"]
        .notna()
        .sum()
    ),

    "education_available": int(
        final["education_years"]
        .notna()
        .sum()
    ),

    "apoe_available": int(
        final["apoe_genotype"]
        .notna()
        .sum()
    ),

    "apoe_missing": int(
        final["apoe_genotype"]
        .isna()
        .sum()
    ),

    "apoe_source_validation_concordant":
        concordant_n,

    "apoe4_count_distribution": {
        ("missing" if pd.isna(k) else str(int(k))):
            int(v)
        for k, v in apoe_counts.items()
    },

    "pet_before_baseline": int(
        (
            final["baseline_to_pet_days"] < 0
        ).sum()
    ),

    "minimum_baseline_to_pet_days": int(
        final["baseline_to_pet_days"]
        .min()
    ),

    "maximum_baseline_to_pet_days": int(
        final["baseline_to_pet_days"]
        .max()
    ),

    "age_at_index": {
        "mean": round(
            float(
                final["age_at_index"]
                .mean()
            ),
            3
        ),

        "std": round(
            float(
                final["age_at_index"]
                .std()
            ),
            3
        ),

        "min": round(
            float(
                final["age_at_index"]
                .min()
            ),
            3
        ),

        "median": round(
            float(
                final["age_at_index"]
                .median()
            ),
            3
        ),

        "max": round(
            float(
                final["age_at_index"]
                .max()
            ),
            3
        ),
    },
}


# ============================================================
# SAVE
# ============================================================

final.to_csv(
    OUT_PATH,
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
print("FINAL DEMOGRAPHICS / GENETICS MANIFEST")
print("=" * 88)

print("\nSubjects:", len(final))

print("\nCoverage:")
print(
    "  Age:",
    final["age_at_index"].notna().sum(),
    "/822"
)
print(
    "  Sex:",
    final["sex"].notna().sum(),
    "/822"
)
print(
    "  Education:",
    final["education_years"].notna().sum(),
    "/822"
)
print(
    "  APOE:",
    final["apoe_genotype"].notna().sum(),
    "/822"
)

print("\nAPOE4 count:")
print(
    final["apoe4_count"]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .to_string()
)

print(
    "\nADSL vs APOERES concordant subjects:",
    concordant_n
)

print(
    "PET before baseline:",
    summary["pet_before_baseline"]
)

print(
    "Minimum PET-baseline difference:",
    summary["minimum_baseline_to_pet_days"],
    "days"
)

print("\nAge at PET index:")
print(
    final["age_at_index"]
    .describe()
    .to_string()
)

print("\nMissing APOE:")
print(
    final.loc[
        final["apoe_genotype"].isna(),
        [
            "RID",
            "PTID",
            "diagnosis",
        ]
    ].to_string(index=False)
)

print(
    "\nSaved manifest:\n",
    OUT_PATH
)

print(
    "\nSaved summary:\n",
    OUT_SUMMARY
)
