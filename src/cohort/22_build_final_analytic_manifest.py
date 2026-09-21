from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BASE_PATH = (
    ROOT / "data/interim/cohort/"
    "primary_post_mri_qc_manifest_v1.csv"
)

DEMO_PATH = (
    ROOT / "data/interim/cohort/"
    "demographics_genetics_manifest_v1.csv"
)

COG_PATH = (
    ROOT / "data/interim/cohort/"
    "cognitive_manifest_v1.csv"
)

OUT_PATH = (
    ROOT / "data/interim/cohort/"
    "analytic_manifest_v1.csv"
)

SUMMARY_PATH = (
    ROOT / "tables/"
    "analytic_manifest_v1_summary.json"
)

# ============================================================
# LOAD
# ============================================================

base = pd.read_csv(
    BASE_PATH,
    low_memory=False
)

demo = pd.read_csv(
    DEMO_PATH,
    low_memory=False
)

cog = pd.read_csv(
    COG_PATH,
    low_memory=False
)

datasets = {
    "base": base,
    "demo": demo,
    "cog": cog,
}

for name, df in datasets.items():

    df["RID"] = pd.to_numeric(
        df["RID"],
        errors="raise"
    ).astype(int)

    if len(df) != 822:
        raise RuntimeError(
            f"{name}: expected 822 rows, got {len(df)}"
        )

    if df["RID"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate RID detected."
        )

# ============================================================
# RID IDENTITY
# ============================================================

base_rids = set(base["RID"])
demo_rids = set(demo["RID"])
cog_rids = set(cog["RID"])

if not (
    base_rids
    == demo_rids
    == cog_rids
):
    raise RuntimeError(
        "RID sets differ across source manifests."
    )

# ============================================================
# CROSS-MANIFEST CONSISTENCY
# ============================================================

date_cols = {
    "pet_date",
    "mri_date",
}

def compare_column(left, right, col, label):

    a = left[
        ["RID", col]
    ].copy()

    b = right[
        ["RID", col]
    ].copy()

    if col in date_cols:

        a[col] = pd.to_datetime(
            a[col],
            errors="coerce"
        )

        b[col] = pd.to_datetime(
            b[col],
            errors="coerce"
        )

    check = a.merge(
        b,
        on="RID",
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right")
    )

    x = check[f"{col}_left"]
    y = check[f"{col}_right"]

    equal = (
        x.eq(y)
        |
        (
            x.isna()
            &
            y.isna()
        )
    )

    if not equal.all():

        bad = check.loc[
            ~equal,
            [
                "RID",
                f"{col}_left",
                f"{col}_right",
            ]
        ]

        raise RuntimeError(
            f"{label}: disagreement in {col}\n"
            f"{bad.head(20).to_string(index=False)}"
        )


for col in [
    "PTID",
    "diagnosis",
    "dx_phase",
    "pet_visit",
    "pet_date",
    "mri_visit",
    "mri_date",
]:
    compare_column(
        base,
        demo,
        col,
        "BASE vs DEMOGRAPHICS"
    )

for col in [
    "PTID",
    "diagnosis",
    "pet_visit",
    "pet_date",
]:
    compare_column(
        base,
        cog,
        col,
        "BASE vs COGNITIVE"
    )

# ============================================================
# BASE IMAGING / DIAGNOSIS MANIFEST
# ============================================================

analytic = base.copy()

# ============================================================
# ADD DEMOGRAPHIC / GENETIC VARIABLES
# ============================================================

demo_extra = [
    "RID",
    "orig_protocol",
    "age_at_baseline",
    "baseline_date",
    "baseline_to_pet_days",
    "age_at_index",
    "sex",
    "education_years",
    "apoe_genotype",
    "apoe4_count",
    "apoe4_carrier",
]

analytic = analytic.merge(
    demo[
        demo_extra
    ],
    how="left",
    on="RID",
    validate="one_to_one"
)

# ============================================================
# ADD COGNITIVE VARIABLES
# ============================================================

cog_extra = [
    c
    for c in cog.columns
    if c not in {
        "RID",
        "PTID",
        "diagnosis",
        "pet_visit",
        "pet_date",
    }
]

analytic = analytic.merge(
    cog[
        ["RID"] + cog_extra
    ],
    how="left",
    on="RID",
    validate="one_to_one"
)

# ============================================================
# LOCKED ANALYTIC BLOCKS
# ============================================================

primary_non_cognitive = [
    "age_at_index",
    "sex",
    "education_years",
    "apoe4_count",
]

secondary_cognitive = [
    "MMSE",
    "ADAS13",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_30MIN",
]

# ============================================================
# VALIDATIONS
# ============================================================

if len(analytic) != 822:
    raise RuntimeError(
        f"Final manifest expected 822 rows, got {len(analytic)}"
    )

if analytic["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate RID in final analytic manifest."
    )

if not analytic["mri_selected"].eq(True).all():
    raise RuntimeError(
        "Final cohort contains a subject without selected MRI."
    )

required_imaging = [
    "pet_image_id",
    "selected_loni_series",
    "selected_image_ids",
    "selected_series_instance_uids",
]

for col in required_imaging:

    if analytic[col].isna().any():

        bad = analytic.loc[
            analytic[col].isna(),
            [
                "RID",
                "PTID",
                col,
            ]
        ]

        raise RuntimeError(
            f"Missing required imaging identifier: {col}\n"
            f"{bad.to_string(index=False)}"
        )

required_primary = [
    "age_at_index",
    "sex",
    "education_years",
]

for col in required_primary:

    if analytic[col].isna().any():
        raise RuntimeError(
            f"Unexpected missing primary covariate: {col}"
        )

# APOE is intentionally allowed to be missing in 2 subjects.

diagnosis_counts = (
    analytic["diagnosis"]
    .value_counts()
    .to_dict()
)

expected_diagnosis = {
    "MCI": 438,
    "CN": 224,
    "Dementia": 160,
}

if diagnosis_counts != expected_diagnosis:
    raise RuntimeError(
        "Diagnosis counts differ from locked post-MRI-QC cohort:\n"
        f"{diagnosis_counts}"
    )

# ============================================================
# SUMMARY
# ============================================================

secondary_complete = (
    analytic[
        secondary_cognitive
    ]
    .notna()
    .all(axis=1)
)

summary = {
    "version": "analytic_manifest_v1",

    "subjects": int(
        len(analytic)
    ),

    "unique_rids": int(
        analytic["RID"].nunique()
    ),

    "diagnosis_counts": {
        str(k): int(v)
        for k, v in diagnosis_counts.items()
    },

    "dx_phase_counts": {
        str(k): int(v)
        for k, v in (
            analytic["dx_phase"]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "primary_non_cognitive_block":
        primary_non_cognitive,

    "primary_non_cognitive_coverage": {
        col: int(
            analytic[col]
            .notna()
            .sum()
        )
        for col in primary_non_cognitive
    },

    "secondary_cognitive_block":
        secondary_cognitive,

    "secondary_cognitive_coverage": {
        col: int(
            analytic[col]
            .notna()
            .sum()
        )
        for col in secondary_cognitive
    },

    "secondary_cognitive_complete_cases":
        int(
            secondary_complete.sum()
        ),

    "secondary_cognitive_complete_percent":
        round(
            100 * secondary_complete.mean(),
            2
        ),

    "imaging_identifier_coverage": {
        col: int(
            analytic[col]
            .notna()
            .sum()
        )
        for col in required_imaging
    },

    "apoe_missing": int(
        analytic["apoe4_count"]
        .isna()
        .sum()
    ),

    "split_status":
        "not yet assigned",

    "model_training_status":
        "not started",

    "notes": [
        (
            "Primary cohort remains N=822."
        ),
        (
            "Cognitive missingness does not remove subjects "
            "from the primary cohort."
        ),
        (
            "No train/validation/test split is created by "
            "this script."
        ),
        (
            "No preprocessing or model-performance criterion "
            "is used in cohort construction."
        ),
    ],
}

# ============================================================
# SAVE
# ============================================================

analytic.to_csv(
    OUT_PATH,
    index=False
)

with open(
    SUMMARY_PATH,
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

print("=" * 96)
print("FINAL ANALYTIC MULTIMODAL MANIFEST V1")
print("=" * 96)

print(
    "\nShape:",
    analytic.shape
)

print(
    "Unique RID:",
    analytic["RID"].nunique()
)

print(
    "Duplicate RID:",
    int(
        analytic["RID"]
        .duplicated()
        .sum()
    )
)

print("\nDiagnosis:")
print(
    analytic["diagnosis"]
    .value_counts()
    .to_string()
)

print("\nDiagnosis phase:")
print(
    analytic["dx_phase"]
    .value_counts()
    .to_string()
)

print("\nImaging identifier coverage:")
for col in required_imaging:
    print(
        f"{col:35s}: "
        f"{analytic[col].notna().sum()}/822"
    )

print("\nPrimary non-cognitive block:")
for col in primary_non_cognitive:
    print(
        f"{col:25s}: "
        f"{analytic[col].notna().sum()}/822"
    )

print("\nSecondary cognitive block:")
for col in secondary_cognitive:
    print(
        f"{col:25s}: "
        f"{analytic[col].notna().sum()}/822"
    )

print(
    "\nSecondary complete cases:",
    int(
        secondary_complete.sum()
    ),
    "/822",
    f"({100*secondary_complete.mean():.2f}%)"
)

print(
    "\nAPOE missing:",
    int(
        analytic["apoe4_count"]
        .isna()
        .sum()
    )
)

print("\nSaved:")
print(OUT_PATH)
print(SUMMARY_PATH)

print(
    "\nNo split created. No model training performed."
)
