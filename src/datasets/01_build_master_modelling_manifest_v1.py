#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

ANALYTIC = (
    ROOT
    / "data/interim/cohort/analytic_manifest_v1.csv"
)

MRI_SELECTION = (
    ROOT
    / "data/interim/imaging_download_manifests/"
    / "mri_final_image_selection_v1.csv"
)

SPLITS = (
    ROOT
    / "splits/split_assignments_v1.csv"
)

OUT = (
    ROOT
    / "data/processed/master_modelling_manifest_v1.csv"
)


analytic = pd.read_csv(
    ANALYTIC,
    low_memory=False
)

mri = pd.read_csv(
    MRI_SELECTION,
    low_memory=False
)

splits = pd.read_csv(
    SPLITS,
    low_memory=False
)


# --------------------------------------------------
# Integrity checks
# --------------------------------------------------

assert len(analytic) == 822
assert analytic["RID"].nunique() == 822

assert len(mri) == 822
assert mri["RID"].nunique() == 822

assert len(splits) == 822
assert splits["RID"].nunique() == 822


# --------------------------------------------------
# MRI identifiers
# --------------------------------------------------

mri["mri_image_id"] = (
    mri["mri_image_id"]
    .astype(int)
    .map(lambda x: f"I{x}")
)


# --------------------------------------------------
# Keep modelling variables
# --------------------------------------------------

clinical_cols = [
    "RID",
    "PTID",
    "diagnosis",
    "dx_phase",
    "age_at_index",
    "sex",
    "education_years",
    "apoe_genotype",
    "apoe4_count",
    "apoe4_carrier",

    # Secondary cognitive block
    "MMSE",
    "ADAS13",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_30MIN",
    "secondary_cognitive_complete",
]

clinical = analytic[clinical_cols].copy()


# --------------------------------------------------
# Merge
# --------------------------------------------------

df = clinical.merge(
    splits[
        [
            "RID",
            "split",
        ]
    ],
    on="RID",
    how="inner",
    validate="one_to_one"
)

df = df.merge(
    mri[
        [
            "RID",
            "mri_image_id",
            "field_strength",
            "selection_family",
            "selection_source",
        ]
    ],
    on="RID",
    how="inner",
    validate="one_to_one"
)


# --------------------------------------------------
# PET identifier from analytic manifest
# --------------------------------------------------

pet_map = analytic[
    [
        "RID",
        "pet_image_id",
    ]
].copy()

pet_map["pet_image_id"] = (
    pet_map["pet_image_id"]
    .astype(int)
    .map(lambda x: f"I{x}")
)

df = df.merge(
    pet_map,
    on="RID",
    how="inner",
    validate="one_to_one"
)


# --------------------------------------------------
# Build final paths
# --------------------------------------------------

df["mri_path"] = df["mri_image_id"].map(
    lambda x:
        f"data/processed/mri_v1/"
        f"{x}/{x}_mri_preprocessed.nii.gz"
)

df["pet_path"] = df["pet_image_id"].map(
    lambda x:
        f"data/processed/pet_normalized_v1/"
        f"{x}/{x}_pet_vermisnorm.nii.gz"
)


# --------------------------------------------------
# File existence checks
# --------------------------------------------------

df["mri_exists"] = df["mri_path"].map(
    lambda p: (ROOT / p).exists()
)

df["pet_exists"] = df["pet_path"].map(
    lambda p: (ROOT / p).exists()
)


# --------------------------------------------------
# Clinical block flags
# --------------------------------------------------

primary_features = [
    "age_at_index",
    "sex",
    "education_years",
    "apoe4_count",
]

cognitive_features = [
    "MMSE",
    "ADAS13",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_30MIN",
]

df["primary_clinical_complete"] = (
    df[primary_features]
    .notna()
    .all(axis=1)
    .astype(int)
)

df["cognitive_block_complete"] = (
    df[cognitive_features]
    .notna()
    .all(axis=1)
    .astype(int)
)


# --------------------------------------------------
# Stable ordering
# --------------------------------------------------

ordered_cols = [
    "RID",
    "PTID",
    "diagnosis",
    "split",
    "dx_phase",

    "mri_image_id",
    "mri_path",
    "mri_exists",

    "pet_image_id",
    "pet_path",
    "pet_exists",

    "field_strength",
    "selection_family",
    "selection_source",

    # Primary leakage-aware clinical block
    "age_at_index",
    "sex",
    "education_years",
    "apoe_genotype",
    "apoe4_count",
    "apoe4_carrier",
    "primary_clinical_complete",

    # Secondary cognitive block
    "MMSE",
    "ADAS13",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_30MIN",
    "secondary_cognitive_complete",
    "cognitive_block_complete",
]

df = df[ordered_cols]

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT,
    index=False
)


# --------------------------------------------------
# Final QC
# --------------------------------------------------

print("=== MASTER MODELLING MANIFEST ===")
print("Rows:", len(df))
print("Unique RID:", df["RID"].nunique())

print("\nDiagnosis:")
print(
    df["diagnosis"]
    .value_counts()
    .to_string()
)

print("\nSplit:")
print(
    df["split"]
    .value_counts()
    .to_string()
)

print("\nMRI files missing:",
      int((~df["mri_exists"]).sum()))

print("PET files missing:",
      int((~df["pet_exists"]).sum()))

print(
    "Primary clinical complete:",
    int(df["primary_clinical_complete"].sum()),
    "/",
    len(df)
)

print(
    "Cognitive block complete:",
    int(df["cognitive_block_complete"].sum()),
    "/",
    len(df)
)

print("\nPrimary missingness:")
for c in primary_features:
    print(
        c,
        ":",
        int(df[c].isna().sum())
    )

print("\nSaved:", OUT)
