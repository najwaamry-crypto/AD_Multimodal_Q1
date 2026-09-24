#!/usr/bin/env python3

from pathlib import Path
import json
import shutil

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

SUMMARY = (
    ROOT
    / "data/interim/pet_lowcoverage_retest_summary_v1.csv"
)

RETEST_ROOT = (
    ROOT
    / "data/interim/pet_registration_retest_auto_v1"
)

PROCESSED_ROOT = (
    ROOT
    / "data/processed/pet_spatial_v1"
)

MRI_ROOT = (
    ROOT
    / "data/processed/mri_v1"
)


df = pd.read_csv(SUMMARY)

print("Retests to accept:", len(df))

for _, row in df.iterrows():

    rid = int(row["RID"])
    pet_id = str(row["pet_image_id"])
    mri_id = str(row["mri_image_id"])

    outdir = PROCESSED_ROOT / pet_id
    retest = RETEST_ROOT / pet_id

    backup = (
        outdir
        / "original_failed_registration"
    )
    backup.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_pet = (
        outdir
        / f"{pet_id}_mni.nii.gz"
    )

    final_support = (
        outdir
        / f"{pet_id}_support_mni.nii.gz"
    )

    final_qc = (
        outdir
        / f"{pet_id}_spatial_qc.json"
    )

    new_pet = (
        retest
        / f"{pet_id}_retest_mni.nii.gz"
    )

    new_support = (
        retest
        / f"{pet_id}_retest_support_mni.nii.gz"
    )

    # Backup original files once
    for src in [
        final_pet,
        final_support,
        final_qc,
    ]:
        dst = backup / src.name

        if (
            src.exists()
            and not dst.exists()
        ):
            shutil.copy2(src, dst)

    # Promote accepted retest
    shutil.copy2(
        new_pet,
        final_pet,
    )

    shutil.copy2(
        new_support,
        final_support,
    )

    # Load previous QC metadata
    with open(final_qc) as f:
        qc = json.load(f)

    pet_img = nib.load(final_pet)

    pet = np.asarray(
        pet_img.dataobj,
        dtype=np.float32,
    )

    support = (
        np.asarray(
            nib.load(final_support).dataobj
        ) > 0.5
    )

    brain_path = (
        MRI_ROOT
        / mri_id
        / f"{mri_id}_brainmask.nii.gz"
    )

    brain_img = nib.load(brain_path)

    brain = (
        np.asarray(
            brain_img.dataobj
        ) > 0
    )

    valid = (
        brain
        & support
        & np.isfinite(pet)
    )

    x = pet[valid]

    qc["brain_voxels"] = int(
        brain.sum()
    )

    qc["valid_brain_voxels"] = int(
        valid.sum()
    )

    qc["brain_coverage_pct"] = float(
        100.0
        * valid.sum()
        / brain.sum()
    )

    qc["brain_intensity_mean"] = float(
        x.mean()
    )

    qc["brain_intensity_std"] = float(
        x.std()
    )

    qc["brain_intensity_median"] = float(
        np.median(x)
    )

    qc["brain_intensity_p05"] = float(
        np.percentile(x, 5)
    )

    qc["brain_intensity_p95"] = float(
        np.percentile(x, 95)
    )

    qc["brain_intensity_min"] = float(
        x.min()
    )

    qc["brain_intensity_max"] = float(
        x.max()
    )

    qc["negative_brain_voxels"] = int(
        (x < 0).sum()
    )

    qc["negative_brain_pct"] = float(
        100.0
        * (x < 0).sum()
        / x.size
    )

    qc["registration_qc"] = (
        "automatic_retest_pass"
    )

    qc["registration_retest_reason"] = (
        "Initial PET-to-MRI registration "
        "had brain coverage below 99%; "
        "accepted robust rigid retest using "
        "brain-extracted MRI."
    )

    qc["initial_brain_coverage_pct"] = float(
        row["old_coverage_pct"]
    )

    with open(final_qc, "w") as f:
        json.dump(
            qc,
            f,
            indent=2,
        )

    print(
        f"RID {rid:4d} | {pet_id} | "
        f"{row['old_coverage_pct']:.3f}% "
        f"-> {qc['brain_coverage_pct']:.3f}% | "
        f"{qc['registration_qc']}"
    )


print("\nDONE")
