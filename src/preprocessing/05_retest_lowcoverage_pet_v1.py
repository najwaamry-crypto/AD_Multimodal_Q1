#!/usr/bin/env python3

from pathlib import Path
import json
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

PET_QC_ROOT = ROOT / "data/processed/pet_spatial_v1"
PET_NATIVE_ROOT = ROOT / "data/interim/pet_nifti_v1"
MRI_WORK_ROOT = ROOT / "data/interim/mri_preprocessing_v1"
MRI_OUT_ROOT = ROOT / "data/processed/mri_v1"

RETEST_ROOT = ROOT / "data/interim/pet_registration_retest_auto_v1"

TPL = (
    ROOT
    / "data/external/templateflow/"
    / "tpl-MNI152NLin2009cAsym/"
    / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)

THRESHOLD = 99.0


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(
        list(map(str, cmd)),
        check=True,
    )


rows = []

for p in PET_QC_ROOT.glob(
    "I*/I*_spatial_qc.json"
):
    with open(p) as f:
        q = json.load(f)

    if q["brain_coverage_pct"] < THRESHOLD:
        rows.append(q)

df = pd.DataFrame(rows).sort_values(
    "brain_coverage_pct"
)

print(
    f"Cases with initial coverage < {THRESHOLD}%:",
    len(df)
)

results = []

for _, row in df.iterrows():

    rid = int(row["RID"])
    pet_name = row["pet_image_id"]
    mri_name = row["mri_image_id"]

    pet_id = int(pet_name[1:])
    mri_id = int(mri_name[1:])

    old_cov = float(
        row["brain_coverage_pct"]
    )

    print(
        f"\n=== RID {rid} | {pet_name} | "
        f"old coverage={old_cov:.3f}% ==="
    )

    outdir = RETEST_ROOT / pet_name
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pet_native = (
        PET_NATIVE_ROOT
        / f"{pet_name}.nii.gz"
    )

    mri_brain = (
        MRI_WORK_ROOT
        / mri_name
        / f"{mri_name}_BrainExtractionBrain.nii.gz"
    )

    mri_aff = (
        MRI_WORK_ROOT
        / mri_name
        / f"{mri_name}_mni_0GenericAffine.mat"
    )

    brain_mask = (
        MRI_OUT_ROOT
        / mri_name
        / f"{mri_name}_brainmask.nii.gz"
    )

    support_native = (
        ROOT
        / "data/interim/pet_spatial_v1"
        / pet_name
        / f"{pet_name}_native_support.nii.gz"
    )

    prefix = (
        outdir
        / f"{pet_name}_to_{mri_name}_retest_"
    )

    rigid_aff = (
        outdir
        / f"{pet_name}_to_{mri_name}_retest_0GenericAffine.mat"
    )

    warped = (
        outdir
        / f"{pet_name}_to_{mri_name}_retest_Warped.nii.gz"
    )

    pet_mni = (
        outdir
        / f"{pet_name}_retest_mni.nii.gz"
    )

    support_mni = (
        outdir
        / f"{pet_name}_retest_support_mni.nii.gz"
    )

    # Robust rigid PET -> brain-extracted MRI
    if not rigid_aff.exists():
        run([
            "antsRegistration",
            "--dimensionality", "3",
            "--float", "0",
            "--output",
            f"[{prefix},{warped}]",
            "--interpolation", "Linear",
            "--winsorize-image-intensities",
            "[0.005,0.995]",
            "--use-histogram-matching", "0",
            "--initial-moving-transform",
            f"[{mri_brain},{pet_native},0]",
            "--transform", "Rigid[0.1]",
            "--metric",
            f"MI[{mri_brain},{pet_native},1,32,Regular,0.25]",
            "--convergence",
            "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
        ])

    # PET -> MRI -> MNI
    run([
        "antsApplyTransforms",
        "-d", "3",
        "-i", pet_native,
        "-r", TPL,
        "-o", pet_mni,
        "-n", "Linear",
        "-t", mri_aff,
        "-t", rigid_aff,
    ])

    # Support -> MNI
    run([
        "antsApplyTransforms",
        "-d", "3",
        "-i", support_native,
        "-r", TPL,
        "-o", support_mni,
        "-n", "NearestNeighbor",
        "-t", mri_aff,
        "-t", rigid_aff,
    ])

    pet = np.asarray(
        nib.load(pet_mni).dataobj,
        dtype=np.float32,
    )

    support = (
        np.asarray(
            nib.load(support_mni).dataobj
        ) > 0.5
    )

    brain = (
        np.asarray(
            nib.load(brain_mask).dataobj
        ) > 0
    )

    valid = (
        brain
        & support
        & np.isfinite(pet)
    )

    new_cov = float(
        100.0
        * valid.sum()
        / brain.sum()
    )

    x = pet[valid]

    results.append({
        "RID": rid,
        "pet_image_id": pet_name,
        "mri_image_id": mri_name,
        "old_coverage_pct": old_cov,
        "new_coverage_pct": new_cov,
        "improvement_pp": new_cov - old_cov,
        "brain_mean": float(x.mean()),
        "negative_pct": float(
            100.0
            * (x < 0).sum()
            / x.size
        ),
    })

    print(
        f"{pet_name}: "
        f"{old_cov:.3f}% -> {new_cov:.3f}%"
    )


res = pd.DataFrame(results)

out_csv = (
    ROOT
    / "data/interim/"
    / "pet_lowcoverage_retest_summary_v1.csv"
)

res.to_csv(
    out_csv,
    index=False,
)

print("\n=== SUMMARY ===")
print(
    res.sort_values(
        "old_coverage_pct"
    ).to_string(index=False)
)

print("\nSaved:", out_csv)
