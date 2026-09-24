#!/usr/bin/env python3

from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

INPUT_ROOT = ROOT / "data/processed/pet_spatial_v1"
OUTPUT_ROOT = ROOT / "data/processed/pet_normalized_v1"

MASK_PATH = (
    ROOT
    / "data/external/pet_reference_masks/"
    / "AAL_SPM12_vermis_MNI2009cAsym_1mm.nii.gz"
)

SUMMARY_CSV = (
    ROOT
    / "data/interim/"
    / "pet_aal_vermis_vs_ucb_top50ponsvermis_v1.csv"
)

mask_img = nib.load(MASK_PATH)
mask = np.asarray(mask_img.dataobj) > 0

summary = pd.read_csv(SUMMARY_CSV)

print("Subjects:", len(summary))
print("Mask voxels:", int(mask.sum()))

results = []

for _, row in summary.iterrows():

    pet_id = str(row["pet_image_id"])

    if not pet_id.startswith("I"):
        pet_id = f"I{int(float(pet_id))}"

    in_path = (
        INPUT_ROOT
        / pet_id
        / f"{pet_id}_mni.nii.gz"
    )

    outdir = OUTPUT_ROOT / pet_id
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = (
        outdir
        / f"{pet_id}_pet_vermisnorm.nii.gz"
    )

    qc_path = (
        outdir
        / f"{pet_id}_pet_vermisnorm_qc.json"
    )

    img = nib.load(in_path)

    pet = np.asarray(
        img.dataobj,
        dtype=np.float32
    )

    vals = pet[
        mask &
        np.isfinite(pet)
    ]

    denom = float(vals.mean())

    if not np.isfinite(denom):
        raise RuntimeError(
            f"{pet_id}: non-finite denominator"
        )

    if denom <= 0:
        raise RuntimeError(
            f"{pet_id}: invalid denominator {denom}"
        )

    norm = pet / denom

    norm = norm.astype(
        np.float32,
        copy=False
    )

    nib.save(
        nib.Nifti1Image(
            norm,
            img.affine,
            img.header
        ),
        out_path
    )

    finite = np.isfinite(norm)

    qc = {
        "pet_image_id": pet_id,
        "reference": "AAL_SPM12_vermis",
        "reference_voxels": int(mask.sum()),
        "reference_mean_raw": denom,
        "normalized_finite_pct": float(
            100.0 * finite.sum() / finite.size
        ),
        "normalized_min": float(
            np.nanmin(norm)
        ),
        "normalized_max": float(
            np.nanmax(norm)
        ),
        "normalized_mean": float(
            np.nanmean(norm)
        ),
        "normalized_std": float(
            np.nanstd(norm)
        ),
        "vermis_mean_after_normalization": float(
            norm[
                mask & np.isfinite(norm)
            ].mean()
        )
    }

    with open(qc_path, "w") as f:
        json.dump(
            qc,
            f,
            indent=2
        )

    results.append(qc)

df = pd.DataFrame(results)

out_summary = (
    ROOT
    / "data/interim/"
    / "pet_vermis_normalization_qc_v1.csv"
)

df.to_csv(
    out_summary,
    index=False
)

print("\n=== NORMALIZATION QC ===")
print("Processed:", len(df))
print(
    "Reference mean range:",
    df["reference_mean_raw"].min(),
    "->",
    df["reference_mean_raw"].max()
)
print(
    "Vermis mean after normalization:",
    df["vermis_mean_after_normalization"].min(),
    "->",
    df["vermis_mean_after_normalization"].max()
)
print(
    "Finite failures:",
    int(
        (df["normalized_finite_pct"] < 100).sum()
    )
)

print("\nSaved:", out_summary)
