#!/usr/bin/env python3

from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = (
    ROOT
    / "data/processed/master_modelling_manifest_v1.csv"
)

INPUT_ROOT = (
    ROOT
    / "data/processed/mri_v1"
)

OUTPUT_ROOT = (
    ROOT
    / "data/processed/mri_model_input_v1"
)

CROP = (
    slice(15, 179),
    slice(16, 214),
    slice(0, 172),
)

ZOOM = (0.5, 0.5, 0.5)

TARGET_SHAPE = np.array(
    [96, 112, 96]
)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

results = []

print("Subjects:", len(df))

for _, row in df.iterrows():

    mri_id = row["mri_image_id"]

    in_path = (
        INPUT_ROOT
        / mri_id
        / f"{mri_id}_mri_preprocessed.nii.gz"
    )

    outdir = (
        OUTPUT_ROOT
        / mri_id
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = (
        outdir
        / f"{mri_id}_model_input.npy"
    )

    qc_path = (
        outdir
        / f"{mri_id}_model_input_qc.json"
    )

    img = nib.load(
        in_path
    )

    x = np.asarray(
        img.dataobj,
        dtype=np.float32
    )

    original_shape = tuple(
        int(v) for v in x.shape
    )

    cropped = x[CROP]

    resampled = zoom(
        cropped,
        zoom=ZOOM,
        order=1
    ).astype(
        np.float32,
        copy=False
    )

    shape = np.array(
        resampled.shape
    )

    total_pad = (
        TARGET_SHAPE
        - shape
    )

    if np.any(total_pad < 0):
        raise RuntimeError(
            f"{mri_id}: resampled shape "
            f"{resampled.shape} exceeds target "
            f"{tuple(TARGET_SHAPE)}"
        )

    before = (
        total_pad // 2
    )

    after = (
        total_pad - before
    )

    final = np.pad(
        resampled,
        [
            (before[0], after[0]),
            (before[1], after[1]),
            (before[2], after[2]),
        ],
        mode="constant",
        constant_values=0
    ).astype(
        np.float32,
        copy=False
    )

    if tuple(final.shape) != tuple(TARGET_SHAPE):
        raise RuntimeError(
            f"{mri_id}: final shape "
            f"{final.shape}"
        )

    np.save(
        out_path,
        final,
        allow_pickle=False
    )

    qc = {
        "RID": int(row["RID"]),
        "mri_image_id": mri_id,
        "original_shape": list(original_shape),
        "cropped_shape": list(cropped.shape),
        "resampled_shape": list(resampled.shape),
        "final_shape": list(final.shape),
        "finite": bool(
            np.isfinite(final).all()
        ),
        "min": float(final.min()),
        "max": float(final.max()),
        "mean": float(final.mean()),
        "std": float(final.std()),
        "padding_before": before.tolist(),
        "padding_after": after.tolist(),
    }

    with open(
        qc_path,
        "w"
    ) as f:
        json.dump(
            qc,
            f,
            indent=2
        )

    results.append(qc)


qc_df = pd.DataFrame(
    results
)

summary_path = (
    ROOT
    / "data/interim/"
    / "mri_model_input_qc_v1.csv"
)

qc_df.to_csv(
    summary_path,
    index=False
)

print("\n=== MRI MODEL INPUT QC ===")
print("Processed:", len(qc_df))

print(
    "Final shapes:",
    qc_df["final_shape"]
    .astype(str)
    .value_counts()
    .to_dict()
)

print(
    "Finite failures:",
    int(
        (~qc_df["finite"]).sum()
    )
)

print(
    "Mean range:",
    qc_df["mean"].min(),
    "->",
    qc_df["mean"].max()
)

print(
    "Std range:",
    qc_df["std"].min(),
    "->",
    qc_df["std"].max()
)

print("\nSaved:", summary_path)
