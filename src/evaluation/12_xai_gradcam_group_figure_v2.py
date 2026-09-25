#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = (
    ROOT
    / "data/processed/master_modelling_manifest_v1.csv"
)

XAI_DIR = (
    ROOT
    / "results/xai/primary_gradcam_group_v1"
)

MRI_ROOT = (
    ROOT
    / "data/processed/mri_model_input_v1"
)

PET_ROOT = (
    ROOT
    / "data/processed/pet_model_input_v1"
)

OUT_DIR = (
    ROOT
    / "figures/xai"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CLASSES = [
    ("CN", "true_cn"),
    ("MCI", "true_mci"),
    ("Dementia", "true_dementia"),
]


def normalize01(x):

    x = x.astype(
        np.float32
    )

    lo = float(
        np.percentile(
            x,
            1
        )
    )

    hi = float(
        np.percentile(
            x,
            99
        )
    )

    if hi <= lo:

        return np.zeros_like(
            x
        )

    x = np.clip(
        x,
        lo,
        hi
    )

    return (
        x - lo
    ) / (
        hi - lo
    )


def load_model_input(
    root,
    image_id
):

    image_id = str(
        image_id
    )

    path = (
        root
        / image_id
        / f"{image_id}_model_input.npy"
    )

    arr = np.load(
        path,
        allow_pickle=False
    ).astype(
        np.float32
    )

    return arr


# ============================================================
# BUILD MEAN MRI / PET BACKGROUND FROM TEST SUBJECTS
# ============================================================

manifest = pd.read_csv(
    MANIFEST,
    low_memory=False
)

test = (
    manifest[
        manifest["split"]
        == "test"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

if len(test) != 164:

    raise RuntimeError(
        f"Expected 164 test subjects, got {len(test)}"
    )


mri_sum = None
pet_sum = None


for _, row in test.iterrows():

    mri = load_model_input(
        MRI_ROOT,
        row["mri_image_id"]
    )

    pet = load_model_input(
        PET_ROOT,
        row["pet_image_id"]
    )

    if mri_sum is None:

        mri_sum = np.zeros_like(
            mri,
            dtype=np.float64
        )

        pet_sum = np.zeros_like(
            pet,
            dtype=np.float64
        )

    mri_sum += mri
    pet_sum += pet


mri_mean = (
    mri_sum
    / len(test)
).astype(
    np.float32
)

pet_mean = (
    pet_sum
    / len(test)
).astype(
    np.float32
)


# ============================================================
# DETERMINE ONE COMMON AXIAL SLICE PER MODALITY
# using overall mean of class-specific mean Grad-CAM maps
# ============================================================

mri_class_maps = []
pet_class_maps = []


for _, class_key in CLASSES:

    mri_class_maps.append(
        np.load(
            XAI_DIR
            / f"{class_key}_mri_mean.npy",
            allow_pickle=False
        )
    )

    pet_class_maps.append(
        np.load(
            XAI_DIR
            / f"{class_key}_pet_mean.npy",
            allow_pickle=False
        )
    )


mri_global_cam = np.mean(
    np.stack(
        mri_class_maps,
        axis=0
    ),
    axis=0
)

pet_global_cam = np.mean(
    np.stack(
        pet_class_maps,
        axis=0
    ),
    axis=0
)


mri_z = int(
    np.argmax(
        mri_global_cam.sum(
            axis=(0, 1)
        )
    )
)

pet_z = int(
    np.argmax(
        pet_global_cam.sum(
            axis=(0, 1)
        )
    )
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(11, 7)
)


for col_idx, (
    class_label,
    class_key
) in enumerate(
    CLASSES
):

    # MRI
    mri_cam = np.load(
        XAI_DIR
        / f"{class_key}_mri_mean.npy",
        allow_pickle=False
    )

    mri_bg_slice = normalize01(
        mri_mean[
            :, :,
            mri_z
        ].T
    )

    mri_cam_slice = normalize01(
        mri_cam[
            :, :,
            mri_z
        ].T
    )

    ax = axes[
        0,
        col_idx
    ]

    ax.imshow(
        mri_bg_slice,
        cmap="gray",
        origin="lower"
    )

    ax.imshow(
        mri_cam_slice,
        cmap="inferno",
        origin="lower",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"MRI — {class_label}\n"
        f"common axial z={mri_z}"
    )

    ax.axis("off")


    # PET
    pet_cam = np.load(
        XAI_DIR
        / f"{class_key}_pet_mean.npy",
        allow_pickle=False
    )

    pet_bg_slice = normalize01(
        pet_mean[
            :, :,
            pet_z
        ].T
    )

    pet_cam_slice = normalize01(
        pet_cam[
            :, :,
            pet_z
        ].T
    )

    ax = axes[
        1,
        col_idx
    ]

    ax.imshow(
        pet_bg_slice,
        cmap="gray",
        origin="lower"
    )

    im = ax.imshow(
        pet_cam_slice,
        cmap="inferno",
        origin="lower",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"PET — {class_label}\n"
        f"common axial z={pet_z}"
    )

    ax.axis("off")


cbar = fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    fraction=0.025,
    pad=0.02
)

cbar.set_label(
    "Normalized mean Grad-CAM attribution"
)


fig.suptitle(
    "Group-level Grad-CAM overlays by true diagnostic class",
    fontsize=14
)


plt.subplots_adjust(
    top=0.88,
    wspace=0.03,
    hspace=0.12,
    right=0.90
)


png_path = (
    OUT_DIR
    / "gradcam_group_true_class_v2.png"
)

pdf_path = (
    OUT_DIR
    / "gradcam_group_true_class_v2.pdf"
)


plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close(
    fig
)


print(
    "=== GROUP GRAD-CAM FIGURE V2 ==="
)

print(
    "Test subjects used for mean backgrounds:",
    len(test)
)

print(
    "MRI common axial slice:",
    mri_z
)

print(
    "PET common axial slice:",
    pet_z
)

print(
    "\nSaved PNG:",
    png_path
)

print(
    "Saved PDF:",
    pdf_path
)
