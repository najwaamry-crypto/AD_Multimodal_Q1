#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

XAI_DIR = (
    ROOT
    / "results/xai/primary_gradcam_group_v1"
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

MODALITIES = [
    ("MRI", "mri"),
    ("PET", "pet"),
]


def normalize01(x):
    x = x.astype(np.float32)
    lo = float(x.min())
    hi = float(x.max())

    if hi <= lo:
        return np.zeros_like(x)

    return (
        (x - lo)
        / (hi - lo)
    )


def best_axial_slice(cam):
    # deterministic:
    # choose z maximizing total attribution
    scores = cam.sum(
        axis=(0, 1)
    )

    return int(
        np.argmax(scores)
    )


fig, axes = plt.subplots(
    2,
    3,
    figsize=(11, 7)
)

slice_records = []


for row_idx, (modality_label, modality_key) in enumerate(
    MODALITIES
):

    for col_idx, (class_label, class_key) in enumerate(
        CLASSES
    ):

        path = (
            XAI_DIR
            / f"{class_key}_{modality_key}_mean.npy"
        )

        cam = np.load(
            path,
            allow_pickle=False
        ).astype(
            np.float32
        )

        z = best_axial_slice(
            cam
        )

        sl = cam[
            :, :,
            z
        ].T

        sl = normalize01(
            sl
        )

        ax = axes[
            row_idx,
            col_idx
        ]

        im = ax.imshow(
            sl,
            cmap="inferno",
            origin="lower",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"{modality_label} — {class_label}\n"
            f"axial z={z}"
        )

        ax.axis("off")

        slice_records.append(
            (
                modality_label,
                class_label,
                z
            )
        )


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
    "Group-level Grad-CAM maps by true diagnostic class",
    fontsize=14
)

plt.subplots_adjust(
    top=0.88,
    wspace=0.05,
    hspace=0.15,
    right=0.90
)

png_path = (
    OUT_DIR
    / "gradcam_group_true_class_v1.png"
)

pdf_path = (
    OUT_DIR
    / "gradcam_group_true_class_v1.pdf"
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
    "=== GROUP GRAD-CAM FIGURE V1 ==="
)

for modality, cls, z in slice_records:

    print(
        f"{modality:3s} | "
        f"{cls:8s} | "
        f"axial z={z}"
    )

print(
    "\nSaved PNG:",
    png_path
)

print(
    "Saved PDF:",
    pdf_path
)
