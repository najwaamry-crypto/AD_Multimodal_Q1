#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

XAI_DIR = (
    ROOT
    / "results/xai/primary_gradcam_v1"
)

OUT_DIR = (
    ROOT
    / "results/xai/primary_gradcam_group_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_PATH = (
    XAI_DIR
    / "cohort_gradcam_summary.csv"
)

df = pd.read_csv(
    SUMMARY_PATH
)

if len(df) != 164:
    raise RuntimeError(
        f"Expected 164 subjects, found {len(df)}"
    )

df["correct"] = (
    df["true_label"]
    ==
    df["predicted_label"]
)


def load_cam(
    rid,
    modality
):

    path = (
        XAI_DIR
        / f"RID_{int(rid)}_{modality}_gradcam.npy"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(
        path,
        allow_pickle=False
    ).astype(
        np.float32
    )

    if arr.shape != (96, 112, 96):
        raise RuntimeError(
            f"Unexpected shape for {path}: {arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"Non-finite values in {path}"
        )

    return arr


def aggregate_group(
    subset,
    modality,
    group_name
):

    maps = np.stack(
        [
            load_cam(
                rid,
                modality
            )
            for rid
            in subset["RID"]
        ],
        axis=0
    )

    mean_map = maps.mean(
        axis=0
    )

    median_map = np.median(
        maps,
        axis=0
    )

    mean_path = (
        OUT_DIR
        / f"{group_name}_{modality}_mean.npy"
    )

    median_path = (
        OUT_DIR
        / f"{group_name}_{modality}_median.npy"
    )

    np.save(
        mean_path,
        mean_map.astype(
            np.float32
        )
    )

    np.save(
        median_path,
        median_map.astype(
            np.float32
        )
    )

    return {
        "group":
            group_name,

        "modality":
            modality.upper(),

        "n":
            int(len(subset)),

        "mean_map_mean":
            float(mean_map.mean()),

        "mean_map_max":
            float(mean_map.max()),

        "median_map_mean":
            float(median_map.mean()),

        "median_map_max":
            float(median_map.max()),

        "mean_map_path":
            str(mean_path),

        "median_map_path":
            str(median_path),
    }


rows = []


# ============================================================
# TRUE-DIAGNOSIS GROUPS
# ============================================================

for diagnosis in [
    "CN",
    "MCI",
    "Dementia",
]:

    subset = df[
        df["true_label"]
        == diagnosis
    ].copy()

    group_name = (
        "true_"
        + diagnosis.lower()
    )

    for modality in [
        "mri",
        "pet",
    ]:

        rows.append(
            aggregate_group(
                subset,
                modality,
                group_name
            )
        )


# ============================================================
# CORRECTNESS GROUPS
# ============================================================

for correct_value, group_name in [
    (True, "correct"),
    (False, "incorrect"),
]:

    subset = df[
        df["correct"]
        == correct_value
    ].copy()

    for modality in [
        "mri",
        "pet",
    ]:

        rows.append(
            aggregate_group(
                subset,
                modality,
                group_name
            )
        )


# ============================================================
# TRUE CLASS × CORRECTNESS
# ============================================================

for diagnosis in [
    "CN",
    "MCI",
    "Dementia",
]:

    for correct_value, status in [
        (True, "correct"),
        (False, "incorrect"),
    ]:

        subset = df[
            (
                df["true_label"]
                == diagnosis
            )
            &
            (
                df["correct"]
                == correct_value
            )
        ].copy()

        if len(subset) == 0:
            continue

        group_name = (
            "true_"
            + diagnosis.lower()
            + "_"
            + status
        )

        for modality in [
            "mri",
            "pet",
        ]:

            rows.append(
                aggregate_group(
                    subset,
                    modality,
                    group_name
                )
            )


summary = pd.DataFrame(
    rows
)

summary.to_csv(
    OUT_DIR
    / "group_gradcam_summary.csv",
    index=False
)


counts = {
    "total":
        int(len(df)),

    "true_class_counts":
        {
            str(k): int(v)
            for k, v
            in df[
                "true_label"
            ].value_counts().items()
        },

    "correct":
        int(df["correct"].sum()),

    "incorrect":
        int((~df["correct"]).sum()),

    "map_shape":
        [96, 112, 96],

    "aggregation":
        [
            "voxelwise mean",
            "voxelwise median",
        ],

    "interpretation":
        "descriptive aggregation of individually normalized Grad-CAM maps",

    "post_test_model_tuning":
        False,
}


with open(
    OUT_DIR
    / "group_gradcam_metadata.json",
    "w"
) as f:

    json.dump(
        counts,
        f,
        indent=2
    )


print(
    "=== GROUP GRAD-CAM SUMMARY V1 ==="
)

print(
    "Subjects:",
    len(df)
)

print(
    "Correct:",
    int(df["correct"].sum())
)

print(
    "Incorrect:",
    int((~df["correct"]).sum())
)

print(
    "\n=== AGGREGATED MAPS ==="
)

print(
    summary[
        [
            "group",
            "modality",
            "n",
            "mean_map_mean",
            "mean_map_max",
            "median_map_mean",
            "median_map_max",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print(
    "\nPOST-TEST MODEL TUNING: NO"
)

print(
    "Saved to:",
    OUT_DIR
)
