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

OUTPUT_ROOT = (
    ROOT
    / "data/processed/pet_model_input_v1"
)

QC_OUT = (
    ROOT
    / "data/interim/pet_model_input_qc_v1.csv"
)


# ============================================================
# LOCKED GEOMETRY
# Must match MRI model-input preparation exactly.
# ============================================================

CROP = (
    slice(15, 179),
    slice(16, 214),
    slice(0, 172),
)

ZOOM_FACTORS = (
    0.5,
    0.5,
    0.5,
)

EXPECTED_CROP_SHAPE = (
    164,
    198,
    172,
)

EXPECTED_RESAMPLED_SHAPE = (
    82,
    99,
    86,
)

FINAL_SHAPE = (
    96,
    112,
    96,
)

PAD_BEFORE = (
    7,
    6,
    5,
)

PAD_AFTER = (
    7,
    7,
    5,
)


def main():

    df = pd.read_csv(
        MANIFEST,
        low_memory=False
    )

    print("=== PET MODEL INPUT PREPARATION V1 ===")
    print("Subjects:", len(df))
    print("Target shape:", FINAL_SHAPE)

    qc_rows = []

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, row in df.iterrows():

        pet_id = str(
            row["pet_image_id"]
        )

        pet_path = Path(
            row["pet_path"]
        )

        if not pet_path.is_absolute():
            pet_path = (
                ROOT / pet_path
            )

        if not pet_path.exists():
            raise FileNotFoundError(
                f"Missing PET: {pet_path}"
            )

        img = nib.load(
            str(pet_path)
        )

        arr = img.get_fdata(
            dtype=np.float32
        )

        original_shape = tuple(
            arr.shape
        )

        if original_shape != (
            193,
            229,
            193,
        ):
            raise RuntimeError(
                f"{pet_id}: unexpected input "
                f"shape {original_shape}"
            )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{pet_id}: non-finite values "
                "before model preparation"
            )

        cropped = arr[CROP]

        if cropped.shape != EXPECTED_CROP_SHAPE:
            raise RuntimeError(
                f"{pet_id}: crop shape "
                f"{cropped.shape}"
            )

        resampled = zoom(
            cropped,
            zoom=ZOOM_FACTORS,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(
            np.float32,
            copy=False
        )

        if (
            resampled.shape
            != EXPECTED_RESAMPLED_SHAPE
        ):
            raise RuntimeError(
                f"{pet_id}: resampled shape "
                f"{resampled.shape}"
            )

        final = np.pad(
            resampled,
            (
                (
                    PAD_BEFORE[0],
                    PAD_AFTER[0],
                ),
                (
                    PAD_BEFORE[1],
                    PAD_AFTER[1],
                ),
                (
                    PAD_BEFORE[2],
                    PAD_AFTER[2],
                ),
            ),
            mode="constant",
            constant_values=0,
        ).astype(
            np.float32,
            copy=False
        )

        if final.shape != FINAL_SHAPE:
            raise RuntimeError(
                f"{pet_id}: final shape "
                f"{final.shape}"
            )

        finite = bool(
            np.isfinite(final).all()
        )

        if not finite:
            raise RuntimeError(
                f"{pet_id}: non-finite final array"
            )

        out_dir = (
            OUTPUT_ROOT
            / pet_id
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        out_path = (
            out_dir
            / f"{pet_id}_model_input.npy"
        )

        np.save(
            out_path,
            final,
            allow_pickle=False
        )

        qc = {
            "RID":
                row["RID"],

            "split":
                row["split"],

            "diagnosis":
                row["diagnosis"],

            "pet_image_id":
                pet_id,

            "input_path":
                str(pet_path),

            "output_path":
                str(out_path),

            "original_shape":
                str(original_shape),

            "crop_shape":
                str(cropped.shape),

            "resampled_shape":
                str(resampled.shape),

            "final_shape":
                str(final.shape),

            "finite":
                finite,

            "min":
                float(final.min()),

            "max":
                float(final.max()),

            "mean":
                float(final.mean()),

            "std":
                float(final.std()),

            "nonzero_fraction":
                float(
                    np.count_nonzero(final)
                    / final.size
                ),
        }

        qc_rows.append(qc)

        if (
            (i + 1) % 50 == 0
            or i == 0
            or i + 1 == len(df)
        ):
            print(
                f"[{i + 1}/{len(df)}] "
                f"{pet_id} "
                f"shape={final.shape}"
            )

    qc_df = pd.DataFrame(
        qc_rows
    )

    QC_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    qc_df.to_csv(
        QC_OUT,
        index=False
    )

    print("\n=== PET MODEL INPUT QC ===")
    print(
        "Processed:",
        len(qc_df)
    )

    print(
        "Final shapes:",
        qc_df[
            "final_shape"
        ].value_counts().to_dict()
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

    print(
        "Nonzero fraction range:",
        qc_df[
            "nonzero_fraction"
        ].min(),
        "->",
        qc_df[
            "nonzero_fraction"
        ].max()
    )

    print(
        "\nSaved:",
        QC_OUT
    )


if __name__ == "__main__":
    main()
