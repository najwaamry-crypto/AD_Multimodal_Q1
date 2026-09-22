#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import os
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "data/interim/imaging_download_manifests/"
    / "mri_final_image_selection_v1.csv"
)

INPUT_DIR = ROOT / "data/interim/mri_nifti_v1"

OUTPUT_ROOT = ROOT / "data/processed/mri_v1"
WORK_ROOT = ROOT / "data/interim/mri_preprocessing_v1"
LOG_ROOT = ROOT / "logs/mri_preprocessing_v1"

TPL_DIR = (
    ROOT
    / "data/external/templateflow/"
    / "tpl-MNI152NLin2009cAsym"
)

TEMPLATE = (
    TPL_DIR
    / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)

TEMPLATE_MASK = (
    TPL_DIR
    / "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"
)


def run(cmd, log_file, env=None):
    with open(log_file, "a") as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()

        subprocess.run(
            list(map(str, cmd)),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )


def preprocess(row, force=False, threads=4):
    image_id = f"I{int(row['mri_image_id'])}"

    src = INPUT_DIR / f"{image_id}.nii.gz"

    work = WORK_ROOT / image_id
    outdir = OUTPUT_ROOT / image_id
    logdir = LOG_ROOT

    work.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    log_file = logdir / f"{image_id}.log"

    n4 = work / f"{image_id}_n4.nii.gz"

    brain_prefix = work / f"{image_id}_"
    native_brain = work / f"{image_id}_BrainExtractionBrain.nii.gz"
    native_mask = work / f"{image_id}_BrainExtractionMask.nii.gz"

    reg_prefix = work / f"{image_id}_mni_"
    registered = work / f"{image_id}_mni_Warped.nii.gz"
    affine = work / f"{image_id}_mni_0GenericAffine.mat"

    mask_mni = work / f"{image_id}_brainmask_mni.nii.gz"

    final_img = outdir / f"{image_id}_mri_preprocessed.nii.gz"
    final_mask = outdir / f"{image_id}_brainmask.nii.gz"
    qc_json = outdir / f"{image_id}_qc.json"

    if final_img.exists() and final_mask.exists() and qc_json.exists() and not force:
        print(f"{image_id}: already complete")
        return {
            "image_id": image_id,
            "status": "already_complete",
        }

    if not src.exists():
        raise FileNotFoundError(src)

    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)

    if force:
        log_file.unlink(missing_ok=True)

    # ---------------------------------------------------------
    # 1. N4 bias correction
    # ---------------------------------------------------------
    run(
        [
            "N4BiasFieldCorrection",
            "-d", "3",
            "-i", src,
            "-o", n4,
            "-s", "2",
            "-b", "[200]",
            "-c", "[50x50x30x20,1e-6]",
        ],
        log_file,
        env,
    )

    # ---------------------------------------------------------
    # 2. Subject-specific brain extraction in native space
    # ---------------------------------------------------------
    run(
        [
            "antsBrainExtraction.sh",
            "-d", "3",
            "-a", n4,
            "-e", TEMPLATE,
            "-m", TEMPLATE_MASK,
            "-o", brain_prefix,
        ],
        log_file,
        env,
    )

    if not native_brain.exists() or not native_mask.exists():
        raise RuntimeError(
            f"{image_id}: brain extraction outputs missing"
        )

    # ---------------------------------------------------------
    # 3. Rigid + affine registration to MNI
    # ---------------------------------------------------------
    run(
        [
            "antsRegistrationSyNQuick.sh",
            "-d", "3",
            "-f", TEMPLATE,
            "-m", n4,
            "-o", reg_prefix,
            "-t", "a",
        ],
        log_file,
        env,
    )

    if not registered.exists() or not affine.exists():
        raise RuntimeError(
            f"{image_id}: affine registration outputs missing"
        )

    # ---------------------------------------------------------
    # 4. Transform subject-specific mask to MNI
    # ---------------------------------------------------------
    run(
        [
            "antsApplyTransforms",
            "-d", "3",
            "-i", native_mask,
            "-r", TEMPLATE,
            "-o", mask_mni,
            "-n", "NearestNeighbor",
            "-t", affine,
        ],
        log_file,
        env,
    )

    # ---------------------------------------------------------
    # 5-6. Mask + within-brain z-score normalization
    # ---------------------------------------------------------
    reg_img = nib.load(registered)
    mask_img = nib.load(mask_mni)
    tpl_img = nib.load(TEMPLATE)

    x = np.asarray(reg_img.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj) > 0.5

    if x.shape != tpl_img.shape:
        raise RuntimeError(
            f"{image_id}: registered shape {x.shape} "
            f"!= template shape {tpl_img.shape}"
        )

    if mask.shape != tpl_img.shape:
        raise RuntimeError(
            f"{image_id}: mask shape {mask.shape} "
            f"!= template shape {tpl_img.shape}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{image_id}: registered image contains NaN/Inf"
        )

    brain = x[mask]

    if brain.size == 0:
        raise RuntimeError(
            f"{image_id}: empty transformed brain mask"
        )

    mean = float(brain.mean())
    std = float(brain.std())

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise RuntimeError(
            f"{image_id}: invalid brain intensity statistics"
        )

    z = np.zeros(x.shape, dtype=np.float32)
    z[mask] = (x[mask] - mean) / std

    mask_u8 = mask.astype(np.uint8)

    final_nii = nib.Nifti1Image(
        z,
        tpl_img.affine,
        tpl_img.header.copy(),
    )
    final_nii.set_data_dtype(np.float32)

    mask_nii = nib.Nifti1Image(
        mask_u8,
        tpl_img.affine,
        tpl_img.header.copy(),
    )
    mask_nii.set_data_dtype(np.uint8)

    nib.save(final_nii, final_img)
    nib.save(mask_nii, final_mask)

    voxel_volume_mm3 = float(
        np.prod(tpl_img.header.get_zooms()[:3])
    )

    mask_volume_ml = float(
        mask.sum() * voxel_volume_mm3 / 1000.0
    )

    qc = {
        "RID": int(row["RID"]),
        "PTID": str(row["PTID"]),
        "image_id": image_id,
        "selected_loni_series": int(
            row["selected_loni_series"]
        ),
        "selection_family": str(row["selection_family"]),
        "field_strength": float(row["field_strength"]),
        "final_shape": list(z.shape),
        "final_voxel_size_mm": [
            float(v)
            for v in tpl_img.header.get_zooms()[:3]
        ],
        "finite": bool(np.isfinite(z).all()),
        "mask_binary": bool(
            set(np.unique(mask_u8)).issubset({0, 1})
        ),
        "mask_voxels": int(mask.sum()),
        "mask_volume_ml": mask_volume_ml,
        "native_registered_brain_mean": mean,
        "native_registered_brain_std": std,
        "normalized_brain_mean": float(z[mask].mean()),
        "normalized_brain_std": float(z[mask].std()),
        "normalized_min": float(z[mask].min()),
        "normalized_max": float(z[mask].max()),
        "final_image": str(final_img.relative_to(ROOT)),
        "final_mask": str(final_mask.relative_to(ROOT)),
        "status": "success",
    }

    with open(qc_json, "w") as f:
        json.dump(qc, f, indent=2)

    print(
        f"{image_id}: SUCCESS | "
        f"mask={mask_volume_ml:.1f} mL | "
        f"mean={qc['normalized_brain_mean']:.4f} | "
        f"std={qc['normalized_brain_std']:.4f}"
    )

    return qc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-id",
        type=int,
        help="Process one numeric ADNI Image ID, e.g. 7070",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all 822 locked MRI selections",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an already completed case",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ITK threads per case (default: 4)",
    )

    args = parser.parse_args()

    if bool(args.image_id) == bool(args.all):
        parser.error(
            "Specify exactly one of --image-id or --all"
        )

    df = pd.read_csv(MANIFEST, low_memory=False)

    assert len(df) == 822
    assert df["RID"].nunique() == 822
    assert df["selected_loni_series"].nunique() == 822

    if args.image_id:
        rows = df[
            df["mri_image_id"].astype(int) == args.image_id
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Image ID {args.image_id}: expected one row, "
                f"found {len(rows)}"
            )

        preprocess(
            rows.iloc[0],
            force=args.force,
            threads=args.threads,
        )
        return

    success = 0
    failed = []

    for _, row in df.iterrows():
        image_id = int(row["mri_image_id"])

        try:
            preprocess(
                row,
                force=args.force,
                threads=args.threads,
            )
            success += 1

        except Exception as e:
            failed.append(
                {
                    "mri_image_id": image_id,
                    "error": str(e),
                }
            )
            print(f"I{image_id}: FAILED — {e}")

    summary = {
        "expected": len(df),
        "completed_without_exception": success,
        "failed": len(failed),
        "failures": failed,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_ROOT / "cohort_preprocessing_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n=== COHORT SUMMARY ===")
    print("Expected:", len(df))
    print("Completed:", success)
    print("Failed:", len(failed))


if __name__ == "__main__":
    main()
