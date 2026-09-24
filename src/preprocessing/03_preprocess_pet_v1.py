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

ANALYTIC_MANIFEST = (
    ROOT / "data/interim/cohort/analytic_manifest_v1.csv"
)

MRI_SELECTION_MANIFEST = (
    ROOT
    / "data/interim/imaging_download_manifests/"
    / "mri_final_image_selection_v1.csv"
)

PET_INPUT = ROOT / "data/interim/pet_nifti_v1"

MRI_WORK = ROOT / "data/interim/mri_preprocessing_v1"
MRI_OUTPUT = ROOT / "data/processed/mri_v1"

PET_WORK = ROOT / "data/interim/pet_preprocessing_v1"
PET_OUTPUT = ROOT / "data/processed/pet_v1"
LOG_ROOT = ROOT / "logs/pet_preprocessing_v1"

TPL = (
    ROOT
    / "data/external/templateflow/"
    / "tpl-MNI152NLin2009cAsym/"
    / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)

VERMIS = (
    ROOT
    / "data/external/pet_reference_masks/"
    / "AAL_SPM12_vermis_MNI2009cAsym_1mm.nii.gz"
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


def build_manifest():
    analytic = pd.read_csv(
        ANALYTIC_MANIFEST,
        low_memory=False,
    )

    mri = pd.read_csv(
        MRI_SELECTION_MANIFEST,
        low_memory=False,
    )

    analytic = analytic[
        [
            "RID",
            "PTID",
            "diagnosis",
            "pet_image_id",
            "pet_date",
            "mri_pet_abs_days",
        ]
    ].copy()

    mri = mri[
        [
            "RID",
            "mri_image_id",
            "selection_family",
            "field_strength",
        ]
    ].copy()

    df = analytic.merge(
        mri,
        on="RID",
        how="inner",
        validate="one_to_one",
    )

    assert len(df) == 822
    assert df["RID"].nunique() == 822
    assert df["pet_image_id"].nunique() == 822
    assert df["mri_image_id"].nunique() == 822

    df["pet_image_id"] = (
        df["pet_image_id"].astype(int)
    )
    df["mri_image_id"] = (
        df["mri_image_id"].astype(int)
    )

    return df


def preprocess(row, force=False, threads=2):
    rid = int(row["RID"])
    pet_id = int(row["pet_image_id"])
    mri_id = int(row["mri_image_id"])

    pet_name = f"I{pet_id}"
    mri_name = f"I{mri_id}"

    pet_src = PET_INPUT / f"{pet_name}.nii.gz"

    mri_n4 = (
        MRI_WORK
        / mri_name
        / f"{mri_name}_n4.nii.gz"
    )

    mri_aff = (
        MRI_WORK
        / mri_name
        / f"{mri_name}_mni_0GenericAffine.mat"
    )

    brain_mask_path = (
        MRI_OUTPUT
        / mri_name
        / f"{mri_name}_brainmask.nii.gz"
    )

    work = PET_WORK / pet_name
    outdir = PET_OUTPUT / pet_name

    work.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    log_file = LOG_ROOT / f"{pet_name}.log"

    rigid_prefix = (
        work / f"{pet_name}_to_{mri_name}_"
    )

    rigid_aff = (
        work
        / f"{pet_name}_to_{mri_name}_0GenericAffine.mat"
    )

    pet_mni = work / f"{pet_name}_mni.nii.gz"

    support_native = (
        work / f"{pet_name}_native_support.nii.gz"
    )

    support_mni = (
        work / f"{pet_name}_support_mni.nii.gz"
    )

    final_pet = (
        outdir / f"{pet_name}_fdg_vermis_norm.nii.gz"
    )

    qc_json = outdir / f"{pet_name}_qc.json"

    if (
        final_pet.exists()
        and qc_json.exists()
        and not force
    ):
        print(f"{pet_name}: already complete")
        return {
            "pet_image_id": pet_name,
            "status": "already_complete",
        }

    required = [
        pet_src,
        mri_n4,
        mri_aff,
        brain_mask_path,
        TPL,
        VERMIS,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{pet_name}: missing required files: "
            + "; ".join(missing)
        )

    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = (
        str(threads)
    )

    if force:
        log_file.unlink(missing_ok=True)

    # ---------------------------------------------------------
    # 1. Rigid PET -> MRI registration
    # ---------------------------------------------------------
    run(
        [
            "antsRegistrationSyNQuick.sh",
            "-d", "3",
            "-f", mri_n4,
            "-m", pet_src,
            "-o", rigid_prefix,
            "-t", "r",
        ],
        log_file,
        env,
    )

    if not rigid_aff.exists():
        raise RuntimeError(
            f"{pet_name}: rigid transform missing"
        )

    # ---------------------------------------------------------
    # 2. PET -> MRI -> MNI in one resampling operation
    # ---------------------------------------------------------
    run(
        [
            "antsApplyTransforms",
            "-d", "3",
            "-i", pet_src,
            "-r", TPL,
            "-o", pet_mni,
            "-n", "Linear",
            "-t", mri_aff,
            "-t", rigid_aff,
        ],
        log_file,
        env,
    )

    # ---------------------------------------------------------
    # 3. Native PET acquisition support mask
    # ---------------------------------------------------------
    src_img = nib.load(pet_src)

    support = np.ones(
        src_img.shape,
        dtype=np.uint8,
    )

    support_img = nib.Nifti1Image(
        support,
        src_img.affine,
        src_img.header.copy(),
    )
    support_img.set_data_dtype(np.uint8)

    nib.save(
        support_img,
        support_native,
    )

    # ---------------------------------------------------------
    # 4. Transform support mask to MNI
    # ---------------------------------------------------------
    run(
        [
            "antsApplyTransforms",
            "-d", "3",
            "-i", support_native,
            "-r", TPL,
            "-o", support_mni,
            "-n", "NearestNeighbor",
            "-t", mri_aff,
            "-t", rigid_aff,
        ],
        log_file,
        env,
    )

    # ---------------------------------------------------------
    # 5. Geometry + masks
    # ---------------------------------------------------------
    tpl_img = nib.load(TPL)
    pet_img = nib.load(pet_mni)

    pet = np.asarray(
        pet_img.dataobj,
        dtype=np.float32,
    )

    brain = (
        np.asarray(
            nib.load(brain_mask_path).dataobj
        )
        > 0
    )

    pet_support = (
        np.asarray(
            nib.load(support_mni).dataobj
        )
        > 0.5
    )

    vermis = (
        np.asarray(
            nib.load(VERMIS).dataobj
        )
        > 0
    )

    expected_shape = tpl_img.shape

    for name, arr in [
        ("PET", pet),
        ("brain mask", brain),
        ("support mask", pet_support),
        ("vermis mask", vermis),
    ]:
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"{pet_name}: {name} shape "
                f"{arr.shape} != {expected_shape}"
            )

    if not np.allclose(
        pet_img.affine,
        tpl_img.affine,
        atol=1e-5,
    ):
        raise RuntimeError(
            f"{pet_name}: PET affine != template affine"
        )

    if not np.isfinite(pet).all():
        raise RuntimeError(
            f"{pet_name}: PET contains NaN/Inf"
        )

    # ---------------------------------------------------------
    # 6. Valid PET brain + vermis reference
    # ---------------------------------------------------------
    valid_brain = (
        brain
        & pet_support
        & np.isfinite(pet)
    )

    valid_ref = (
        vermis
        & valid_brain
        & (pet > 0)
    )

    ref_vals = pet[valid_ref]

    if ref_vals.size == 0:
        raise RuntimeError(
            f"{pet_name}: empty vermis reference"
        )

    ref_mean = float(ref_vals.mean())
    ref_std = float(ref_vals.std())

    if (
        not np.isfinite(ref_mean)
        or ref_mean <= 0
    ):
        raise RuntimeError(
            f"{pet_name}: invalid vermis mean"
        )

    ref_cv = (
        float(ref_std / ref_mean)
        if ref_mean > 0
        else float("nan")
    )

    # ---------------------------------------------------------
    # 7. Vermis-reference normalization
    # ---------------------------------------------------------
    final = np.zeros(
        pet.shape,
        dtype=np.float32,
    )

    final[valid_brain] = (
        pet[valid_brain] / ref_mean
    )

    final_img = nib.Nifti1Image(
        final,
        tpl_img.affine,
        tpl_img.header.copy(),
    )
    final_img.set_data_dtype(np.float32)

    nib.save(
        final_img,
        final_pet,
    )

    # ---------------------------------------------------------
    # 8. QC metrics
    # ---------------------------------------------------------
    x = final[valid_brain]

    brain_voxels = int(brain.sum())
    valid_brain_voxels = int(valid_brain.sum())

    coverage_pct = float(
        100.0
        * valid_brain_voxels
        / brain_voxels
    )

    negative_voxels = int(
        (x < 0).sum()
    )

    negative_pct = float(
        100.0
        * negative_voxels
        / x.size
    )

    qc = {
        "RID": rid,
        "PTID": str(row["PTID"]),
        "diagnosis": str(row["diagnosis"]),
        "pet_image_id": pet_name,
        "mri_image_id": mri_name,
        "selection_family": str(
            row["selection_family"]
        ),
        "field_strength": float(
            row["field_strength"]
        ),
        "pet_date": str(row["pet_date"]),
        "mri_pet_abs_days": int(
            row["mri_pet_abs_days"]
        ),

        "final_shape": list(final.shape),
        "final_voxel_size_mm": [
            float(v)
            for v in tpl_img.header.get_zooms()[:3]
        ],

        "finite": bool(
            np.isfinite(final).all()
        ),
        "same_template_affine": bool(
            np.allclose(
                final_img.affine,
                tpl_img.affine,
                atol=1e-5,
            )
        ),

        "brain_voxels": brain_voxels,
        "valid_brain_voxels": valid_brain_voxels,
        "brain_coverage_pct": coverage_pct,

        "vermis_reference_voxels": int(
            valid_ref.sum()
        ),
        "vermis_reference_mean": ref_mean,
        "vermis_reference_std": ref_std,
        "vermis_reference_cv": ref_cv,

        "normalized_brain_mean": float(
            x.mean()
        ),
        "normalized_brain_std": float(
            x.std()
        ),
        "normalized_brain_median": float(
            np.median(x)
        ),
        "normalized_brain_p05": float(
            np.percentile(x, 5)
        ),
        "normalized_brain_p95": float(
            np.percentile(x, 95)
        ),
        "normalized_brain_min": float(
            x.min()
        ),
        "normalized_brain_max": float(
            x.max()
        ),

        "negative_brain_voxels": negative_voxels,
        "negative_brain_pct": negative_pct,

        "outside_valid_brain_nonzero": int(
            np.count_nonzero(
                final[~valid_brain]
            )
        ),

        "final_pet": str(
            final_pet.relative_to(ROOT)
        ),

        "status": "success",
    }

    with open(qc_json, "w") as f:
        json.dump(
            qc,
            f,
            indent=2,
        )

    print(
        f"{pet_name}: SUCCESS | "
        f"coverage={coverage_pct:.2f}% | "
        f"ref_mean={ref_mean:.4f} | "
        f"ref_cv={ref_cv:.3f} | "
        f"brain_mean={qc['normalized_brain_mean']:.4f}"
    )

    return qc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pet-image-id",
        type=int,
        help=(
            "Process one numeric ADNI PET Image ID, "
            "e.g. 1594687"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all 822 locked PET selections"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocess an already completed PET"
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help=(
            "ITK threads per PET case "
            "(default: 2)"
        ),
    )

    args = parser.parse_args()

    if bool(args.pet_image_id) == bool(args.all):
        parser.error(
            "Specify exactly one of "
            "--pet-image-id or --all"
        )

    df = build_manifest()

    if args.pet_image_id:
        rows = df[
            df["pet_image_id"]
            == args.pet_image_id
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"PET Image ID {args.pet_image_id}: "
                f"expected one row, found {len(rows)}"
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
        pet_id = int(row["pet_image_id"])

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
                    "pet_image_id": pet_id,
                    "RID": int(row["RID"]),
                    "error": str(e),
                }
            )

            print(
                f"I{pet_id}: FAILED — {e}"
            )

    summary = {
        "expected": len(df),
        "completed_without_exception": success,
        "failed": len(failed),
        "failures": failed,
    }

    PET_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        PET_OUTPUT
        / "cohort_preprocessing_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n=== COHORT SUMMARY ===")
    print("Expected:", len(df))
    print("Completed:", success)
    print("Failed:", len(failed))


if __name__ == "__main__":
    main()
