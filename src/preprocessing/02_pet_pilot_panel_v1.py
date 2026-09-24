#!/usr/bin/env python3

from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np

ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

TPL = (
    ROOT
    / "data/external/templateflow/tpl-MNI152NLin2009cAsym/"
    / "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)

VERMIS = (
    ROOT
    / "data/external/pet_reference_masks/"
    / "AAL_SPM12_vermis_MNI2009cAsym_1mm.nii.gz"
)

CASES = [
    ("CN", 5, 1594687, 7070),
    ("CN", 89, 1599526, 202166),
    ("MCI", 33, 1597335, 84235),
    ("MCI", 1155, 1593107, 352948),
    ("Dementia", 3, 1594605, 16547),
    ("Dementia", 994, 1602405, 346547),
]


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


for dx, rid, pet_id, mri_id in CASES:
    pet = ROOT / f"data/interim/pet_nifti_v1/I{pet_id}.nii.gz"

    mri_n4 = (
        ROOT
        / f"data/interim/mri_preprocessing_v1/I{mri_id}/"
        / f"I{mri_id}_n4.nii.gz"
    )

    mri_aff = (
        ROOT
        / f"data/interim/mri_preprocessing_v1/I{mri_id}/"
        / f"I{mri_id}_mni_0GenericAffine.mat"
    )

    brain_mask = (
        ROOT
        / f"data/processed/mri_v1/I{mri_id}/"
        / f"I{mri_id}_brainmask.nii.gz"
    )

    outdir = (
        ROOT
        / f"data/interim/pet_preprocessing_pilot/I{pet_id}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rigid_prefix = outdir / f"I{pet_id}_to_I{mri_id}_"
    rigid_aff = outdir / f"I{pet_id}_to_I{mri_id}_0GenericAffine.mat"

    pet_mni = outdir / f"I{pet_id}_mni.nii.gz"
    support_native = outdir / f"I{pet_id}_native_support.nii.gz"
    support_mni = outdir / f"I{pet_id}_support_mni.nii.gz"
    final_pet = outdir / f"I{pet_id}_fdg_vermis_norm.nii.gz"

    print(f"\n=== {dx} RID={rid} PET=I{pet_id} MRI=I{mri_id} ===")

    if not rigid_aff.exists():
        run([
            "antsRegistrationSyNQuick.sh",
            "-d", "3",
            "-f", mri_n4,
            "-m", pet,
            "-o", rigid_prefix,
            "-t", "r",
        ])

    if not pet_mni.exists():
        run([
            "antsApplyTransforms",
            "-d", "3",
            "-i", pet,
            "-r", TPL,
            "-o", pet_mni,
            "-n", "Linear",
            "-t", mri_aff,
            "-t", rigid_aff,
        ])

    if not support_native.exists():
        src = nib.load(pet)
        ones = np.ones(src.shape, dtype=np.uint8)

        nii = nib.Nifti1Image(
            ones,
            src.affine,
            src.header.copy(),
        )
        nii.set_data_dtype(np.uint8)
        nib.save(nii, support_native)

    if not support_mni.exists():
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

    pet_img = nib.load(pet_mni)
    pet_data = np.asarray(pet_img.dataobj, dtype=np.float32)

    brain = np.asarray(
        nib.load(brain_mask).dataobj
    ) > 0

    support = np.asarray(
        nib.load(support_mni).dataobj
    ) > 0.5

    vermis = np.asarray(
        nib.load(VERMIS).dataobj
    ) > 0

    valid_ref = (
        vermis
        & brain
        & support
        & np.isfinite(pet_data)
        & (pet_data > 0)
    )

    valid_brain = (
        brain
        & support
        & np.isfinite(pet_data)
    )

    ref_vals = pet_data[valid_ref]

    if ref_vals.size == 0:
        raise RuntimeError(
            f"I{pet_id}: empty vermis reference"
        )

    ref_mean = float(ref_vals.mean())

    out = np.zeros(pet_data.shape, dtype=np.float32)
    out[valid_brain] = pet_data[valid_brain] / ref_mean

    nii = nib.Nifti1Image(
        out,
        pet_img.affine,
        pet_img.header.copy(),
    )
    nii.set_data_dtype(np.float32)
    nib.save(nii, final_pet)

    coverage = 100 * valid_brain.sum() / brain.sum()

    vals = out[valid_brain]

    print(
        f"I{pet_id}: SUCCESS | "
        f"coverage={coverage:.2f}% | "
        f"ref_mean={ref_mean:.4f} | "
        f"brain_mean={vals.mean():.4f} | "
        f"brain_std={vals.std():.4f}"
    )
