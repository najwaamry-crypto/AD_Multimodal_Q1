#!/usr/bin/env python3

from pathlib import Path
import subprocess


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

SOURCE = (
    ROOT
    / "src/training/10_train_mri_pet_clinical_primary_fusion_v1.py"
)

SEEDS = [
    11,
    42,
    1337,
    31415,
]

TMP_DIR = (
    ROOT
    / "experiments/seed_robustness_v1"
)

TMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

source_text = SOURCE.read_text()


for seed in SEEDS:

    print("\n" + "=" * 70)
    print("RUNNING SEED:", seed)
    print("=" * 70)

    text = source_text

    text = text.replace(
        "SEED = 20260921",
        f"SEED = {seed}"
    )

    text = text.replace(
        '''RESULTS_DIR = (
    ROOT
    / "results/multimodal/mri_pet_clinical_primary_fusion_v1"
)''',
        f'''RESULTS_DIR = (
    ROOT
    / "results/robustness/primary_seed_v1/seed_{seed}"
)'''
    )

    out_script = (
        TMP_DIR
        / f"train_seed_{seed}.py"
    )

    out_script.write_text(
        text
    )

    result = subprocess.run(
        [
            "python",
            str(out_script),
        ]
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Seed {seed} failed "
            f"with return code "
            f"{result.returncode}"
        )

    print(
        f"\nSeed {seed}: COMPLETE"
    )


print("\n" + "=" * 70)
print("ALL ADDITIONAL SEEDS COMPLETE")
print("=" * 70)

print(
    "\nTEST SET STATUS: NOT ACCESSED"
)
