#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

ORIGINAL = (
    ROOT
    / "results/multimodal/mri_pet_clinical_primary_fusion_v1"
    / "best_validation_result.json"
)

ROBUSTNESS_ROOT = (
    ROOT
    / "results/robustness/primary_seed_v1"
)

OUT_DIR = (
    ROOT
    / "results/summary/primary_seed_robustness_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = [
    20260921,
    11,
    42,
    1337,
    31415,
]

rows = []

for seed in SEEDS:

    if seed == 20260921:
        path = ORIGINAL
    else:
        path = (
            ROBUSTNESS_ROOT
            / f"seed_{seed}"
            / "best_validation_result.json"
        )

    with open(path) as f:
        d = json.load(f)

    rows.append({
        "seed": seed,
        "epoch": d["epoch"],
        "accuracy": d["accuracy"],
        "balanced_accuracy": d["balanced_accuracy"],
        "macro_f1": d["macro_f1"],
        "macro_auc_ovr": d["macro_auc_ovr"],
        "log_loss": d["log_loss"],
    })


df = pd.DataFrame(rows)

metrics = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auc_ovr",
    "log_loss",
]

summary_rows = []

for metric in metrics:

    summary_rows.append({
        "metric": metric,
        "mean": df[metric].mean(),
        "std": df[metric].std(ddof=1),
        "min": df[metric].min(),
        "max": df[metric].max(),
    })

summary = pd.DataFrame(
    summary_rows
)

df.to_csv(
    OUT_DIR / "seed_results.csv",
    index=False
)

summary.to_csv(
    OUT_DIR / "seed_summary.csv",
    index=False
)

with open(
    OUT_DIR / "seed_summary.json",
    "w"
) as f:

    json.dump(
        {
            "primary_seed": 20260921,
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "selection_rule":
                "Original locked primary seed remains primary; "
                "additional seeds are sensitivity analysis only.",
            "test_accessed": False,
            "metrics": {
                row["metric"]: {
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
                for row in summary_rows
            }
        },
        f,
        indent=2
    )


print("=== SEED RESULTS ===")

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n=== SEED ROBUSTNESS SUMMARY ===")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(
    "\nPRIMARY SEED REMAINS: 20260921"
)

print(
    "TEST SET STATUS: NOT ACCESSED"
)

print(
    "\nSaved to:",
    OUT_DIR
)
