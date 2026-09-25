#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

OUT_DIR = ROOT / "results/summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    {
        "model": "Clinical primary",
        "modalities": "Clinical",
        "path": ROOT / "results/baselines/clinical_primary_logreg_v1/best_validation_result.json",
        "status": "primary",
        "type": "standard",
    },
    {
        "model": "MRI-only V2",
        "modalities": "MRI",
        "path": ROOT / "results/baselines/mri_spatial_3dcnn_v2/best_validation_result.json",
        "status": "primary",
        "type": "standard",
    },
    {
        "model": "PET-only",
        "modalities": "PET",
        "path": ROOT / "results/baselines/pet_spatial_3dcnn_v1/best_validation_result.json",
        "status": "primary",
        "type": "standard",
    },
    {
        "model": "MRI + Clinical",
        "modalities": "MRI + Clinical",
        "path": ROOT / "results/multimodal/mri_clinical_primary_fusion_v1/best_validation_result.json",
        "status": "primary",
        "type": "standard",
    },
    {
        "model": "PET + Clinical",
        "modalities": "PET + Clinical",
        "path": ROOT / "results/multimodal/pet_clinical_primary_fusion_v1/best_validation_result.json",
        "status": "primary",
        "type": "standard",
    },
    {
        "model": "MRI + PET simple fusion",
        "modalities": "MRI + PET",
        "path": ROOT / "results/baselines/mri_pet_simple_fusion_v1/best_validation_result.json",
        "status": "primary",
        "type": "standard",
    },
    {
        "model": "MRI + PET late fusion",
        "modalities": "MRI + PET",
        "path": ROOT / "results/baselines/mri_pet_late_fusion_v1/validation_result.json",
        "status": "primary",
        "type": "late_fusion",
    },
    {
        "model": "MRI ↔ PET cross-modal attention",
        "modalities": "MRI + PET",
        "path": ROOT / "results/multimodal/mri_pet_crossmodal_attention_v1/best_validation_result.json",
        "status": "primary",
        "type": "standard",
    },
    {
        "model": "MRI + PET + Clinical",
        "modalities": "MRI + PET + Clinical",
        "path": ROOT / "results/multimodal/mri_pet_clinical_primary_fusion_v1/best_validation_result.json",
        "status": "selected_primary_model",
        "type": "standard",
    },

    # Exploratory MRI architectures retained for audit trail
    {
        "model": "MRI-only V1",
        "modalities": "MRI",
        "path": ROOT / "results/baselines/mri_3dcnn_v1/best_validation_result.json",
        "status": "exploratory",
        "type": "standard",
    },
    {
        "model": "MRI-only V3",
        "modalities": "MRI",
        "path": ROOT / "results/baselines/mri_spatial_3dcnn_v3/best_validation_result.json",
        "status": "exploratory",
        "type": "standard",
    },
]


def get_metric(d, *names):
    for name in names:
        if name in d:
            return d[name]
    return None


rows = []

for exp in EXPERIMENTS:

    path = exp["path"]

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path) as f:
        d = json.load(f)

    if exp["type"] == "late_fusion":
        d = d["late_fusion"]

    row = {
        "model": exp["model"],
        "modalities": exp["modalities"],
        "status": exp["status"],
        "epoch": get_metric(d, "epoch"),
        "accuracy": get_metric(
            d,
            "accuracy",
            "val_accuracy",
        ),
        "balanced_accuracy": get_metric(
            d,
            "balanced_accuracy",
            "val_balanced_accuracy",
        ),
        "macro_f1": get_metric(
            d,
            "macro_f1",
            "val_macro_f1",
        ),
        "macro_auc_ovr": get_metric(
            d,
            "macro_auc_ovr",
            "val_macro_auc_ovr",
        ),
        "log_loss": get_metric(
            d,
            "log_loss",
            "val_log_loss",
        ),
        "source_file": str(
            path.relative_to(ROOT)
        ),
    }

    rows.append(row)


df = pd.DataFrame(rows)

primary = df[
    df["status"] != "exploratory"
].copy()

primary = primary.sort_values(
    "macro_auc_ovr",
    ascending=False
).reset_index(drop=True)

full = df.sort_values(
    [
        "status",
        "macro_auc_ovr",
    ],
    ascending=[
        True,
        False,
    ]
).reset_index(drop=True)


primary.to_csv(
    OUT_DIR / "validation_primary_models_v1.csv",
    index=False
)

full.to_csv(
    OUT_DIR / "validation_all_experiments_v1.csv",
    index=False
)


with open(
    OUT_DIR / "validation_summary_v1.json",
    "w"
) as f:

    json.dump(
        {
            "selection_metric":
                "macro_auc_ovr",

            "test_accessed":
                False,

            "selected_primary_model":
                "MRI + PET + Clinical",

            "selected_primary_model_validation_auc":
                float(
                    primary.loc[
                        primary["model"]
                        == "MRI + PET + Clinical",
                        "macro_auc_ovr"
                    ].iloc[0]
                ),

            "n_primary_models":
                int(len(primary)),

            "n_all_experiments":
                int(len(full)),
        },
        f,
        indent=2
    )


print("=== PRIMARY VALIDATION MODELS ===")

print(
    primary[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_auc_ovr",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n=== SELECTED PRIMARY MODEL ===")

best = primary.loc[
    primary["model"]
    == "MRI + PET + Clinical"
].iloc[0]

print(
    "Model:",
    best["model"]
)

print(
    "Macro AUC:",
    f"{best['macro_auc_ovr']:.4f}"
)

print(
    "Balanced accuracy:",
    f"{best['balanced_accuracy']:.4f}"
)

print(
    "Macro F1:",
    f"{best['macro_f1']:.4f}"
)

print(
    "Accuracy:",
    f"{best['accuracy']:.4f}"
)

print("\nTEST SET STATUS: NOT ACCESSED")

print(
    "\nSaved:",
    OUT_DIR / "validation_primary_models_v1.csv"
)
