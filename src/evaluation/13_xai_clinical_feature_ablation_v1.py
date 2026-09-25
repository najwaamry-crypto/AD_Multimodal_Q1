#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = (
    ROOT
    / "data/processed/master_modelling_manifest_v1.csv"
)

MRI_ROOT = (
    ROOT
    / "data/processed/mri_model_input_v1"
)

PET_ROOT = (
    ROOT
    / "data/processed/pet_model_input_v1"
)

CHECKPOINT = (
    ROOT
    / "results/multimodal/mri_pet_clinical_primary_fusion_v1"
    / "best_validation_model.pt"
)

OUT_DIR = (
    ROOT
    / "results/xai/clinical_feature_ablation_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "Dementia": 2,
}


class SpatialEncoder3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.GroupNorm(4, 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),

            nn.AdaptiveAvgPool3d((3, 4, 3)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 4 * 3, 128),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return x


class MRIPETClinicalFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.mri_encoder = SpatialEncoder3D()
        self.pet_encoder = SpatialEncoder3D()

        self.clinical_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, 3),
        )

    def forward(self, mri, pet, clinical):

        mri_feat = self.mri_encoder(mri)
        pet_feat = self.pet_encoder(pet)
        clinical_feat = self.clinical_encoder(clinical)

        fused = torch.cat(
            [mri_feat, pet_feat, clinical_feat],
            dim=1
        )

        return self.classifier(fused)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

train = df[
    df["split"] == "train"
].copy()

test = df[
    df["split"] == "test"
].copy().reset_index(drop=True)

if len(test) != 164:
    raise RuntimeError(
        f"Expected 164 test subjects, got {len(test)}"
    )


age_mean = float(
    train["age_at_index"].mean()
)

age_std = float(
    train["age_at_index"].std(ddof=0)
)

edu_mean = float(
    train["education_years"].mean()
)

edu_std = float(
    train["education_years"].std(ddof=0)
)

apoe_median = float(
    train["apoe4_count"].median()
)

apoe_train = (
    train["apoe4_count"]
    .fillna(apoe_median)
    .astype(float)
)

apoe_mean = float(
    apoe_train.mean()
)

apoe_std = float(
    apoe_train.std(ddof=0)
)


def clinical_vector(row):

    age = float(row["age_at_index"])
    edu = float(row["education_years"])

    apoe = row["apoe4_count"]

    if pd.isna(apoe):
        apoe = apoe_median

    apoe = float(apoe)

    sex = (
        1.0
        if str(row["sex"]).lower() == "male"
        else 0.0
    )

    return np.asarray(
        [
            (age - age_mean) / age_std,
            sex,
            (edu - edu_mean) / edu_std,
            (apoe - apoe_mean) / apoe_std,
        ],
        dtype=np.float32
    )


device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

model = MRIPETClinicalFusion()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(device)
model.eval()


FEATURES = {
    "none": None,
    "age": 0,
    "sex": 1,
    "education": 2,
    "apoe4": 3,
}


all_probs = {
    key: []
    for key in FEATURES
}

y_true = []


with torch.no_grad():

    for _, row in test.iterrows():

        mri_id = str(
            row["mri_image_id"]
        )

        pet_id = str(
            row["pet_image_id"]
        )

        mri = np.load(
            MRI_ROOT
            / mri_id
            / f"{mri_id}_model_input.npy",
            allow_pickle=False
        ).astype(np.float32)

        pet = np.load(
            PET_ROOT
            / pet_id
            / f"{pet_id}_model_input.npy",
            allow_pickle=False
        ).astype(np.float32)

        clin = clinical_vector(
            row
        )

        mri_t = (
            torch.from_numpy(
                mri[None, None, ...]
            )
            .to(device)
        )

        pet_t = (
            torch.from_numpy(
                pet[None, None, ...]
            )
            .to(device)
        )

        y_true.append(
            LABEL_MAP[
                row["diagnosis"]
            ]
        )

        for name, idx in FEATURES.items():

            clin_mod = clin.copy()

            if idx is not None:
                clin_mod[idx] = 0.0

            clin_t = (
                torch.from_numpy(
                    clin_mod[None, ...]
                )
                .to(device)
            )

            logits = model(
                mri_t,
                pet_t,
                clin_t
            )

            probs = torch.softmax(
                logits,
                dim=1
            )[0].cpu().numpy()

            all_probs[
                name
            ].append(
                probs
            )


y_true = np.asarray(
    y_true,
    dtype=int
)


def metrics(
    y,
    probs
):

    probs = np.asarray(
        probs,
        dtype=np.float64
    )

    probs = (
        probs
        / probs.sum(
            axis=1,
            keepdims=True
        )
    )

    pred = probs.argmax(
        axis=1
    )

    macro_auc = roc_auc_score(
        y,
        probs,
        multi_class="ovr",
        average="macro"
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "macro_auc_ovr":
            float(
                macro_auc
            ),

        "log_loss":
            float(
                log_loss(
                    y,
                    probs,
                    labels=[0, 1, 2]
                )
            ),
    }


baseline = metrics(
    y_true,
    all_probs["none"]
)

rows = []


for name in FEATURES:

    m = metrics(
        y_true,
        all_probs[name]
    )

    rows.append({
        "condition":
            name,

        **m,

        "delta_macro_auc_vs_baseline":
            m["macro_auc_ovr"]
            - baseline["macro_auc_ovr"],

        "delta_balanced_accuracy_vs_baseline":
            m["balanced_accuracy"]
            - baseline["balanced_accuracy"],

        "delta_macro_f1_vs_baseline":
            m["macro_f1"]
            - baseline["macro_f1"],

        "delta_log_loss_vs_baseline":
            m["log_loss"]
            - baseline["log_loss"],
    })


results = pd.DataFrame(
    rows
)

results.to_csv(
    OUT_DIR
    / "clinical_feature_ablation_results.csv",
    index=False
)


metadata = {
    "population":
        "164 held-out test subjects",

    "model":
        "MRI + PET + Primary Clinical Fusion v1",

    "primary_seed":
        20260921,

    "ablation_definition":
        "one clinical feature at a time set to zero in standardized model input space",

    "clinical_input_order":
        [
            "age_z",
            "sex_binary_male_1",
            "education_z",
            "apoe4_z",
        ],

    "interpretation":
        "descriptive post-hoc sensitivity analysis; not causal feature importance",

    "post_test_model_tuning":
        False,
}


with open(
    OUT_DIR
    / "clinical_feature_ablation_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print(
    "=== CLINICAL FEATURE ABLATION V1 ==="
)

print(
    "Subjects:",
    len(test)
)

print(
    "\nBaseline Macro AUC:",
    f"{baseline['macro_auc_ovr']:.4f}"
)

print(
    "\n=== RESULTS ==="
)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(
    "\nPOST-TEST MODEL TUNING: NO"
)

print(
    "Saved to:",
    OUT_DIR
)
