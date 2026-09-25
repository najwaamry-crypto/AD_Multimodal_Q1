#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = ROOT / "data/processed/master_modelling_manifest_v1.csv"

PREDICTIONS = (
    ROOT
    / "results/final_test/test_predictions_v1"
    / "test_predictions_all_models.csv"
)

MRI_ROOT = ROOT / "data/processed/mri_model_input_v1"
PET_ROOT = ROOT / "data/processed/pet_model_input_v1"

CHECKPOINT = (
    ROOT
    / "results/multimodal/mri_pet_clinical_primary_fusion_v1"
    / "best_validation_model.pt"
)

OUT_DIR = ROOT / "results/xai/primary_gradcam_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "Dementia": 2,
}

INDEX_TO_LABEL = {
    0: "CN",
    1: "MCI",
    2: "Dementia",
}


# ============================================================
# LOCKED MODEL ARCHITECTURE
# ============================================================

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
            [
                mri_feat,
                pet_feat,
                clinical_feat,
            ],
            dim=1
        )

        return self.classifier(fused)


# ============================================================
# LOAD MANIFEST + RECOMPUTE LOCKED TRAIN-ONLY CLINICAL STATS
# ============================================================

manifest = pd.read_csv(
    MANIFEST,
    low_memory=False
)

train = manifest[
    manifest["split"] == "train"
].copy()

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

apoe_train_imputed = (
    train["apoe4_count"]
    .fillna(apoe_median)
    .astype(float)
)

apoe_mean = float(
    apoe_train_imputed.mean()
)

apoe_std = float(
    apoe_train_imputed.std(ddof=0)
)


def clinical_vector(row):

    age = float(
        row["age_at_index"]
    )

    edu = float(
        row["education_years"]
    )

    apoe = row[
        "apoe4_count"
    ]

    if pd.isna(apoe):
        apoe = apoe_median

    apoe = float(apoe)

    sex = (
        1.0
        if str(
            row["sex"]
        ).lower() == "male"
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


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM3D:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output.detach()

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0]
            .detach()
        )

    def remove(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        mri,
        pet,
        clinical,
        target_class
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            mri,
            pet,
            clinical
        )

        score = logits[
            0,
            target_class
        ]

        score.backward()

        activations = (
            self.activations[0]
        )

        gradients = (
            self.gradients[0]
        )

        weights = gradients.mean(
            dim=(1, 2, 3),
            keepdim=True
        )

        cam = (
            weights
            * activations
        ).sum(
            dim=0
        )

        cam = F.relu(
            cam
        )

        cam = (
            cam[None, None, ...]
        )

        cam = F.interpolate(
            cam,
            size=tuple(
                mri.shape[2:]
            ),
            mode="trilinear",
            align_corners=False
        )

        cam = (
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )

        probs = (
            torch.softmax(
                logits,
                dim=1
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        return cam, probs


# ============================================================
# CHECKPOINT
# ============================================================

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

if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    state_dict = checkpoint[
        "model_state_dict"
    ]

elif isinstance(
    checkpoint,
    dict
) and "state_dict" in checkpoint:

    state_dict = checkpoint[
        "state_dict"
    ]

else:

    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(device)
model.eval()


# ============================================================
# SELECT 3 DETERMINISTIC TECHNICAL PILOT CASES
# ============================================================

pred = pd.read_csv(
    PREDICTIONS
)

prob_cols = [
    "tri_p_cn",
    "tri_p_mci",
    "tri_p_dementia",
]

pred_probs = pred[
    prob_cols
].to_numpy(
    dtype=float
)

pred[
    "pred_idx"
] = pred_probs.argmax(
    axis=1
)

pred[
    "pred_label"
] = pred[
    "pred_idx"
].map(
    INDEX_TO_LABEL
)

pred[
    "confidence"
] = pred_probs.max(
    axis=1
)

pred[
    "correct"
] = (
    pred[
        "pred_label"
    ]
    ==
    pred[
        "diagnosis"
    ]
)

merged = pred.merge(
    manifest,
    on="RID",
    how="left",
    suffixes=("", "_manifest")
)

selected = []

for diagnosis in [
    "CN",
    "MCI",
    "Dementia",
]:

    subset = merged[
        (
            merged["diagnosis"]
            == diagnosis
        )
        &
        (
            merged["correct"]
        )
    ].copy()

    if len(subset) == 0:

        raise RuntimeError(
            f"No correctly classified {diagnosis} case."
        )

    subset = subset.sort_values(
        "confidence"
    ).reset_index(
        drop=True
    )

    # deterministic median-confidence correct case
    row = subset.iloc[
        len(subset) // 2
    ]

    selected.append(
        row
    )


selected_df = pd.DataFrame(
    selected
)

selected_df[
    [
        "RID",
        "diagnosis",
        "pred_label",
        "confidence",
    ]
].to_csv(
    OUT_DIR
    / "pilot_selected_cases.csv",
    index=False
)


# ============================================================
# RUN MRI + PET GRAD-CAM
# ============================================================

results = []

for _, row in selected_df.iterrows():

    rid = int(
        row["RID"]
    )

    mri_id = str(
        row["mri_image_id"]
    )

    pet_id = str(
        row["pet_image_id"]
    )

    mri_path = (
        MRI_ROOT
        / mri_id
        / f"{mri_id}_model_input.npy"
    )

    pet_path = (
        PET_ROOT
        / pet_id
        / f"{pet_id}_model_input.npy"
    )

    if not mri_path.exists():

        raise FileNotFoundError(
            mri_path
        )

    if not pet_path.exists():

        raise FileNotFoundError(
            pet_path
        )

    mri_np = np.load(
        mri_path,
        allow_pickle=False
    ).astype(
        np.float32
    )

    pet_np = np.load(
        pet_path,
        allow_pickle=False
    ).astype(
        np.float32
    )

    clinical_np = clinical_vector(
        row
    )

    mri = (
        torch.from_numpy(
            mri_np[None, None, ...]
        )
        .to(device)
    )

    pet = (
        torch.from_numpy(
            pet_np[None, None, ...]
        )
        .to(device)
    )

    clinical = (
        torch.from_numpy(
            clinical_np[None, ...]
        )
        .to(device)
    )

    target_class = int(
        row["pred_idx"]
    )

    mri_cam_engine = GradCAM3D(
        model,
        model.mri_encoder.features[12]
    )

    mri_cam, probs = (
        mri_cam_engine.generate(
            mri,
            pet,
            clinical,
            target_class
        )
    )

    mri_cam_engine.remove()

    pet_cam_engine = GradCAM3D(
        model,
        model.pet_encoder.features[12]
    )

    pet_cam, probs_pet = (
        pet_cam_engine.generate(
            mri,
            pet,
            clinical,
            target_class
        )
    )

    pet_cam_engine.remove()

    np.save(
        OUT_DIR
        / f"RID_{rid}_mri_gradcam.npy",
        mri_cam
    )

    np.save(
        OUT_DIR
        / f"RID_{rid}_pet_gradcam.npy",
        pet_cam
    )

    results.append({
        "RID":
            rid,

        "true_label":
            row["diagnosis"],

        "predicted_label":
            row["pred_label"],

        "target_class":
            INDEX_TO_LABEL[
                target_class
            ],

        "p_cn":
            float(probs[0]),

        "p_mci":
            float(probs[1]),

        "p_dementia":
            float(probs[2]),

        "mri_cam_min":
            float(mri_cam.min()),

        "mri_cam_max":
            float(mri_cam.max()),

        "mri_cam_mean":
            float(mri_cam.mean()),

        "pet_cam_min":
            float(pet_cam.min()),

        "pet_cam_max":
            float(pet_cam.max()),

        "pet_cam_mean":
            float(pet_cam.mean()),
    })


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUT_DIR
    / "pilot_gradcam_summary.csv",
    index=False
)


metadata = {
    "checkpoint":
        str(CHECKPOINT),

    "target_layer_mri":
        "mri_encoder.features[12]",

    "target_layer_pet":
        "pet_encoder.features[12]",

    "target_definition":
        "predicted class",

    "pilot_selection":
        "median-confidence correctly classified test subject within each true class",

    "purpose":
        "technical Grad-CAM sanity check before cohort-wide XAI",

    "post_test_model_tuning":
        False,
}


with open(
    OUT_DIR
    / "pilot_xai_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print(
    "=== PRIMARY MODEL GRAD-CAM PILOT V1 ==="
)

print(
    "Device:",
    device
)

print(
    "\nTarget layers:"
)

print(
    "MRI:",
    "model.mri_encoder.features[12]"
)

print(
    "PET:",
    "model.pet_encoder.features[12]"
)

print(
    "\nSelected deterministic pilot cases:"
)

print(
    results_df[
        [
            "RID",
            "true_label",
            "predicted_label",
            "p_cn",
            "p_mci",
            "p_dementia",
            "mri_cam_max",
            "mri_cam_mean",
            "pet_cam_max",
            "pet_cam_mean",
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
