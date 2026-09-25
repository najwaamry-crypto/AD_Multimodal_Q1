#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize


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

MRI_CKPT = (
    ROOT
    / "results/baselines/mri_spatial_3dcnn_v2/"
    / "best_validation_model.pt"
)

PET_CKPT = (
    ROOT
    / "results/baselines/pet_spatial_3dcnn_v1/"
    / "best_validation_model.pt"
)

RESULTS_DIR = (
    ROOT
    / "results/baselines/mri_pet_late_fusion_v1"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "Dementia": 2,
}


class SingleModalityDataset(Dataset):

    def __init__(
        self,
        dataframe,
        modality
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.modality = modality

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        if self.modality == "MRI":

            image_id = row["mri_image_id"]

            path = (
                MRI_ROOT
                / image_id
                / f"{image_id}_model_input.npy"
            )

        else:

            image_id = row["pet_image_id"]

            path = (
                PET_ROOT
                / image_id
                / f"{image_id}_model_input.npy"
            )

        x = np.load(
            path,
            allow_pickle=False
        ).astype(
            np.float32,
            copy=False
        )

        x = x[None, ...]

        y = LABEL_MAP[
            row["diagnosis"]
        ]

        return (
            torch.from_numpy(x),
            torch.tensor(
                y,
                dtype=torch.long
            ),
        )


class Spatial3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1, 8,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(4, 8),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.MaxPool3d(2),

            nn.Conv3d(
                8, 16,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.MaxPool3d(2),

            nn.Conv3d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.MaxPool3d(2),

            nn.Conv3d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
            nn.MaxPool3d(2),

            nn.AdaptiveAvgPool3d(
                (3, 4, 3)
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 3 * 4 * 3,
                128
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Dropout(0.40),

            nn.Linear(
                128,
                3
            ),
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


@torch.no_grad()
def predict(
    model,
    loader
):

    model.eval()

    ys = []
    probs = []

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(x)

        p = torch.softmax(
            logits,
            dim=1
        )

        ys.extend(
            y.numpy()
        )

        probs.append(
            p.cpu().numpy()
        )

    return (
        np.asarray(ys),
        np.concatenate(
            probs,
            axis=0
        )
    )


def metrics(
    y_true,
    y_prob
):

    y_pred = y_prob.argmax(
        axis=1
    )

    y_bin = label_binarize(
        y_true,
        classes=[0, 1, 2]
    )

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "macro_auc_ovr":
            roc_auc_score(
                y_bin,
                y_prob,
                average="macro",
                multi_class="ovr"
            ),

        "log_loss":
            log_loss(
                y_true,
                y_prob,
                labels=[0, 1, 2]
            ),
    }


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

val_df = df[
    df["split"] == "val"
].copy()

print(
    "=== MRI + PET LATE FUSION V1 ==="
)

print(
    "Validation subjects:",
    len(val_df)
)

print(
    "Fusion rule: "
    "0.50 * MRI probability + "
    "0.50 * PET probability"
)

print(
    "TEST SET STATUS: NOT ACCESSED"
)


mri_loader = DataLoader(
    SingleModalityDataset(
        val_df,
        "MRI"
    ),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

pet_loader = DataLoader(
    SingleModalityDataset(
        val_df,
        "PET"
    ),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


mri_model = Spatial3DCNN().to(
    DEVICE
)

pet_model = Spatial3DCNN().to(
    DEVICE
)


mri_ckpt = torch.load(
    MRI_CKPT,
    map_location=DEVICE,
    weights_only=False
)

pet_ckpt = torch.load(
    PET_CKPT,
    map_location=DEVICE,
    weights_only=False
)


mri_model.load_state_dict(
    mri_ckpt[
        "model_state_dict"
    ]
)

pet_model.load_state_dict(
    pet_ckpt[
        "model_state_dict"
    ]
)


y_mri, p_mri = predict(
    mri_model,
    mri_loader
)

y_pet, p_pet = predict(
    pet_model,
    pet_loader
)

if not np.array_equal(
    y_mri,
    y_pet
):
    raise RuntimeError(
        "MRI/PET validation labels "
        "are not aligned."
    )


p_fused = (
    0.5 * p_mri
    +
    0.5 * p_pet
)


mri_metrics = metrics(
    y_mri,
    p_mri
)

pet_metrics = metrics(
    y_pet,
    p_pet
)

fused_metrics = metrics(
    y_mri,
    p_fused
)


print(
    "\n=== MRI VALIDATION ==="
)

for k, v in mri_metrics.items():
    print(
        f"{k}: {v}"
    )


print(
    "\n=== PET VALIDATION ==="
)

for k, v in pet_metrics.items():
    print(
        f"{k}: {v}"
    )


print(
    "\n=== FIXED 50/50 LATE FUSION ==="
)

for k, v in fused_metrics.items():
    print(
        f"{k}: {v}"
    )


result = {
    "fusion_rule":
        "0.5 MRI + 0.5 PET",

    "mri":
        {
            k: float(v)
            for k, v
            in mri_metrics.items()
        },

    "pet":
        {
            k: float(v)
            for k, v
            in pet_metrics.items()
        },

    "late_fusion":
        {
            k: float(v)
            for k, v
            in fused_metrics.items()
        },

    "test_accessed":
        False,
}


with open(
    RESULTS_DIR
    / "validation_result.json",
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )


pred_df = val_df[
    [
        "RID",
        "diagnosis"
    ]
].copy()

pred_df["p_mri_cn"] = (
    p_mri[:, 0]
)

pred_df["p_mri_mci"] = (
    p_mri[:, 1]
)

pred_df["p_mri_dementia"] = (
    p_mri[:, 2]
)

pred_df["p_pet_cn"] = (
    p_pet[:, 0]
)

pred_df["p_pet_mci"] = (
    p_pet[:, 1]
)

pred_df["p_pet_dementia"] = (
    p_pet[:, 2]
)

pred_df["p_fused_cn"] = (
    p_fused[:, 0]
)

pred_df["p_fused_mci"] = (
    p_fused[:, 1]
)

pred_df["p_fused_dementia"] = (
    p_fused[:, 2]
)

pred_df.to_csv(
    RESULTS_DIR
    / "validation_predictions.csv",
    index=False
)

print(
    "\nSaved to:",
    RESULTS_DIR
)

print(
    "\nTEST SET STATUS: NOT ACCESSED"
)
