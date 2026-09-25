#!/usr/bin/env python3

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = ROOT / "data/processed/master_modelling_manifest_v1.csv"

MRI_ROOT = ROOT / "data/processed/mri_model_input_v1"
PET_ROOT = ROOT / "data/processed/pet_model_input_v1"

OUT_DIR = ROOT / "results/summary/validation_classwise_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "Dementia": 2,
}

CLASS_NAMES = [
    "CN",
    "MCI",
    "Dementia",
]


# ============================================================
# DATA
# ============================================================

df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

val_df = df[
    df["split"] == "val"
].copy().reset_index(drop=True)

print("=== VALIDATION CLASSWISE ANALYSIS V1 ===")
print("Validation subjects:", len(val_df))
print("TEST SET STATUS: NOT ACCESSED")


# ============================================================
# CLINICAL PREPROCESSING
# ============================================================

clinical_ckpt = joblib.load(
    ROOT
    / "results/baselines/clinical_primary_logreg_v1"
    / "best_validation_model.joblib"
)

clinical_features = [
    "age_at_index",
    "sex",
    "education_years",
    "apoe4_count",
]


# ============================================================
# DATASETS
# ============================================================

class ImagingDataset(Dataset):

    def __init__(
        self,
        dataframe,
        modality
    ):
        self.df = dataframe
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

        y = LABEL_MAP[
            row["diagnosis"]
        ]

        return (
            torch.from_numpy(
                x[None, ...]
            ),
            torch.tensor(
                y,
                dtype=torch.long
            ),
        )


class ImagingClinicalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        modality,
        preprocessing
    ):
        self.df = dataframe
        self.modality = modality
        self.pp = preprocessing

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

        clin = make_clinical_vector(
            row,
            self.pp
        )

        y = LABEL_MAP[
            row["diagnosis"]
        ]

        return (
            torch.from_numpy(
                x[None, ...]
            ),
            torch.from_numpy(
                clin
            ),
            torch.tensor(
                y,
                dtype=torch.long
            ),
        )


class MRIPETClinicalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        preprocessing
    ):
        self.df = dataframe
        self.pp = preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        mri_id = row["mri_image_id"]
        pet_id = row["pet_image_id"]

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

        clin = make_clinical_vector(
            row,
            self.pp
        )

        y = LABEL_MAP[
            row["diagnosis"]
        ]

        return (
            torch.from_numpy(
                mri[None, ...]
            ),
            torch.from_numpy(
                pet[None, ...]
            ),
            torch.from_numpy(
                clin
            ),
            torch.tensor(
                y,
                dtype=torch.long
            ),
        )


# ============================================================
# MODELS
# ============================================================

class Spatial3DCNN(nn.Module):

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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 3 * 4 * 3,
                128
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.classifier(
            self.features(x)
        )


class SingleClinicalFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
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
            nn.Linear(
                64 * 3 * 4 * 3,
                128
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        image,
        clinical
    ):

        feat = self.encoder(image)
        feat = self.projection(feat)

        clin = self.clinical_encoder(
            clinical
        )

        return self.classifier(
            torch.cat(
                [feat, clin],
                dim=1
            )
        )


class MRIPETClinicalFusion(nn.Module):

    def __init__(self):

        super().__init__()

        def encoder():
            return nn.ModuleDict({
                "features": nn.Sequential(
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
                ),

                "projection": nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        64 * 3 * 4 * 3,
                        128
                    ),
                    nn.LeakyReLU(
                        0.1,
                        inplace=True
                    ),
                ),
            })

        self.mri_encoder = encoder()
        self.pet_encoder = encoder()

        self.clinical_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                128 + 128 + 32,
                128
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, 3),
        )

    def run_encoder(
        self,
        enc,
        x
    ):

        x = enc["features"](x)
        x = enc["projection"](x)

        return x

    def forward(
        self,
        mri,
        pet,
        clinical
    ):

        mri_feat = self.run_encoder(
            self.mri_encoder,
            mri
        )

        pet_feat = self.run_encoder(
            self.pet_encoder,
            pet
        )

        clin_feat = self.clinical_encoder(
            clinical
        )

        return self.classifier(
            torch.cat(
                [
                    mri_feat,
                    pet_feat,
                    clin_feat,
                ],
                dim=1
            )
        )


# ============================================================
# CLINICAL PREPROCESSING
# ============================================================

def make_clinical_vector(
    row,
    pp
):

    apoe = row["apoe4_count"]

    if pd.isna(apoe):
        apoe = pp["apoe_median"]

    sex = (
        1.0
        if str(
            row["sex"]
        ).lower() == "male"
        else 0.0
    )

    return np.asarray(
        [
            (
                row["age_at_index"]
                - pp["age_mean"]
            )
            / pp["age_std"],

            sex,

            (
                row["education_years"]
                - pp["edu_mean"]
            )
            / pp["edu_std"],

            (
                apoe
                - pp["apoe_mean"]
            )
            / pp["apoe_std"],
        ],
        dtype=np.float32
    )


# ============================================================
# PREDICTION HELPERS
# ============================================================

@torch.no_grad()
def predict_single(
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


@torch.no_grad()
def predict_single_clinical(
    model,
    loader
):

    model.eval()

    ys = []
    probs = []

    for x, clin, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        clin = clin.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            x,
            clin
        )

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


@torch.no_grad()
def predict_tri(
    model,
    loader
):

    model.eval()

    ys = []
    probs = []

    for (
        mri,
        pet,
        clin,
        y
    ) in loader:

        mri = mri.to(
            DEVICE,
            non_blocking=True
        )

        pet = pet.to(
            DEVICE,
            non_blocking=True
        )

        clin = clin.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            mri,
            pet,
            clin
        )

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


def load_ckpt(
    model,
    path,
    encoder_prefix=None
):

    ckpt = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    state = ckpt[
        "model_state_dict"
    ]

    # MRI+Clinical and PET+Clinical checkpoints were
    # trained with modality-specific encoder names:
    # mri_encoder.* or pet_encoder.*.
    # The generic evaluation class uses encoder/projection.
    if encoder_prefix is not None:

        remapped = {}

        for key, value in state.items():

            if key.startswith(
                f"{encoder_prefix}.features."
            ):

                new_key = (
                    "encoder."
                    + key[
                        len(
                            f"{encoder_prefix}.features."
                        ):
                    ]
                )

            elif key.startswith(
                f"{encoder_prefix}.projection."
            ):

                new_key = (
                    "projection."
                    + key[
                        len(
                            f"{encoder_prefix}.projection."
                        ):
                    ]
                )

            else:
                new_key = key

            remapped[
                new_key
            ] = value

        state = remapped

    model.load_state_dict(
        state
    )

    return model.to(DEVICE), ckpt


# ============================================================
# METRICS
# ============================================================

def classwise_metrics(
    model_name,
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

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    rows = []

    for i, cls in enumerate(
        CLASS_NAMES
    ):

        auc = roc_auc_score(
            y_bin[:, i],
            y_prob[:, i]
        )

        rows.append({
            "model":
                model_name,

            "class":
                cls,

            "auc_ovr":
                float(auc),

            "precision":
                float(precision[i]),

            "recall":
                float(recall[i]),

            "f1":
                float(f1[i]),

            "support":
                int(support[i]),
        })

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )

    return rows, cm


# ============================================================
# RUN MODELS
# ============================================================

all_predictions = {
    "RID":
        val_df["RID"].to_numpy(),

    "diagnosis":
        val_df["diagnosis"].to_numpy(),
}

classwise_rows = []
confusions = {}


# ------------------------------------------------------------
# Clinical primary
# ------------------------------------------------------------

x_clin = val_df[
    clinical_features
].copy()

y_true = (
    val_df["diagnosis"]
    .map(LABEL_MAP)
    .to_numpy()
)

p_clin = clinical_ckpt.predict_proba(
    x_clin
)

rows, cm = classwise_metrics(
    "Clinical primary",
    y_true,
    p_clin
)

classwise_rows.extend(rows)
confusions[
    "Clinical primary"
] = cm

all_predictions[
    "clinical_p_cn"
] = p_clin[:, 0]

all_predictions[
    "clinical_p_mci"
] = p_clin[:, 1]

all_predictions[
    "clinical_p_dementia"
] = p_clin[:, 2]


# ------------------------------------------------------------
# MRI-only
# ------------------------------------------------------------

mri_model = Spatial3DCNN()

mri_model, _ = load_ckpt(
    mri_model,
    ROOT
    / "results/baselines/mri_spatial_3dcnn_v2"
    / "best_validation_model.pt"
)

mri_loader = DataLoader(
    ImagingDataset(
        val_df,
        "MRI"
    ),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

y_mri, p_mri = predict_single(
    mri_model,
    mri_loader
)

rows, cm = classwise_metrics(
    "MRI-only",
    y_mri,
    p_mri
)

classwise_rows.extend(rows)
confusions[
    "MRI-only"
] = cm


# ------------------------------------------------------------
# PET-only
# ------------------------------------------------------------

pet_model = Spatial3DCNN()

pet_model, _ = load_ckpt(
    pet_model,
    ROOT
    / "results/baselines/pet_spatial_3dcnn_v1"
    / "best_validation_model.pt"
)

pet_loader = DataLoader(
    ImagingDataset(
        val_df,
        "PET"
    ),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

y_pet, p_pet = predict_single(
    pet_model,
    pet_loader
)

rows, cm = classwise_metrics(
    "PET-only",
    y_pet,
    p_pet
)

classwise_rows.extend(rows)
confusions[
    "PET-only"
] = cm


# ------------------------------------------------------------
# MRI + Clinical
# ------------------------------------------------------------

mri_clin_path = (
    ROOT
    / "results/multimodal/mri_clinical_primary_fusion_v1"
    / "best_validation_model.pt"
)

mri_clin_model = (
    SingleClinicalFusion()
)

mri_clin_model, mri_clin_ckpt = (
    load_ckpt(
        mri_clin_model,
        mri_clin_path,
        encoder_prefix="mri_encoder"
    )
)

mri_clin_pp = (
    mri_clin_ckpt[
        "clinical_preprocessing"
    ]
)

mri_clin_loader = DataLoader(
    ImagingClinicalDataset(
        val_df,
        "MRI",
        mri_clin_pp
    ),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

y_mc, p_mc = (
    predict_single_clinical(
        mri_clin_model,
        mri_clin_loader
    )
)

rows, cm = classwise_metrics(
    "MRI + Clinical",
    y_mc,
    p_mc
)

classwise_rows.extend(rows)
confusions[
    "MRI + Clinical"
] = cm


# ------------------------------------------------------------
# PET + Clinical
# ------------------------------------------------------------

pet_clin_path = (
    ROOT
    / "results/multimodal/pet_clinical_primary_fusion_v1"
    / "best_validation_model.pt"
)

pet_clin_model = (
    SingleClinicalFusion()
)

pet_clin_model, pet_clin_ckpt = (
    load_ckpt(
        pet_clin_model,
        pet_clin_path,
        encoder_prefix="pet_encoder"
    )
)

pet_clin_pp = (
    pet_clin_ckpt[
        "clinical_preprocessing"
    ]
)

pet_clin_loader = DataLoader(
    ImagingClinicalDataset(
        val_df,
        "PET",
        pet_clin_pp
    ),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

y_pc, p_pc = (
    predict_single_clinical(
        pet_clin_model,
        pet_clin_loader
    )
)

rows, cm = classwise_metrics(
    "PET + Clinical",
    y_pc,
    p_pc
)

classwise_rows.extend(rows)
confusions[
    "PET + Clinical"
] = cm


# ------------------------------------------------------------
# MRI + PET + Clinical
# ------------------------------------------------------------

tri_path = (
    ROOT
    / "results/multimodal/mri_pet_clinical_primary_fusion_v1"
    / "best_validation_model.pt"
)

tri_model = (
    MRIPETClinicalFusion()
)

tri_model, tri_ckpt = load_ckpt(
    tri_model,
    tri_path
)

tri_pp = (
    tri_ckpt[
        "clinical_preprocessing"
    ]
)

tri_loader = DataLoader(
    MRIPETClinicalDataset(
        val_df,
        tri_pp
    ),
    batch_size=2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

y_tri, p_tri = predict_tri(
    tri_model,
    tri_loader
)

rows, cm = classwise_metrics(
    "MRI + PET + Clinical",
    y_tri,
    p_tri
)

classwise_rows.extend(rows)
confusions[
    "MRI + PET + Clinical"
] = cm


# ============================================================
# SAVE
# ============================================================

pred_df = pd.DataFrame(
    all_predictions
)

for prefix, probs in [
    ("mri", p_mri),
    ("pet", p_pet),
    ("mri_clin", p_mc),
    ("pet_clin", p_pc),
    ("tri", p_tri),
]:

    pred_df[
        f"{prefix}_p_cn"
    ] = probs[:, 0]

    pred_df[
        f"{prefix}_p_mci"
    ] = probs[:, 1]

    pred_df[
        f"{prefix}_p_dementia"
    ] = probs[:, 2]


pred_df.to_csv(
    OUT_DIR
    / "validation_predictions_all_models.csv",
    index=False
)


classwise_df = pd.DataFrame(
    classwise_rows
)

classwise_df.to_csv(
    OUT_DIR
    / "classwise_metrics.csv",
    index=False
)


with open(
    OUT_DIR
    / "confusion_matrices.json",
    "w"
) as f:

    json.dump(
        {
            k: v.tolist()
            for k, v
            in confusions.items()
        },
        f,
        indent=2
    )


print(
    "\n=== CLASSWISE AUC ==="
)

auc_table = (
    classwise_df
    .pivot(
        index="model",
        columns="class",
        values="auc_ovr"
    )
    [
        [
            "CN",
            "MCI",
            "Dementia",
        ]
    ]
)

print(
    auc_table.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print(
    "\n=== CLASSWISE F1 ==="
)

f1_table = (
    classwise_df
    .pivot(
        index="model",
        columns="class",
        values="f1"
    )
    [
        [
            "CN",
            "MCI",
            "Dementia",
        ]
    ]
)

print(
    f1_table.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print(
    "\n=== CONFUSION MATRICES ==="
)

for name, cm in confusions.items():

    print(
        "\n",
        name,
        sep=""
    )

    print(cm)

print(
    "\nRows=true [CN, MCI, Dementia]"
)

print(
    "Cols=pred [CN, MCI, Dementia]"
)

print(
    "\nTEST SET STATUS: NOT ACCESSED"
)

print(
    "\nSaved to:",
    OUT_DIR
)
