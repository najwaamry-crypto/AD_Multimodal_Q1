#!/usr/bin/env python3

from pathlib import Path
import json
import random

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

PET_ROOT = (
    ROOT
    / "data/processed/pet_model_input_v1"
)

PET_CKPT = (
    ROOT
    / "results/baselines/pet_spatial_3dcnn_v1"
    / "best_validation_model.pt"
)

RESULTS_DIR = (
    ROOT
    / "results/multimodal/pet_clinical_primary_fusion_v1"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 20260921

BATCH_SIZE = 4
NUM_WORKERS = 4

MAX_EPOCHS = 50
PATIENCE = 10

LR = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "CN",
    "MCI",
    "Dementia",
]

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "Dementia": 2,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

print("=== PET + PRIMARY CLINICAL FUSION V1 ===")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("TEST SET STATUS: NOT ACCESSED")


# ============================================================
# TRAIN-ONLY CLINICAL PREPROCESSING
# ============================================================

apoe_median = float(
    train_df["apoe4_count"].median()
)

age_mean = float(
    train_df["age_at_index"].mean()
)

age_std = float(
    train_df["age_at_index"].std(ddof=0)
)

edu_mean = float(
    train_df["education_years"].mean()
)

edu_std = float(
    train_df["education_years"].std(ddof=0)
)

apoe_train_imputed = (
    train_df["apoe4_count"]
    .fillna(apoe_median)
)

apoe_mean = float(
    apoe_train_imputed.mean()
)

apoe_std = float(
    apoe_train_imputed.std(ddof=0)
)

print("\n=== TRAIN-ONLY CLINICAL STATS ===")
print("APOE median:", apoe_median)
print("Age mean/std:", age_mean, age_std)
print("Education mean/std:", edu_mean, edu_std)
print("APOE mean/std:", apoe_mean, apoe_std)


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

    age_z = (
        (age - age_mean)
        / age_std
    )

    edu_z = (
        (edu - edu_mean)
        / edu_std
    )

    apoe_z = (
        (apoe - apoe_mean)
        / apoe_std
    )

    return np.asarray(
        [
            age_z,
            sex,
            edu_z,
            apoe_z,
        ],
        dtype=np.float32
    )


# ============================================================
# DATASET
# ============================================================

class PETClinicalDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        pet_id = row["pet_image_id"]

        pet = np.load(
            PET_ROOT
            / pet_id
            / f"{pet_id}_model_input.npy",
            allow_pickle=False
        ).astype(
            np.float32,
            copy=False
        )

        pet = pet[None, ...]

        clin = clinical_vector(
            row
        )

        y = LABEL_MAP[
            row["diagnosis"]
        ]

        return (
            torch.from_numpy(pet),
            torch.from_numpy(clin),
            torch.tensor(
                y,
                dtype=torch.long
            ),
        )


# ============================================================
# PET ENCODER
# ============================================================

class PETEncoder(nn.Module):

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

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 3 * 4 * 3,
                128
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.projection(x)

        return x


# ============================================================
# FULL MODEL
# ============================================================

class PETClinicalFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.pet_encoder = PETEncoder()

        self.clinical_encoder = nn.Sequential(

            nn.Linear(
                4,
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Linear(
                32,
                32
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                128 + 32,
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.40
            ),

            nn.Linear(
                128,
                3
            ),
        )

    def forward(
        self,
        pet,
        clinical
    ):

        pet_feat = self.pet_encoder(
            pet
        )

        clinical_feat = (
            self.clinical_encoder(
                clinical
            )
        )

        fused = torch.cat(
            [
                pet_feat,
                clinical_feat
            ],
            dim=1
        )

        return self.classifier(
            fused
        )


# ============================================================
# INITIALIZE PET ENCODER FROM PET-ONLY MODEL
# ============================================================

def initialize_pet_encoder(
    encoder
):

    ckpt = torch.load(
        PET_CKPT,
        map_location="cpu",
        weights_only=False
    )

    source = ckpt[
        "model_state_dict"
    ]

    target = encoder.state_dict()

    transferred = {}

    for key in target:

        if key in source:
            if source[key].shape == target[key].shape:
                transferred[key] = source[key]

    missing, unexpected = (
        encoder.load_state_dict(
            transferred,
            strict=False
        )
    )

    print(
        "\nPET encoder transferred tensors:",
        len(transferred)
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    ys = []
    preds = []
    probs = []

    for pet, clin, y in loader:

        pet = pet.to(
            DEVICE,
            non_blocking=True
        )

        clin = clin.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            pet,
            clin
        )

        loss = criterion(
            logits,
            y
        )

        prob = torch.softmax(
            logits,
            dim=1
        )

        pred = prob.argmax(
            dim=1
        )

        total_loss += (
            loss.item()
            * y.size(0)
        )

        ys.extend(
            y.cpu().numpy()
        )

        preds.extend(
            pred.cpu().numpy()
        )

        probs.append(
            prob.cpu().numpy()
        )

    y_true = np.asarray(ys)
    y_pred = np.asarray(preds)

    y_prob = np.concatenate(
        probs,
        axis=0
    )

    y_bin = label_binarize(
        y_true,
        classes=[0, 1, 2]
    )

    return {
        "loss":
            total_loss / len(loader.dataset),

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


# ============================================================
# CLASS WEIGHTS
# ============================================================

counts = np.array([
    (
        train_df["diagnosis"] == c
    ).sum()
    for c in CLASS_NAMES
], dtype=np.float32)

weights = (
    len(train_df)
    /
    (
        len(CLASS_NAMES)
        * counts
    )
)


# ============================================================
# LOADERS
# ============================================================

generator = torch.Generator()

generator.manual_seed(
    SEED
)

train_loader = DataLoader(
    PETClinicalDataset(
        train_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    generator=generator,
)

val_loader = DataLoader(
    PETClinicalDataset(
        val_df
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


# ============================================================
# MODEL
# ============================================================

model = PETClinicalFusion()

initialize_pet_encoder(
    model.pet_encoder
)

model = model.to(
    DEVICE
)

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=1e-6
    )
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# TRAIN
# ============================================================

history = []

best_auc = -np.inf
best_f1 = -np.inf

epochs_without_improvement = 0

best_path = (
    RESULTS_DIR
    / "best_validation_model.pt"
)


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    total_train_loss = 0.0

    for pet, clin, y in train_loader:

        pet = pet.to(
            DEVICE,
            non_blocking=True
        )

        clin = clin.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available()
        ):

            logits = model(
                pet,
                clin
            )

            loss = criterion(
                logits,
                y
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_train_loss += (
            loss.item()
            * y.size(0)
        )

    scheduler.step()

    train_loss = (
        total_train_loss
        / len(train_df)
    )

    val_metrics = evaluate(
        model,
        val_loader,
        criterion
    )

    lr_now = (
        optimizer.param_groups[0]["lr"]
    )

    row = {
        "epoch": epoch,
        "lr": float(lr_now),
        "train_loss":
            float(train_loss),
        **{
            f"val_{k}":
                float(v)
            for k, v
            in val_metrics.items()
        }
    }

    history.append(row)

    print(
        f"Epoch {epoch:02d} | "
        f"LR={lr_now:.6f} | "
        f"TrainLoss={train_loss:.4f} | "
        f"ValLoss={val_metrics['loss']:.4f} | "
        f"Acc={val_metrics['accuracy']:.4f} | "
        f"BalAcc={val_metrics['balanced_accuracy']:.4f} | "
        f"F1={val_metrics['macro_f1']:.4f} | "
        f"AUC={val_metrics['macro_auc_ovr']:.4f}"
    )

    current_auc = (
        val_metrics[
            "macro_auc_ovr"
        ]
    )

    current_f1 = (
        val_metrics[
            "macro_f1"
        ]
    )

    improved = (
        current_auc > best_auc
        or (
            np.isclose(
                current_auc,
                best_auc
            )
            and
            current_f1 > best_f1
        )
    )

    if improved:

        best_auc = current_auc
        best_f1 = current_f1

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "best_val_macro_auc":
                    best_auc,

                "best_val_macro_f1":
                    best_f1,

                "label_map":
                    LABEL_MAP,

                "clinical_preprocessing":
                    {
                        "apoe_median":
                            apoe_median,
                        "age_mean":
                            age_mean,
                        "age_std":
                            age_std,
                        "edu_mean":
                            edu_mean,
                        "edu_std":
                            edu_std,
                        "apoe_mean":
                            apoe_mean,
                        "apoe_std":
                            apoe_std,
                        "sex_encoding":
                            "Female=0, Male=1",
                    },
            },
            best_path
        )

    else:

        epochs_without_improvement += 1

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# SAVE
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    RESULTS_DIR
    / "training_history.csv",
    index=False
)

best_row = (
    history_df
    .sort_values(
        [
            "val_macro_auc_ovr",
            "val_macro_f1"
        ],
        ascending=False
    )
    .iloc[0]
)

best_result = {
    "epoch":
        int(
            best_row[
                "epoch"
            ]
        ),

    "accuracy":
        float(
            best_row[
                "val_accuracy"
            ]
        ),

    "balanced_accuracy":
        float(
            best_row[
                "val_balanced_accuracy"
            ]
        ),

    "macro_f1":
        float(
            best_row[
                "val_macro_f1"
            ]
        ),

    "macro_auc_ovr":
        float(
            best_row[
                "val_macro_auc_ovr"
            ]
        ),

    "log_loss":
        float(
            best_row[
                "val_log_loss"
            ]
        ),
}

with open(
    RESULTS_DIR
    / "best_validation_result.json",
    "w"
) as f:

    json.dump(
        best_result,
        f,
        indent=2
    )


print(
    "\n=== BEST PET + CLINICAL PRIMARY RESULT ==="
)

for k, v in best_result.items():
    print(
        f"{k}: {v}"
    )

print(
    "\nREFERENCE:"
)

print(
    "Clinical-only AUC ~0.645"
)

print(
    "PET-only AUC ~0.767"
)

print(
    "\nTEST SET STATUS: NOT ACCESSED"
)

print(
    "\nSaved to:",
    RESULTS_DIR
)
