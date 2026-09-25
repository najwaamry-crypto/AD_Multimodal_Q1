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

MANIFEST = ROOT / "data/processed/master_modelling_manifest_v1.csv"

MRI_ROOT = ROOT / "data/processed/mri_model_input_v1"
PET_ROOT = ROOT / "data/processed/pet_model_input_v1"

MRI_CKPT = (
    ROOT
    / "results/baselines/mri_spatial_3dcnn_v2"
    / "best_validation_model.pt"
)

PET_CKPT = (
    ROOT
    / "results/baselines/pet_spatial_3dcnn_v1"
    / "best_validation_model.pt"
)

RESULTS_DIR = (
    ROOT
    / "results/multimodal/mri_pet_crossmodal_attention_v1"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 20260921

BATCH_SIZE = 2
NUM_WORKERS = 4

MAX_EPOCHS = 50
PATIENCE = 10

LR = 1e-4
WEIGHT_DECAY = 1e-4

TOKEN_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.20

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
# DATASET
# ============================================================

class MRIPETDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

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
        ).astype(
            np.float32,
            copy=False
        )

        pet = np.load(
            PET_ROOT
            / pet_id
            / f"{pet_id}_model_input.npy",
            allow_pickle=False
        ).astype(
            np.float32,
            copy=False
        )

        if mri.shape != pet.shape:
            raise RuntimeError(
                f"Shape mismatch RID={row['RID']}: "
                f"MRI={mri.shape}, PET={pet.shape}"
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
            torch.tensor(
                y,
                dtype=torch.long
            ),
        )


# ============================================================
# SPATIAL ENCODER
# ============================================================

class SpatialEncoder3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1, 8,
                3,
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
                3,
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
                3,
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
                3,
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

    def forward(self, x):

        x = self.features(x)

        # B,64,3,4,3
        # -> B,36,64

        x = x.flatten(
            2
        ).transpose(
            1, 2
        )

        return x


# ============================================================
# CROSS-MODAL BLOCK
# ============================================================

class CrossModalBlock(nn.Module):

    def __init__(
        self,
        dim=64,
        heads=4,
        dropout=0.2
    ):

        super().__init__()

        self.mri_to_pet = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.pet_to_mri = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm_mri_1 = nn.LayerNorm(
            dim
        )

        self.norm_pet_1 = nn.LayerNorm(
            dim
        )

        self.ffn_mri = nn.Sequential(
            nn.Linear(
                dim,
                dim * 2
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                dim * 2,
                dim
            ),
        )

        self.ffn_pet = nn.Sequential(
            nn.Linear(
                dim,
                dim * 2
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                dim * 2,
                dim
            ),
        )

        self.norm_mri_2 = nn.LayerNorm(
            dim
        )

        self.norm_pet_2 = nn.LayerNorm(
            dim
        )

    def forward(
        self,
        mri_tokens,
        pet_tokens
    ):

        mri_cross, _ = self.mri_to_pet(
            query=mri_tokens,
            key=pet_tokens,
            value=pet_tokens,
            need_weights=False
        )

        pet_cross, _ = self.pet_to_mri(
            query=pet_tokens,
            key=mri_tokens,
            value=mri_tokens,
            need_weights=False
        )

        mri = self.norm_mri_1(
            mri_tokens
            + mri_cross
        )

        pet = self.norm_pet_1(
            pet_tokens
            + pet_cross
        )

        mri = self.norm_mri_2(
            mri
            + self.ffn_mri(mri)
        )

        pet = self.norm_pet_2(
            pet
            + self.ffn_pet(pet)
        )

        return mri, pet


# ============================================================
# FULL MODEL
# ============================================================

class MRIPETCrossModalAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.mri_encoder = (
            SpatialEncoder3D()
        )

        self.pet_encoder = (
            SpatialEncoder3D()
        )

        # Same spatial grid for both modalities.
        # One shared positional embedding preserves
        # homologous location identity.
        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                36,
                TOKEN_DIM
            )
        )

        nn.init.trunc_normal_(
            self.positional_embedding,
            std=0.02
        )

        self.cross_modal = (
            CrossModalBlock(
                dim=TOKEN_DIM,
                heads=NUM_HEADS,
                dropout=DROPOUT
            )
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                TOKEN_DIM * 2,
                128
            ),

            nn.GELU(),

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
        mri,
        pet
    ):

        mri_tokens = self.mri_encoder(
            mri
        )

        pet_tokens = self.pet_encoder(
            pet
        )

        mri_tokens = (
            mri_tokens
            + self.positional_embedding
        )

        pet_tokens = (
            pet_tokens
            + self.positional_embedding
        )

        mri_tokens, pet_tokens = (
            self.cross_modal(
                mri_tokens,
                pet_tokens
            )
        )

        mri_global = (
            mri_tokens.mean(
                dim=1
            )
        )

        pet_global = (
            pet_tokens.mean(
                dim=1
            )
        )

        fused = torch.cat(
            [
                mri_global,
                pet_global
            ],
            dim=1
        )

        return self.classifier(
            fused
        )


# ============================================================
# PRETRAINED ENCODER INITIALIZATION
# ============================================================

def initialize_encoder_from_checkpoint(
    encoder,
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    source_state = checkpoint[
        "model_state_dict"
    ]

    encoder_state = (
        encoder.state_dict()
    )

    transferred = {}

    for key in encoder_state:

        source_key = (
            "features."
            + key[len("features."):]
            if key.startswith(
                "features."
            )
            else key
        )

        if (
            source_key in source_state
            and
            source_state[source_key].shape
            ==
            encoder_state[key].shape
        ):

            transferred[key] = (
                source_state[
                    source_key
                ]
            )

    missing, unexpected = (
        encoder.load_state_dict(
            transferred,
            strict=False
        )
    )

    print(
        f"Initialized {checkpoint_path.name}: "
        f"{len(transferred)} tensors transferred"
    )

    if unexpected:
        print(
            "Unexpected:",
            unexpected
        )

    return len(transferred)


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

    for mri, pet, y in loader:

        mri = mri.to(
            DEVICE,
            non_blocking=True
        )

        pet = pet.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            mri,
            pet
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

    y_true = np.asarray(
        ys
    )

    y_pred = np.asarray(
        preds
    )

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
            total_loss
            / len(loader.dataset),

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
# DATA
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

print(
    "=== MRI PET CROSS-MODAL ATTENTION V1 ==="
)

print(
    "Device:",
    DEVICE
)

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Spatial tokens per modality:",
    36
)

print(
    "Token dimension:",
    TOKEN_DIM
)

print(
    "Attention heads:",
    NUM_HEADS
)

print(
    "TEST SET STATUS: NOT ACCESSED"
)


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
    MRIPETDataset(
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
    MRIPETDataset(
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

model = (
    MRIPETCrossModalAttention()
)

initialize_encoder_from_checkpoint(
    model.mri_encoder,
    MRI_CKPT
)

initialize_encoder_from_checkpoint(
    model.pet_encoder,
    PET_CKPT
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

    running_loss = 0.0

    for mri, pet, y in train_loader:

        mri = mri.to(
            DEVICE,
            non_blocking=True
        )

        pet = pet.to(
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
                mri,
                pet
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

        running_loss += (
            loss.item()
            * y.size(0)
        )

    scheduler.step()

    train_loss = (
        running_loss
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
        "epoch":
            epoch,

        "lr":
            float(lr_now),

        "train_loss":
            float(train_loss),

        **{
            f"val_{k}":
                float(v)
            for k, v
            in val_metrics.items()
        }
    }

    history.append(
        row
    )

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

        best_auc = (
            current_auc
        )

        best_f1 = (
            current_f1
        )

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

                "token_grid":
                    [3, 4, 3],

                "token_dim":
                    TOKEN_DIM,

                "num_heads":
                    NUM_HEADS,
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
    "\n=== BEST CROSS-MODAL ATTENTION RESULT ==="
)

for k, v in best_result.items():

    print(
        f"{k}: {v}"
    )

print(
    "\nPET-only reference AUC: "
    "0.7675"
)

print(
    "Fixed late-fusion reference AUC: "
    "0.7680"
)

print(
    "\nTEST SET STATUS: NOT ACCESSED"
)

print(
    "\nSaved to:",
    RESULTS_DIR
)
