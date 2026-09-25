#!/usr/bin/env python3

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = (
    ROOT
    / "data/processed/master_modelling_manifest_v1.csv"
)

MRI_ROOT = (
    ROOT
    / "data/processed/mri_model_input_v1"
)

SEED = 20260921

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "Dementia": 2,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class MRIDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        mri_id = row["mri_image_id"]

        x = np.load(
            MRI_ROOT
            / mri_id
            / f"{mri_id}_model_input.npy",
            allow_pickle=False
        ).astype(np.float32)

        x = x[None, ...]

        y = LABEL_MAP[row["diagnosis"]]

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
        )


class SpatialMRI3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        # Input 96 x 112 x 96
        # after four pools -> 6 x 7 x 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * 6 * 7 * 6,
                128
            ),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

train = df[
    df["split"] == "train"
].copy()

parts = []

for diagnosis in [
    "CN",
    "MCI",
    "Dementia",
]:

    parts.append(
        train[
            train["diagnosis"] == diagnosis
        ].sample(
            n=8,
            random_state=SEED
        )
    )

small = pd.concat(
    parts,
    ignore_index=True
)

print("=== SPATIAL OVERFIT SANITY V2 ===")
print("Device:", DEVICE)
print(
    small["diagnosis"]
    .value_counts()
    .to_string()
)

loader = DataLoader(
    MRIDataset(small),
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

model = SpatialMRI3DCNN().to(DEVICE)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(1, 101):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(
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

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)

        correct += int(
            (pred == y).sum()
        )

        total += y.size(0)

        total_loss += (
            loss.item()
            * y.size(0)
        )

    acc = correct / total
    avg_loss = total_loss / total

    if (
        epoch == 1
        or epoch % 5 == 0
        or acc >= 0.95
    ):
        print(
            f"Epoch {epoch:03d} | "
            f"Loss={avg_loss:.4f} | "
            f"TrainAcc={acc:.4f}"
        )

    if acc >= 0.99:

        print("\nSANITY CHECK V2: PASS")
        print(
            "Spatial CNN can memorize the "
            "24-subject training subset."
        )
        break

else:

    print("\nSANITY CHECK V2: FAIL")

print("\nTEST SET STATUS: NOT ACCESSED")
