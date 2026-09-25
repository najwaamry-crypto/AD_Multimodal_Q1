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
            / f"{mri_id}_model_input.npy"
        ).astype(np.float32)

        x = x[None, ...]

        y = LABEL_MAP[row["diagnosis"]]

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
        )


class MRI3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(1, 16, 3, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Linear(
            128,
            3
        )

    def forward(self, x):
        return self.classifier(
            torch.flatten(
                self.features(x),
                1
            )
        )


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

    g = train[
        train["diagnosis"] == diagnosis
    ].sample(
        n=8,
        random_state=SEED
    )

    parts.append(g)

small = pd.concat(
    parts,
    ignore_index=True
)

print("=== OVERFIT SANITY SET ===")
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
    pin_memory=True
)

model = MRI3DCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0
)

for epoch in range(1, 81):

    model.train()

    correct = 0
    total = 0
    total_loss = 0.0

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

        pred = logits.argmax(
            dim=1
        )

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
            f"Epoch {epoch:02d} | "
            f"Loss={avg_loss:.4f} | "
            f"TrainAcc={acc:.4f}"
        )

    if acc >= 0.99:
        print("\nSANITY CHECK: PASS")
        print(
            "Network can memorize the small training subset."
        )
        break

else:
    print("\nSANITY CHECK: FAIL")
    print(
        "Network did not memorize the small training subset."
    )

print("\nTEST SET STATUS: NOT ACCESSED")
