#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

SUMMARY_PATH = (
    ROOT
    / "results/xai/clinical_permutation_importance_v1"
    / "permutation_importance_summary.csv"
)

OUT_DIR = (
    ROOT
    / "figures/xai"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    SUMMARY_PATH
)

label_map = {
    "age": "Age",
    "apoe4": "APOE4",
    "education": "Education",
    "sex": "Sex",
}

df["label"] = df["feature"].map(
    label_map
)

# Importance shown as positive AUC drop
df["importance"] = (
    -df["mean_delta_macro_auc"]
)

# Convert empirical delta interval into interval
# on positive AUC drop scale.
df["importance_low"] = (
    -df["ci95_high_delta_macro_auc"]
)

df["importance_high"] = (
    -df["ci95_low_delta_macro_auc"]
)

df = df.sort_values(
    "importance",
    ascending=True
).reset_index(drop=True)


y = np.arange(
    len(df)
)

x = df["importance"].to_numpy()

xerr_low = (
    x
    - df["importance_low"].to_numpy()
)

xerr_high = (
    df["importance_high"].to_numpy()
    - x
)


fig, ax = plt.subplots(
    figsize=(7.5, 4.8)
)

ax.barh(
    y,
    x,
    xerr=np.vstack(
        [
            xerr_low,
            xerr_high,
        ]
    ),
    capsize=4
)

ax.set_yticks(
    y
)

ax.set_yticklabels(
    df["label"]
)

ax.axvline(
    0.0,
    linewidth=1
)

ax.set_xlabel(
    "Mean decrease in macro OvR ROC-AUC after permutation"
)

ax.set_title(
    "Clinical permutation importance in the locked trimodal model"
)

ax.text(
    0.99,
    0.02,
    "Bars: mean over 100 permutations\n"
    "Error bars: empirical 2.5th–97.5th percentile interval",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9
)

plt.tight_layout()


png_path = (
    OUT_DIR
    / "clinical_permutation_importance_v1.png"
)

pdf_path = (
    OUT_DIR
    / "clinical_permutation_importance_v1.pdf"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close(
    fig
)


print(
    "=== CLINICAL PERMUTATION IMPORTANCE FIGURE V1 ==="
)

print(
    df[
        [
            "label",
            "importance",
            "importance_low",
            "importance_high",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(
    "\nSaved PNG:",
    png_path
)

print(
    "Saved PDF:",
    pdf_path
)
