#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

PRED_PATH = (
    ROOT
    / "results/summary/validation_classwise_v1"
    / "validation_predictions_all_models.csv"
)

OUT_DIR = (
    ROOT
    / "results/summary/primary_validation_calibration_v1"
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


df = pd.read_csv(
    PRED_PATH
)

print("=== PRIMARY MODEL VALIDATION CALIBRATION V1 ===")
print("Subjects:", len(df))
print("TEST SET STATUS: NOT ACCESSED")


y_true = (
    df["diagnosis"]
    .map(LABEL_MAP)
    .to_numpy()
)

probs = df[
    [
        "tri_p_cn",
        "tri_p_mci",
        "tri_p_dementia",
    ]
].to_numpy(
    dtype=np.float64
)


# ============================================================
# BASIC CHECKS
# ============================================================

if not np.isfinite(
    probs
).all():

    raise RuntimeError(
        "Non-finite probabilities detected."
    )

row_sums = probs.sum(
    axis=1,
    keepdims=True
)

if np.any(
    row_sums <= 0
):
    raise RuntimeError(
        "Invalid probability row sum."
    )

# CSV serialization can introduce tiny floating-point
# deviations from an exact sum of one.
probs = probs / row_sums

if not np.allclose(
    probs.sum(axis=1),
    1.0,
    atol=1e-12
):
    raise RuntimeError(
        "Probability normalization failed."
    )


# ============================================================
# LOG LOSS
# ============================================================

multiclass_log_loss = log_loss(
    y_true,
    probs,
    labels=[0, 1, 2]
)


# ============================================================
# MULTICLASS BRIER SCORE
# mean over subjects of sum_k (p_ik - y_ik)^2
# ============================================================

y_onehot = np.eye(
    3,
    dtype=np.float64
)[y_true]

brier_per_subject = np.sum(
    (
        probs
        - y_onehot
    ) ** 2,
    axis=1
)

multiclass_brier = float(
    brier_per_subject.mean()
)


# ============================================================
# CONFIDENCE-BASED ECE
# ============================================================

pred = probs.argmax(
    axis=1
)

confidence = probs.max(
    axis=1
)

correct = (
    pred == y_true
).astype(
    np.float64
)


N_BINS = 10

edges = np.linspace(
    0.0,
    1.0,
    N_BINS + 1
)

bin_rows = []

ece = 0.0


for i in range(
    N_BINS
):

    lower = edges[i]
    upper = edges[i + 1]

    if i == N_BINS - 1:

        mask = (
            (confidence >= lower)
            &
            (confidence <= upper)
        )

    else:

        mask = (
            (confidence >= lower)
            &
            (confidence < upper)
        )

    n = int(
        mask.sum()
    )

    if n == 0:

        bin_rows.append({
            "bin":
                i + 1,

            "lower":
                lower,

            "upper":
                upper,

            "n":
                0,

            "mean_confidence":
                np.nan,

            "accuracy":
                np.nan,

            "abs_gap":
                np.nan,
        })

        continue

    bin_conf = float(
        confidence[
            mask
        ].mean()
    )

    bin_acc = float(
        correct[
            mask
        ].mean()
    )

    gap = abs(
        bin_acc
        - bin_conf
    )

    weight = (
        n
        / len(df)
    )

    ece += (
        weight
        * gap
    )

    bin_rows.append({
        "bin":
            i + 1,

        "lower":
            lower,

        "upper":
            upper,

        "n":
            n,

        "mean_confidence":
            bin_conf,

        "accuracy":
            bin_acc,

        "abs_gap":
            gap,
    })


bins_df = pd.DataFrame(
    bin_rows
)


# ============================================================
# ADDITIONAL DESCRIPTIVE CALIBRATION METRICS
# ============================================================

mean_confidence = float(
    confidence.mean()
)

accuracy = float(
    correct.mean()
)

confidence_gap = float(
    mean_confidence
    - accuracy
)


# ============================================================
# SAVE
# ============================================================

summary = {
    "model":
        "MRI + PET + Primary Clinical Fusion v1",

    "primary_seed":
        20260921,

    "n_validation":
        int(len(df)),

    "accuracy":
        accuracy,

    "mean_confidence":
        mean_confidence,

    "mean_confidence_minus_accuracy":
        confidence_gap,

    "multiclass_log_loss":
        float(
            multiclass_log_loss
        ),

    "multiclass_brier_score":
        multiclass_brier,

    "ece_10bin":
        float(ece),

    "ece_definition":
        "confidence-based expected calibration error "
        "using 10 equal-width bins",

    "test_accessed":
        False,
}


with open(
    OUT_DIR
    / "calibration_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


bins_df.to_csv(
    OUT_DIR
    / "reliability_bins.csv",
    index=False
)


subject_df = pd.DataFrame({
    "RID":
        df["RID"],

    "diagnosis":
        df["diagnosis"],

    "predicted_class":
        [
            [
                "CN",
                "MCI",
                "Dementia",
            ][i]
            for i in pred
        ],

    "confidence":
        confidence,

    "correct":
        correct.astype(int),

    "brier_contribution":
        brier_per_subject,
})


subject_df.to_csv(
    OUT_DIR
    / "subject_calibration.csv",
    index=False
)


print("\n=== CALIBRATION SUMMARY ===")

print(
    "Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Mean confidence:",
    f"{mean_confidence:.4f}"
)

print(
    "Mean confidence - accuracy:",
    f"{confidence_gap:.4f}"
)

print(
    "Multiclass log loss:",
    f"{multiclass_log_loss:.4f}"
)

print(
    "Multiclass Brier score:",
    f"{multiclass_brier:.4f}"
)

print(
    "ECE (10 bins):",
    f"{ece:.4f}"
)


print(
    "\n=== RELIABILITY BINS ==="
)

print(
    bins_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print(
    "\nTEST SET STATUS: NOT ACCESSED"
)

print(
    "\nSaved to:",
    OUT_DIR
)
