#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

PRED_PATH = (
    ROOT
    / "results/final_test/test_predictions_v1"
    / "test_predictions_all_models.csv"
)

OUT_DIR = (
    ROOT
    / "results/final_test/statistics_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

N_BOOT = 10000
BOOT_SEED = 20260925

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

MODELS = {
    "Clinical primary": [
        "clinical_p_cn",
        "clinical_p_mci",
        "clinical_p_dementia",
    ],

    "MRI-only": [
        "mri_p_cn",
        "mri_p_mci",
        "mri_p_dementia",
    ],

    "PET-only": [
        "pet_p_cn",
        "pet_p_mci",
        "pet_p_dementia",
    ],

    "MRI + Clinical": [
        "mri_clin_p_cn",
        "mri_clin_p_mci",
        "mri_clin_p_dementia",
    ],

    "PET + Clinical": [
        "pet_clin_p_cn",
        "pet_clin_p_mci",
        "pet_clin_p_dementia",
    ],

    "MRI + PET + Clinical": [
        "tri_p_cn",
        "tri_p_mci",
        "tri_p_dementia",
    ],
}


# Added-value orientation:
# positive delta favors the multimodal / augmented model.
COMPARISONS = [
    (
        "MRI + PET + Clinical",
        "PET + Clinical",
        "Primary: Tri-modal - PET+Clinical",
    ),
    (
        "MRI + PET + Clinical",
        "Clinical primary",
        "Tri-modal - Clinical",
    ),
    (
        "MRI + Clinical",
        "MRI-only",
        "MRI+Clinical - MRI",
    ),
    (
        "PET + Clinical",
        "PET-only",
        "PET+Clinical - PET",
    ),
    (
        "MRI + PET + Clinical",
        "PET-only",
        "Tri-modal - PET",
    ),
]


def normalize_probabilities(p):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    if not np.isfinite(p).all():
        raise RuntimeError(
            "Non-finite probability detected."
        )

    row_sums = p.sum(
        axis=1,
        keepdims=True
    )

    if np.any(row_sums <= 0):
        raise RuntimeError(
            "Invalid probability row sum."
        )

    return p / row_sums


def compute_metrics(
    y_true,
    probs
):

    probs = normalize_probabilities(
        probs
    )

    pred = probs.argmax(
        axis=1
    )

    y_bin = label_binarize(
        y_true,
        classes=[0, 1, 2]
    )

    class_aucs = {}

    for i, cls in enumerate(
        CLASS_NAMES
    ):

        class_aucs[
            f"auc_{cls.lower()}"
        ] = roc_auc_score(
            y_bin[:, i],
            probs[:, i]
        )

    macro_auc = float(
        np.mean(
            list(
                class_aucs.values()
            )
        )
    )

    y_onehot = np.eye(
        3,
        dtype=np.float64
    )[y_true]

    brier = float(
        np.mean(
            np.sum(
                (
                    probs
                    - y_onehot
                ) ** 2,
                axis=1
            )
        )
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    pred
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    pred
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "macro_auc_ovr":
            macro_auc,

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probs,
                    labels=[0, 1, 2]
                )
            ),

        "multiclass_brier":
            brier,

        **{
            k: float(v)
            for k, v
            in class_aucs.items()
        },
    }


def percentile_ci(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),
        float(
            np.percentile(
                values,
                97.5
            )
        ),
    )


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================

df = pd.read_csv(
    PRED_PATH
)

y_true = (
    df["diagnosis"]
    .map(LABEL_MAP)
    .to_numpy()
)

if len(df) != 164:
    raise RuntimeError(
        f"Expected 164 test subjects, got {len(df)}"
    )


print(
    "=== FINAL HELD-OUT TEST STATISTICS V1 ==="
)

print(
    "Test subjects:",
    len(df)
)

print(
    "Class counts:"
)

print(
    df["diagnosis"]
    .value_counts()
    .to_string()
)

print(
    "\nBootstrap replicates:",
    N_BOOT
)

print(
    "Bootstrap seed:",
    BOOT_SEED
)

print(
    "Resampling: paired, diagnosis-stratified"
)


# ============================================================
# PROBABILITY MATRICES
# ============================================================

model_probs = {}

for model, cols in MODELS.items():

    model_probs[
        model
    ] = normalize_probabilities(
        df[cols].to_numpy()
    )


# ============================================================
# POINT ESTIMATES
# ============================================================

point_rows = []

point_metrics = {}

for model, probs in model_probs.items():

    metrics = compute_metrics(
        y_true,
        probs
    )

    point_metrics[
        model
    ] = metrics

    point_rows.append({
        "model":
            model,
        **metrics,
    })


point_df = pd.DataFrame(
    point_rows
)

point_df.to_csv(
    OUT_DIR
    / "test_point_estimates.csv",
    index=False
)


print(
    "\n=== TEST POINT ESTIMATES ==="
)

print(
    point_df[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_auc_ovr",
            "log_loss",
            "multiclass_brier",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# FIXED STRATIFIED BOOTSTRAP INDICES
# ============================================================

rng = np.random.default_rng(
    BOOT_SEED
)

class_indices = {
    c: np.flatnonzero(
        y_true == c
    )
    for c in [0, 1, 2]
}

bootstrap_indices = []

for _ in range(
    N_BOOT
):

    sampled_parts = []

    for c in [0, 1, 2]:

        idx = class_indices[c]

        sampled = rng.choice(
            idx,
            size=len(idx),
            replace=True
        )

        sampled_parts.append(
            sampled
        )

    boot_idx = np.concatenate(
        sampled_parts
    )

    bootstrap_indices.append(
        boot_idx
    )


# ============================================================
# BOOTSTRAP ALL MODEL METRICS
# ============================================================

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auc_ovr",
    "log_loss",
    "multiclass_brier",
    "auc_cn",
    "auc_mci",
    "auc_dementia",
]

boot_values = {
    model: {
        metric: []
        for metric in METRIC_NAMES
    }
    for model in MODELS
}


for b, idx in enumerate(
    bootstrap_indices,
    start=1
):

    y_b = y_true[
        idx
    ]

    for model, probs in model_probs.items():

        m = compute_metrics(
            y_b,
            probs[idx]
        )

        for metric in METRIC_NAMES:

            boot_values[
                model
            ][
                metric
            ].append(
                m[metric]
            )

    if b % 1000 == 0:

        print(
            f"Bootstrap {b}/{N_BOOT}"
        )


# ============================================================
# MODEL CONFIDENCE INTERVALS
# ============================================================

ci_rows = []

for model in MODELS:

    for metric in METRIC_NAMES:

        low, high = percentile_ci(
            boot_values[
                model
            ][
                metric
            ]
        )

        ci_rows.append({
            "model":
                model,

            "metric":
                metric,

            "estimate":
                point_metrics[
                    model
                ][
                    metric
                ],

            "ci95_low":
                low,

            "ci95_high":
                high,
        })


ci_df = pd.DataFrame(
    ci_rows
)

ci_df.to_csv(
    OUT_DIR
    / "test_bootstrap_ci.csv",
    index=False
)


# ============================================================
# PAIRED AUC DIFFERENCES
# ============================================================

comparison_rows = []


for (
    model_a,
    model_b,
    label
) in COMPARISONS:

    point_delta = (
        point_metrics[
            model_a
        ][
            "macro_auc_ovr"
        ]
        -
        point_metrics[
            model_b
        ][
            "macro_auc_ovr"
        ]
    )

    a = np.asarray(
        boot_values[
            model_a
        ][
            "macro_auc_ovr"
        ]
    )

    b = np.asarray(
        boot_values[
            model_b
        ][
            "macro_auc_ovr"
        ]
    )

    delta = (
        a - b
    )

    low, high = percentile_ci(
        delta
    )

    comparison_rows.append({
        "comparison":
            label,

        "model_a":
            model_a,

        "model_b":
            model_b,

        "delta_macro_auc":
            float(
                point_delta
            ),

        "ci95_low":
            low,

        "ci95_high":
            high,
    })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUT_DIR
    / "paired_macro_auc_comparisons.csv",
    index=False
)


# ============================================================
# PRIMARY MODEL SUMMARY
# ============================================================

primary = (
    "MRI + PET + Clinical"
)

primary_rows = (
    ci_df[
        ci_df[
            "model"
        ] == primary
    ]
    .copy()
)

primary_rows.to_csv(
    OUT_DIR
    / "primary_model_test_summary.csv",
    index=False
)


summary_json = {
    "test_n":
        int(len(df)),

    "test_class_counts":
        {
            k: int(v)
            for k, v
            in df[
                "diagnosis"
            ].value_counts().items()
        },

    "primary_model":
        primary,

    "primary_seed":
        20260921,

    "bootstrap_replicates":
        N_BOOT,

    "bootstrap_seed":
        BOOT_SEED,

    "bootstrap_method":
        "paired diagnosis-stratified subject bootstrap with replacement",

    "confidence_interval":
        "95% percentile bootstrap",

    "post_test_tuning":
        False,
}


with open(
    OUT_DIR
    / "final_test_statistics_metadata.json",
    "w"
) as f:

    json.dump(
        summary_json,
        f,
        indent=2
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n=== PRIMARY MODEL 95% CI ==="
)

print(
    primary_rows[
        [
            "metric",
            "estimate",
            "ci95_low",
            "ci95_high",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print(
    "\n=== PAIRED MACRO AUC COMPARISONS ==="
)

print(
    comparison_df[
        [
            "comparison",
            "delta_macro_auc",
            "ci95_low",
            "ci95_high",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print(
    "\nFINAL TEST STATUS: "
    "EVALUATED WITH LOCKED MODELS"
)

print(
    "POST-TEST MODEL TUNING: NO"
)

print(
    "\nSaved to:",
    OUT_DIR
)
