#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = (
    ROOT
    / "data/processed/master_modelling_manifest_v1.csv"
)

RESULTS_DIR = (
    ROOT
    / "results/baselines/cognitive_logreg_v1"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 20260921

COGNITIVE_COLS = [
    "MMSE",
    "ADAS13",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_30MIN",
]

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "Dementia": 2,
}

C_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


def evaluate(
    model,
    x,
    y
):

    prob = model.predict_proba(x)
    pred = prob.argmax(axis=1)

    y_bin = label_binarize(
        y,
        classes=[0, 1, 2]
    )

    return {
        "accuracy":
            accuracy_score(
                y,
                pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred
            ),

        "macro_f1":
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0
            ),

        "macro_auc_ovr":
            roc_auc_score(
                y_bin,
                prob,
                average="macro",
                multi_class="ovr"
            ),

        "log_loss":
            log_loss(
                y,
                prob,
                labels=[0, 1, 2]
            ),
    }


df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

print(
    "=== COGNITIVE LOGISTIC REGRESSION V1 ==="
)

print(
    "Cognitive variables:"
)

for col in COGNITIVE_COLS:
    print(
        "-",
        col
    )


# ============================================================
# COMPLETE-CASE SECONDARY COHORT
# ============================================================

complete_mask = (
    df[COGNITIVE_COLS]
    .notna()
    .all(axis=1)
)

complete = df[
    complete_mask
].copy()

print(
    "\nComplete cognitive cases:",
    len(complete),
    "/",
    len(df)
)

print(
    "\nComplete cases by split:"
)

print(
    complete[
        "split"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nComplete cases by diagnosis:"
)

print(
    complete[
        "diagnosis"
    ]
    .value_counts()
    .to_string()
)


train_df = complete[
    complete["split"] == "train"
].copy()

val_df = complete[
    complete["split"] == "val"
].copy()

# Test deliberately not loaded/evaluated.


x_train = train_df[
    COGNITIVE_COLS
].copy()

x_val = val_df[
    COGNITIVE_COLS
].copy()

y_train = (
    train_df["diagnosis"]
    .map(LABEL_MAP)
    .to_numpy()
)

y_val = (
    val_df["diagnosis"]
    .map(LABEL_MAP)
    .to_numpy()
)


print(
    "\nTrain complete cases:",
    len(train_df)
)

print(
    "Validation complete cases:",
    len(val_df)
)

print(
    "\nTrain classes:"
)

print(
    train_df[
        "diagnosis"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nValidation classes:"
)

print(
    val_df[
        "diagnosis"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# GRID SEARCH ON VALIDATION ONLY
# ============================================================

results = []

best_auc = -np.inf
best_f1 = -np.inf
best_model = None
best_c = None
best_metrics = None


for C in C_GRID:

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                )
            ),
        ]
    )

    model.fit(
        x_train,
        y_train
    )

    metrics = evaluate(
        model,
        x_val,
        y_val
    )

    row = {
        "C":
            C,
        **metrics,
    }

    results.append(
        row
    )

    print(
        f"C={C:<6} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"BalAcc={metrics['balanced_accuracy']:.4f} | "
        f"F1={metrics['macro_f1']:.4f} | "
        f"AUC={metrics['macro_auc_ovr']:.4f} | "
        f"LogLoss={metrics['log_loss']:.4f}"
    )

    improved = (
        metrics[
            "macro_auc_ovr"
        ] > best_auc
        or (
            np.isclose(
                metrics[
                    "macro_auc_ovr"
                ],
                best_auc
            )
            and
            metrics[
                "macro_f1"
            ] > best_f1
        )
    )

    if improved:

        best_auc = (
            metrics[
                "macro_auc_ovr"
            ]
        )

        best_f1 = (
            metrics[
                "macro_f1"
            ]
        )

        best_model = model
        best_c = C
        best_metrics = metrics


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_DIR
    / "validation_grid_results.csv",
    index=False
)

summary = {
    "analysis_type":
        "secondary cognitive benchmark",

    "cognitive_variables":
        COGNITIVE_COLS,

    "complete_case_policy":
        True,

    "n_complete_total":
        int(len(complete)),

    "n_train_complete":
        int(len(train_df)),

    "n_val_complete":
        int(len(val_df)),

    "best_C":
        float(best_c),

    **{
        k: float(v)
        for k, v
        in best_metrics.items()
    },

    "test_accessed":
        False,
}

with open(
    RESULTS_DIR
    / "best_validation_result.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print(
    "\n=== BEST COGNITIVE VALIDATION RESULT ==="
)

print(
    "Best C:",
    best_c
)

for k, v in best_metrics.items():

    print(
        f"{k}: {v}"
    )

print(
    "\nComplete-case population only."
)

print(
    "SECONDARY BENCHMARK: YES"
)

print(
    "TEST SET STATUS: NOT ACCESSED"
)

print(
    "\nSaved to:",
    RESULTS_DIR
)
