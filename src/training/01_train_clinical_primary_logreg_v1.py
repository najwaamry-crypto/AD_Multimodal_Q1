#!/usr/bin/env python3

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


ROOT = Path("/home/najwa-amry/AD_Multimodal_Q1")

MANIFEST = (
    ROOT
    / "data/processed/master_modelling_manifest_v1.csv"
)

RESULTS_DIR = (
    ROOT
    / "results/baselines/clinical_primary_logreg_v1"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    MANIFEST,
    low_memory=False
)

features = [
    "age_at_index",
    "sex",
    "education_years",
    "apoe4_count",
]

target = "diagnosis"

classes = [
    "CN",
    "MCI",
    "Dementia",
]


train = df[
    df["split"] == "train"
].copy()

val = df[
    df["split"] == "val"
].copy()

# IMPORTANT:
# test split is intentionally not loaded/evaluated here.


X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]


print("=== DATA ===")
print("Train:", len(train))
print("Validation:", len(val))
print("Features:", features)

print("\nTrain diagnosis:")
print(
    y_train.value_counts().to_string()
)

print("\nValidation diagnosis:")
print(
    y_val.value_counts().to_string()
)


# --------------------------------------------------
# Leakage-safe preprocessing
# --------------------------------------------------

numeric_features = [
    "age_at_index",
    "education_years",
    "apoe4_count",
]

categorical_features = [
    "sex",
]


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
])


categorical_pipeline = Pipeline([
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            drop=None,
        )
    ),
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features,
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features,
    ),
])


# --------------------------------------------------
# Validation-only model selection
# --------------------------------------------------

C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

results = []

best_score = -np.inf
best_model = None
best_result = None


for C in C_VALUES:

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=20260921,
            )
        ),
    ])

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_val
    )

    prob = model.predict_proba(
        X_val
    )

    model_classes = (
        model
        .named_steps["classifier"]
        .classes_
    )

    acc = accuracy_score(
        y_val,
        pred
    )

    bal_acc = balanced_accuracy_score(
        y_val,
        pred
    )

    macro_f1 = f1_score(
        y_val,
        pred,
        average="macro"
    )

    macro_auc = roc_auc_score(
        y_val,
        prob,
        labels=model_classes,
        multi_class="ovr",
        average="macro",
    )

    ll = log_loss(
        y_val,
        prob,
        labels=model_classes,
    )

    row = {
        "C": C,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "macro_auc_ovr": float(macro_auc),
        "log_loss": float(ll),
    }

    results.append(row)

    print(
        f"C={C:<6} | "
        f"Acc={acc:.4f} | "
        f"BalAcc={bal_acc:.4f} | "
        f"MacroF1={macro_f1:.4f} | "
        f"MacroAUC={macro_auc:.4f} | "
        f"LogLoss={ll:.4f}"
    )

    # Primary selection metric:
    # macro one-vs-rest ROC-AUC.
    # Macro F1 used as first tie-break.
    score = (
        macro_auc,
        macro_f1,
    )

    if (
        best_result is None
        or score >
        (
            best_result["macro_auc_ovr"],
            best_result["macro_f1"],
        )
    ):
        best_result = row
        best_model = model


# --------------------------------------------------
# Save validation results
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_DIR
    / "validation_grid_results.csv",
    index=False
)

joblib.dump(
    best_model,
    RESULTS_DIR
    / "best_validation_model.joblib"
)

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


# --------------------------------------------------
# Inspect train-fitted preprocessing parameters
# --------------------------------------------------

num_pipe = (
    best_model
    .named_steps["preprocessor"]
    .named_transformers_["numeric"]
)

imputer = (
    num_pipe
    .named_steps["imputer"]
)

scaler = (
    num_pipe
    .named_steps["scaler"]
)


print("\n=== BEST VALIDATION MODEL ===")

for k, v in best_result.items():
    print(f"{k}: {v}")

print("\n=== TRAIN-FITTED PREPROCESSING ===")

print(
    "Numeric features:",
    numeric_features
)

print(
    "Training medians:",
    dict(
        zip(
            numeric_features,
            imputer.statistics_
        )
    )
)

print(
    "Training means after imputation:",
    dict(
        zip(
            numeric_features,
            scaler.mean_
        )
    )
)

print(
    "Training scales:",
    dict(
        zip(
            numeric_features,
            scaler.scale_
        )
    )
)

print(
    "\nTEST SET STATUS: NOT ACCESSED"
)

print(
    "\nSaved to:",
    RESULTS_DIR
)
