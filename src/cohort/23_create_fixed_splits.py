from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data/interim/cohort/"
    "analytic_manifest_v1.csv"
)

SPLIT_DIR = ROOT / "splits"
SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT = (
    SPLIT_DIR /
    "split_assignments_v1.csv"
)

SUMMARY = (
    ROOT / "tables/"
    "split_assignments_v1_summary.json"
)

SEED = 20260921

# Fixed absolute sizes:
# 822 total
# train = 576
# val   = 82
# test  = 164

N_TOTAL = 822
N_TEST = 164
N_VAL = 82
N_TRAIN = 576

df = pd.read_csv(
    INPUT,
    low_memory=False
)

df["RID"] = pd.to_numeric(
    df["RID"],
    errors="raise"
).astype(int)

# ============================================================
# SANITY CHECKS
# ============================================================

if len(df) != N_TOTAL:
    raise RuntimeError(
        f"Expected {N_TOTAL} subjects, got {len(df)}"
    )

if df["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate RID detected."
    )

if df["diagnosis"].isna().any():
    raise RuntimeError(
        "Missing diagnosis detected."
    )

# ============================================================
# TEST SPLIT
# ============================================================

trainval, test = train_test_split(
    df,
    test_size=N_TEST,
    random_state=SEED,
    stratify=df["diagnosis"],
)

# ============================================================
# VALIDATION SPLIT
# ============================================================

train, val = train_test_split(
    trainval,
    test_size=N_VAL,
    random_state=SEED,
    stratify=trainval["diagnosis"],
)

if len(train) != N_TRAIN:
    raise RuntimeError(
        f"Expected {N_TRAIN} train subjects, got {len(train)}"
    )

if len(val) != N_VAL:
    raise RuntimeError(
        f"Expected {N_VAL} validation subjects, got {len(val)}"
    )

if len(test) != N_TEST:
    raise RuntimeError(
        f"Expected {N_TEST} test subjects, got {len(test)}"
    )

# ============================================================
# ASSIGNMENTS
# ============================================================

assignments = pd.concat(
    [
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ],
    ignore_index=True
)

if len(assignments) != N_TOTAL:
    raise RuntimeError(
        "Split assignment lost subjects."
    )

if assignments["RID"].duplicated().any():
    raise RuntimeError(
        "A RID appears in more than one split."
    )

if set(assignments["RID"]) != set(df["RID"]):
    raise RuntimeError(
        "RID set differs after splitting."
    )

# Keep only identifiers + audit metadata.
split_manifest = assignments[
    [
        "RID",
        "PTID",
        "diagnosis",
        "dx_phase",
        "sex",
        "secondary_cognitive_complete",
        "split",
    ]
].copy()

order = {
    "train": 0,
    "val": 1,
    "test": 2,
}

split_manifest["_order"] = (
    split_manifest["split"]
    .map(order)
)

split_manifest = (
    split_manifest
    .sort_values(
        [
            "_order",
            "RID",
        ]
    )
    .drop(
        columns="_order"
    )
    .reset_index(
        drop=True
    )
)

# ============================================================
# SUMMARY HELPERS
# ============================================================

def nested_counts(frame, row_col, col_col):

    tab = pd.crosstab(
        frame[row_col],
        frame[col_col]
    )

    return {
        str(idx): {
            str(col): int(tab.loc[idx, col])
            for col in tab.columns
        }
        for idx in tab.index
    }


summary = {
    "version": "split_assignments_v1",
    "seed": SEED,
    "strategy": (
        "Subject-level fixed 70/10/20 split, "
        "stratified on diagnosis only."
    ),
    "sizes": {
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
    },
    "diagnosis_by_split": nested_counts(
        split_manifest,
        "diagnosis",
        "split"
    ),
    "phase_by_split": nested_counts(
        split_manifest,
        "dx_phase",
        "split"
    ),
    "sex_by_split": nested_counts(
        split_manifest,
        "sex",
        "split"
    ),
    "cognitive_complete_by_split": nested_counts(
        split_manifest,
        "secondary_cognitive_complete",
        "split"
    ),
    "notes": [
        (
            "Diagnosis was the only stratification variable."
        ),
        (
            "ADNI phase, sex, and cognitive completeness "
            "were audited but did not influence assignment."
        ),
        (
            "No model performance or preprocessing result "
            "was used to select the seed or assignments."
        ),
    ],
}

# ============================================================
# SAVE
# ============================================================

split_manifest.to_csv(
    OUT,
    index=False
)

with open(
    SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

# ============================================================
# REPORT
# ============================================================

print("=" * 96)
print("FIXED TRAIN / VALIDATION / TEST SPLIT V1")
print("=" * 96)

print("\nSeed:", SEED)

print("\nSplit sizes:")
print(
    split_manifest["split"]
    .value_counts()
    .reindex(
        [
            "train",
            "val",
            "test",
        ]
    )
    .to_string()
)

print("\n" + "=" * 96)
print("DIAGNOSIS x SPLIT")
print("=" * 96)

print(
    pd.crosstab(
        split_manifest["diagnosis"],
        split_manifest["split"]
    )[
        [
            "train",
            "val",
            "test",
        ]
    ]
    .to_string()
)

print("\n" + "=" * 96)
print("PHASE x SPLIT")
print("=" * 96)

print(
    pd.crosstab(
        split_manifest["dx_phase"],
        split_manifest["split"]
    )[
        [
            "train",
            "val",
            "test",
        ]
    ]
    .to_string()
)

print("\n" + "=" * 96)
print("SEX x SPLIT")
print("=" * 96)

print(
    pd.crosstab(
        split_manifest["sex"],
        split_manifest["split"]
    )[
        [
            "train",
            "val",
            "test",
        ]
    ]
    .to_string()
)

print("\n" + "=" * 96)
print("COGNITIVE COMPLETE x SPLIT")
print("=" * 96)

print(
    pd.crosstab(
        split_manifest[
            "secondary_cognitive_complete"
        ],
        split_manifest["split"]
    )[
        [
            "train",
            "val",
            "test",
        ]
    ]
    .to_string()
)

print("\nSaved:")
print(OUT)
print(SUMMARY)

print(
    "\nNo preprocessing or model training performed."
)
