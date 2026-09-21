from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PAIRS_PATH = (
    ROOT / "data/interim/cohort"
    / "mri_fdg_candidate_pairs_21Sep2026.csv"
)

DX_PATH = (
    ROOT / "data/raw/metadata/extracted"
    / "DXSUM_18Sep2026.csv"
)

OUT_JSON = (
    ROOT / "tables"
    / "candidate_definition_audit_21Sep2026.json"
)

pairs = pd.read_csv(PAIRS_PATH, low_memory=False)
dx = pd.read_csv(DX_PATH, low_memory=False)

pairs["RID"] = pd.to_numeric(
    pairs["RID"], errors="coerce"
).astype("Int64")

pairs["pet_date"] = pd.to_datetime(
    pairs["pet_date"], errors="coerce"
)

pairs["nearest_mri_date"] = pd.to_datetime(
    pairs["nearest_mri_date"], errors="coerce"
)

pairs["mri_pet_abs_days"] = pd.to_numeric(
    pairs["mri_pet_abs_days"], errors="coerce"
)

pairs["same_visit_mri_available"] = (
    pairs["same_visit_mri_available"]
    .astype(str)
    .str.lower()
    .eq("true")
)

dx["RID"] = pd.to_numeric(
    dx["RID"], errors="coerce"
).astype("Int64")

dx["DIAGNOSIS"] = pd.to_numeric(
    dx["DIAGNOSIS"], errors="coerce"
).astype("Int64")

dx["EXAMDATE"] = pd.to_datetime(
    dx["EXAMDATE"], errors="coerce"
)

dx = dx[
    dx["DIAGNOSIS"].isin([1, 2, 3])
].copy()

dx = dx.dropna(
    subset=["RID", "VISCODE2", "EXAMDATE"]
)

# ---------------------------------------------------
# Build diagnosis lookup
# ---------------------------------------------------

dx_groups = {
    (int(rid), str(vis)): grp.copy()
    for (rid, vis), grp in
    dx.groupby(["RID", "VISCODE2"], dropna=False)
}

records = []
ambiguous_nearest_dx = 0

for row in pairs.itertuples(index=False):

    key = (
        int(row.RID),
        str(row.pet_visit)
    )

    grp = dx_groups.get(key)

    rec = {
        "dx_date_nearest": pd.NaT,
        "dx_gap_days": np.nan,
        "diagnosis": pd.NA,
        "dx_phase": pd.NA,
    }

    if grp is not None and len(grp):

        grp = grp.copy()

        grp["gap"] = (
            grp["EXAMDATE"] - row.pet_date
        ).dt.days.abs()

        min_gap = grp["gap"].min()

        nearest = grp[
            grp["gap"] == min_gap
        ].copy()

        classes = nearest[
            "DIAGNOSIS"
        ].dropna().unique()

        if len(classes) > 1:
            ambiguous_nearest_dx += 1
        else:
            chosen = (
                nearest
                .sort_values(
                    ["EXAMDATE", "PHASE"]
                )
                .iloc[0]
            )

            rec = {
                "dx_date_nearest":
                    chosen["EXAMDATE"],
                "dx_gap_days":
                    int(chosen["gap"]),
                "diagnosis":
                    int(chosen["DIAGNOSIS"]),
                "dx_phase":
                    chosen["PHASE"],
            }

    records.append(rec)

dx_match = pd.DataFrame(records)

pairs = pd.concat(
    [
        pairs.reset_index(drop=True),
        dx_match.reset_index(drop=True)
    ],
    axis=1
)

pairs["has_valid_dx"] = (
    pairs["diagnosis"].isin([1, 2, 3])
)

CLASS_NAMES = {
    1: "CN",
    2: "MCI",
    3: "Dementia",
}

# ---------------------------------------------------
# Summary helper
# ---------------------------------------------------

def summarize(df):

    if len(df) == 0:
        return {
            "subjects": 0,
            "class_counts": {},
            "phase_counts": {},
        }

    classes = (
        df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    class_counts = {
        CLASS_NAMES.get(int(k), str(k)): int(v)
        for k, v in classes.items()
    }

    phases = (
        df["dx_phase"]
        .value_counts(dropna=False)
    )

    phase_counts = {
        str(k): int(v)
        for k, v in phases.items()
    }

    return {
        "subjects": int(df["RID"].nunique()),
        "class_counts": class_counts,
        "phase_counts": phase_counts,

        "mri_pet_gap_median": float(
            df["mri_pet_abs_days"].median()
        ),

        "mri_pet_gap_p90": float(
            df["mri_pet_abs_days"].quantile(0.90)
        ),

        "dx_pet_gap_median": float(
            df["dx_gap_days"].median()
        ),

        "dx_pet_gap_p90": float(
            df["dx_gap_days"].quantile(0.90)
        ),

        "same_visit_mri_subjects": int(
            df.loc[
                df["same_visit_mri_available"],
                "RID"
            ].nunique()
        ),
    }


# ---------------------------------------------------
# One-observation-per-RID strategies
# ---------------------------------------------------

def earliest_eligible(df):

    return (
        df.sort_values(
            [
                "RID",
                "pet_date",
                "mri_pet_abs_days",
                "pet_image_id",
            ]
        )
        .drop_duplicates("RID", keep="first")
    )


def closest_pair(df):

    return (
        df.sort_values(
            [
                "RID",
                "mri_pet_abs_days",
                "pet_date",
                "pet_image_id",
            ]
        )
        .drop_duplicates("RID", keep="first")
    )


thresholds = [14, 30, 60, 90]

results = {}

for days in thresholds:

    eligible = pairs[
        (pairs["mri_pet_abs_days"] <= days)
        & pairs["has_valid_dx"]
    ].copy()

    earliest = earliest_eligible(eligible)
    closest = closest_pair(eligible)

    baseline = eligible[
        eligible["pet_visit"].astype(str)
        .str.lower()
        .eq("bl")
    ].copy()

    baseline = earliest_eligible(baseline)

    results[str(days)] = {
        "eligible_pet_exams": int(len(eligible)),
        "eligible_subjects": int(
            eligible["RID"].nunique()
        ),

        "earliest_eligible":
            summarize(earliest),

        "baseline_only":
            summarize(baseline),

        "closest_pair_sensitivity":
            summarize(closest),
    }


# ---------------------------------------------------
# Same-visit MRI cohort
# ---------------------------------------------------

same_visit = pairs[
    pairs["same_visit_mri_available"]
    & pairs["has_valid_dx"]
].copy()

same_visit_index = earliest_eligible(
    same_visit
)

# ---------------------------------------------------
# Overall diagnosis-date audit
# ---------------------------------------------------

valid_dx_pairs = pairs[
    pairs["has_valid_dx"]
]

dx_gap_summary = {
    "median": float(
        valid_dx_pairs["dx_gap_days"].median()
    ),
    "p75": float(
        valid_dx_pairs["dx_gap_days"].quantile(.75)
    ),
    "p90": float(
        valid_dx_pairs["dx_gap_days"].quantile(.90)
    ),
    "p95": float(
        valid_dx_pairs["dx_gap_days"].quantile(.95)
    ),
    "max": int(
        valid_dx_pairs["dx_gap_days"].max()
    ),
}

summary = {
    "pet_mri_pairs_input": int(len(pairs)),

    "pairs_with_nearest_same_visit_valid_dx":
        int(pairs["has_valid_dx"].sum()),

    "subjects_with_nearest_same_visit_valid_dx":
        int(
            pairs.loc[
                pairs["has_valid_dx"],
                "RID"
            ].nunique()
        ),

    "ambiguous_nearest_diagnosis_ties":
        int(ambiguous_nearest_dx),

    "diagnosis_pet_gap_days":
        dx_gap_summary,

    "same_visit_mri_index_cohort":
        summarize(same_visit_index),

    "threshold_analysis":
        results,
}


# ---------------------------------------------------
# Console
# ---------------------------------------------------

print("\n" + "=" * 76)
print("CANDIDATE COHORT DEFINITION AUDIT")
print("=" * 76)

print(
    "Input PET exams with an MRI available:",
    len(pairs)
)

print(
    "PET exams with nearest same-visit valid diagnosis:",
    summary[
        "pairs_with_nearest_same_visit_valid_dx"
    ]
)

print(
    "Subjects with valid diagnosis:",
    summary[
        "subjects_with_nearest_same_visit_valid_dx"
    ]
)

print(
    "Ambiguous nearest-diagnosis ties:",
    ambiguous_nearest_dx
)

print("\nPET ↔ diagnosis date gap:")
for k, v in dx_gap_summary.items():
    print(f"  {k}: {v}")

print("\nSAME-VISIT MRI STRATEGY")
sv = summary["same_visit_mri_index_cohort"]
print("Subjects:", sv["subjects"])
print("Classes:", sv["class_counts"])
print("Phases:", sv["phase_counts"])

for days in thresholds:

    r = results[str(days)]

    print("\n" + "-" * 76)
    print(f"MRI ↔ PET <= {days} DAYS")
    print("-" * 76)

    print(
        "Eligible subjects:",
        r["eligible_subjects"]
    )

    e = r["earliest_eligible"]

    print("\nEarliest eligible observation:")
    print("  Subjects:", e["subjects"])
    print("  Classes:", e["class_counts"])
    print("  Phases:", e["phase_counts"])
    print(
        "  Median MRI-PET gap:",
        e["mri_pet_gap_median"]
    )
    print(
        "  P90 MRI-PET gap:",
        e["mri_pet_gap_p90"]
    )
    print(
        "  Median PET-DX gap:",
        e["dx_pet_gap_median"]
    )

    b = r["baseline_only"]

    print("\nBaseline-only:")
    print("  Subjects:", b["subjects"])
    print("  Classes:", b["class_counts"])
    print("  Phases:", b["phase_counts"])

    c = r["closest_pair_sensitivity"]

    print(
        "\nClosest-pair sensitivity subjects:",
        c["subjects"]
    )
    print(
        "Closest-pair classes:",
        c["class_counts"]
    )


# ---------------------------------------------------
# Save aggregate-only output
# ---------------------------------------------------

with open(
    OUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    "\nSaved aggregate candidate audit to:\n"
    f"{OUT_JSON}"
)
