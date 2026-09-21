from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

AUDIT = (
    ROOT / "data/interim/cohort/"
    "cognitive_temporal_audit_v1.csv"
)

OUT = (
    ROOT / "data/interim/cohort/"
    "cognitive_manifest_v1.csv"
)

SUMMARY = (
    ROOT / "tables/"
    "cognitive_manifest_v1_summary.json"
)

WINDOW = 60

df = pd.read_csv(
    AUDIT,
    low_memory=False
)

if len(df) != 822:
    raise RuntimeError(
        f"Expected 822 subjects, got {len(df)}"
    )

if df["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate RID in cognitive audit."
    )

# ------------------------------------------------------------
# Variables retained in the final subject-level manifest
# ------------------------------------------------------------

variables = [
    "MMSE",
    "ADAS11",
    "ADAS13",
    "CDR_GLOBAL",
    "CDR_SB",
    "LOGMEM_IMMEDIATE",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_TRIAL6",
    "RAVLT_30MIN",
    "RAVLT_RECOGNITION",
]

base_cols = [
    "RID",
    "PTID",
    "diagnosis",
    "pet_visit",
    "pet_date",
]

out = df[base_cols].copy()

summary = {
    "subjects": 822,
    "index_date": "PET acquisition date",
    "window_days": WINDOW,
    "selection_rule": (
        "Nearest available cognitive score within ±60 days "
        "of PET. Equal-distance conflicting scores are treated "
        "as missing."
    ),
    "subject_exclusion": "none",
    "variables": {},
}

for var in variables:

    score_col = f"{var}_nearest_score"
    date_col = f"{var}_nearest_date"
    visit_col = f"{var}_nearest_visit"
    gap_col = f"{var}_nearest_gap_days"
    abs_gap_col = f"{var}_nearest_abs_gap_days"
    status_col = f"{var}_nearest_status"

    required = [
        score_col,
        date_col,
        visit_col,
        gap_col,
        abs_gap_col,
        status_col,
    ]

    missing_cols = [
        c for c in required
        if c not in df.columns
    ]

    if missing_cols:
        raise RuntimeError(
            f"{var}: missing columns {missing_cols}"
        )

    valid = (
        pd.to_numeric(
            df[abs_gap_col],
            errors="coerce"
        ).le(WINDOW)
        &
        df[score_col].notna()
        &
        ~df[status_col].eq(
            "conflicting_equal_distance_scores"
        )
    )

    out[var] = df[score_col].where(valid)
    out[f"{var}_date"] = df[date_col].where(valid)
    out[f"{var}_visit"] = df[visit_col].where(valid)
    out[f"{var}_gap_days"] = df[gap_col].where(valid)

    summary["variables"][var] = {
        "available_within_60_days":
            int(valid.sum()),

        "missing_after_window":
            int((~valid).sum()),

        "coverage_percent":
            round(
                100 * valid.mean(),
                2
            ),
    }

# ------------------------------------------------------------
# Pre-specified secondary cognitive panel
# ------------------------------------------------------------

secondary_panel = [
    "MMSE",
    "ADAS13",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_30MIN",
]

complete_secondary = (
    out[secondary_panel]
    .notna()
    .all(axis=1)
)

summary["secondary_cognitive_panel"] = {
    "variables": secondary_panel,
    "complete_cases": int(
        complete_secondary.sum()
    ),
    "complete_case_percent": round(
        100 * complete_secondary.mean(),
        2
    ),
}

out["secondary_cognitive_complete"] = (
    complete_secondary.astype(int)
)

# ------------------------------------------------------------
# Descriptive-only / auxiliary variables
# ------------------------------------------------------------

summary["auxiliary_cognitive_variables"] = [
    "ADAS11",
    "CDR_GLOBAL",
    "LOGMEM_IMMEDIATE",
    "RAVLT_TRIAL6",
    "RAVLT_RECOGNITION",
]

summary["primary_non_cognitive_block"] = [
    "age_at_index",
    "sex",
    "education_years",
    "apoe4_count",
]

summary["cognitive_role"] = (
    "Secondary/ablation analysis only. Cognitive scores are "
    "not part of the primary non-cognitive clinical block."
)

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

if len(out) != 822:
    raise RuntimeError(
        "Final cognitive manifest does not contain 822 rows."
    )

if out["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate RID in final cognitive manifest."
    )

# Known audit cases
rid292 = out.loc[
    out["RID"].eq(292),
    "LOGMEM_IMMEDIATE"
]

if len(rid292) == 1 and rid292.notna().any():
    raise RuntimeError(
        "RID 292 LOGMEM_IMMEDIATE should be missing "
        "under ±60-day rule."
    )

rid4001 = out.loc[
    out["RID"].eq(4001),
    "RAVLT_T1_5_SUM"
]

if len(rid4001) == 1 and rid4001.notna().any():
    raise RuntimeError(
        "RID 4001 RAVLT_T1_5_SUM should remain missing."
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out.to_csv(
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

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("=" * 92)
print("FINAL COGNITIVE MANIFEST - ±60 DAYS")
print("=" * 92)

print("\nSubjects:", len(out))

print("\nCoverage:")
for var in variables:
    n = int(
        out[var]
        .notna()
        .sum()
    )

    print(
        f"{var:22s}: "
        f"{n:3d}/822 "
        f"({100*n/822:.2f}%)"
    )

print("\nSecondary cognitive panel:")
print(
    " + ".join(
        secondary_panel
    )
)

print(
    "Complete cases:",
    int(
        complete_secondary.sum()
    ),
    "/822",
    f"({100*complete_secondary.mean():.2f}%)"
)

print("\nKnown special cases:")

print(
    "RID 292 LOGMEM_IMMEDIATE:",
    out.loc[
        out["RID"].eq(292),
        "LOGMEM_IMMEDIATE"
    ].tolist()
)

print(
    "RID 4001 RAVLT_T1_5_SUM:",
    out.loc[
        out["RID"].eq(4001),
        "RAVLT_T1_5_SUM"
    ].tolist()
)

print("\nSaved:")
print(OUT)
print(SUMMARY)

print(
    "\nNo subjects were excluded."
)
