from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PATH = (
    ROOT / "data/interim/cohort/"
    "cognitive_temporal_audit_v1.csv"
)

df = pd.read_csv(
    PATH,
    low_memory=False
)

# Add diagnostic phase from the locked post-MRI-QC cohort.
cohort = pd.read_csv(
    ROOT / "data/interim/cohort/"
    "primary_post_mri_qc_manifest_v1.csv",
    low_memory=False
)

cohort["RID"] = pd.to_numeric(
    cohort["RID"],
    errors="raise"
).astype(int)

df["RID"] = pd.to_numeric(
    df["RID"],
    errors="raise"
).astype(int)

df = df.merge(
    cohort[
        [
            "RID",
            "dx_phase",
        ]
    ],
    how="left",
    on="RID",
    validate="one_to_one"
)

if df["dx_phase"].isna().any():
    raise RuntimeError(
        "Missing dx_phase after merge."
    )

variables = [
    "MMSE",
    "ADAS11",
    "ADAS13",
    "CDR_GLOBAL",
    "CDR_SB",
    "LOGMEM_IMMEDIATE",
    "LOGMEM_DELAYED",
    "RAVLT_T1_5_SUM",
    "RAVLT_30MIN",
    "RAVLT_RECOGNITION",
]

print("=" * 100)
print("COGNITIVE WINDOW / MISSINGNESS BIAS AUDIT")
print("=" * 100)

# ============================================================
# COVERAGE BY DIAGNOSIS
# ============================================================

for var in variables:

    gap = f"{var}_nearest_abs_gap_days"

    print("\n" + "=" * 100)
    print(var)
    print("=" * 100)

    rows = []

    for diagnosis, grp in df.groupby("diagnosis"):

        n = len(grp)

        available = grp[gap].notna()

        rows.append({
            "diagnosis": diagnosis,
            "N": n,
            "any": int(available.sum()),
            "within30": int(
                (grp[gap] <= 30).sum()
            ),
            "within60": int(
                (grp[gap] <= 60).sum()
            ),
            "within90": int(
                (grp[gap] <= 90).sum()
            ),
        })

    print(
        pd.DataFrame(rows)
        .to_string(index=False)
    )

# ============================================================
# COVERAGE BY ADNI DIAGNOSIS PHASE
# ============================================================

for var in [
    "MMSE",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "ADAS13",
    "RAVLT_T1_5_SUM",
]:

    gap = f"{var}_nearest_abs_gap_days"

    print("\n" + "=" * 100)
    print(f"{var} - BY DX PHASE")
    print("=" * 100)

    rows = []

    for phase, grp in df.groupby("dx_phase"):

        rows.append({
            "dx_phase": phase,
            "N": len(grp),
            "within30": int(
                (grp[gap] <= 30).sum()
            ),
            "within60": int(
                (grp[gap] <= 60).sum()
            ),
            "within90": int(
                (grp[gap] <= 90).sum()
            ),
            "median_gap": (
                float(
                    grp[gap].median()
                )
                if grp[gap].notna().any()
                else None
            ),
        })

    print(
        pd.DataFrame(rows)
        .to_string(index=False)
    )

# ============================================================
# PET VISIT -> NEAREST COGNITIVE VISIT PAIRS
# ============================================================

for var in [
    "MMSE",
    "CDR_SB",
    "LOGMEM_DELAYED",
    "ADAS13",
    "RAVLT_T1_5_SUM",
]:

    visit_col = f"{var}_nearest_visit"
    gap_col = f"{var}_nearest_gap_days"

    tmp = df[
        [
            "pet_visit",
            visit_col,
            gap_col,
        ]
    ].copy()

    tmp = tmp.dropna(
        subset=[visit_col]
    )

    pairs = (
        tmp.groupby(
            [
                "pet_visit",
                visit_col,
            ],
            dropna=False
        )
        .agg(
            n=(gap_col, "size"),
            median_signed_gap=(
                gap_col,
                "median"
            ),
            min_gap=(
                gap_col,
                "min"
            ),
            max_gap=(
                gap_col,
                "max"
            ),
        )
        .reset_index()
        .sort_values(
            "n",
            ascending=False
        )
    )

    print("\n" + "=" * 100)
    print(f"{var} - TOP PET/Cognitive VISIT PAIRS")
    print("=" * 100)

    print(
        pairs.head(20)
        .to_string(index=False)
    )

# ============================================================
# JOINT AVAILABILITY
# ============================================================

panels = {
    "diagnostic_core": [
        "MMSE",
        "ADAS13",
        "CDR_SB",
    ],

    "memory_panel": [
        "LOGMEM_DELAYED",
        "RAVLT_T1_5_SUM",
        "RAVLT_30MIN",
    ],

    "all_candidate_cognitive": [
        "MMSE",
        "ADAS11",
        "ADAS13",
        "CDR_SB",
        "LOGMEM_DELAYED",
        "RAVLT_T1_5_SUM",
        "RAVLT_30MIN",
        "RAVLT_RECOGNITION",
    ],
}

print("\n" + "=" * 100)
print("JOINT AVAILABILITY")
print("=" * 100)

for panel_name, vars_ in panels.items():

    print(f"\n{panel_name}")

    for window in [30, 60, 90]:

        mask = pd.Series(
            True,
            index=df.index
        )

        for var in vars_:
            mask &= (
                df[
                    f"{var}_nearest_abs_gap_days"
                ]
                <= window
            )

        print(
            f"  ±{window:2d} days: "
            f"{int(mask.sum())}/822 "
            f"({100*mask.mean():.2f}%)"
        )

# ============================================================
# LOGMEM IMMEDIATE CONFLICT
# ============================================================

print("\n" + "=" * 100)
print("LOGMEM IMMEDIATE CONFLICT")
print("=" * 100)

conflict = df[
    df[
        "LOGMEM_IMMEDIATE_nearest_status"
    ]
    .eq(
        "conflicting_equal_distance_scores"
    )
]

print("Count:", len(conflict))

if len(conflict):
    print(
        conflict[
            [
                "RID",
                "PTID",
                "diagnosis",
                "pet_visit",
                "pet_date",
                "LOGMEM_IMMEDIATE_nearest_abs_gap_days",
            ]
        ].to_string(index=False)
    )

# ============================================================
# RAVLT SUBJECT WITHOUT T1-5 SUM
# ============================================================

print("\n" + "=" * 100)
print("RAVLT T1-5 SUM MISSING")
print("=" * 100)

missing = df[
    df[
        "RAVLT_T1_5_SUM_nearest_score"
    ].isna()
]

print("Count:", len(missing))

if len(missing):
    print(
        missing[
            [
                "RID",
                "PTID",
                "diagnosis",
                "pet_visit",
                "pet_date",
            ]
        ].to_string(index=False)
    )
