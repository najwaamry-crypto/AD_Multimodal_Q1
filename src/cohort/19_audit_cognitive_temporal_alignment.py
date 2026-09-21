from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

COHORT = (
    ROOT / "data/interim/cohort/"
    "primary_post_mri_qc_manifest_v1.csv"
)

EXPORT = (
    ROOT / "data/interim/adnimerge2_exports"
)

OUT_SUBJECT = (
    ROOT / "data/interim/cohort/"
    "cognitive_temporal_audit_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables/"
    "cognitive_temporal_audit_v1_summary.json"
)

# ============================================================
# COHORT
# ============================================================

cohort = pd.read_csv(
    COHORT,
    low_memory=False
)

if len(cohort) != 822:
    raise RuntimeError(
        f"Expected 822 subjects, got {len(cohort)}"
    )

if cohort["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate RID in cohort."
    )

cohort["RID"] = pd.to_numeric(
    cohort["RID"],
    errors="raise"
).astype(int)

cohort["pet_date"] = pd.to_datetime(
    cohort["pet_date"],
    errors="raise"
)

cohort["pet_visit_norm"] = (
    cohort["pet_visit"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ============================================================
# TABLES
# ============================================================

mmse = pd.read_csv(
    EXPORT / "MMSE_cognitive_v1.csv",
    low_memory=False
)

cdr = pd.read_csv(
    EXPORT / "CDR_cognitive_v1.csv",
    low_memory=False
)

adas = pd.read_csv(
    EXPORT / "ADAS_cognitive_v1.csv",
    low_memory=False
)

neuro = pd.read_csv(
    EXPORT / "NEUROBAT_cognitive_v1.csv",
    low_memory=False
)

tables = {
    "MMSE": mmse,
    "CDR": cdr,
    "ADAS": adas,
    "NEUROBAT": neuro,
}

for name, df in tables.items():

    df["RID"] = pd.to_numeric(
        df["RID"],
        errors="coerce"
    )

    df["VISDATE"] = pd.to_datetime(
        df["VISDATE"],
        errors="coerce"
    )

    df["VISCODE2_norm"] = (
        df["VISCODE2"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df.drop(
        df[
            ~df["RID"].isin(
                cohort["RID"]
            )
        ].index,
        inplace=True
    )

    df["RID"] = df["RID"].astype(int)

# ============================================================
# NUMERIC SCORES
# ============================================================

for col in ["MMSCORE"]:
    mmse[col] = pd.to_numeric(
        mmse[col],
        errors="coerce"
    )

for col in ["CDGLOBAL", "CDRSB"]:
    cdr[col] = pd.to_numeric(
        cdr[col],
        errors="coerce"
    )

for col in ["TOTSCORE", "TOTAL13"]:
    adas[col] = pd.to_numeric(
        adas[col],
        errors="coerce"
    )

neuro_cols = [
    "LIMMTOTAL",
    "LDELTOTAL",
    "AVTOT1",
    "AVTOT2",
    "AVTOT3",
    "AVTOT4",
    "AVTOT5",
    "AVTOT6",
    "AVTOTB",
    "AVDEL30MIN",
    "AVDELTOT",
]

for col in neuro_cols:
    neuro[col] = pd.to_numeric(
        neuro[col],
        errors="coerce"
    )

# Provisional derived RAVLT candidate:
# total learning across trials 1-5.
# This is AUDIT ONLY and is not yet a locked modeling variable.

trial_cols = [
    "AVTOT1",
    "AVTOT2",
    "AVTOT3",
    "AVTOT4",
    "AVTOT5",
]

neuro["RAVLT_T1_5_SUM"] = (
    neuro[trial_cols]
    .sum(
        axis=1,
        min_count=5
    )
)

# ============================================================
# VARIABLE DEFINITIONS
# ============================================================

variables = {
    "MMSE": (
        mmse,
        "MMSCORE"
    ),

    "ADAS11": (
        adas,
        "TOTSCORE"
    ),

    "ADAS13": (
        adas,
        "TOTAL13"
    ),

    "CDR_GLOBAL": (
        cdr,
        "CDGLOBAL"
    ),

    "CDR_SB": (
        cdr,
        "CDRSB"
    ),

    "LOGMEM_IMMEDIATE": (
        neuro,
        "LIMMTOTAL"
    ),

    "LOGMEM_DELAYED": (
        neuro,
        "LDELTOTAL"
    ),

    "RAVLT_T1_5_SUM": (
        neuro,
        "RAVLT_T1_5_SUM"
    ),

    "RAVLT_TRIAL6": (
        neuro,
        "AVTOT6"
    ),

    "RAVLT_30MIN": (
        neuro,
        "AVDEL30MIN"
    ),

    "RAVLT_RECOGNITION": (
        neuro,
        "AVDELTOT"
    ),
}

# ============================================================
# AUDIT FUNCTION
# ============================================================

subject_output = cohort[
    [
        "RID",
        "PTID",
        "diagnosis",
        "pet_visit",
        "pet_date",
    ]
].copy()

summary = {}

print("=" * 100)
print("COGNITIVE TEMPORAL ALIGNMENT AUDIT")
print("=" * 100)

for var_name, (source, score_col) in variables.items():

    x = source[
        [
            "RID",
            "VISCODE2_norm",
            "VISDATE",
            score_col,
        ]
    ].copy()

    x = x[
        x[score_col].notna()
        &
        x["VISDATE"].notna()
    ].copy()

    # Remove exact duplicate records only.
    x = x.drop_duplicates(
        subset=[
            "RID",
            "VISCODE2_norm",
            "VISDATE",
            score_col,
        ]
    )

    x = x.merge(
        cohort[
            [
                "RID",
                "pet_visit_norm",
                "pet_date",
            ]
        ],
        how="inner",
        on="RID",
        validate="many_to_one"
    )

    x["gap_days"] = (
        x["VISDATE"]
        -
        x["pet_date"]
    ).dt.days

    x["abs_gap_days"] = (
        x["gap_days"]
        .abs()
    )

    x["same_visit"] = (
        x["VISCODE2_norm"]
        ==
        x["pet_visit_norm"]
    )

    # --------------------------------------------------------
    # Coverage using ANY available score
    # --------------------------------------------------------

    any_rids = set(
        x["RID"]
        .unique()
    )

    exact_date_rids = set(
        x.loc[
            x["abs_gap_days"].eq(0),
            "RID"
        ]
        .unique()
    )

    same_visit_rids = set(
        x.loc[
            x["same_visit"],
            "RID"
        ]
        .unique()
    )

    within30_rids = set(
        x.loc[
            x["abs_gap_days"].le(30),
            "RID"
        ]
        .unique()
    )

    within60_rids = set(
        x.loc[
            x["abs_gap_days"].le(60),
            "RID"
        ]
        .unique()
    )

    within90_rids = set(
        x.loc[
            x["abs_gap_days"].le(90),
            "RID"
        ]
        .unique()
    )

    # --------------------------------------------------------
    # Nearest score to PET per subject
    # --------------------------------------------------------

    nearest_records = []

    for rid, grp in x.groupby("RID"):

        min_gap = grp[
            "abs_gap_days"
        ].min()

        nearest = grp[
            grp["abs_gap_days"].eq(
                min_gap
            )
        ].copy()

        distinct_scores = (
            nearest[score_col]
            .dropna()
            .unique()
        )

        if len(distinct_scores) == 1:

            status = "unique_nearest_score"

            score = distinct_scores[0]

            # If duplicate rows carry the same score,
            # use the first only for descriptive metadata.
            row = nearest.iloc[0]

            nearest_date = row["VISDATE"]
            nearest_visit = row["VISCODE2_norm"]
            signed_gap = int(row["gap_days"])

        else:

            status = "conflicting_equal_distance_scores"

            score = pd.NA
            nearest_date = pd.NaT
            nearest_visit = pd.NA
            signed_gap = pd.NA

        nearest_records.append(
            {
                "RID": int(rid),
                f"{var_name}_nearest_score":
                    score,
                f"{var_name}_nearest_date":
                    nearest_date,
                f"{var_name}_nearest_visit":
                    nearest_visit,
                f"{var_name}_nearest_gap_days":
                    signed_gap,
                f"{var_name}_nearest_abs_gap_days":
                    int(min_gap),
                f"{var_name}_nearest_status":
                    status,
            }
        )

    nearest_df = pd.DataFrame(
        nearest_records
    )

    if len(nearest_df):

        subject_output = (
            subject_output.merge(
                nearest_df,
                how="left",
                on="RID",
                validate="one_to_one"
            )
        )

    conflicts = 0

    if len(nearest_df):
        conflicts = int(
            nearest_df[
                f"{var_name}_nearest_status"
            ]
            .eq(
                "conflicting_equal_distance_scores"
            )
            .sum()
        )

    nearest_gaps = (
        nearest_df[
            f"{var_name}_nearest_abs_gap_days"
        ]
        if len(nearest_df)
        else pd.Series(
            dtype=float
        )
    )

    info = {
        "any_score":
            len(any_rids),

        "same_pet_visit":
            len(same_visit_rids),

        "exact_pet_date":
            len(exact_date_rids),

        "within_30_days":
            len(within30_rids),

        "within_60_days":
            len(within60_rids),

        "within_90_days":
            len(within90_rids),

        "nearest_score_conflicts":
            conflicts,

        "nearest_abs_gap_median_days":
            (
                float(
                    nearest_gaps.median()
                )
                if len(nearest_gaps)
                else None
            ),

        "nearest_abs_gap_p90_days":
            (
                float(
                    nearest_gaps.quantile(0.90)
                )
                if len(nearest_gaps)
                else None
            ),
    }

    summary[var_name] = info

    print("\n" + "-" * 100)
    print(var_name)
    print("-" * 100)

    print(
        f"Any score:          {info['any_score']:3d}/822"
    )

    print(
        f"Same PET visit:     {info['same_pet_visit']:3d}/822"
    )

    print(
        f"Exact PET date:     {info['exact_pet_date']:3d}/822"
    )

    print(
        f"Within ±30 days:    {info['within_30_days']:3d}/822"
    )

    print(
        f"Within ±60 days:    {info['within_60_days']:3d}/822"
    )

    print(
        f"Within ±90 days:    {info['within_90_days']:3d}/822"
    )

    print(
        "Nearest conflicts: ",
        info["nearest_score_conflicts"]
    )

    print(
        "Nearest abs gap median:",
        info[
            "nearest_abs_gap_median_days"
        ]
    )

    print(
        "Nearest abs gap p90:",
        info[
            "nearest_abs_gap_p90_days"
        ]
    )

# ============================================================
# TABLE-LEVEL COVERAGE
# ============================================================

print("\n" + "=" * 100)
print("SOURCE TABLE COVERAGE AMONG 822 SUBJECTS")
print("=" * 100)

table_summary = {}

for name, df in tables.items():

    n_rows = len(df)
    n_rids = df["RID"].nunique()

    table_summary[name] = {
        "rows_for_target_subjects":
            int(n_rows),

        "unique_target_subjects":
            int(n_rids),
    }

    print(
        f"{name:10s}: "
        f"{n_rows:5d} rows, "
        f"{n_rids:3d}/822 subjects"
    )

# ============================================================
# SAVE
# ============================================================

full_summary = {
    "subjects": 822,
    "index_date": "PET acquisition date",
    "note": (
        "Audit only. No cognitive feature selection, "
        "time window, imputation rule, or subject exclusion "
        "is locked by this script."
    ),
    "tables": table_summary,
    "variables": summary,
}

subject_output.to_csv(
    OUT_SUBJECT,
    index=False
)

with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        full_summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(
    "Subject audit:",
    OUT_SUBJECT
)

print(
    "Summary:",
    OUT_SUMMARY
)

print(
    "\nNo subjects were excluded."
)
