from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT / "data/interim/cohort"
    / "primary_candidate_manifest_v1.csv"
)

MRI_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/extracted"
    / "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

RANK_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/qc/extracted"
    / "MRIMPRANK_21Sep2026.csv"
)

LOCAL_OUT = (
    ROOT / "data/interim/cohort"
    / "adni1_mri_rank_audit_v1.csv"
)

SUMMARY_OUT = (
    ROOT / "tables"
    / "adni1_mri_rank_audit_v1_summary.json"
)


# ============================================================
# LOAD
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False
)

mri = pd.read_csv(
    MRI_PATH,
    low_memory=False
)

rank = pd.read_csv(
    RANK_PATH,
    low_memory=False
)

manifest["mri_date"] = pd.to_datetime(
    manifest["mri_date"],
    errors="coerce"
)

mri["image_date"] = pd.to_datetime(
    mri["image_date"],
    errors="coerce"
)

rank["EXAMDATE"] = pd.to_datetime(
    rank["EXAMDATE"],
    errors="coerce"
)

manifest["RID"] = pd.to_numeric(
    manifest["RID"],
    errors="coerce"
).astype("Int64")

rank["RID"] = pd.to_numeric(
    rank["RID"],
    errors="coerce"
).astype("Int64")

rank["RANK"] = pd.to_numeric(
    rank["RANK"],
    errors="coerce"
)


# ============================================================
# NORMALIZED VISIT KEYS
# ============================================================

manifest["mri_visit_key"] = (
    manifest["mri_visit"]
    .astype(str)
    .str.strip()
)

mri["mri_visit_key"] = (
    mri["image_visit"]
    .astype(str)
    .str.strip()
)


# ============================================================
# EXPAND MANIFEST TO ACTUAL MRI SERIES
# ============================================================

candidate_series = manifest[
    [
        "RID",
        "PTID",
        "diagnosis",
        "dx_phase",
        "mri_visit",
        "mri_visit_key",
        "mri_date",
        "mri_candidate_series_count",
    ]
].merge(
    mri,
    how="left",
    left_on=[
        "PTID",
        "mri_visit_key",
        "mri_date",
    ],
    right_on=[
        "subject_id",
        "mri_visit_key",
        "image_date",
    ],
    validate="one_to_many"
)


# ============================================================
# HARD CHECK:
# expanded series count must reproduce manifest count
# ============================================================

actual_counts = (
    candidate_series
    .groupby("RID")["image_id"]
    .nunique()
)

expected_counts = (
    manifest
    .set_index("RID")[
        "mri_candidate_series_count"
    ]
)

comparison = pd.concat(
    [
        expected_counts.rename("expected"),
        actual_counts.rename("actual"),
    ],
    axis=1
)

bad_counts = comparison[
    comparison["expected"]
    != comparison["actual"]
]

if len(bad_counts):
    print(bad_counts.head(20))
    raise RuntimeError(
        f"{len(bad_counts)} RIDs have inconsistent "
        "MRI candidate counts."
    )


# ============================================================
# MRI PHASE INVENTORY IN THE 882-SUBJECT COHORT
# ============================================================

print("\n" + "=" * 76)
print("MRI PHASES IN PRIMARY CANDIDATE COHORT")
print("=" * 76)

print("\nManifest-level MRI protocol phase:")
print(
    manifest["mri_protocol_phases"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nMRI phase x candidate-series count:")
print(
    pd.crosstab(
        manifest["mri_protocol_phases"],
        manifest["mri_candidate_series_count"],
        margins=True
    ).to_string()
)


# ============================================================
# PREPARE MRIMPRANK SERIES IDENTIFIER
# ============================================================

rank["loni_series_num"] = pd.to_numeric(
    rank["LONIUID"]
    .astype(str)
    .str.extract(
        r"(\d+)",
        expand=False
    ),
    errors="coerce"
)

candidate_series["loni_series_num"] = pd.to_numeric(
    candidate_series["loni_series"],
    errors="coerce"
)


# ============================================================
# CHECK WHETHER ONE LONI SERIES HAS CONFLICTING RANKS
# ============================================================

rank_series_stats = (
    rank
    .dropna(
        subset=["loni_series_num"]
    )
    .groupby("loni_series_num")
    .agg(
        rank_rows=("RANK", "size"),
        rank_nunique=("RANK", "nunique"),
        rank_value=("RANK", "first"),
        rank_rid_nunique=("RID", "nunique"),
        rank_rid=("RID", "first"),
        rank_ptid=("PTID", "first"),
        rank_visit=("VISCODE", "first"),
        rank_examdate=("EXAMDATE", "first"),
    )
    .reset_index()
)

conflicting_rank_series = rank_series_stats[
    (rank_series_stats["rank_nunique"] > 1)
    | (rank_series_stats["rank_rid_nunique"] > 1)
]

print(
    "\nMRIMPRANK series with conflicting rank/RID:",
    len(conflicting_rank_series)
)


# ============================================================
# ADNI1 CANDIDATE SERIES ONLY
# ============================================================

adni1 = candidate_series[
    candidate_series[
        "mri_protocol_phase"
    ].astype(str).eq("ADNI1")
].copy()

print("\n" + "=" * 76)
print("ADNI1 TARGETED RANK COVERAGE")
print("=" * 76)

print(
    "ADNI1 subjects:",
    adni1["RID"].nunique()
)

print(
    "ADNI1 candidate series:",
    len(adni1)
)


# ============================================================
# MERGE RANK BY EXACT LONI SERIES
# ============================================================

adni1 = adni1.merge(
    rank_series_stats,
    how="left",
    on="loni_series_num",
    validate="many_to_one"
)

adni1["rank_matches_same_rid"] = (
    adni1["rank_rid"]
    .astype("Int64")
    .eq(
        adni1["RID"]
        .astype("Int64")
    )
)

adni1["rank_usable"] = (
    adni1["rank_value"].notna()
    & adni1["rank_matches_same_rid"]
    & adni1["rank_nunique"].eq(1)
    & adni1["rank_rid_nunique"].eq(1)
)

adni1["passing_rank"] = (
    adni1["rank_usable"]
    & adni1["rank_value"].isin(
        [1, 2, 3]
    )
)


print(
    "Series with exact usable MRIMPRANK match:",
    int(adni1["rank_usable"].sum())
)

print(
    "Series with passing rank 1-3:",
    int(adni1["passing_rank"].sum())
)

print(
    "Series without usable rank:",
    int((~adni1["rank_usable"]).sum())
)


# ============================================================
# SUBJECT-LEVEL RESOLUTION STATUS
# ============================================================

subject_records = []

for rid, grp in adni1.groupby("RID"):

    usable = grp[
        grp["rank_usable"]
    ].copy()

    passing = grp[
        grp["passing_rank"]
    ].copy()

    if passing.empty:
        best_rank = np.nan
        best_count = 0
        status = "no_passing_rank"

    else:
        best_rank = int(
            passing["rank_value"].min()
        )

        best = passing[
            passing["rank_value"]
            == best_rank
        ]

        best_count = len(best)

        if best_count == 1:
            status = "unique_best_rank"
        else:
            status = "best_rank_tie"

    subject_records.append(
        {
            "RID": int(rid),

            "candidate_series":
                int(len(grp)),

            "usable_ranked_series":
                int(len(usable)),

            "passing_ranked_series":
                int(len(passing)),

            "best_rank":
                best_rank,

            "best_rank_series_count":
                int(best_count),

            "status":
                status,
        }
    )

subject_audit = pd.DataFrame(
    subject_records
)

status_counts = (
    subject_audit["status"]
    .value_counts()
    .to_dict()
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "primary_manifest_subjects":
        int(len(manifest)),

    "mri_protocol_phase_counts": {
        str(k): int(v)
        for k, v in
        manifest[
            "mri_protocol_phases"
        ]
        .value_counts(
            dropna=False
        )
        .items()
    },

    "adni1_subjects":
        int(
            adni1["RID"].nunique()
        ),

    "adni1_candidate_series":
        int(len(adni1)),

    "adni1_series_with_usable_rank":
        int(
            adni1["rank_usable"].sum()
        ),

    "adni1_series_with_passing_rank":
        int(
            adni1["passing_rank"].sum()
        ),

    "adni1_subject_resolution_status": {
        str(k): int(v)
        for k, v in status_counts.items()
    },

    "adni1_unique_best_rank_subjects":
        int(
            (
                subject_audit["status"]
                == "unique_best_rank"
            ).sum()
        ),

    "adni1_best_rank_tie_subjects":
        int(
            (
                subject_audit["status"]
                == "best_rank_tie"
            ).sum()
        ),

    "adni1_no_passing_rank_subjects":
        int(
            (
                subject_audit["status"]
                == "no_passing_rank"
            ).sum()
        ),

    "mrimprank_conflicting_series":
        int(
            len(
                conflicting_rank_series
            )
        ),
}


# ============================================================
# SAVE
# ============================================================

subject_audit.to_csv(
    LOCAL_OUT,
    index=False
)

with open(
    SUMMARY_OUT,
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
# CONSOLE
# ============================================================

print("\nSUBJECT-LEVEL ADNI1 RESOLUTION")
print("-" * 76)

for k, v in status_counts.items():
    print(f"{k}: {v}")

print(
    "\nUnique-best-rank subjects:",
    summary[
        "adni1_unique_best_rank_subjects"
    ]
)

print(
    "Best-rank ties:",
    summary[
        "adni1_best_rank_tie_subjects"
    ]
)

print(
    "No passing rank:",
    summary[
        "adni1_no_passing_rank_subjects"
    ]
)

print(
    "\nSaved local audit:\n",
    LOCAL_OUT
)

print(
    "\nSaved aggregate summary:\n",
    SUMMARY_OUT
)
