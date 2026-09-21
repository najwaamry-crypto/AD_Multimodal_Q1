from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

STD_ROOT = (
    ROOT / "data/raw/imaging_metadata/mri/qc/adni1/"
    "standardized_lists/extracted/1.5T"
)

MRI_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/extracted/"
    "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

MANIFEST_PATH = (
    ROOT / "data/interim/cohort/"
    "primary_candidate_manifest_v1.csv"
)

RANK_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/qc/extracted/"
    "MRIMPRANK_21Sep2026.csv"
)

# ============================================================
# STANDARDIZED LISTS
# ============================================================

frames = []

for path in sorted(STD_ROOT.glob("*.csv")):
    df = pd.read_csv(path, low_memory=False)
    df["source_file"] = path.name
    frames.append(df)

std = pd.concat(
    frames,
    ignore_index=True
)

std["RID"] = pd.to_numeric(
    std["RID"],
    errors="coerce"
)

std["Series.ID"] = pd.to_numeric(
    std["Series.ID"],
    errors="coerce"
)

std["Image.ID"] = pd.to_numeric(
    std["Image.ID"],
    errors="coerce"
)

std["scan_date"] = pd.to_datetime(
    std["Scan.Date"],
    errors="coerce"
)

# Same standardized scan appears in several standardized-list files.
std = (
    std.dropna(
        subset=[
            "RID",
            "scan_date",
            "Series.ID",
        ]
    )
    .drop_duplicates(
        [
            "RID",
            "scan_date",
            "Series.ID",
            "Image.ID",
        ]
    )
)

# ============================================================
# PRIMARY ADNI1 MRI CANDIDATES
# ============================================================

mri = pd.read_csv(
    MRI_PATH,
    low_memory=False
)

manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False
)

mri["image_date"] = pd.to_datetime(
    mri["image_date"],
    errors="coerce"
)

mri["loni_series_num"] = pd.to_numeric(
    mri["loni_series"],
    errors="coerce"
)

manifest["mri_date"] = pd.to_datetime(
    manifest["mri_date"],
    errors="coerce"
)

adni1 = manifest[
    manifest["mri_protocol_phases"]
    .astype(str)
    .eq("ADNI1")
][
    [
        "RID",
        "PTID",
        "diagnosis",
        "mri_visit",
        "mri_date",
    ]
].copy()

target = mri.merge(
    adni1,
    left_on=[
        "subject_id",
        "image_visit",
        "image_date",
    ],
    right_on=[
        "PTID",
        "mri_visit",
        "mri_date",
    ],
    how="inner",
    validate="many_to_one"
)

# ============================================================
# MRIMPRANK
# ============================================================

rank = pd.read_csv(
    RANK_PATH,
    low_memory=False
)

rank["loni_series_num"] = pd.to_numeric(
    rank["LONIUID"]
    .astype(str)
    .str.extract(
        r"(\d+)",
        expand=False
    ),
    errors="coerce"
)

rank["rank_num"] = pd.to_numeric(
    rank["RANK"],
    errors="coerce"
)

# Check whether any candidate series has conflicting rank records.
rank_series = (
    rank.dropna(
        subset=["loni_series_num"]
    )
    .groupby("loni_series_num")
    .agg(
        rank_nunique=(
            "rank_num",
            lambda x:
                x.dropna().nunique()
        ),
        rank_value=(
            "rank_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),
    )
    .reset_index()
)

target = target.merge(
    rank_series,
    how="left",
    on="loni_series_num",
    validate="many_to_one"
)

conflicting_target = target[
    target["rank_nunique"].gt(1)
]

print("=" * 88)
print("ADNI1 STANDARDIZED-LIST TIE RESOLUTION")
print("=" * 88)

print(
    "\nCandidate series with conflicting MRIMPRANK records:",
    conflicting_target["loni_series_num"].nunique()
)

# ============================================================
# SUBJECT-BY-SUBJECT RESOLUTION
# ============================================================

records = []

for rid, grp in target.groupby("RID"):

    rid = int(rid)

    diagnosis = grp["diagnosis"].iloc[0]
    mri_date = grp["mri_date"].iloc[0]
    mri_visit = grp["mri_visit"].iloc[0]

    # Distinct raw candidate LONI series
    candidate_series = set(
        grp[
            "loni_series_num"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    # Passing rank = 1, 2 or 3
    passing = grp[
        grp["rank_value"].isin(
            [1, 2, 3]
        )
    ].copy()

    if passing.empty:

        rank_status = "no_passing_rank"
        best_rank = pd.NA
        best_series = set()

    else:

        best_rank = int(
            passing["rank_value"].min()
        )

        best_series = set(
            passing.loc[
                passing["rank_value"].eq(
                    best_rank
                ),
                "loni_series_num"
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        if len(best_series) == 1:
            rank_status = "unique_best_rank"
        else:
            rank_status = "best_rank_tie"

    # Exact same RID + MRI acquisition date in standardized lists
    std_obs = std[
        std["RID"].eq(rid)
        &
        std["scan_date"].eq(mri_date)
    ].copy()

    std_series = set(
        std_obs[
            "Series.ID"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    # Important intersections
    std_candidate_overlap = (
        std_series
        &
        candidate_series
    )

    std_best_overlap = (
        std_series
        &
        best_series
    )

    # --------------------------------------------------------
    # Resolution status
    # --------------------------------------------------------

    if rank_status == "unique_best_rank":

        resolution_status = (
            "already_unique_best_rank"
        )

        selected_series = next(
            iter(best_series)
        )

    elif rank_status == "no_passing_rank":

        resolution_status = (
            "no_passing_rank"
        )

        selected_series = pd.NA

    else:
        # TIED BEST RANK

        if len(std_best_overlap) == 1:

            resolution_status = (
                "tie_resolved_by_standardized_list"
            )

            selected_series = next(
                iter(std_best_overlap)
            )

        elif len(std_best_overlap) > 1:

            resolution_status = (
                "tie_still_multiple_standardized_best"
            )

            selected_series = pd.NA

        elif len(std_series) == 0:

            resolution_status = (
                "tie_no_standardized_exact_date"
            )

            selected_series = pd.NA

        elif len(std_candidate_overlap) > 0:

            resolution_status = (
                "tie_standardized_points_to_nonbest_candidate"
            )

            selected_series = pd.NA

        else:

            resolution_status = (
                "tie_standardized_no_raw_candidate_overlap"
            )

            selected_series = pd.NA

    records.append(
        {
            "RID": rid,
            "diagnosis": diagnosis,
            "mri_visit": mri_visit,
            "mri_date": mri_date,

            "rank_status":
                rank_status,

            "best_rank":
                best_rank,

            "n_candidate_series":
                len(candidate_series),

            "candidate_series":
                "|".join(
                    map(
                        str,
                        sorted(candidate_series)
                    )
                ),

            "n_best_series":
                len(best_series),

            "best_series":
                "|".join(
                    map(
                        str,
                        sorted(best_series)
                    )
                ),

            "n_standardized_series":
                len(std_series),

            "standardized_series":
                "|".join(
                    map(
                        str,
                        sorted(std_series)
                    )
                ),

            "n_std_candidate_overlap":
                len(std_candidate_overlap),

            "std_candidate_overlap":
                "|".join(
                    map(
                        str,
                        sorted(std_candidate_overlap)
                    )
                ),

            "n_std_best_overlap":
                len(std_best_overlap),

            "std_best_overlap":
                "|".join(
                    map(
                        str,
                        sorted(std_best_overlap)
                    )
                ),

            "resolution_status":
                resolution_status,

            "selected_loni_series":
                selected_series,
        }
    )

audit = pd.DataFrame(records)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 88)
print("ORIGINAL RANK STATUS")
print("=" * 88)

print(
    audit[
        "rank_status"
    ]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 88)
print("FINAL RESOLUTION STATUS")
print("=" * 88)

print(
    audit[
        "resolution_status"
    ]
    .value_counts()
    .to_string()
)

ties = audit[
    audit["rank_status"]
    .eq("best_rank_tie")
]

resolved = ties[
    ties["resolution_status"]
    .eq(
        "tie_resolved_by_standardized_list"
    )
]

print("\n" + "=" * 88)
print("TIED SUBJECTS")
print("=" * 88)

print(
    "Total tied subjects:",
    len(ties)
)

print(
    "Resolved by standardized list:",
    len(resolved)
)

print(
    "Still unresolved:",
    len(ties) - len(resolved)
)

print(
    "\nClasses among standardized-list-resolved ties:"
)

print(
    resolved[
        "diagnosis"
    ]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 88)
print("TIE RESOLUTION DETAILS")
print("=" * 88)

print(
    ties[
        "resolution_status"
    ]
    .value_counts()
    .to_string()
)

# ============================================================
# CONFLICT CHECK
# ============================================================

conflicts = ties[
    ties["resolution_status"].isin(
        [
            "tie_standardized_points_to_nonbest_candidate",
            "tie_standardized_no_raw_candidate_overlap",
            "tie_still_multiple_standardized_best",
        ]
    )
]

print("\nPotential conflicts:", len(conflicts))

if len(conflicts):

    print("\nConflict cases:")

    print(
        conflicts[
            [
                "RID",
                "diagnosis",
                "mri_visit",
                "mri_date",
                "best_rank",
                "best_series",
                "standardized_series",
                "std_candidate_overlap",
                "std_best_overlap",
                "resolution_status",
            ]
        ]
        .to_string(
            index=False
        )
    )

# ============================================================
# UNRESOLVED TIES
# ============================================================

unresolved = ties[
    ~ties["resolution_status"]
    .eq(
        "tie_resolved_by_standardized_list"
    )
]

print("\n" + "=" * 88)
print("UNRESOLVED TIES")
print("=" * 88)

print(
    "Count:",
    len(unresolved)
)

print(
    "\nReason:"
)

print(
    unresolved[
        "resolution_status"
    ]
    .value_counts()
    .to_string()
)

print("\nFirst 40 unresolved:")

print(
    unresolved[
        [
            "RID",
            "diagnosis",
            "mri_visit",
            "mri_date",
            "best_rank",
            "best_series",
            "standardized_series",
            "resolution_status",
        ]
    ]
    .head(40)
    .to_string(
        index=False
    )
)
