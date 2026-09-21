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
# LOAD STANDARDIZED LISTS
# ============================================================

std_files = sorted(STD_ROOT.glob("*.csv"))

frames = []

for path in std_files:
    df = pd.read_csv(path, low_memory=False)
    df["source_file"] = path.name
    frames.append(df)

std_all = pd.concat(
    frames,
    ignore_index=True
)

std_all["RID"] = pd.to_numeric(
    std_all["RID"],
    errors="coerce"
)

std_all["Image.ID"] = pd.to_numeric(
    std_all["Image.ID"],
    errors="coerce"
)

std_all["Study.ID"] = pd.to_numeric(
    std_all["Study.ID"],
    errors="coerce"
)

std_all["Series.ID"] = pd.to_numeric(
    std_all["Series.ID"],
    errors="coerce"
)

std_all["scan_date"] = pd.to_datetime(
    std_all["Scan.Date"],
    errors="coerce"
)

# One standardized image may appear in several list variants.
std_unique = (
    std_all
    .sort_values(
        [
            "RID",
            "scan_date",
            "Image.ID",
            "Series.ID",
        ]
    )
    .drop_duplicates(
        [
            "RID",
            "scan_date",
            "Image.ID",
            "Series.ID",
        ]
    )
    .copy()
)

# ============================================================
# LOAD PRIMARY ADNI1 TARGET
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

mri["image_id_num"] = pd.to_numeric(
    mri["image_id"],
    errors="coerce"
)

mri["loni_series_num"] = pd.to_numeric(
    mri["loni_series"],
    errors="coerce"
)

mri["loni_study_num"] = pd.to_numeric(
    mri["loni_study"],
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
# STANDARDIZED LIST COVERAGE BY RID + DATE
# ============================================================

std_primary_rids = std_unique[
    std_unique["RID"].isin(
        adni1["RID"]
    )
].copy()

obs = (
    adni1[
        [
            "RID",
            "PTID",
            "diagnosis",
            "mri_visit",
            "mri_date",
        ]
    ]
    .drop_duplicates()
)

date_match = obs.merge(
    std_unique,
    left_on=[
        "RID",
        "mri_date",
    ],
    right_on=[
        "RID",
        "scan_date",
    ],
    how="left",
    indicator=True
)

per_obs = (
    date_match
    .groupby(
        [
            "RID",
            "PTID_x",
            "diagnosis",
            "mri_visit",
            "mri_date",
        ],
        dropna=False
    )
    .agg(
        standardized_rows=(
            "_merge",
            lambda x:
                int(
                    (x == "both").sum()
                )
        ),

        standardized_images=(
            "Image.ID",
            lambda x:
                x.dropna().nunique()
        ),

        standardized_series=(
            "Series.ID",
            lambda x:
                x.dropna().nunique()
        ),
    )
    .reset_index()
)

# ============================================================
# DIRECT ID OVERLAP TEST
# ============================================================

target_image_ids = set(
    target[
        "image_id_num"
    ].dropna()
)

target_series_ids = set(
    target[
        "loni_series_num"
    ].dropna()
)

target_study_ids = set(
    target[
        "loni_study_num"
    ].dropna()
)

std_unique["image_id_matches_raw"] = (
    std_unique["Image.ID"]
    .isin(target_image_ids)
)

std_unique["series_id_matches_raw"] = (
    std_unique["Series.ID"]
    .isin(target_series_ids)
)

std_unique["study_id_matches_raw"] = (
    std_unique["Study.ID"]
    .isin(target_study_ids)
)

# ============================================================
# LOAD MRIMPRANK TO RECREATE TIED BEST-RANK SUBJECTS
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

target_ranked = target.merge(
    rank[
        [
            "loni_series_num",
            "rank_num",
        ]
    ]
    .drop_duplicates(),
    how="left",
    on="loni_series_num"
)

tie_records = []

for rid, grp in target_ranked.groupby("RID"):

    passing = grp[
        grp["rank_num"].isin(
            [1, 2, 3]
        )
    ].copy()

    if passing.empty:
        status = "no_passing_rank"
        best_rank = pd.NA
        n_best = 0

    else:
        best_rank = passing["rank_num"].min()

        n_best = (
            passing.loc[
                passing["rank_num"].eq(
                    best_rank
                ),
                "loni_series_num"
            ]
            .nunique()
        )

        if n_best == 1:
            status = "unique_best_rank"
        else:
            status = "best_rank_tie"

    tie_records.append(
        {
            "RID": rid,
            "rank_status": status,
            "best_rank": best_rank,
            "n_best_series": n_best,
        }
    )

rank_status = pd.DataFrame(
    tie_records
)

per_obs = per_obs.merge(
    rank_status,
    how="left",
    on="RID",
    validate="one_to_one"
)

# ============================================================
# REPORT
# ============================================================

print("=" * 84)
print("ADNI1 1.5T STANDARDIZED LIST AUDIT")
print("=" * 84)

print("\nStandardized list files:", len(std_files))
print("Rows across all files:", len(std_all))
print("Unique standardized rows:", len(std_unique))
print(
    "Unique standardized subjects:",
    std_unique["RID"].nunique()
)

print("\n" + "=" * 84)
print("PRIMARY ADNI1 COHORT")
print("=" * 84)

print("Subjects:", adni1["RID"].nunique())
print("Candidate raw T1 rows:", len(target))
print(
    "Distinct raw LONI series:",
    target["loni_series_num"].nunique()
)

print("\nRank status:")
print(
    rank_status[
        "rank_status"
    ]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 84)
print("STANDARDIZED COVERAGE AT EXACT PRIMARY MRI DATE")
print("=" * 84)

print(
    per_obs[
        "standardized_rows"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "\nSubjects with >=1 standardized row on exact MRI date:",
    int(
        (
            per_obs[
                "standardized_rows"
            ] > 0
        ).sum()
    )
)

print(
    "Subjects with no standardized row on exact MRI date:",
    int(
        (
            per_obs[
                "standardized_rows"
            ] == 0
        ).sum()
    )
)

print("\nStandardized SERIES count per observation:")
print(
    per_obs[
        "standardized_series"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "=" * 84)
print("DIRECT ID OVERLAP WITH RAW KEY MRI")
print("=" * 84)

relevant_std = std_unique[
    std_unique["RID"].isin(
        adni1["RID"]
    )
]

print(
    "Standardized rows for our 204 RIDs:",
    len(relevant_std)
)

print(
    "Image.ID directly matching raw Key MRI image_id:",
    int(
        relevant_std[
            "image_id_matches_raw"
        ].sum()
    )
)

print(
    "Series.ID directly matching raw Key MRI loni_series:",
    int(
        relevant_std[
            "series_id_matches_raw"
        ].sum()
    )
)

print(
    "Study.ID directly matching raw Key MRI loni_study:",
    int(
        relevant_std[
            "study_id_matches_raw"
        ].sum()
    )
)

print("\n" + "=" * 84)
print("COVERAGE OF THE 127 BEST-RANK TIES")
print("=" * 84)

ties = per_obs[
    per_obs[
        "rank_status"
    ].eq(
        "best_rank_tie"
    )
].copy()

print(
    "Tied subjects:",
    len(ties)
)

print(
    "Tied subjects with standardized row on exact MRI date:",
    int(
        (
            ties[
                "standardized_rows"
            ] > 0
        ).sum()
    )
)

print(
    "Tied subjects without standardized row on exact MRI date:",
    int(
        (
            ties[
                "standardized_rows"
            ] == 0
        ).sum()
    )
)

print("\nStandardized series count among tied subjects:")
print(
    ties[
        "standardized_series"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nFirst 30 tied subjects:")
print(
    ties[
        [
            "RID",
            "mri_visit",
            "mri_date",
            "best_rank",
            "n_best_series",
            "standardized_rows",
            "standardized_images",
            "standardized_series",
        ]
    ]
    .head(30)
    .to_string(index=False)
)
