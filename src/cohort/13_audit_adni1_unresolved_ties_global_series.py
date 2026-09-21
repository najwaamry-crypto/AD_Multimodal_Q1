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

std = (
    std.dropna(
        subset=[
            "RID",
            "Series.ID",
        ]
    )
    .drop_duplicates(
        [
            "RID",
            "Series.ID",
            "Image.ID",
            "scan_date",
        ]
    )
)

# ============================================================
# PRIMARY ADNI1 CANDIDATES
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

rank_series = (
    rank.dropna(
        subset=["loni_series_num"]
    )
    .groupby("loni_series_num")
    .agg(
        rank_value=(
            "rank_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        )
    )
    .reset_index()
)

target = target.merge(
    rank_series,
    how="left",
    on="loni_series_num",
    validate="many_to_one"
)

# ============================================================
# IDENTIFY THE 26 UNRESOLVED TIES
# ============================================================

records = []

for rid, grp in target.groupby("RID"):

    passing = grp[
        grp["rank_value"].isin(
            [1, 2, 3]
        )
    ].copy()

    if passing.empty:
        continue

    best_rank = passing[
        "rank_value"
    ].min()

    best_series = sorted(
        set(
            passing.loc[
                passing["rank_value"].eq(
                    best_rank
                ),
                "loni_series_num"
            ]
            .dropna()
            .astype(int)
        )
    )

    if len(best_series) <= 1:
        continue

    mri_date = grp[
        "mri_date"
    ].iloc[0]

    # Exact-date standardized series
    exact_std = std[
        std["RID"].eq(rid)
        &
        std["scan_date"].eq(
            mri_date
        )
    ]

    exact_overlap = set(
        exact_std[
            "Series.ID"
        ]
        .dropna()
        .astype(int)
    ).intersection(
        best_series
    )

    # We only want those that were NOT resolved on exact date
    if len(exact_overlap) == 1:
        continue

    # Search exact Series.ID anywhere in standardized lists
    # for the same RID, regardless of date.
    global_std = std[
        std["RID"].eq(rid)
        &
        std["Series.ID"].isin(
            best_series
        )
    ].copy()

    matched_series = sorted(
        set(
            global_std[
                "Series.ID"
            ]
            .dropna()
            .astype(int)
        )
    )

    if len(matched_series) == 1:
        status = (
            "one_best_series_found_elsewhere"
        )
        proposed_series = matched_series[0]

    elif len(matched_series) > 1:
        status = (
            "multiple_best_series_found_elsewhere"
        )
        proposed_series = pd.NA

    else:
        status = (
            "no_best_series_in_standardized_lists"
        )
        proposed_series = pd.NA

    matched_dates = (
        global_std[
            "scan_date"
        ]
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist()
    )

    records.append(
        {
            "RID": int(rid),
            "diagnosis":
                grp["diagnosis"].iloc[0],

            "mri_visit":
                grp["mri_visit"].iloc[0],

            "primary_mri_date":
                mri_date,

            "best_rank":
                int(best_rank),

            "best_series":
                "|".join(
                    map(
                        str,
                        best_series
                    )
                ),

            "matched_best_series_elsewhere":
                "|".join(
                    map(
                        str,
                        matched_series
                    )
                ),

            "matched_standardized_dates":
                "|".join(
                    matched_dates
                ),

            "status":
                status,

            "proposed_series":
                proposed_series,
        }
    )

audit = pd.DataFrame(records)

# ============================================================
# REPORT
# ============================================================

print("=" * 88)
print("ADNI1 - GLOBAL STANDARDIZED SERIES SEARCH FOR 26 TIES")
print("=" * 88)

print(
    "\nUnresolved ties evaluated:",
    len(audit)
)

print("\nStatus:")
print(
    audit[
        "status"
    ]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 88)
print("CASES FOUND ELSEWHERE")
print("=" * 88)

found = audit[
    audit[
        "status"
    ].ne(
        "no_best_series_in_standardized_lists"
    )
]

print(
    "Count:",
    len(found)
)

if len(found):
    print(
        found.to_string(
            index=False
        )
    )

print("\n" + "=" * 88)
print("NO STANDARDIZED BEST SERIES ANYWHERE")
print("=" * 88)

missing = audit[
    audit[
        "status"
    ].eq(
        "no_best_series_in_standardized_lists"
    )
]

print(
    "Count:",
    len(missing)
)

if len(missing):
    print(
        missing[
            [
                "RID",
                "diagnosis",
                "mri_visit",
                "primary_mri_date",
                "best_rank",
                "best_series",
            ]
        ]
        .to_string(
            index=False
        )
    )
