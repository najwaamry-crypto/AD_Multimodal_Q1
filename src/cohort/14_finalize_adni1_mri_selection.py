from pathlib import Path
import json
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

OUT_SELECTION = (
    ROOT / "data/interim/cohort/"
    "adni1_mri_selection_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables/"
    "adni1_mri_selection_v1_summary.json"
)

# ============================================================
# STANDARDIZED 1.5T LISTS
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
    format="mixed",
    errors="coerce"
)

std = (
    std.dropna(
        subset=[
            "RID",
            "Series.ID",
            "scan_date",
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
        "dx_phase",
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

# No candidate series may have conflicting rank records.
candidate_rank_conflicts = rank_series[
    rank_series["rank_nunique"] > 1
]

target_series_ids = set(
    target[
        "loni_series_num"
    ]
    .dropna()
)

candidate_rank_conflicts = (
    candidate_rank_conflicts[
        candidate_rank_conflicts[
            "loni_series_num"
        ].isin(
            target_series_ids
        )
    ]
)

if len(candidate_rank_conflicts):
    raise RuntimeError(
        "Conflicting MRIMPRANK records found "
        "for ADNI1 candidate series."
    )

target = target.merge(
    rank_series[
        [
            "loni_series_num",
            "rank_value",
        ]
    ],
    how="left",
    on="loni_series_num",
    validate="many_to_one"
)

# ============================================================
# SUBJECT-LEVEL FINAL SELECTION
# ============================================================

records = []

for rid, grp in target.groupby("RID"):

    rid = int(rid)

    diagnosis = grp["diagnosis"].iloc[0]
    dx_phase = grp["dx_phase"].iloc[0]
    mri_visit = grp["mri_visit"].iloc[0]
    mri_date = grp["mri_date"].iloc[0]

    passing = grp[
        grp["rank_value"].isin(
            [1, 2, 3]
        )
    ].copy()

    selected_series = pd.NA
    selection_source = pd.NA
    best_rank = pd.NA

    # --------------------------------------------------------
    # NO PASSING RANK
    # --------------------------------------------------------

    if passing.empty:

        status = "exclude_no_passing_rank"

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

        # ----------------------------------------------------
        # UNIQUE BEST MRIMPRANK
        # ----------------------------------------------------

        if len(best_series) == 1:

            selected_series = next(
                iter(best_series)
            )

            status = "selected"

            selection_source = (
                "MRIMPRANK_unique_best"
            )

        # ----------------------------------------------------
        # TIED BEST MRIMPRANK
        # ----------------------------------------------------

        else:

            std_obs = std[
                std["RID"].eq(rid)
                &
                std["scan_date"].eq(
                    mri_date
                )
            ].copy()

            standardized_series = set(
                std_obs[
                    "Series.ID"
                ]
                .dropna()
                .astype(int)
                .tolist()
            )

            best_overlap = (
                best_series
                &
                standardized_series
            )

            if len(best_overlap) == 1:

                selected_series = next(
                    iter(best_overlap)
                )

                status = "selected"

                selection_source = (
                    "standardized_1.5T_list"
                )

            elif len(best_overlap) > 1:

                status = (
                    "exclude_tie_multiple_standardized_best"
                )

            elif len(standardized_series) == 0:

                status = (
                    "exclude_tie_no_standardized_series"
                )

            else:

                status = (
                    "exclude_tie_no_best_standardized_overlap"
                )

    # ========================================================
    # GET EXACT KEY MRI ROW FOR SELECTED SERIES
    # ========================================================

    selected_image_id = pd.NA
    selected_series_description = pd.NA
    selected_series_uid = pd.NA
    selected_loni_image = pd.NA

    if status == "selected":

        selected_rows = grp[
            grp["loni_series_num"].eq(
                selected_series
            )
        ].copy()

        if len(selected_rows) != 1:
            raise RuntimeError(
                f"RID {rid}: expected exactly one "
                f"Key MRI row for selected LONI series "
                f"{selected_series}, got {len(selected_rows)}"
            )

        row = selected_rows.iloc[0]

        selected_image_id = row["image_id"]

        selected_series_description = (
            row["series_description"]
        )

        selected_series_uid = (
            row["series_instance_uid"]
        )

        selected_loni_image = (
            row["loni_image"]
        )

    records.append(
        {
            "RID": rid,

            "PTID":
                grp["PTID"].iloc[0],

            "diagnosis":
                diagnosis,

            "dx_phase":
                dx_phase,

            "mri_visit":
                mri_visit,

            "mri_date":
                mri_date,

            "best_rank":
                best_rank,

            "n_best_rank_series":
                len(best_series),

            "best_rank_series":
                "|".join(
                    map(
                        str,
                        sorted(best_series)
                    )
                ),

            "status":
                status,

            "selection_source":
                selection_source,

            "selected_loni_series":
                selected_series,

            "selected_image_id":
                selected_image_id,

            "selected_loni_image":
                selected_loni_image,

            "selected_series_instance_uid":
                selected_series_uid,

            "selected_series_description":
                selected_series_description,
        }
    )

selection = pd.DataFrame(
    records
)

# ============================================================
# VALIDATION
# ============================================================

if len(selection) != 204:
    raise RuntimeError(
        f"Expected 204 ADNI1 subjects, "
        f"got {len(selection)}"
    )

selected = selection[
    selection["status"].eq(
        "selected"
    )
].copy()

if selected[
    "selected_loni_series"
].isna().any():
    raise RuntimeError(
        "Selected subject without selected LONI series."
    )

if selected[
    "selected_loni_series"
].duplicated().any():
    raise RuntimeError(
        "Selected LONI series reused across subjects."
    )

# ============================================================
# SUMMARY
# ============================================================

status_counts = (
    selection[
        "status"
    ]
    .value_counts()
    .to_dict()
)

source_counts = (
    selected[
        "selection_source"
    ]
    .value_counts()
    .to_dict()
)

class_counts = (
    selected[
        "diagnosis"
    ]
    .value_counts()
    .to_dict()
)

excluded = selection[
    ~selection["status"].eq(
        "selected"
    )
]

excluded_class_counts = (
    excluded[
        "diagnosis"
    ]
    .value_counts()
    .to_dict()
)

summary = {
    "subjects_total":
        int(len(selection)),

    "subjects_selected":
        int(len(selected)),

    "subjects_excluded":
        int(len(excluded)),

    "status_counts": {
        str(k): int(v)
        for k, v in status_counts.items()
    },

    "selection_source_counts": {
        str(k): int(v)
        for k, v in source_counts.items()
    },

    "selected_class_counts": {
        str(k): int(v)
        for k, v in class_counts.items()
    },

    "excluded_class_counts": {
        str(k): int(v)
        for k, v in excluded_class_counts.items()
    },
}

# ============================================================
# SAVE
# ============================================================

selection.to_csv(
    OUT_SELECTION,
    index=False
)

with open(
    OUT_SUMMARY,
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

print("\n" + "=" * 80)
print("FINAL ADNI1 MRI SELECTION")
print("=" * 80)

print(
    "\nTotal subjects:",
    summary["subjects_total"]
)

print(
    "Selected:",
    summary["subjects_selected"]
)

print(
    "Excluded:",
    summary["subjects_excluded"]
)

print("\nSelection source:")
for k, v in summary[
    "selection_source_counts"
].items():
    print(f"  {k}: {v}")

print("\nStatus:")
for k, v in summary[
    "status_counts"
].items():
    print(f"  {k}: {v}")

print("\nClasses among selected:")
for k, v in summary[
    "selected_class_counts"
].items():
    print(f"  {k}: {v}")

print("\nClasses among excluded:")
for k, v in summary[
    "excluded_class_counts"
].items():
    print(f"  {k}: {v}")

print(
    "\nSaved local selection:\n",
    OUT_SELECTION
)

print(
    "\nSaved aggregate summary:\n",
    OUT_SUMMARY
)
