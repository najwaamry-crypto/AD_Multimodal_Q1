from pathlib import Path
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# PATHS
# ============================================================

OLD_QC_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/qc/go2/extracted"
    / "MAYOADIRL_MRI_IMAGEQC_05_07_15_21Sep2026.csv"
)

CURRENT_QC_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/qc/adni3"
    / "MAYOADIRL_MRI_QUALITY_ADNI3.rda"
)

MRI_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/extracted"
    / "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

MANIFEST_PATH = (
    ROOT / "data/interim/cohort"
    / "primary_candidate_manifest_v1.csv"
)


# ============================================================
# LOAD
# ============================================================

old = pd.read_csv(
    OLD_QC_PATH,
    low_memory=False
)

cur = next(
    iter(
        pyreadr.read_r(
            str(CURRENT_QC_PATH)
        ).values()
    )
)

mri = pd.read_csv(
    MRI_PATH,
    low_memory=False
)

manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False
)


# ============================================================
# NORMALIZE MANIFEST / KEY MRI
# ============================================================

manifest["mri_date"] = pd.to_datetime(
    manifest["mri_date"],
    errors="coerce"
)

mri["image_date"] = pd.to_datetime(
    mri["image_date"],
    errors="coerce"
)

mri["loni_series_num"] = pd.to_numeric(
    mri["loni_series"],
    errors="coerce"
)

mri["series_uid_key"] = (
    mri["series_instance_uid"]
    .astype("string")
    .str.strip()
)


# ============================================================
# NORMALIZE 2015 GO/2 MAYO QC
# ============================================================

old["loni_series_num"] = pd.to_numeric(
    old["loni_series"]
    .astype(str)
    .str.extract(
        r"(\d+)",
        expand=False
    ),
    errors="coerce"
)

old["selected_num"] = pd.to_numeric(
    old["series_selected"],
    errors="coerce"
)

old["quality_num"] = pd.to_numeric(
    old["series_quality"],
    errors="coerce"
)

old["update_dt"] = pd.to_datetime(
    old["update_stamp"],
    errors="coerce"
)


# ============================================================
# COLLAPSE HISTORICAL DUPLICATE QC ROWS
#
# The historical file can contain more than one record for
# the same LONI series. We use the latest recorded state.
# ============================================================

old_latest = (
    old
    .sort_values(
        [
            "loni_series_num",
            "update_dt",
        ],
        na_position="first"
    )
    .drop_duplicates(
        "loni_series_num",
        keep="last"
    )
    [
        [
            "loni_series_num",
            "selected_num",
            "quality_num",
        ]
    ]
    .rename(
        columns={
            "selected_num":
                "old_selected",

            "quality_num":
                "old_quality",
        }
    )
)


# ============================================================
# NORMALIZE CURRENT GO/2/3 MAYO QC
#
# IMPORTANT:
# This RDA has NO update_stamp.
# Therefore duplicate SeriesInstanceUID rows are collapsed only
# if their QC values are internally consistent.
# ============================================================

cur["series_uid_key"] = (
    cur["SERIESINSTANCEUID"]
    .astype("string")
    .str.strip()
)

cur["current_quality_num"] = pd.to_numeric(
    cur["SERIES_QUALITY"],
    errors="coerce"
)

cur["current_selected_num"] = (
    cur["SERIES_SELECTED"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(
        {
            "true": 1,
            "1": 1,
            "yes": 1,
            "y": 1,

            "false": 0,
            "0": 0,
            "no": 0,
            "n": 0,
        }
    )
)


# ============================================================
# CURRENT QC DUPLICATE AUDIT
# ============================================================

current = (
    cur
    .dropna(
        subset=["series_uid_key"]
    )
    .groupby(
        "series_uid_key"
    )
    .agg(
        current_qc_rows=(
            "series_uid_key",
            "size"
        ),

        selected_nunique=(
            "current_selected_num",
            lambda x:
                x.dropna().nunique()
        ),

        current_selected=(
            "current_selected_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),

        quality_nunique=(
            "current_quality_num",
            lambda x:
                x.dropna().nunique()
        ),

        current_quality=(
            "current_quality_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),

        current_origprot=(
            "ORIGPROT",
            "first"
        ),

        current_t1_accelerated=(
            "T1_ACCELERATED",
            "first"
        ),
    )
    .reset_index()
)

current["current_qc_usable"] = (
    current["selected_nunique"].le(1)
    &
    current["quality_nunique"].le(1)
)


# ============================================================
# EXACT 574-SUBJECT ADNIGO/ADNI2 PROTOCOL-FAMILY COHORT
# ============================================================

go2 = manifest[
    manifest[
        "mri_protocol_phases"
    ]
    .astype(str)
    .eq("ADNIGO/ADNI2")
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
    go2,
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
# MERGE HISTORICAL MAYO QC
# ============================================================

target = target.merge(
    old_latest,
    how="left",
    on="loni_series_num",
    validate="many_to_one"
)


# ============================================================
# MERGE CURRENT QC BY DICOM SERIESINSTANCEUID
# ============================================================

target = target.merge(
    current,
    how="left",
    on="series_uid_key",
    validate="many_to_one"
)


# ============================================================
# DETERMINE QC SOURCE
#
# Historical 2015 GO/2 QC = primary source.
# Current GO/2/3 table = fallback only when historical row
# is missing and current QC record is internally consistent.
# ============================================================

target["qc_source"] = "missing"

mask_old = (
    target["old_selected"]
    .notna()
)

target.loc[
    mask_old,
    "qc_source"
] = "Mayo2015"

mask_current = (
    ~mask_old
    &
    target[
        "current_selected"
    ].notna()
    &
    target[
        "current_qc_usable"
    ].fillna(False)
)

target.loc[
    mask_current,
    "qc_source"
] = "CurrentQC"


# ============================================================
# BUILD UNIFIED QC DECISION
# ============================================================

target["official_selected"] = False
target["official_quality"] = pd.NA


# Historical GO/2 QC has priority
target.loc[
    mask_old,
    "official_selected"
] = (
    target.loc[
        mask_old,
        "old_selected"
    ]
    .eq(1)
)

target.loc[
    mask_old,
    "official_quality"
] = (
    target.loc[
        mask_old,
        "old_quality"
    ]
)


# Current QC only fills historical gaps
target.loc[
    mask_current,
    "official_selected"
] = (
    target.loc[
        mask_current,
        "current_selected"
    ]
    .eq(1)
)

target.loc[
    mask_current,
    "official_quality"
] = (
    target.loc[
        mask_current,
        "current_quality"
    ]
)

target[
    "official_quality"
] = pd.to_numeric(
    target[
        "official_quality"
    ],
    errors="coerce"
)


# ============================================================
# PASSING QC RULE
#
# Quality:
# 1, 2, 3 = usable
# 4 / -1 / missing = not retained for primary selection
#
# Must also be SERIES_SELECTED by Mayo.
# ============================================================

target["selected_passing"] = (
    target[
        "official_selected"
    ]
    &
    target[
        "official_quality"
    ].isin(
        [1, 2, 3]
    )
)


# ============================================================
# SERIES-LEVEL REPORT
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "GO/ADNI2 PROTOCOL FAMILY - FINAL SELECTION AUDIT"
)

print(
    "=" * 78
)

print(
    "\nSubjects:",
    target["RID"].nunique()
)

print(
    "Candidate series:",
    len(target)
)

print(
    "\nQC source:"
)

print(
    target[
        "qc_source"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print(
    "\nKey MRI acceleration:"
)

print(
    target[
        "acceleration"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# SELECTED + PASSING SERIES
# ============================================================

sel = target[
    target[
        "selected_passing"
    ]
].copy()

print(
    "\nPassing + officially selected series:",
    len(sel)
)

print(
    "\nSelected passing series by acceleration:"
)

print(
    sel[
        "acceleration"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print(
    "\nSelected passing quality:"
)

print(
    sel[
        "official_quality"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .to_string()
)


# ============================================================
# SUBJECT-LEVEL AVAILABILITY
# ============================================================

def count_type(
    grp,
    acceleration_type
):

    return int(
        (
            grp[
                "selected_passing"
            ]
            &
            grp[
                "acceleration"
            ].eq(
                acceleration_type
            )
        ).sum()
    )


records = []

for rid, grp in target.groupby(
    "RID"
):

    records.append(
        {
            "RID":
                rid,

            "diagnosis":
                grp[
                    "diagnosis"
                ].iloc[0],

            "selected_accelerated":
                count_type(
                    grp,
                    "Accelerated"
                ),

            "selected_unaccelerated":
                count_type(
                    grp,
                    "Unaccelerated"
                ),

            "selected_other":
                int(
                    (
                        grp[
                            "selected_passing"
                        ]
                        &
                        ~grp[
                            "acceleration"
                        ].isin(
                            [
                                "Accelerated",
                                "Unaccelerated",
                            ]
                        )
                    ).sum()
                ),

            "missing_qc_series":
                int(
                    grp[
                        "qc_source"
                    ]
                    .eq(
                        "missing"
                    )
                    .sum()
                ),
        }
    )

sub = pd.DataFrame(
    records
)


# ============================================================
# SUBJECT-LEVEL REPORT
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "SUBJECT-LEVEL AVAILABILITY"
)

print(
    "=" * 78
)

print(
    "\nSelected accelerated count:"
)

print(
    sub[
        "selected_accelerated"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "\nSelected unaccelerated count:"
)

print(
    sub[
        "selected_unaccelerated"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# ACCELERATED STRATEGY
# ============================================================

acc = sub[
    sub[
        "selected_accelerated"
    ]
    .eq(1)
].copy()

print(
    "\n"
    + "=" * 78
)

print(
    "ACCELERATED STRATEGY"
)

print(
    "=" * 78
)

print(
    "Subjects with exactly one:",
    len(acc)
)

print(
    "\nClasses:"
)

print(
    acc[
        "diagnosis"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nZero accelerated:",
    int(
        (
            sub[
                "selected_accelerated"
            ] == 0
        ).sum()
    )
)

print(
    ">1 accelerated:",
    int(
        (
            sub[
                "selected_accelerated"
            ] > 1
        ).sum()
    )
)


# ============================================================
# UNACCELERATED STRATEGY
# ============================================================

unacc = sub[
    sub[
        "selected_unaccelerated"
    ]
    .eq(1)
].copy()

print(
    "\n"
    + "=" * 78
)

print(
    "UNACCELERATED STRATEGY"
)

print(
    "=" * 78
)

print(
    "Subjects with exactly one:",
    len(unacc)
)

print(
    "\nClasses:"
)

print(
    unacc[
        "diagnosis"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nZero unaccelerated:",
    int(
        (
            sub[
                "selected_unaccelerated"
            ] == 0
        ).sum()
    )
)

print(
    ">1 unaccelerated:",
    int(
        (
            sub[
                "selected_unaccelerated"
            ] > 1
        ).sum()
    )
)


# ============================================================
# NON-STANDARD SUBJECTS
# ============================================================

problem = sub[
    (
        sub[
            "selected_accelerated"
        ] != 1
    )
    |
    (
        sub[
            "selected_unaccelerated"
        ] != 1
    )
].copy()

print(
    "\n"
    + "=" * 78
)

print(
    "NON-STANDARD AVAILABILITY"
)

print(
    "=" * 78
)

print(
    "Subjects with some non-standard availability pattern:",
    len(problem)
)

if len(problem):

    print(
        "\nFirst cases:"
    )

    print(
        problem
        .head(40)
        .to_string(
            index=False
        )
    )


# ============================================================
# MISSING QC SUBJECTS
# ============================================================

missing_subjects = sub[
    sub[
        "missing_qc_series"
    ] > 0
].copy()

print(
    "\nSubjects with at least one candidate series lacking QC:",
    len(missing_subjects)
)

if len(missing_subjects):

    print(
        missing_subjects
        .head(30)
        .to_string(
            index=False
        )
    )
