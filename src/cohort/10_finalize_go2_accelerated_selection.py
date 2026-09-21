from pathlib import Path
import json
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[2]

OLD_QC = (
    ROOT / "data/raw/imaging_metadata/mri/qc/go2/extracted"
    / "MAYOADIRL_MRI_IMAGEQC_05_07_15_21Sep2026.csv"
)

CURRENT_QC = (
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

OUT_LOCAL = (
    ROOT / "data/interim/cohort"
    / "go2_accelerated_selection_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables"
    / "go2_accelerated_selection_v1_summary.json"
)

# ============================================================
# LOAD
# ============================================================

old = pd.read_csv(OLD_QC, low_memory=False)

cur = next(
    iter(
        pyreadr.read_r(
            str(CURRENT_QC)
        ).values()
    )
)

mri = pd.read_csv(MRI_PATH, low_memory=False)
manifest = pd.read_csv(MANIFEST_PATH, low_memory=False)

# ============================================================
# NORMALIZE MRI / MANIFEST
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
# HISTORICAL MAYO GO/2 QC
# ============================================================

old["loni_series_num"] = pd.to_numeric(
    old["loni_series"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False),
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

old_latest = (
    old
    .sort_values(
        ["loni_series_num", "update_dt"],
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
            "selected_num": "old_selected",
            "quality_num": "old_quality",
        }
    )
)

# ============================================================
# CURRENT QC FALLBACK
# ============================================================

cur["series_uid_key"] = (
    cur["SERIESINSTANCEUID"]
    .astype("string")
    .str.strip()
)

cur["current_quality"] = pd.to_numeric(
    cur["SERIES_QUALITY"],
    errors="coerce"
)

cur["current_selected"] = (
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

current = (
    cur
    .dropna(subset=["series_uid_key"])
    .groupby("series_uid_key")
    .agg(
        selected_nunique=(
            "current_selected",
            lambda x: x.dropna().nunique()
        ),
        current_selected=(
            "current_selected",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),
        quality_nunique=(
            "current_quality",
            lambda x: x.dropna().nunique()
        ),
        current_quality=(
            "current_quality",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),
    )
    .reset_index()
)

current["current_usable"] = (
    current["selected_nunique"].le(1)
    &
    current["quality_nunique"].le(1)
)

# ============================================================
# EXACT GO/2 PROTOCOL-FAMILY OBSERVATIONS
# ============================================================

go2 = manifest[
    manifest["mri_protocol_phases"]
    .astype(str)
    .eq("ADNIGO/ADNI2")
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
# IMPORTANT:
# COLLAPSE KEY-MRI DUPLICATES TO LONI SERIES LEVEL
# ============================================================

series = (
    target
    .sort_values(
        [
            "RID",
            "loni_series_num",
            "image_id",
        ]
    )
    .groupby(
        [
            "RID",
            "loni_series_num",
        ],
        as_index=False
    )
    .agg(
        PTID=("PTID", "first"),
        diagnosis=("diagnosis", "first"),
        dx_phase=("dx_phase", "first"),
        mri_visit=("mri_visit", "first"),
        mri_date=("mri_date", "first"),

        acceleration=("acceleration", "first"),

        series_description=(
            "series_description",
            "first"
        ),

        series_uid_key=(
            "series_uid_key",
            "first"
        ),

        key_mri_image_count=(
            "image_id",
            "nunique"
        ),

        key_mri_image_ids=(
            "image_id",
            lambda x:
                "|".join(
                    map(
                        str,
                        sorted(
                            pd.unique(x)
                        )
                    )
                )
        ),
    )
)

# ============================================================
# ATTACH HISTORICAL + CURRENT QC
# ============================================================

series = series.merge(
    old_latest,
    how="left",
    on="loni_series_num",
    validate="many_to_one"
)

series = series.merge(
    current,
    how="left",
    on="series_uid_key",
    validate="many_to_one"
)

series["qc_source"] = "missing"

mask_old = series["old_selected"].notna()

series.loc[
    mask_old,
    "qc_source"
] = "Mayo2015"

mask_current = (
    ~mask_old
    &
    series["current_selected"].notna()
    &
    series["current_usable"].fillna(False)
)

series.loc[
    mask_current,
    "qc_source"
] = "CurrentQC"

series["official_selected"] = False
series["official_quality"] = pd.NA

series.loc[
    mask_old,
    "official_selected"
] = (
    series.loc[
        mask_old,
        "old_selected"
    ].eq(1)
)

series.loc[
    mask_old,
    "official_quality"
] = series.loc[
    mask_old,
    "old_quality"
]

series.loc[
    mask_current,
    "official_selected"
] = (
    series.loc[
        mask_current,
        "current_selected"
    ].eq(1)
)

series.loc[
    mask_current,
    "official_quality"
] = series.loc[
    mask_current,
    "current_quality"
]

series["official_quality"] = pd.to_numeric(
    series["official_quality"],
    errors="coerce"
)

series["eligible_accelerated"] = (
    series["acceleration"].eq("Accelerated")
    &
    series["official_selected"]
    &
    series["official_quality"].isin(
        [1, 2, 3]
    )
)

# ============================================================
# SUBJECT-LEVEL FINAL STATUS
# ============================================================

records = []

for rid, grp in series.groupby("RID"):

    eligible = grp[
        grp["eligible_accelerated"]
    ].copy()

    if len(eligible) == 1:

        selected = eligible.iloc[0]

        status = "selected"

        selected_loni_series = int(
            selected["loni_series_num"]
        )

        selected_quality = float(
            selected["official_quality"]
        )

        selected_source = (
            selected["qc_source"]
        )

        selected_image_ids = (
            selected["key_mri_image_ids"]
        )

    elif len(eligible) > 1:

        status = "multiple_distinct_accelerated_series"

        selected_loni_series = pd.NA
        selected_quality = pd.NA
        selected_source = pd.NA
        selected_image_ids = pd.NA

    else:

        accelerated = grp[
            grp["acceleration"]
            .eq("Accelerated")
        ]

        if accelerated.empty:
            status = "exclude_no_accelerated_candidate"

        elif accelerated["qc_source"].eq(
            "missing"
        ).all():
            status = "exclude_accelerated_qc_missing"

        elif not accelerated[
            "official_selected"
        ].any():
            status = "exclude_accelerated_not_selected"

        elif not (
            accelerated["official_selected"]
            &
            accelerated["official_quality"].isin(
                [1, 2, 3]
            )
        ).any():
            status = "exclude_accelerated_failed_qc"

        else:
            status = "exclude_other"

        selected_loni_series = pd.NA
        selected_quality = pd.NA
        selected_source = pd.NA
        selected_image_ids = pd.NA

    records.append(
        {
            "RID": int(rid),
            "diagnosis":
                grp["diagnosis"].iloc[0],

            "dx_phase":
                grp["dx_phase"].iloc[0],

            "status": status,

            "selected_loni_series":
                selected_loni_series,

            "selected_quality":
                selected_quality,

            "selected_qc_source":
                selected_source,

            "selected_key_mri_image_ids":
                selected_image_ids,
        }
    )

subject = pd.DataFrame(records)

# ============================================================
# VALIDATION
# ============================================================

if len(subject) != 574:
    raise RuntimeError(
        f"Expected 574 subjects, got {len(subject)}"
    )

selected_subjects = subject[
    subject["status"].eq("selected")
]

multiple = subject[
    subject["status"]
    .eq("multiple_distinct_accelerated_series")
]

# ============================================================
# SUMMARY
# ============================================================

status_counts = (
    subject["status"]
    .value_counts()
    .to_dict()
)

class_selected = (
    selected_subjects[
        "diagnosis"
    ]
    .value_counts()
    .to_dict()
)

summary = {
    "subjects_total": int(len(subject)),

    "selected_subjects":
        int(len(selected_subjects)),

    "status_counts": {
        str(k): int(v)
        for k, v in status_counts.items()
    },

    "selected_class_counts": {
        str(k): int(v)
        for k, v in class_selected.items()
    },

    "multiple_distinct_accelerated_series":
        int(len(multiple)),

    "subjects_with_duplicate_key_mri_rows_for_selected_series":
        int(
            selected_subjects[
                "selected_key_mri_image_ids"
            ]
            .astype(str)
            .str.contains(r"\|")
            .sum()
        ),
}

# ============================================================
# SAVE
# ============================================================

subject.to_csv(
    OUT_LOCAL,
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
print("FINAL GO/ADNI2 ACCELERATED SELECTION AUDIT")
print("=" * 80)

print(
    "\nTotal subjects:",
    summary["subjects_total"]
)

print(
    "Selected subjects:",
    summary["selected_subjects"]
)

print("\nStatus:")
for k, v in summary["status_counts"].items():
    print(f"  {k}: {v}")

print("\nClasses among selected subjects:")
for k, v in summary[
    "selected_class_counts"
].items():
    print(f"  {k}: {v}")

print(
    "\nMultiple DISTINCT accelerated LONI series:",
    summary[
        "multiple_distinct_accelerated_series"
    ]
)

print(
    "Selected subjects whose LONI series has >1 Key-MRI image_id:",
    summary[
        "subjects_with_duplicate_key_mri_rows_for_selected_series"
    ]
)

print(
    "\nSaved local selection table:\n",
    OUT_LOCAL
)

print(
    "\nSaved aggregate summary:\n",
    OUT_SUMMARY
)
