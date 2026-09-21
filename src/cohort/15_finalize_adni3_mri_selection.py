from pathlib import Path
import json
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[2]

QC_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/qc/adni3/"
    "MAYOADIRL_MRI_QUALITY_ADNI3.rda"
)

MRI_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/extracted/"
    "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

MANIFEST_PATH = (
    ROOT / "data/interim/cohort/"
    "primary_candidate_manifest_v1.csv"
)

OUT_SELECTION = (
    ROOT / "data/interim/cohort/"
    "adni3_mri_selection_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables/"
    "adni3_mri_selection_v1_summary.json"
)

# ============================================================
# LOAD
# ============================================================

qc = next(
    iter(
        pyreadr.read_r(
            str(QC_PATH)
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
# NORMALIZE MRI / MANIFEST
# ============================================================

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

# ============================================================
# NORMALIZE MAYO QC
# ============================================================

qc["loni_series_num"] = pd.to_numeric(
    qc["LONI_SERIES"],
    errors="coerce"
)

qc["qc_rid"] = pd.to_numeric(
    qc["RID"],
    errors="coerce"
)

qc["selected_num"] = (
    qc["SERIES_SELECTED"]
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

qc["quality_num"] = pd.to_numeric(
    qc["SERIES_QUALITY"],
    errors="coerce"
)

qc["protocol_status_num"] = pd.to_numeric(
    qc["PROTOCOL_STATUS"],
    errors="coerce"
)

qc["overallpass_num"] = pd.to_numeric(
    qc["STUDY_OVERALLPASS"],
    errors="coerce"
)

# ============================================================
# COLLAPSE QC TO DISTINCT LONI SERIES
# ============================================================

qc_series = (
    qc.dropna(
        subset=["loni_series_num"]
    )
    .groupby("loni_series_num")
    .agg(
        qc_rows=(
            "loni_series_num",
            "size"
        ),

        qc_rid_nunique=(
            "qc_rid",
            "nunique"
        ),

        qc_rid=(
            "qc_rid",
            "first"
        ),

        selected_nunique=(
            "selected_num",
            lambda x:
                x.dropna().nunique()
        ),

        selected_value=(
            "selected_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),

        quality_nunique=(
            "quality_num",
            lambda x:
                x.dropna().nunique()
        ),

        quality_value=(
            "quality_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),

        protocol_nunique=(
            "protocol_status_num",
            lambda x:
                x.dropna().nunique()
        ),

        protocol_value=(
            "protocol_status_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),

        overallpass_nunique=(
            "overallpass_num",
            lambda x:
                x.dropna().nunique()
        ),

        overallpass_value=(
            "overallpass_num",
            lambda x:
                x.dropna().iloc[0]
                if len(x.dropna())
                else pd.NA
        ),
    )
    .reset_index()
)

qc_series["qc_consistent"] = (
    qc_series["qc_rid_nunique"].le(1)
    &
    qc_series["selected_nunique"].le(1)
    &
    qc_series["quality_nunique"].le(1)
    &
    qc_series["protocol_nunique"].le(1)
    &
    qc_series["overallpass_nunique"].le(1)
)

# ============================================================
# EXACT ADNI3 PRIMARY OBSERVATIONS
# ============================================================

adni3 = manifest[
    manifest["mri_protocol_phases"]
    .astype(str)
    .eq("ADNI3")
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
    adni3,
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

print("=" * 82)
print("ADNI3 FINAL MRI SELECTION")
print("=" * 82)

print(
    "\nSubjects:",
    adni3["RID"].nunique()
)

print(
    "Candidate Key MRI rows:",
    len(target)
)

print(
    "Distinct candidate LONI series:",
    target["loni_series_num"].nunique()
)

# ============================================================
# COLLAPSE KEY MRI TO DISTINCT SERIES LEVEL
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

        image_id=("image_id", "first"),

        key_mri_rows=(
            "image_id",
            "size"
        ),

        series_description=(
            "series_description",
            "first"
        ),

        magnetic_field_strength=(
            "magnetic_field_strength",
            "first"
        ),

        series_instance_uid=(
            "series_instance_uid",
            "first"
        ),

        loni_image=(
            "loni_image",
            "first"
        ),
    )
)

# ============================================================
# MERGE OFFICIAL QC
# ============================================================

series = series.merge(
    qc_series,
    how="left",
    on="loni_series_num",
    validate="many_to_one"
)

series["qc_match"] = (
    series["qc_rows"].notna()
)

series["rid_match"] = (
    series["qc_match"]
    &
    series["qc_rid"].eq(
        pd.to_numeric(
            series["RID"],
            errors="coerce"
        )
    )
)

# Strict primary ADNI3 rule
# Primary ADNI3 QC rule.
#
# PROTOCOL_STATUS is retained as descriptive QC metadata but is NOT used
# as an exclusion criterion because its 1/2/3 coding is not defined in
# the available ADNI data dictionary. SERIES_SELECTED, SERIES_QUALITY,
# and STUDY_OVERALLPASS provide the explicit selection/quality criteria.
series["eligible"] = (
    series["qc_match"]
    &
    series["rid_match"]
    &
    series["qc_consistent"].fillna(False)
    &
    series["selected_value"].eq(1)
    &
    series["quality_value"].isin(
        [1, 2, 3]
    )
    &
    series["overallpass_value"].eq(1)
)

# ============================================================
# SERIES-LEVEL AUDIT
# ============================================================

print("\nSeries with exact Mayo QC match:",
      int(series["qc_match"].sum()))

print(
    "Series without Mayo QC match:",
    int((~series["qc_match"]).sum())
)

print(
    "QC matches with RID mismatch:",
    int(
        (
            series["qc_match"]
            &
            ~series["rid_match"]
        ).sum()
    )
)

print("\nSERIES_SELECTED among matched candidates:")
print(
    series.loc[
        series["qc_match"],
        "selected_value"
    ]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nSERIES_QUALITY among selected candidates:")
print(
    series.loc[
        series["selected_value"].eq(1),
        "quality_value"
    ]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nPROTOCOL_STATUS among selected candidates:")
print(
    series.loc[
        series["selected_value"].eq(1),
        "protocol_value"
    ]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nSTUDY_OVERALLPASS among selected candidates:")
print(
    series.loc[
        series["selected_value"].eq(1),
        "overallpass_value"
    ]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

# ============================================================
# SUBJECT-LEVEL FINAL SELECTION
# ============================================================

records = []

for rid, grp in series.groupby("RID"):

    eligible = grp[
        grp["eligible"]
    ].copy()

    if len(eligible) == 1:

        row = eligible.iloc[0]

        status = "selected"

        selected_loni_series = int(
            row["loni_series_num"]
        )

        selected_image_id = (
            row["image_id"]
        )

        selected_loni_image = (
            row["loni_image"]
        )

        selected_uid = (
            row["series_instance_uid"]
        )

        selected_description = (
            row["series_description"]
        )

        selected_quality = (
            row["quality_value"]
        )

    elif len(eligible) > 1:

        status = (
            "exclude_multiple_selected_passing_series"
        )

        selected_loni_series = pd.NA
        selected_image_id = pd.NA
        selected_loni_image = pd.NA
        selected_uid = pd.NA
        selected_description = pd.NA
        selected_quality = pd.NA

    else:

        if not grp["qc_match"].any():

            status = "exclude_qc_match_missing"

        elif not grp["selected_value"].eq(1).any():

            status = "exclude_no_series_selected"

        elif not (
            grp["selected_value"].eq(1)
            &
            grp["quality_value"].isin(
                [1, 2, 3]
            )
        ).any():

            status = "exclude_selected_series_failed_quality"

        elif not (
            grp["selected_value"].eq(1)
            &
            grp["protocol_value"].eq(1)
        ).any():

            status = "exclude_selected_series_protocol_fail"

        elif not (
            grp["selected_value"].eq(1)
            &
            grp["overallpass_value"].eq(1)
        ).any():

            status = "exclude_study_overall_fail"

        else:

            status = "exclude_other"

        selected_loni_series = pd.NA
        selected_image_id = pd.NA
        selected_loni_image = pd.NA
        selected_uid = pd.NA
        selected_description = pd.NA
        selected_quality = pd.NA

    records.append(
        {
            "RID": int(rid),

            "PTID":
                grp["PTID"].iloc[0],

            "diagnosis":
                grp["diagnosis"].iloc[0],

            "dx_phase":
                grp["dx_phase"].iloc[0],

            "mri_visit":
                grp["mri_visit"].iloc[0],

            "mri_date":
                grp["mri_date"].iloc[0],

            "status":
                status,

            "selected_loni_series":
                selected_loni_series,

            "selected_image_id":
                selected_image_id,

            "selected_loni_image":
                selected_loni_image,

            "selected_series_instance_uid":
                selected_uid,

            "selected_series_description":
                selected_description,

            "selected_quality":
                selected_quality,
        }
    )

selection = pd.DataFrame(records)

# ============================================================
# VALIDATION
# ============================================================

if len(selection) != 104:
    raise RuntimeError(
        f"Expected 104 ADNI3 subjects, "
        f"got {len(selection)}"
    )

selected = selection[
    selection["status"].eq(
        "selected"
    )
].copy()

excluded = selection[
    ~selection["status"].eq(
        "selected"
    )
].copy()

if selected[
    "selected_loni_series"
].isna().any():
    raise RuntimeError(
        "Selected ADNI3 subject without LONI series."
    )

if selected[
    "selected_loni_series"
].duplicated().any():
    raise RuntimeError(
        "Selected ADNI3 LONI series reused across subjects."
    )

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "subjects_total":
        int(len(selection)),

    "subjects_selected":
        int(len(selected)),

    "subjects_excluded":
        int(len(excluded)),

    "status_counts": {
        str(k): int(v)
        for k, v in
        selection[
            "status"
        ]
        .value_counts()
        .to_dict()
        .items()
    },

    "selected_class_counts": {
        str(k): int(v)
        for k, v in
        selected[
            "diagnosis"
        ]
        .value_counts()
        .to_dict()
        .items()
    },

    "excluded_class_counts": {
        str(k): int(v)
        for k, v in
        excluded[
            "diagnosis"
        ]
        .value_counts()
        .to_dict()
        .items()
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
# FINAL REPORT
# ============================================================

print("\n" + "=" * 82)
print("FINAL SUBJECT-LEVEL ADNI3 SELECTION")
print("=" * 82)

print(
    "\nTotal:",
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
