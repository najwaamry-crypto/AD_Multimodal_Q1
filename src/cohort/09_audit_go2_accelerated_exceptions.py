from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

QC_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/qc/go2/extracted"
    / "MAYOADIRL_MRI_IMAGEQC_05_07_15_21Sep2026.csv"
)

MRI_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/extracted"
    / "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

MANIFEST_PATH = (
    ROOT / "data/interim/cohort"
    / "primary_candidate_manifest_v1.csv"
)

qc = pd.read_csv(QC_PATH, low_memory=False)
mri = pd.read_csv(MRI_PATH, low_memory=False)
manifest = pd.read_csv(MANIFEST_PATH, low_memory=False)

# ------------------------------------------------------------
# Normalize
# ------------------------------------------------------------

qc["loni_series_num"] = pd.to_numeric(
    qc["loni_series"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False),
    errors="coerce"
)

qc["selected_num"] = pd.to_numeric(
    qc["series_selected"],
    errors="coerce"
)

qc["quality_num"] = pd.to_numeric(
    qc["series_quality"],
    errors="coerce"
)

qc["update_dt"] = pd.to_datetime(
    qc["update_stamp"],
    errors="coerce"
)

qc["series_number_num"] = pd.to_numeric(
    qc["series_number"],
    errors="coerce"
)

# latest historical state per LONI series
qc_latest = (
    qc.sort_values(
        ["loni_series_num", "update_dt"],
        na_position="first"
    )
    .drop_duplicates(
        "loni_series_num",
        keep="last"
    )
)

mri["loni_series_num"] = pd.to_numeric(
    mri["loni_series"],
    errors="coerce"
)

mri["image_date"] = pd.to_datetime(
    mri["image_date"],
    errors="coerce"
)

manifest["mri_date"] = pd.to_datetime(
    manifest["mri_date"],
    errors="coerce"
)

# ------------------------------------------------------------
# Exact GO/ADNI2 observations
# ------------------------------------------------------------

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

target = target.merge(
    qc_latest[
        [
            "loni_series_num",
            "selected_num",
            "quality_num",
            "series_number_num",
            "series_time",
            "series_description",
            "T1_ACCELERATED",
            "update_stamp",
        ]
    ],
    how="left",
    on="loni_series_num",
    suffixes=("_keymri", "_qc"),
    validate="many_to_one"
)

target["selected_passing_accelerated"] = (
    target["acceleration"].eq("Accelerated")
    & target["selected_num"].eq(1)
    & target["quality_num"].isin([1, 2, 3])
)

counts = (
    target.groupby("RID")
    ["selected_passing_accelerated"]
    .sum()
)

zero_rids = counts[
    counts == 0
].index

multi_rids = counts[
    counts > 1
].index

print("=" * 84)
print("GO/ADNI2 ACCELERATED EXCEPTIONS")
print("=" * 84)

print("\nExactly one accelerated:",
      int((counts == 1).sum()))

print("Zero accelerated:",
      len(zero_rids))

print("Multiple accelerated:",
      len(multi_rids))

# ------------------------------------------------------------
# MULTIPLE selected accelerated
# ------------------------------------------------------------

print("\n" + "=" * 84)
print("MULTIPLE SELECTED ACCELERATED CASES")
print("=" * 84)

multi = target[
    target["RID"].isin(multi_rids)
    & target["acceleration"].eq("Accelerated")
].copy()

cols = [
    "RID",
    "PTID",
    "diagnosis",
    "mri_visit",
    "mri_date",
    "image_id",
    "series_description_keymri",
    "loni_series",
    "selected_num",
    "quality_num",
    "series_number_num",
    "series_time",
    "series_description_qc",
    "T1_ACCELERATED",
    "update_stamp",
]

cols = [c for c in cols if c in multi.columns]

print(
    multi[cols]
    .sort_values(
        [
            "RID",
            "quality_num",
            "series_number_num",
        ]
    )
    .to_string(index=False)
)

# ------------------------------------------------------------
# ZERO accelerated
# ------------------------------------------------------------

print("\n" + "=" * 84)
print("ZERO SELECTED PASSING ACCELERATED CASES")
print("=" * 84)

zero = target[
    target["RID"].isin(zero_rids)
].copy()

cols = [
    "RID",
    "PTID",
    "diagnosis",
    "dx_phase",
    "mri_visit",
    "mri_date",
    "image_id",
    "series_description_keymri",
    "acceleration",
    "loni_series",
    "selected_num",
    "quality_num",
    "series_number_num",
    "series_time",
    "series_description_qc",
    "T1_ACCELERATED",
]

cols = [c for c in cols if c in zero.columns]

print(
    zero[cols]
    .sort_values(
        [
            "RID",
            "acceleration",
            "series_number_num",
        ]
    )
    .to_string(index=False)
)

# ------------------------------------------------------------
# Reason categories for zero cases
# ------------------------------------------------------------

records = []

for rid, grp in zero.groupby("RID"):

    accelerated = grp[
        grp["acceleration"]
        .eq("Accelerated")
    ]

    if accelerated.empty:
        reason = "no_accelerated_candidate"

    elif accelerated["selected_num"].isna().all():
        reason = "accelerated_qc_missing"

    elif not accelerated["selected_num"].eq(1).any():
        reason = "accelerated_not_selected"

    elif not (
        accelerated["selected_num"].eq(1)
        & accelerated["quality_num"].isin([1, 2, 3])
    ).any():
        reason = "accelerated_selected_but_failed_qc"

    else:
        reason = "other"

    records.append(
        {
            "RID": rid,
            "diagnosis":
                grp["diagnosis"].iloc[0],
            "reason": reason,
        }
    )

reasons = pd.DataFrame(records)

print("\n" + "=" * 84)
print("ZERO-ACCELERATED REASON SUMMARY")
print("=" * 84)

print(
    reasons["reason"]
    .value_counts()
    .to_string()
)

print("\nReason x diagnosis:")
print(
    pd.crosstab(
        reasons["reason"],
        reasons["diagnosis"],
        margins=True
    ).to_string()
)
