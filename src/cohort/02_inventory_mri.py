from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MRI_PATH = (
    ROOT / "data" / "raw" / "imaging_metadata" / "mri" / "extracted"
    / "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

ROSTER_PATH = (
    ROOT / "data" / "raw" / "metadata" / "extracted"
    / "ROSTER_18Sep2026.csv"
)

OUT_PATH = ROOT / "tables" / "mri_inventory_19Sep2026.json"


# ---------------------------------------------------
# Load
# ---------------------------------------------------

mri = pd.read_csv(MRI_PATH, low_memory=False)
roster = pd.read_csv(ROSTER_PATH, low_memory=False)

mri["image_date"] = pd.to_datetime(
    mri["image_date"],
    errors="coerce"
)

roster["RID"] = pd.to_numeric(
    roster["RID"],
    errors="coerce"
).astype("Int64")


# ---------------------------------------------------
# PTID -> RID mapping from official ROSTER
# ---------------------------------------------------

ptid_rid = (
    roster[["PTID", "RID"]]
    .dropna()
    .drop_duplicates()
)

mapping_conflicts = (
    ptid_rid.groupby("PTID")["RID"]
    .nunique()
    .gt(1)
    .sum()
)

if mapping_conflicts:
    raise RuntimeError(
        f"ROSTER contains {mapping_conflicts} PTID -> multiple RID conflicts"
    )

ptid_rid = ptid_rid.drop_duplicates("PTID")

mri = mri.merge(
    ptid_rid,
    how="left",
    left_on="subject_id",
    right_on="PTID",
    validate="many_to_one"
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def value_counts_dict(series):
    counts = series.value_counts(dropna=False)
    return {
        str(k): int(v)
        for k, v in counts.items()
    }


summary = {
    "source": MRI_PATH.name,
    "rows": int(len(mri)),
    "columns": int(len(mri.columns)),
    "unique_subject_ids": int(mri["subject_id"].nunique()),
    "unique_rids": int(mri["RID"].nunique(dropna=True)),
    "missing_rid_after_roster_join": int(mri["RID"].isna().sum()),
    "unique_image_ids": int(mri["image_id"].nunique()),
    "duplicate_image_id_rows": int(
        mri.duplicated("image_id", keep=False).sum()
    ),
    "unique_series_instance_uid": int(
        mri["series_instance_uid"].nunique(dropna=True)
    ),
    "duplicate_series_uid_rows": int(
        mri.duplicated("series_instance_uid", keep=False).sum()
    ),
    "missing_image_date": int(mri["image_date"].isna().sum()),
}


# ---------------------------------------------------
# Dates
# ---------------------------------------------------

valid_dates = mri["image_date"].dropna()

summary["image_date_min"] = (
    str(valid_dates.min().date()) if len(valid_dates) else None
)

summary["image_date_max"] = (
    str(valid_dates.max().date()) if len(valid_dates) else None
)


# ---------------------------------------------------
# Core distributions
# ---------------------------------------------------

fields = [
    "series_type",
    "mri_protocol_phase",
    "acceleration",
    "acquisition_type",
    "acquisition_plane",
    "scanner_manufacturer",
    "scanner_model",
    "magnetic_field_strength",
    "series_description",
]

for field in fields:
    if field in mri.columns:
        summary[field] = value_counts_dict(mri[field])


# ---------------------------------------------------
# Multiplicity / repeated scans
# ---------------------------------------------------

visit_key = [
    "subject_id",
    "image_visit",
    "image_date",
]

visit_counts = (
    mri.groupby(visit_key, dropna=False)
    .size()
)

summary["subject_visit_date_groups"] = int(len(visit_counts))

summary["groups_with_multiple_t1w_series"] = int(
    (visit_counts > 1).sum()
)

summary["max_t1w_series_same_subject_visit_date"] = int(
    visit_counts.max()
)

summary["t1w_series_per_subject_visit_date"] = {
    str(k): int(v)
    for k, v in visit_counts.value_counts().sort_index().items()
}


# ---------------------------------------------------
# Subjects with repeated longitudinal visits
# ---------------------------------------------------

visits_per_subject = (
    mri.groupby("subject_id")["image_visit"]
    .nunique(dropna=True)
)

summary["visits_per_subject"] = {
    "median": float(visits_per_subject.median()),
    "min": int(visits_per_subject.min()),
    "max": int(visits_per_subject.max()),
}


# ---------------------------------------------------
# Protocol phase x field strength
# ---------------------------------------------------

phase_field = pd.crosstab(
    mri["mri_protocol_phase"],
    mri["magnetic_field_strength"],
    dropna=False,
)

summary["phase_by_field_strength"] = {
    str(idx): {
        str(col): int(value)
        for col, value in row.items()
    }
    for idx, row in phase_field.iterrows()
}


# ---------------------------------------------------
# Protocol phase x acquisition type
# ---------------------------------------------------

phase_acq = pd.crosstab(
    mri["mri_protocol_phase"],
    mri["acquisition_type"],
    dropna=False,
)

summary["phase_by_acquisition_type"] = {
    str(idx): {
        str(col): int(value)
        for col, value in row.items()
    }
    for idx, row in phase_acq.iterrows()
}


# ---------------------------------------------------
# Console
# ---------------------------------------------------

print("\n" + "=" * 72)
print("MRI T1W INVENTORY")
print("=" * 72)

print(f"Rows: {summary['rows']}")
print(f"Subjects: {summary['unique_subject_ids']}")
print(f"Mapped RIDs: {summary['unique_rids']}")
print(
    "Missing RID after ROSTER join:",
    summary["missing_rid_after_roster_join"]
)
print(f"Unique image IDs: {summary['unique_image_ids']}")
print(
    "Duplicate image_id rows:",
    summary["duplicate_image_id_rows"]
)
print(
    "Date range:",
    summary["image_date_min"],
    "to",
    summary["image_date_max"]
)

for field in [
    "mri_protocol_phase",
    "acceleration",
    "acquisition_type",
    "acquisition_plane",
    "magnetic_field_strength",
    "scanner_manufacturer",
]:
    print("\n" + field.upper())
    print("-" * len(field))
    for k, v in summary[field].items():
        print(f"{k}: {v}")

print("\nMULTIPLE T1W SERIES PER SUBJECT/VISIT/DATE")
print(
    "Groups with >1 T1w series:",
    summary["groups_with_multiple_t1w_series"]
)
print(
    "Maximum number:",
    summary["max_t1w_series_same_subject_visit_date"]
)
print(
    "Distribution:",
    summary["t1w_series_per_subject_visit_date"]
)

print("\nPHASE x FIELD STRENGTH")
print(phase_field.to_string())

print("\nPHASE x ACQUISITION TYPE")
print(phase_acq.to_string())


# ---------------------------------------------------
# Save aggregate-only report
# ---------------------------------------------------

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"\nSaved aggregate MRI inventory to:\n{OUT_PATH}")
