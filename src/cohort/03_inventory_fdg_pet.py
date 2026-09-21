from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PET_PATH = (
    ROOT / "data" / "raw" / "imaging_metadata" / "pet" / "extracted"
    / "FDG_PET_Inventory_21Sep2026_Key_PET_21Sep2026.csv"
)

ROSTER_PATH = (
    ROOT / "data" / "raw" / "metadata" / "extracted"
    / "ROSTER_18Sep2026.csv"
)

OUT_PATH = ROOT / "tables" / "fdg_pet_inventory_21Sep2026.json"


# ---------------------------------------------------
# Load
# ---------------------------------------------------

pet = pd.read_csv(PET_PATH, low_memory=False)
roster = pd.read_csv(ROSTER_PATH, low_memory=False)

pet["image_date"] = pd.to_datetime(
    pet["image_date"],
    errors="coerce"
)

roster["RID"] = pd.to_numeric(
    roster["RID"],
    errors="coerce"
).astype("Int64")


# ---------------------------------------------------
# Official PTID -> RID mapping
# ---------------------------------------------------

ptid_rid = (
    roster[["PTID", "RID"]]
    .dropna()
    .drop_duplicates()
)

conflicts = (
    ptid_rid.groupby("PTID")["RID"]
    .nunique()
    .gt(1)
    .sum()
)

if conflicts:
    raise RuntimeError(
        f"ROSTER contains {conflicts} PTID -> multiple RID conflicts"
    )

ptid_rid = ptid_rid.drop_duplicates("PTID")

pet = pet.merge(
    ptid_rid,
    how="left",
    left_on="subject_id",
    right_on="PTID",
    validate="many_to_one"
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def counts(series):
    return {
        str(k): int(v)
        for k, v in series.value_counts(dropna=False).items()
    }


summary = {
    "source": PET_PATH.name,
    "rows": int(len(pet)),
    "unique_subject_ids": int(pet["subject_id"].nunique()),
    "unique_rids": int(pet["RID"].nunique(dropna=True)),
    "missing_rid_after_roster_join": int(pet["RID"].isna().sum()),
    "unique_image_ids": int(pet["image_id"].nunique()),
    "duplicate_image_id_rows": int(
        pet.duplicated("image_id", keep=False).sum()
    ),
    "missing_image_date": int(pet["image_date"].isna().sum()),
}


# ---------------------------------------------------
# Dates
# ---------------------------------------------------

valid_dates = pet["image_date"].dropna()

summary["image_date_min"] = (
    str(valid_dates.min().date())
    if len(valid_dates) else None
)

summary["image_date_max"] = (
    str(valid_dates.max().date())
    if len(valid_dates) else None
)


# ---------------------------------------------------
# Core PET fields
# ---------------------------------------------------

for field in [
    "radiopharmaceutical",
    "pet_description",
    "tau_pet",
    "amyloid_pet",
]:
    if field in pet.columns:
        summary[field] = counts(pet[field])


# ---------------------------------------------------
# Multiplicity
# ---------------------------------------------------

visit_date_counts = (
    pet.groupby(
        ["subject_id", "image_visit", "image_date"],
        dropna=False
    )
    .size()
)

summary["subject_visit_date_groups"] = int(
    len(visit_date_counts)
)

summary["groups_with_multiple_fdg_images"] = int(
    (visit_date_counts > 1).sum()
)

summary["max_fdg_images_same_subject_visit_date"] = int(
    visit_date_counts.max()
)

summary["fdg_images_per_subject_visit_date"] = {
    str(k): int(v)
    for k, v in
    visit_date_counts.value_counts().sort_index().items()
}


# ---------------------------------------------------
# Visits per subject
# ---------------------------------------------------

visits_per_subject = (
    pet.groupby("subject_id")["image_visit"]
    .nunique(dropna=True)
)

summary["visits_per_subject"] = {
    "median": float(visits_per_subject.median()),
    "min": int(visits_per_subject.min()),
    "max": int(visits_per_subject.max()),
}


# ---------------------------------------------------
# Multiple dates for same subject + visit code
# ---------------------------------------------------

subject_visit_dates = (
    pet.groupby(
        ["subject_id", "image_visit"],
        dropna=False
    )["image_date"]
    .nunique(dropna=True)
)

summary["subject_visit_pairs_with_multiple_dates"] = int(
    (subject_visit_dates > 1).sum()
)


# ---------------------------------------------------
# Console report
# ---------------------------------------------------

print("\n" + "=" * 72)
print("FDG-PET INVENTORY")
print("=" * 72)

print("Rows:", summary["rows"])
print("Subjects:", summary["unique_subject_ids"])
print("Mapped RIDs:", summary["unique_rids"])
print(
    "Missing RID after ROSTER join:",
    summary["missing_rid_after_roster_join"]
)
print("Unique image IDs:", summary["unique_image_ids"])
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

print("\nRADIOPHARMACEUTICAL")
for k, v in summary["radiopharmaceutical"].items():
    print(f"{k}: {v}")

print("\nPET_DESCRIPTION")
for k, v in summary["pet_description"].items():
    print(f"{k}: {v}")

print("\nMULTIPLICITY")
print(
    "Subject/visit/date groups:",
    summary["subject_visit_date_groups"]
)
print(
    "Groups with >1 FDG image:",
    summary["groups_with_multiple_fdg_images"]
)
print(
    "Maximum images same subject/visit/date:",
    summary["max_fdg_images_same_subject_visit_date"]
)
print(
    "Distribution:",
    summary["fdg_images_per_subject_visit_date"]
)

print(
    "\nSubject/visit pairs with multiple scan dates:",
    summary["subject_visit_pairs_with_multiple_dates"]
)

print("\nVisits per subject:")
print(summary["visits_per_subject"])


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

print(f"\nSaved aggregate FDG-PET inventory to:\n{OUT_PATH}")
