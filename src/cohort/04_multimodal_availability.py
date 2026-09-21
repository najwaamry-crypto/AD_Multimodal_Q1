from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MRI_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/extracted"
    / "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

PET_PATH = (
    ROOT / "data/raw/imaging_metadata/pet/extracted"
    / "FDG_PET_Inventory_21Sep2026_Key_PET_21Sep2026.csv"
)

ROSTER_PATH = (
    ROOT / "data/raw/metadata/extracted"
    / "ROSTER_18Sep2026.csv"
)

DX_PATH = (
    ROOT / "data/raw/metadata/extracted"
    / "DXSUM_18Sep2026.csv"
)

OUT_JSON = ROOT / "tables/multimodal_availability_21Sep2026.json"

LOCAL_OUT_DIR = ROOT / "data/interim/cohort"
LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_CANDIDATES = (
    LOCAL_OUT_DIR / "mri_fdg_candidate_pairs_21Sep2026.csv"
)


# ===================================================
# Load
# ===================================================

mri = pd.read_csv(MRI_PATH, low_memory=False)
pet = pd.read_csv(PET_PATH, low_memory=False)
roster = pd.read_csv(ROSTER_PATH, low_memory=False)
dx = pd.read_csv(DX_PATH, low_memory=False)

mri["image_date"] = pd.to_datetime(
    mri["image_date"], errors="coerce"
)

pet["image_date"] = pd.to_datetime(
    pet["image_date"], errors="coerce"
)

dx["EXAMDATE"] = pd.to_datetime(
    dx["EXAMDATE"], errors="coerce"
)

roster["RID"] = pd.to_numeric(
    roster["RID"], errors="coerce"
).astype("Int64")

dx["RID"] = pd.to_numeric(
    dx["RID"], errors="coerce"
).astype("Int64")

dx["DIAGNOSIS"] = pd.to_numeric(
    dx["DIAGNOSIS"], errors="coerce"
).astype("Int64")


# ===================================================
# Official PTID -> RID mapping
# ===================================================

mapping = (
    roster[["PTID", "RID"]]
    .dropna()
    .drop_duplicates()
)

conflicts = (
    mapping.groupby("PTID")["RID"]
    .nunique()
    .gt(1)
    .sum()
)

if conflicts:
    raise RuntimeError(
        f"{conflicts} PTID values map to multiple RIDs"
    )

mapping = mapping.drop_duplicates("PTID")


def attach_rid(df):
    return df.merge(
        mapping,
        how="left",
        left_on="subject_id",
        right_on="PTID",
        validate="many_to_one"
    )


mri = attach_rid(mri)
pet = attach_rid(pet)


# ===================================================
# Collapse MRI to acquisition opportunities
#
# We deliberately do NOT choose one MRI series yet.
# Multiple T1w series acquired on the same subject,
# visit and date represent one MRI opportunity here.
# ===================================================

mri_dates = (
    mri[
        ["RID", "subject_id", "image_visit", "image_date"]
    ]
    .dropna(subset=["RID", "image_date"])
    .drop_duplicates()
    .rename(
        columns={
            "image_visit": "mri_visit",
            "image_date": "mri_date",
        }
    )
)

pet_dates = (
    pet[
        [
            "RID",
            "subject_id",
            "image_id",
            "image_visit",
            "image_date",
        ]
    ]
    .dropna(subset=["RID", "image_date"])
    .drop_duplicates()
    .rename(
        columns={
            "image_id": "pet_image_id",
            "image_visit": "pet_visit",
            "image_date": "pet_date",
        }
    )
)


# ===================================================
# Subject-level overlap
# ===================================================

mri_rids = set(mri_dates["RID"].dropna().astype(int))
pet_rids = set(pet_dates["RID"].dropna().astype(int))

valid_dx = dx[
    dx["DIAGNOSIS"].isin([1, 2, 3])
].copy()

dx_rids = set(valid_dx["RID"].dropna().astype(int))

mri_pet_rids = mri_rids & pet_rids
mri_pet_dx_rids = mri_pet_rids & dx_rids


# ===================================================
# Diagnosis consistency by RID + VISCODE2
# ===================================================

dx_visit = valid_dx[
    ["RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS"]
].dropna(subset=["RID", "VISCODE2"])

class_counts = (
    dx_visit
    .groupby(["RID", "VISCODE2"])["DIAGNOSIS"]
    .nunique()
)

ambiguous_dx_keys = int((class_counts > 1).sum())

# Only keys with one unique valid class are safe to use
safe_keys = class_counts[class_counts == 1].index

dx_safe = (
    dx_visit
    .set_index(["RID", "VISCODE2"])
    .loc[safe_keys]
    .reset_index()
)

# If duplicate rows exist with same diagnosis,
# keep one representative row only for coverage audit.
dx_safe = (
    dx_safe
    .sort_values("EXAMDATE")
    .drop_duplicates(
        ["RID", "VISCODE2"],
        keep="first"
    )
    .rename(
        columns={
            "VISCODE2": "pet_visit",
            "EXAMDATE": "dx_date",
        }
    )
)


# ===================================================
# For every PET exam:
# nearest MRI date on same RID
# and nearest MRI date with same visit code
# ===================================================

mri_by_rid = {
    int(rid): grp.copy()
    for rid, grp in mri_dates.groupby("RID")
}

records = []

for row in pet_dates.itertuples(index=False):

    rid = int(row.RID)

    if rid not in mri_by_rid:
        continue

    candidates = mri_by_rid[rid].copy()

    candidates["abs_days"] = (
        candidates["mri_date"] - row.pet_date
    ).dt.days.abs()

    min_gap = int(candidates["abs_days"].min())

    nearest = candidates[
        candidates["abs_days"] == min_gap
    ].sort_values(["mri_date", "mri_visit"])

    chosen_for_audit = nearest.iloc[0]

    same_visit = candidates[
        candidates["mri_visit"].astype(str)
        == str(row.pet_visit)
    ].copy()

    if len(same_visit):
        same_visit["abs_days"] = (
            same_visit["mri_date"] - row.pet_date
        ).dt.days.abs()

        same_visit_gap = int(
            same_visit["abs_days"].min()
        )

        same_visit_exists = True
    else:
        same_visit_gap = np.nan
        same_visit_exists = False

    records.append(
        {
            "RID": rid,
            "subject_id": row.subject_id,
            "pet_image_id": row.pet_image_id,
            "pet_visit": row.pet_visit,
            "pet_date": row.pet_date,
            "nearest_mri_visit": chosen_for_audit["mri_visit"],
            "nearest_mri_date": chosen_for_audit["mri_date"],
            "mri_pet_abs_days": min_gap,
            "nearest_mri_date_ties": int(len(nearest)),
            "same_visit_mri_available": same_visit_exists,
            "same_visit_mri_abs_days": same_visit_gap,
        }
    )

pairs = pd.DataFrame(records)


# ===================================================
# Exact PET-visit diagnostic availability
# ===================================================

pairs = pairs.merge(
    dx_safe[
        [
            "RID",
            "pet_visit",
            "dx_date",
            "DIAGNOSIS",
        ]
    ],
    how="left",
    on=["RID", "pet_visit"],
    validate="many_to_one"
)

pairs["has_exact_visit_diagnosis"] = (
    pairs["DIAGNOSIS"].isin([1, 2, 3])
)

pairs["pet_dx_abs_days"] = (
    pairs["pet_date"] - pairs["dx_date"]
).dt.days.abs()


# ===================================================
# Aggregate statistics
# ===================================================

thresholds = [0, 7, 14, 30, 60, 90, 180, 365]

threshold_counts = {}

for d in thresholds:
    eligible = pairs["mri_pet_abs_days"] <= d

    threshold_counts[str(d)] = {
        "pet_exams": int(eligible.sum()),
        "unique_rids": int(
            pairs.loc[eligible, "RID"].nunique()
        ),
        "pet_exams_with_exact_visit_diagnosis": int(
            (
                eligible
                & pairs["has_exact_visit_diagnosis"]
            ).sum()
        ),
        "unique_rids_with_exact_visit_diagnosis": int(
            pairs.loc[
                eligible
                & pairs["has_exact_visit_diagnosis"],
                "RID"
            ].nunique()
        ),
    }


summary = {
    "mri_unique_rids": len(mri_rids),
    "pet_unique_rids": len(pet_rids),
    "valid_diagnosis_unique_rids": len(dx_rids),

    "mri_pet_subject_overlap": len(mri_pet_rids),
    "mri_pet_diagnosis_subject_overlap": len(
        mri_pet_dx_rids
    ),

    "pet_exams_with_any_mri": int(len(pairs)),

    "pet_exams_with_same_visit_mri": int(
        pairs["same_visit_mri_available"].sum()
    ),

    "unique_rids_with_same_visit_mri": int(
        pairs.loc[
            pairs["same_visit_mri_available"],
            "RID"
        ].nunique()
    ),

    "ambiguous_valid_diagnosis_rid_visit_keys": (
        ambiguous_dx_keys
    ),

    "pet_exams_with_exact_visit_diagnosis": int(
        pairs["has_exact_visit_diagnosis"].sum()
    ),

    "unique_rids_with_exact_visit_diagnosis": int(
        pairs.loc[
            pairs["has_exact_visit_diagnosis"],
            "RID"
        ].nunique()
    ),

    "nearest_mri_date_ties_gt1": int(
        (pairs["nearest_mri_date_ties"] > 1).sum()
    ),

    "mri_pet_gap_days": {
        "median": float(
            pairs["mri_pet_abs_days"].median()
        ),
        "mean": float(
            pairs["mri_pet_abs_days"].mean()
        ),
        "p25": float(
            pairs["mri_pet_abs_days"].quantile(0.25)
        ),
        "p75": float(
            pairs["mri_pet_abs_days"].quantile(0.75)
        ),
        "p90": float(
            pairs["mri_pet_abs_days"].quantile(0.90)
        ),
        "max": int(
            pairs["mri_pet_abs_days"].max()
        ),
    },

    "thresholds_days": threshold_counts,
}


# ===================================================
# Console
# ===================================================

print("\n" + "=" * 72)
print("MULTIMODAL AVAILABILITY AUDIT")
print("=" * 72)

print("MRI subjects:", summary["mri_unique_rids"])
print("FDG-PET subjects:", summary["pet_unique_rids"])
print(
    "Subjects with valid diagnosis:",
    summary["valid_diagnosis_unique_rids"]
)

print(
    "\nMRI ∩ PET subjects:",
    summary["mri_pet_subject_overlap"]
)

print(
    "MRI ∩ PET ∩ any valid diagnosis subjects:",
    summary["mri_pet_diagnosis_subject_overlap"]
)

print(
    "\nPET exams having at least one MRI:",
    summary["pet_exams_with_any_mri"]
)

print(
    "PET exams with same-visit MRI:",
    summary["pet_exams_with_same_visit_mri"]
)

print(
    "Subjects with same-visit MRI:",
    summary["unique_rids_with_same_visit_mri"]
)

print(
    "\nPET exams with exact-visit CN/MCI/Dementia:",
    summary["pet_exams_with_exact_visit_diagnosis"]
)

print(
    "Subjects with exact-visit CN/MCI/Dementia:",
    summary["unique_rids_with_exact_visit_diagnosis"]
)

print(
    "\nAmbiguous RID/visit diagnosis keys:",
    summary["ambiguous_valid_diagnosis_rid_visit_keys"]
)

print("\nMRI-PET nearest-date gap:")
for k, v in summary["mri_pet_gap_days"].items():
    print(f"  {k}: {v}")

print("\nTEMPORAL THRESHOLDS")
print("-" * 72)

for d, stats in threshold_counts.items():
    print(
        f"<= {d:>3} days | "
        f"PET exams={stats['pet_exams']:>4} | "
        f"RIDs={stats['unique_rids']:>4} | "
        f"with exact dx RIDs="
        f"{stats['unique_rids_with_exact_visit_diagnosis']:>4}"
    )


# ===================================================
# Save
#
# Aggregate report -> Git-safe
# Participant-level candidate file -> local data only
# ===================================================

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

pairs.to_csv(
    LOCAL_CANDIDATES,
    index=False
)

print(
    "\nSaved aggregate report to:\n"
    f"{OUT_JSON}"
)

print(
    "\nSaved participant-level candidate table locally to:\n"
    f"{LOCAL_CANDIDATES}"
)

print(
    "\nIMPORTANT: candidate table is exploratory only. "
    "No temporal threshold has been selected."
)
