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

OUT_DIR = ROOT / "data/interim/cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANIFEST = (
    OUT_DIR / "primary_candidate_manifest_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables/primary_candidate_manifest_v1_summary.json"
)

MRI_PET_MAX_DAYS = 30
PET_DX_MAX_DAYS = 30

EXPECTED_N = 882


# ============================================================
# LOAD
# ============================================================

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


# ============================================================
# OFFICIAL PTID -> RID
# ============================================================

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
        f"{conflicts} PTIDs map to multiple RIDs."
    )

mapping = mapping.drop_duplicates("PTID")


def attach_rid(df):
    out = df.merge(
        mapping,
        left_on="subject_id",
        right_on="PTID",
        how="left",
        validate="many_to_one"
    )

    if out["RID"].isna().any():
        raise RuntimeError(
            "Some imaging rows could not be mapped to RID."
        )

    return out


mri = attach_rid(mri)
pet = attach_rid(pet)


# ============================================================
# VALID DIAGNOSES ONLY
# ============================================================

dx = dx[
    dx["DIAGNOSIS"].isin([1, 2, 3])
].dropna(
    subset=["RID", "VISCODE2", "EXAMDATE"]
).copy()


# ============================================================
# MRI ACQUISITION OPPORTUNITIES
#
# Do NOT select an MRI series yet.
# Multiple T1w series on the same visit/date remain candidates.
# ============================================================

mri_opportunities = (
    mri[
        [
            "RID",
            "subject_id",
            "image_visit",
            "image_date",
        ]
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

mri_by_rid = {
    int(rid): grp.copy()
    for rid, grp in mri_opportunities.groupby("RID")
}

dx_by_key = {
    (int(rid), str(vis)): grp.copy()
    for (rid, vis), grp in
    dx.groupby(["RID", "VISCODE2"])
}


# ============================================================
# BUILD ALL ELIGIBLE PET-CENTERED OBSERVATIONS
# ============================================================

records = []
ambiguous_dx_ties = 0

for prow in pet.itertuples(index=False):

    rid = int(prow.RID)

    # --------------------------------------------------------
    # MRI matching
    # --------------------------------------------------------

    mri_candidates = mri_by_rid.get(rid)

    if mri_candidates is None:
        continue

    mc = mri_candidates.copy()

    mc["mri_pet_abs_days"] = (
        mc["mri_date"] - prow.image_date
    ).dt.days.abs()

    mc = mc[
        mc["mri_pet_abs_days"]
        <= MRI_PET_MAX_DAYS
    ].copy()

    if mc.empty:
        continue

    # Primary MRI acquisition opportunity:
    # 1. minimum absolute temporal distance
    # 2. prefer matching visit code if tied
    # 3. earlier acquisition date
    # 4. lexical visit-code tie-break
    mc["same_visit"] = (
        mc["mri_visit"].astype(str)
        == str(prow.image_visit)
    )

    mc = mc.sort_values(
        [
            "mri_pet_abs_days",
            "same_visit",
            "mri_date",
            "mri_visit",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ]
    )

    selected_mri = mc.iloc[0]

    # --------------------------------------------------------
    # Diagnosis matching
    # --------------------------------------------------------

    key = (
        rid,
        str(prow.image_visit)
    )

    dg = dx_by_key.get(key)

    if dg is None or dg.empty:
        continue

    dg = dg.copy()

    dg["pet_dx_abs_days"] = (
        dg["EXAMDATE"] - prow.image_date
    ).dt.days.abs()

    dg = dg[
        dg["pet_dx_abs_days"]
        <= PET_DX_MAX_DAYS
    ].copy()

    if dg.empty:
        continue

    min_dx_gap = dg["pet_dx_abs_days"].min()

    nearest_dx = dg[
        dg["pet_dx_abs_days"] == min_dx_gap
    ].copy()

    # Nearest equally-distant records must agree in class.
    if nearest_dx["DIAGNOSIS"].nunique() != 1:
        ambiguous_dx_ties += 1
        continue

    selected_dx = (
        nearest_dx
        .sort_values(
            ["EXAMDATE", "PHASE"]
        )
        .iloc[0]
    )

    records.append(
        {
            "RID": rid,
            "PTID": prow.subject_id,

            "pet_image_id": prow.image_id,
            "pet_visit": prow.image_visit,
            "pet_date": prow.image_date,
            "radiopharmaceutical":
                prow.radiopharmaceutical,
            "pet_description":
                prow.pet_description,

            "mri_visit":
                selected_mri["mri_visit"],
            "mri_date":
                selected_mri["mri_date"],
            "mri_pet_abs_days":
                int(
                    selected_mri[
                        "mri_pet_abs_days"
                    ]
                ),

            "mri_visit_matches_pet":
                bool(
                    selected_mri[
                        "same_visit"
                    ]
                ),

            "dx_visit":
                selected_dx["VISCODE2"],
            "dx_date":
                selected_dx["EXAMDATE"],
            "pet_dx_abs_days":
                int(
                    selected_dx[
                        "pet_dx_abs_days"
                    ]
                ),

            "diagnosis_code":
                int(
                    selected_dx["DIAGNOSIS"]
                ),

            "dx_phase":
                selected_dx["PHASE"],
        }
    )

eligible = pd.DataFrame(records)


# ============================================================
# ONE INDEX OBSERVATION PER RID
#
# Earliest eligible PET-centered observation.
# ============================================================

manifest = (
    eligible
    .sort_values(
        [
            "RID",
            "pet_date",
            "mri_pet_abs_days",
            "pet_image_id",
        ]
    )
    .drop_duplicates(
        "RID",
        keep="first"
    )
    .reset_index(drop=True)
)


# ============================================================
# DIAGNOSIS LABEL
# ============================================================

LABELS = {
    1: "CN",
    2: "MCI",
    3: "Dementia",
}

manifest["diagnosis"] = (
    manifest["diagnosis_code"]
    .map(LABELS)
)


# ============================================================
# MRI SERIES STILL AVAILABLE ON SELECTED DATE/VISIT
#
# This is intentionally NOT a scan-selection step.
# ============================================================

mri_series = mri.rename(
    columns={
        "image_id": "mri_image_id",
        "image_visit": "mri_visit",
        "image_date": "mri_date",
    }
)

candidate_counts = (
    mri_series
    .groupby(
        [
            "RID",
            "mri_visit",
            "mri_date",
        ],
        dropna=False
    )
    .agg(
        mri_candidate_series_count=(
            "mri_image_id",
            "size"
        ),

        mri_candidate_image_ids=(
            "mri_image_id",
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

        mri_protocol_phases=(
            "mri_protocol_phase",
            lambda x:
                "|".join(
                    sorted(
                        {
                            str(v)
                            for v in x.dropna()
                        }
                    )
                )
        ),
    )
    .reset_index()
)

manifest = manifest.merge(
    candidate_counts,
    how="left",
    on=[
        "RID",
        "mri_visit",
        "mri_date",
    ],
    validate="one_to_one"
)


# ============================================================
# AUDIT CONSTANTS
# ============================================================

manifest["cohort_version"] = (
    "primary_candidate_v1"
)

manifest["mri_pet_max_days"] = (
    MRI_PET_MAX_DAYS
)

manifest["pet_dx_max_days"] = (
    PET_DX_MAX_DAYS
)

manifest["index_rule"] = (
    "earliest_eligible_pet_centered_observation"
)


# ============================================================
# HARD VALIDATION
# ============================================================

if len(manifest) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} participants "
        f"but obtained {len(manifest)}. "
        "STOP: investigate before proceeding."
    )

if manifest["RID"].duplicated().any():
    raise RuntimeError(
        "Duplicate RID in primary manifest."
    )

if manifest["pet_image_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate PET image_id in primary manifest."
    )

if (
    manifest["mri_pet_abs_days"]
    > MRI_PET_MAX_DAYS
).any():
    raise RuntimeError(
        "MRI-PET temporal threshold violation."
    )

if (
    manifest["pet_dx_abs_days"]
    > PET_DX_MAX_DAYS
).any():
    raise RuntimeError(
        "PET-diagnosis temporal threshold violation."
    )

if manifest["diagnosis"].isna().any():
    raise RuntimeError(
        "Missing diagnosis label."
    )

if (
    manifest["mri_candidate_series_count"]
    .isna()
    .any()
):
    raise RuntimeError(
        "MRI candidate-series lookup failed."
    )


# ============================================================
# SUMMARY
# ============================================================

class_counts = (
    manifest["diagnosis"]
    .value_counts()
    .to_dict()
)

phase_counts = (
    manifest["dx_phase"]
    .value_counts()
    .to_dict()
)

candidate_series_distribution = (
    manifest[
        "mri_candidate_series_count"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
)

summary = {
    "cohort_version":
        "primary_candidate_v1",

    "participants":
        int(len(manifest)),

    "unique_rids":
        int(manifest["RID"].nunique()),

    "class_counts": {
        str(k): int(v)
        for k, v in class_counts.items()
    },

    "diagnosis_phase_counts": {
        str(k): int(v)
        for k, v in phase_counts.items()
    },

    "mri_pet_gap_days": {
        "median":
            float(
                manifest[
                    "mri_pet_abs_days"
                ].median()
            ),

        "p90":
            float(
                manifest[
                    "mri_pet_abs_days"
                ].quantile(.90)
            ),

        "max":
            int(
                manifest[
                    "mri_pet_abs_days"
                ].max()
            ),
    },

    "pet_dx_gap_days": {
        "median":
            float(
                manifest[
                    "pet_dx_abs_days"
                ].median()
            ),

        "p90":
            float(
                manifest[
                    "pet_dx_abs_days"
                ].quantile(.90)
            ),

        "max":
            int(
                manifest[
                    "pet_dx_abs_days"
                ].max()
            ),
    },

    "mri_visit_matches_pet_subjects":
        int(
            manifest[
                "mri_visit_matches_pet"
            ].sum()
        ),

    "mri_candidate_series_distribution": {
        str(k): int(v)
        for k, v in
        candidate_series_distribution.items()
    },

    "subjects_with_multiple_mri_series":
        int(
            (
                manifest[
                    "mri_candidate_series_count"
                ] > 1
            ).sum()
        ),

    "ambiguous_dx_ties_excluded":
        int(ambiguous_dx_ties),
}


# ============================================================
# SAVE
# ============================================================

manifest.to_csv(
    OUT_MANIFEST,
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
# CONSOLE
# ============================================================

print("\n" + "=" * 76)
print("PRIMARY CANDIDATE MANIFEST V1")
print("=" * 76)

print(
    "Participants:",
    summary["participants"]
)

print(
    "Classes:",
    summary["class_counts"]
)

print(
    "Diagnosis phases:",
    summary["diagnosis_phase_counts"]
)

print(
    "\nMRI-PET gap:",
    summary["mri_pet_gap_days"]
)

print(
    "PET-DX gap:",
    summary["pet_dx_gap_days"]
)

print(
    "\nMRI visit matches PET:",
    summary[
        "mri_visit_matches_pet_subjects"
    ]
)

print(
    "\nMRI candidate-series distribution:",
    summary[
        "mri_candidate_series_distribution"
    ]
)

print(
    "Subjects with >1 MRI candidate series:",
    summary[
        "subjects_with_multiple_mri_series"
    ]
)

print(
    "\nAmbiguous diagnosis ties excluded:",
    summary[
        "ambiguous_dx_ties_excluded"
    ]
)

print(
    "\nSaved local participant-level manifest:\n"
    f"{OUT_MANIFEST}"
)

print(
    "\nSaved Git-safe aggregate summary:\n"
    f"{OUT_SUMMARY}"
)

print(
    "\nSUCCESS: primary candidate cohort "
    "contains exactly 882 unique RIDs."
)
