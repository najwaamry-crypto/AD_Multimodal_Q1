from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PRIMARY_PATH = (
    ROOT / "data/interim/cohort/"
    "primary_candidate_manifest_v1.csv"
)

MRI_PATH = (
    ROOT / "data/raw/imaging_metadata/mri/extracted/"
    "T1w_Structural_Imaging_Group_Key_MRI_19Sep2026.csv"
)

ADNI1_PATH = (
    ROOT / "data/interim/cohort/"
    "adni1_mri_selection_v1.csv"
)

GO2_PATH = (
    ROOT / "data/interim/cohort/"
    "go2_accelerated_selection_v1.csv"
)

ADNI3_PATH = (
    ROOT / "data/interim/cohort/"
    "adni3_mri_selection_v1.csv"
)

OUT_ALL = (
    ROOT / "data/interim/cohort/"
    "mri_selection_manifest_v1.csv"
)

OUT_SELECTED = (
    ROOT / "data/interim/cohort/"
    "primary_post_mri_qc_manifest_v1.csv"
)

OUT_SUMMARY = (
    ROOT / "tables/"
    "mri_selection_manifest_v1_summary.json"
)

# ============================================================
# LOAD
# ============================================================

primary = pd.read_csv(
    PRIMARY_PATH,
    low_memory=False
)

mri = pd.read_csv(
    MRI_PATH,
    low_memory=False
)

adni1 = pd.read_csv(
    ADNI1_PATH,
    low_memory=False
)

go2 = pd.read_csv(
    GO2_PATH,
    low_memory=False
)

adni3 = pd.read_csv(
    ADNI3_PATH,
    low_memory=False
)

primary["mri_date"] = pd.to_datetime(
    primary["mri_date"],
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

# ============================================================
# BASIC VALIDATION OF PHASE FILES
# ============================================================

for name, df, expected in [
    ("ADNI1", adni1, 204),
    ("GO2", go2, 574),
    ("ADNI3", adni3, 104),
]:
    if len(df) != expected:
        raise RuntimeError(
            f"{name}: expected {expected} rows, got {len(df)}"
        )

    if df["RID"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate RID found."
        )

# ============================================================
# NORMALIZE PHASE-SPECIFIC OUTPUTS
# ============================================================

adni1_common = pd.DataFrame({
    "RID": adni1["RID"],
    "mri_selection_family": "ADNI1",
    "mri_qc_status": adni1["status"],
    "mri_selected": adni1["status"].eq("selected"),
    "selected_loni_series": pd.to_numeric(
        adni1["selected_loni_series"],
        errors="coerce"
    ),
    "mri_selection_source": adni1["selection_source"],
})

go2_common = pd.DataFrame({
    "RID": go2["RID"],
    "mri_selection_family": "ADNIGO/ADNI2",
    "mri_qc_status": go2["status"],
    "mri_selected": go2["status"].eq("selected"),
    "selected_loni_series": pd.to_numeric(
        go2["selected_loni_series"],
        errors="coerce"
    ),
    "mri_selection_source": go2["selected_qc_source"],
})

adni3_common = pd.DataFrame({
    "RID": adni3["RID"],
    "mri_selection_family": "ADNI3",
    "mri_qc_status": adni3["status"],
    "mri_selected": adni3["status"].eq("selected"),
    "selected_loni_series": pd.to_numeric(
        adni3["selected_loni_series"],
        errors="coerce"
    ),
    "mri_selection_source": pd.Series(
        ["Mayo_GO2_3_QC"] * len(adni3)
    ),
})

selection = pd.concat(
    [
        adni1_common,
        go2_common,
        adni3_common,
    ],
    ignore_index=True
)

# ============================================================
# GLOBAL RID VALIDATION
# ============================================================

if len(selection) != 882:
    raise RuntimeError(
        f"Expected 882 phase-selection rows, got {len(selection)}"
    )

if selection["RID"].duplicated().any():
    duplicates = selection.loc[
        selection["RID"].duplicated(
            keep=False
        ),
        ["RID", "mri_selection_family"]
    ]

    raise RuntimeError(
        "RID appears in more than one MRI selection family:\n"
        + duplicates.to_string(index=False)
    )

primary_rids = set(
    primary["RID"]
    .astype(int)
)

selection_rids = set(
    selection["RID"]
    .astype(int)
)

missing_from_selection = (
    primary_rids
    - selection_rids
)

extra_in_selection = (
    selection_rids
    - primary_rids
)

if missing_from_selection:
    raise RuntimeError(
        f"Primary RIDs missing from MRI selection: "
        f"{sorted(missing_from_selection)}"
    )

if extra_in_selection:
    raise RuntimeError(
        f"Unexpected RIDs in MRI selection: "
        f"{sorted(extra_in_selection)}"
    )

# ============================================================
# MERGE BACK TO PRIMARY 882-SUBJECT MANIFEST
# ============================================================

final = primary.merge(
    selection,
    how="left",
    on="RID",
    validate="one_to_one"
)

# Protocol-family identity must agree with original manifest.
phase_mismatch = final[
    final["mri_protocol_phases"].astype(str)
    !=
    final["mri_selection_family"].astype(str)
]

if len(phase_mismatch):
    raise RuntimeError(
        "MRI protocol-family mismatch detected:\n"
        +
        phase_mismatch[
            [
                "RID",
                "mri_protocol_phases",
                "mri_selection_family",
            ]
        ].to_string(index=False)
    )

# ============================================================
# VALIDATE SELECTED SERIES
# ============================================================

selected = final[
    final["mri_selected"]
].copy()

excluded = final[
    ~final["mri_selected"]
].copy()

if selected["selected_loni_series"].isna().any():
    bad = selected[
        selected["selected_loni_series"].isna()
    ]

    raise RuntimeError(
        "Selected subject missing selected LONI series:\n"
        +
        bad[
            [
                "RID",
                "mri_selection_family",
            ]
        ].to_string(index=False)
    )

if excluded["selected_loni_series"].notna().any():
    raise RuntimeError(
        "Excluded subject unexpectedly has a selected LONI series."
    )

# ============================================================
# LINK SELECTED SERIES BACK TO KEY MRI
# ============================================================

selected_lookup = selected[
    [
        "RID",
        "PTID",
        "mri_visit",
        "mri_date",
        "selected_loni_series",
    ]
].copy()

selected_lookup["selected_loni_series"] = (
    pd.to_numeric(
        selected_lookup["selected_loni_series"],
        errors="coerce"
    )
)

selected_key = mri.merge(
    selected_lookup,
    left_on=[
        "subject_id",
        "image_visit",
        "image_date",
        "loni_series_num",
    ],
    right_on=[
        "PTID",
        "mri_visit",
        "mri_date",
        "selected_loni_series",
    ],
    how="inner",
    validate="many_to_one"
)

# Every selected subject must resolve to >=1 Key MRI row.
resolved_rids = set(
    selected_key["RID"]
    .astype(int)
)

selected_rids = set(
    selected["RID"]
    .astype(int)
)

not_resolved = (
    selected_rids
    - resolved_rids
)

if not_resolved:
    raise RuntimeError(
        "Selected subjects not found back in Key MRI: "
        f"{sorted(not_resolved)}"
    )

# ============================================================
# AGGREGATE KEY MRI DETAILS AT SUBJECT / LONI SERIES LEVEL
# ============================================================

def pipe_unique(values):
    values = (
        pd.Series(values)
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return "|".join(values)


details = (
    selected_key
    .groupby(
        [
            "RID",
            "selected_loni_series",
        ],
        as_index=False
    )
    .agg(
        selected_key_mri_row_count=(
            "image_id",
            "size"
        ),

        selected_image_ids=(
            "image_id",
            pipe_unique
        ),

        selected_loni_images=(
            "loni_image",
            pipe_unique
        ),

        selected_series_instance_uids=(
            "series_instance_uid",
            pipe_unique
        ),

        selected_series_descriptions=(
            "series_description",
            pipe_unique
        ),

        selected_magnetic_field_strengths=(
            "magnetic_field_strength",
            pipe_unique
        ),
    )
)

if details["RID"].duplicated().any():
    raise RuntimeError(
        "A selected subject resolved to multiple distinct "
        "selected LONI series."
    )

final = final.merge(
    details,
    how="left",
    on=[
        "RID",
        "selected_loni_series",
    ],
    validate="one_to_one"
)

# ============================================================
# ADD SIMPLE EXCLUSION FLAG
# ============================================================

final["mri_exclusion_reason"] = pd.NA

mask_excluded = ~final["mri_selected"]

final.loc[
    mask_excluded,
    "mri_exclusion_reason"
] = final.loc[
    mask_excluded,
    "mri_qc_status"
]

# ============================================================
# FINAL VALIDATION
# ============================================================

selected = final[
    final["mri_selected"]
].copy()

excluded = final[
    ~final["mri_selected"]
].copy()

if len(final) != 882:
    raise RuntimeError(
        f"Final manifest should contain 882 rows, got {len(final)}"
    )

if final["RID"].duplicated().any():
    raise RuntimeError(
        "Final MRI manifest contains duplicated RIDs."
    )

if selected[
    "selected_key_mri_row_count"
].isna().any():
    raise RuntimeError(
        "Selected subject missing Key MRI detail aggregation."
    )

# Multiple Key MRI rows are permitted only when they refer to
# the same already-selected LONI series.
multiple_key_rows = selected[
    selected[
        "selected_key_mri_row_count"
    ] > 1
].copy()

# ============================================================
# SUMMARY
# ============================================================

phase_counts = (
    final.groupby(
        [
            "mri_selection_family",
            "mri_selected",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

selected_class_counts = (
    selected[
        "diagnosis"
    ]
    .value_counts()
    .to_dict()
)

excluded_class_counts = (
    excluded[
        "diagnosis"
    ]
    .value_counts()
    .to_dict()
)

status_counts = (
    final[
        "mri_qc_status"
    ]
    .value_counts()
    .to_dict()
)

summary = {
    "candidate_subjects": int(len(final)),
    "mri_qc_selected": int(len(selected)),
    "mri_qc_excluded": int(len(excluded)),

    "selected_class_counts": {
        str(k): int(v)
        for k, v in selected_class_counts.items()
    },

    "excluded_class_counts": {
        str(k): int(v)
        for k, v in excluded_class_counts.items()
    },

    "phase_counts": {
        str(phase): {
            "excluded": int(
                row.get(False, 0)
            ),
            "selected": int(
                row.get(True, 0)
            ),
        }
        for phase, row in phase_counts.iterrows()
    },

    "status_counts": {
        str(k): int(v)
        for k, v in status_counts.items()
    },

    "selected_subjects_with_multiple_key_mri_rows":
        int(len(multiple_key_rows)),
}

# ============================================================
# SAVE
# ============================================================

final.to_csv(
    OUT_ALL,
    index=False
)

selected.to_csv(
    OUT_SELECTED,
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

print("\n" + "=" * 82)
print("GLOBAL MRI SELECTION MANIFEST")
print("=" * 82)

print(
    "\nCandidate subjects:",
    summary["candidate_subjects"]
)

print(
    "Selected after MRI QC:",
    summary["mri_qc_selected"]
)

print(
    "Excluded by MRI QC:",
    summary["mri_qc_excluded"]
)

print("\nBy MRI protocol family:")
print(
    phase_counts.to_string()
)

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

print("\nExclusion / selection status:")
for k, v in summary[
    "status_counts"
].items():
    print(f"  {k}: {v}")

print(
    "\nSelected subjects with >1 Key MRI row "
    "for the same selected LONI series:",
    summary[
        "selected_subjects_with_multiple_key_mri_rows"
    ]
)

if len(multiple_key_rows):
    print("\nThose subjects:")
    print(
        multiple_key_rows[
            [
                "RID",
                "mri_selection_family",
                "selected_loni_series",
                "selected_key_mri_row_count",
                "selected_image_ids",
            ]
        ]
        .to_string(index=False)
    )

print(
    "\nSaved full 882-subject MRI manifest:\n",
    OUT_ALL
)

print(
    "\nSaved post-MRI-QC selected manifest:\n",
    OUT_SELECTED
)

print(
    "\nSaved aggregate summary:\n",
    OUT_SUMMARY
)
