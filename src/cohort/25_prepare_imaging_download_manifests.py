from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ANALYTIC = (
    ROOT / "data/interim/cohort/"
    "analytic_manifest_v1.csv"
)

SPLITS = (
    ROOT / "splits/"
    "split_assignments_v1.csv"
)

OUT_DIR = (
    ROOT / "data/interim/"
    "imaging_download_manifests"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY = (
    ROOT / "tables/"
    "imaging_download_manifests_v1_summary.json"
)

df = pd.read_csv(
    ANALYTIC,
    low_memory=False
)

sp = pd.read_csv(
    SPLITS,
    low_memory=False
)

df["RID"] = pd.to_numeric(
    df["RID"],
    errors="raise"
).astype(int)

sp["RID"] = pd.to_numeric(
    sp["RID"],
    errors="raise"
).astype(int)

df = df.merge(
    sp[
        [
            "RID",
            "split",
        ]
    ],
    on="RID",
    how="left",
    validate="one_to_one"
)

if len(df) != 822:
    raise RuntimeError(
        f"Expected 822 subjects, got {len(df)}"
    )

if df["split"].isna().any():
    raise RuntimeError(
        "Missing split assignment."
    )

# ============================================================
# PET DOWNLOAD MANIFEST
# ============================================================

pet = df[
    [
        "RID",
        "PTID",
        "split",
        "diagnosis",
        "pet_image_id",
        "pet_visit",
        "pet_date",
        "pet_description",
    ]
].copy()

pet["pet_image_id"] = pd.to_numeric(
    pet["pet_image_id"],
    errors="raise"
).astype(int)

if pet["pet_image_id"].nunique() != 822:
    raise RuntimeError(
        "Expected 822 unique PET image IDs."
    )

pet = pet.sort_values(
    "pet_image_id"
).reset_index(
    drop=True
)

# ============================================================
# MRI SERIES MANIFEST
# ============================================================

mri_series = df[
    [
        "RID",
        "PTID",
        "split",
        "diagnosis",
        "selected_loni_series",
        "selected_image_ids",
        "selected_loni_images",
        "selected_series_instance_uids",
        "mri_visit",
        "mri_date",
        "selected_series_descriptions",
        "selected_magnetic_field_strengths",
        "mri_selection_family",
        "mri_selection_source",
    ]
].copy()

mri_series["selected_loni_series"] = pd.to_numeric(
    mri_series["selected_loni_series"],
    errors="raise"
).astype(int)

if mri_series["selected_loni_series"].nunique() != 822:
    raise RuntimeError(
        "Expected 822 unique MRI LONI series."
    )

mri_series = mri_series.sort_values(
    "selected_loni_series"
).reset_index(
    drop=True
)

# ============================================================
# MRI IMAGE-ID MANIFEST
# ============================================================

rows = []

for _, row in mri_series.iterrows():

    image_ids = [
        x.strip()
        for x in str(
            row["selected_image_ids"]
        ).split("|")
        if x.strip()
    ]

    if not image_ids:
        raise RuntimeError(
            f"RID {row['RID']} has no MRI image ID."
        )

    for image_id in image_ids:

        image_id = int(
            float(image_id)
        )

        rows.append(
            {
                "RID": int(row["RID"]),
                "PTID": row["PTID"],
                "split": row["split"],
                "diagnosis": row["diagnosis"],
                "selected_loni_series":
                    int(row["selected_loni_series"]),
                "mri_image_id":
                    image_id,
                "mri_visit":
                    row["mri_visit"],
                "mri_date":
                    row["mri_date"],
                "series_description":
                    row["selected_series_descriptions"],
                "field_strength":
                    row["selected_magnetic_field_strengths"],
                "selection_family":
                    row["mri_selection_family"],
                "selection_source":
                    row["mri_selection_source"],
            }
        )

mri_images = pd.DataFrame(
    rows
)

if len(mri_images) != 825:
    raise RuntimeError(
        f"Expected 825 MRI image-ID rows, got {len(mri_images)}"
    )

if mri_images["mri_image_id"].nunique() != 825:
    raise RuntimeError(
        "MRI image IDs are not unique."
    )

# ============================================================
# MULTI-IMAGE SERIES
# ============================================================

counts = (
    mri_images
    .groupby("RID")
    .size()
)

multi_rids = (
    counts[
        counts > 1
    ]
    .index
)

multi = (
    mri_images[
        mri_images["RID"].isin(
            multi_rids
        )
    ]
    .sort_values(
        [
            "RID",
            "mri_image_id",
        ]
    )
    .reset_index(
        drop=True
    )
)

if len(multi_rids) != 3:
    raise RuntimeError(
        f"Expected 3 multi-image MRI subjects, got {len(multi_rids)}"
    )

# ============================================================
# SAVE CSV MANIFESTS
# ============================================================

pet.to_csv(
    OUT_DIR / "pet_download_manifest_v1.csv",
    index=False
)

mri_series.to_csv(
    OUT_DIR / "mri_series_download_manifest_v1.csv",
    index=False
)

mri_images.to_csv(
    OUT_DIR / "mri_image_download_manifest_v1.csv",
    index=False
)

multi.to_csv(
    OUT_DIR / "mri_multi_image_series_v1.csv",
    index=False
)

# ============================================================
# SAVE SIMPLE TEXT LISTS
# ============================================================

with open(
    OUT_DIR / "pet_image_ids_v1.txt",
    "w"
) as f:
    for x in pet["pet_image_id"]:
        f.write(f"{x}\n")

with open(
    OUT_DIR / "mri_series_ids_v1.txt",
    "w"
) as f:
    for x in mri_series["selected_loni_series"]:
        f.write(f"{x}\n")

with open(
    OUT_DIR / "mri_image_ids_v1.txt",
    "w"
) as f:
    for x in mri_images["mri_image_id"]:
        f.write(f"{x}\n")

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "subjects": 822,

    "pet": {
        "subjects": 822,
        "unique_image_ids": int(
            pet["pet_image_id"].nunique()
        ),
    },

    "mri": {
        "subjects": 822,
        "unique_loni_series": int(
            mri_series[
                "selected_loni_series"
            ].nunique()
        ),
        "image_id_rows": int(
            len(mri_images)
        ),
        "unique_image_ids": int(
            mri_images[
                "mri_image_id"
            ].nunique()
        ),
        "subjects_with_multiple_image_ids":
            int(len(multi_rids)),
    },

    "download_policy": {
        "preferred_mri_unit":
            "selected LONI series",
        "fallback":
            (
                "If series-level download is unavailable, "
                "download all 825 selected MRI image IDs. "
                "Do not arbitrarily choose among duplicate "
                "Key MRI rows belonging to the same selected series."
            ),
    },
}

with open(
    SUMMARY,
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

print("=" * 96)
print("IMAGING DOWNLOAD MANIFESTS V1")
print("=" * 96)

print("\nPET:")
print(
    "  subjects:",
    pet["RID"].nunique()
)
print(
    "  image IDs:",
    pet["pet_image_id"].nunique()
)

print("\nMRI:")
print(
    "  subjects:",
    mri_series["RID"].nunique()
)
print(
    "  LONI series:",
    mri_series[
        "selected_loni_series"
    ].nunique()
)
print(
    "  MRI image IDs:",
    len(mri_images)
)
print(
    "  subjects with >1 image ID:",
    len(multi_rids)
)

print("\nMulti-image MRI series:")
print(
    multi[
        [
            "RID",
            "PTID",
            "selected_loni_series",
            "mri_image_id",
        ]
    ].to_string(
        index=False
    )
)

print("\nFiles created:")
for p in sorted(
    OUT_DIR.iterdir()
):
    print(" ", p)

print("\nSummary:")
print(SUMMARY)

print(
    "\nNo images downloaded yet."
)
