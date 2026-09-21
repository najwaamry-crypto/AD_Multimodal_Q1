from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PATH = (
    ROOT / "data/interim/cohort/"
    "analytic_manifest_v1.csv"
)

df = pd.read_csv(
    PATH,
    low_memory=False
)

if len(df) != 822:
    raise RuntimeError(
        f"Expected 822 subjects, got {len(df)}"
    )

df["RID"] = pd.to_numeric(
    df["RID"],
    errors="raise"
).astype(int)

# ============================================================
# PET
# ============================================================

pet = df[
    [
        "RID",
        "PTID",
        "diagnosis",
        "pet_image_id",
        "pet_date",
        "pet_visit",
        "pet_description",
    ]
].copy()

pet["pet_image_id"] = pd.to_numeric(
    pet["pet_image_id"],
    errors="raise"
).astype(int)

# ============================================================
# MRI SERIES
# ============================================================

mri_series = df[
    [
        "RID",
        "PTID",
        "diagnosis",
        "selected_loni_series",
        "selected_image_ids",
        "selected_loni_images",
        "selected_series_instance_uids",
        "mri_date",
        "mri_visit",
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

# ============================================================
# EXPLODE MRI IMAGE IDS
# ============================================================

rows = []

for _, row in mri_series.iterrows():

    raw = str(
        row["selected_image_ids"]
    ).strip()

    ids = [
        x.strip()
        for x in raw.split("|")
        if x.strip()
    ]

    if not ids:
        raise RuntimeError(
            f"RID {row['RID']} has no selected_image_ids."
        )

    for image_id in ids:

        try:
            image_id_int = int(
                float(image_id)
            )
        except Exception as e:
            raise RuntimeError(
                f"Invalid MRI image ID {image_id} "
                f"for RID {row['RID']}"
            ) from e

        rows.append(
            {
                "RID": int(row["RID"]),
                "PTID": row["PTID"],
                "diagnosis": row["diagnosis"],
                "selected_loni_series":
                    int(row["selected_loni_series"]),
                "mri_image_id":
                    image_id_int,
                "mri_date":
                    row["mri_date"],
                "mri_visit":
                    row["mri_visit"],
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

mri_images = pd.DataFrame(rows)

# ============================================================
# DUPLICATION CHECKS
# ============================================================

pet_cross_rid = (
    pet.groupby("pet_image_id")["RID"]
    .nunique()
)

pet_cross_rid = pet_cross_rid[
    pet_cross_rid > 1
]

series_cross_rid = (
    mri_series
    .groupby("selected_loni_series")["RID"]
    .nunique()
)

series_cross_rid = series_cross_rid[
    series_cross_rid > 1
]

image_cross_rid = (
    mri_images
    .groupby("mri_image_id")["RID"]
    .nunique()
)

image_cross_rid = image_cross_rid[
    image_cross_rid > 1
]

# ============================================================
# REPORT
# ============================================================

print("=" * 96)
print("IMAGING DOWNLOAD IDENTIFIER AUDIT")
print("=" * 96)

print("\nPET")
print("-" * 96)

print(
    "Subjects:",
    pet["RID"].nunique()
)

print(
    "Rows:",
    len(pet)
)

print(
    "Unique PET image IDs:",
    pet["pet_image_id"].nunique()
)

print(
    "PET image IDs shared across RID:",
    len(pet_cross_rid)
)

print("\nMRI")
print("-" * 96)

print(
    "Subjects:",
    mri_series["RID"].nunique()
)

print(
    "Selected LONI series:",
    mri_series[
        "selected_loni_series"
    ].nunique()
)

print(
    "LONI series shared across RID:",
    len(series_cross_rid)
)

print(
    "Exploded MRI image-ID rows:",
    len(mri_images)
)

print(
    "Unique MRI image IDs:",
    mri_images[
        "mri_image_id"
    ].nunique()
)

print(
    "MRI image IDs shared across RID:",
    len(image_cross_rid)
)

counts_per_rid = (
    mri_images
    .groupby("RID")
    .size()
)

print("\nMRI image IDs per subject:")
print(
    counts_per_rid
    .value_counts()
    .sort_index()
    .to_string()
)

multi = (
    counts_per_rid[
        counts_per_rid > 1
    ]
    .index
)

print(
    "\nSubjects with >1 selected MRI image ID:",
    len(multi)
)

if len(multi):

    print(
        mri_images[
            mri_images["RID"].isin(
                multi
            )
        ][
            [
                "RID",
                "PTID",
                "selected_loni_series",
                "mri_image_id",
                "mri_date",
                "series_description",
            ]
        ]
        .sort_values(
            [
                "RID",
                "mri_image_id",
            ]
        )
        .to_string(
            index=False
        )
    )

print("\nPET description:")
print(
    pet["pet_description"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nMRI selection family:")
print(
    mri_series[
        "mri_selection_family"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nMRI selection source:")
print(
    mri_series[
        "mri_selection_source"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nMRI field strength:")
print(
    mri_series[
        "selected_magnetic_field_strengths"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)
