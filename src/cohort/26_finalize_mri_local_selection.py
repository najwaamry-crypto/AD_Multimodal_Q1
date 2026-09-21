from pathlib import Path
import pandas as pd

BASE = Path("data/interim/imaging_download_manifests")

SRC = BASE / "mri_image_download_manifest_v1.csv"
OUT = BASE / "mri_final_image_selection_v1.csv"

df = pd.read_csv(SRC, low_memory=False)

# Three QC-selected LONI series contained two downloadable Image IDs.
# Both acquisitions were geometrically valid and visually acceptable.
# Deterministic tie-break fixed before model training:
# retain the earliest same-day acquisition.

keep_ids = {389652, 416046, 420552}
drop_ids = {389660, 416053, 420554}

final = df[~df["mri_image_id"].isin(drop_ids)].copy()

assert len(df) == 825
assert len(final) == 822
assert final["RID"].nunique() == 822
assert final["selected_loni_series"].nunique() == 822

assert keep_ids.issubset(set(final["mri_image_id"]))
assert not drop_ids.intersection(set(final["mri_image_id"]))

final["local_nifti_path"] = final["mri_image_id"].apply(
    lambda x: f"data/interim/mri_nifti_v1/I{int(x)}.nii.gz"
)

final["selection_tiebreak"] = "single_image_id"

final.loc[
    final["mri_image_id"].isin(keep_ids),
    "selection_tiebreak"
] = "earliest_acquisition_after_equivalent_local_qc"

final.to_csv(OUT, index=False)

print("Original MRI Image IDs:", len(df))
print("Final MRI selections:", len(final))
print("Unique RIDs:", final["RID"].nunique())
print("Unique LONI series:", final["selected_loni_series"].nunique())

print("\nTie-break selections:")
print(
    final[
        final["mri_image_id"].isin(keep_ids)
    ][
        [
            "RID",
            "PTID",
            "selected_loni_series",
            "mri_image_id",
            "series_description",
        ]
    ].to_string(index=False)
)

print("\nSaved:", OUT)
