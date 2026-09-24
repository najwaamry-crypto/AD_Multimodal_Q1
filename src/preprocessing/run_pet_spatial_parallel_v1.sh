#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/najwa-amry/AD_Multimodal_Q1"
PYTHON="/home/najwa-amry/.local/share/mamba/envs/adq1/bin/python"
SCRIPT="$ROOT/src/preprocessing/03a_preprocess_pet_spatial_v1.py"

LOCK="$ROOT/logs/pet_spatial_v1_batch.lock"
LOG="$ROOT/logs/pet_spatial_v1_batch.log"
IDLIST="$ROOT/data/interim/pet_spatial_image_ids_v1.txt"

mkdir -p "$ROOT/logs"

exec 9>"$LOCK"

if ! flock -n 9; then
    echo "PET spatial batch already running."
    exit 0
fi

cd "$ROOT"

"$PYTHON" -c '
from pathlib import Path
import pandas as pd

df = pd.read_csv(
    "data/interim/cohort/analytic_manifest_v1.csv",
    low_memory=False
)

mri = pd.read_csv(
    "data/interim/imaging_download_manifests/mri_final_image_selection_v1.csv",
    low_memory=False
)[["RID", "mri_image_id"]]

df = df.merge(
    mri,
    on="RID",
    how="inner",
    validate="one_to_one"
)

ready = []

for _, row in df.iterrows():
    pet_id = int(row["pet_image_id"])
    mri_id = int(row["mri_image_id"])

    qc = Path(
        f"data/processed/mri_v1/I{mri_id}/I{mri_id}_qc.json"
    )

    if qc.exists():
        ready.append(pet_id)

for pet_id in ready:
    print(pet_id)
' > "$IDLIST"

READY_COUNT=$(wc -l < "$IDLIST")

echo "=== PET spatial batch started $(date) ===" >> "$LOG"
echo "MRI-ready PET subjects: $READY_COUNT" >> "$LOG"

xargs -P 8 -I {} \
  "$PYTHON" "$SCRIPT" \
  --pet-image-id {} \
  --threads 2 \
  < "$IDLIST" \
  >> "$LOG" 2>&1

echo "=== PET spatial batch finished $(date) ===" >> "$LOG"
