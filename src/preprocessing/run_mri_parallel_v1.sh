#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/najwa-amry/AD_Multimodal_Q1"
PY="/home/najwa-amry/.local/share/mamba/envs/adq1/bin/python"

export HOME="/home/najwa-amry"
export PATH="/home/najwa-amry/.local/share/mamba/envs/adq1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

cd "$ROOT"

LOGDIR="$ROOT/logs/mri_preprocessing_v1"
LOG="$LOGDIR/parallel_run.log"
LOCK="$LOGDIR/parallel_run.lock"
IDS="$ROOT/data/interim/imaging_download_manifests/mri_image_ids_v1.txt"

mkdir -p "$LOGDIR"

exec 9>"$LOCK"

if ! flock -n 9; then
    echo "$(date -Is) MRI preprocessing already running; exiting." >> "$LOG"
    exit 0
fi

echo "" >> "$LOG"
echo "==================================================" >> "$LOG"
echo "$(date -Is) Starting MRI preprocessing batch" >> "$LOG"
echo "Workers: 12 | Threads per subject: 4" >> "$LOG"
echo "==================================================" >> "$LOG"

cat "$IDS" |
xargs -P 12 -I{} "$PY" -u \
    src/preprocessing/01_preprocess_mri_v1.py \
    --image-id {} \
    --threads 4 \
    >> "$LOG" 2>&1

echo "$(date -Is) MRI preprocessing batch finished." >> "$LOG"
