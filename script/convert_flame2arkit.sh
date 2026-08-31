#!/usr/bin/env bash
# Run FLAME -> ARKit conversion for all DualTalk splits, driven by metadata.
#
# The script only needs the metadata folder and the output folder: the source
# FLAME paths are already recorded in the metadata jsonl files, and the
# converted output paths are written back into the same metadata.
#
# Hyperparameters (edit below or override via env):
#   METADATA_DIR : folder containing train/test/ood .jsonl metadata
#   OUTPUT_DIR   : folder where converted .npy files are saved
#   MATRIX       : path to mat_final.npy (ARKit51->FLAME103 forward matrix)
#   NUM_WORKERS  : parallel worker processes per split (default 16; 1 = serial)
#
# Extra flags after the script name are forwarded to convert_flame2arkit.py,
# e.g. smooth/head controls (see flame2arkit.doc, sections 8-9):
#   bash convert_flame2arkit.sh --no-smooth
#   bash convert_flame2arkit.sh --no-head-stabilize
#   bash convert_flame2arkit.sh --head-median-window 7 --head-smooth-window 9 \
#       --head-max-delta 0.15
#   bash convert_flame2arkit.sh --head-center median --head-signs -1 1 1 \
#       --head-gains 1.2 1 1 --head-limits 0.5 0.5 0.5

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/xuxuanyang/DATA/_data/DualTalk_Dataset}"
SCRIPT_DIR="${SCRIPT_DIR:-${DATASET_ROOT}/script}"
METADATA_DIR="${METADATA_DIR:-${DATASET_ROOT}/metadata}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_ROOT}/ARKit_npy}"
MATRIX="${MATRIX:-${SCRIPT_DIR}/mat_final.npy}"
SPLITS="${SPLITS:-train test ood}"
NUM_WORKERS="${NUM_WORKERS:-16}"

mkdir -p "${OUTPUT_DIR}"

for split in ${SPLITS}; do
    metadata="${METADATA_DIR}/${split}.jsonl"
    if [[ ! -f "${metadata}" ]]; then
        echo "[skip] metadata not found: ${metadata}" >&2
        continue
    fi
    echo "=== converting split=${split} metadata=${metadata} output=${OUTPUT_DIR} workers=${NUM_WORKERS} ==="
    python3 "${SCRIPT_DIR}/convert_flame2arkit.py" \
        --matrix "${MATRIX}" \
        --metadata "${metadata}" \
        --dataset-root "${DATASET_ROOT}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-workers "${NUM_WORKERS}" \
        "$@"
done

echo "all splits done. output_dir=${OUTPUT_DIR}"
