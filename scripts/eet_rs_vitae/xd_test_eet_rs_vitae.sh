#!/bin/bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# custom config
DATA="${DATA:-data/}"
OUTPUT_ROOT="${EET_OUTPUT_ROOT:-output}"
TRAINER=EET_RS_ViTAE
CONFIG_DIR=EET_rs_vitae

DATASET=${1:?Usage: script DATASET SEED [config overrides]}
SEED=${2:?Usage: script DATASET SEED [config overrides]}
shift 2

if [ -n "${EXPERT_CKPT:-}" ]; then
    set -- MODEL.EXPERT_CHECKPOINT "$EXPERT_CKPT" "$@"
fi

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16


DIR=${OUTPUT_ROOT}/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${CONFIG_DIR}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    --model-dir "${OUTPUT_ROOT}/MLRSNet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}" \
    --load-epoch 10 \
    --eval-only \
    "$@"
fi
