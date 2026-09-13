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

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=10
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=${OUTPUT_ROOT}/base2new/train_base/${COMMON_DIR}
DIR=${OUTPUT_ROOT}/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${CONFIG_DIR}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    --model-dir "${MODEL_DIR}" \
    --load-epoch "${LOADEP}" \
    --eval-only \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES "${SUB}" \
    "$@"

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${CONFIG_DIR}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    --model-dir "${MODEL_DIR}" \
    --load-epoch "${LOADEP}" \
    --eval-only \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES "${SUB}" \
    "$@"
fi
