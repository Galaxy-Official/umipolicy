#!/bin/bash
set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_erase_board_wrist}"
EXP_NAME="${EXP_NAME:-erase_board_wrist_0429_handcap_pi05}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"

python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

python scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --no-wandb-enabled \
  --fsdp-devices "${FSDP_DEVICES}" \
  --overwrite
