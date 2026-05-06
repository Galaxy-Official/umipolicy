#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_505_stiring_tactile_force_align}"
EXP_NAME="${EXP_NAME:-505_stiring_handcap_pi05_4gpu_tactile_force_align}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
NUM_WORKERS="${NUM_WORKERS:-72}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"

export XLA_PYTHON_CLIENT_PREALLOCATE="true"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
export TF_ENABLE_ONEDNN_OPTS=1
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16"

echo "=========================================="
echo "Starting OpenPI PI05 Vision + Tactile + Force Align training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: Data/505_stiring_lerobot"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Num Workers: ${NUM_WORKERS}"
echo "=========================================="

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
