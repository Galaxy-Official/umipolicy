#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_512_stiring_490}"
EXP_NAME="${EXP_NAME:-512_stiring_490_handcap_pi05_4gpu_vision_only}"
export WANDB_DIR="${WANDB_DIR:-checkpoints/${CONFIG_NAME}/${EXP_NAME}}"
DATA_ROOT="${DATA_ROOT:-Data/512_stiring_lerobot_490}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
NUM_WORKERS="${NUM_WORKERS:-32}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: Neither python nor python3 was found. Activate the openpi env or set PYTHON_BIN=/path/to/python."
    exit 1
  fi
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=16}"

echo "=========================================="
echo "Starting OpenPI PI05 Vision Only training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: ${DATA_ROOT}"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Save interval: ${SAVE_INTERVAL}"
echo "Num Workers: ${NUM_WORKERS}"
echo "Python: ${PYTHON_BIN}"
echo "W&B mode: ${WANDB_MODE}"
echo "W&B dir: ${WANDB_DIR}"
echo "=========================================="

if [[ ! -d "${DATA_ROOT}/data" || ! -d "${DATA_ROOT}/meta" || ! -d "${DATA_ROOT}/videos" ]]; then
  echo "ERROR: Expected LeRobot dataset folders data/meta/videos under: ${DATA_ROOT}"
  exit 1
fi

"${PYTHON_BIN}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

"${PYTHON_BIN}" scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --fsdp-devices "${FSDP_DEVICES}" \
  --overwrite
