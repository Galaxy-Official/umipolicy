#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# 自动保存终端日志到 logs/
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_430_clamp_seal_tactile_concat_t3}"
EXP_NAME="${EXP_NAME:-clamp_seal_handcap_pi05_4gpu_tactile_concat_t3}"
DATA_ROOT="${DATA_ROOT:-Data/clamp_seal_lerobot}"
T3_ROOT="${T3_ROOT:-ckpt/t3_tiny}"

# ==============================================================================
# Same data and throughput defaults as train_vision_tactile.sh.
# This script uses a separate config/experiment name so it will not touch prior
# vision-only, vision+tactile, force-predict, or health-distill checkpoints.
# ==============================================================================
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
NUM_WORKERS="${NUM_WORKERS:-32}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

export XLA_PYTHON_CLIENT_PREALLOCATE="true"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
export TF_ENABLE_ONEDNN_OPTS=1
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16"
# ==============================================================================

echo "=========================================="
echo "Starting OpenPI PI05 TactileConcat + T3 training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: ${DATA_ROOT}"
echo "Experiment: ${EXP_NAME}"
echo "T3 root: ${T3_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Save interval: ${SAVE_INTERVAL}"
echo "Num Workers: ${NUM_WORKERS}"
echo "Resume: ${RESUME}"
echo "Overwrite: ${OVERWRITE}"
echo "=========================================="

if [[ ! -d "${DATA_ROOT}/data" || ! -d "${DATA_ROOT}/meta" || ! -d "${DATA_ROOT}/videos" ]]; then
  echo "ERROR: Expected LeRobot dataset folders data/meta/videos under: ${DATA_ROOT}"
  exit 1
fi

if [[ ! -f "${T3_ROOT}/trunk.pth" || ! -f "${T3_ROOT}/encoders/mini.pth" ]]; then
  echo "ERROR: Expected T3 tiny checkpoints:"
  echo "  ${T3_ROOT}/trunk.pth"
  echo "  ${T3_ROOT}/encoders/mini.pth"
  echo "Download them first from alanz-mit/FoundationTactile, or set T3_ROOT to the local ckpt/t3_tiny directory."
  exit 1
fi

CHECKPOINT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"
LATEST_CKPT="$(
  find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' 2>/dev/null \
    | awk -F/ '{print $NF}' \
    | sort -n \
    | tail -n 1 \
    || true
)"

if [[ "${RESUME}" == "1" && "${OVERWRITE}" == "1" ]]; then
  echo "ERROR: RESUME=1 and OVERWRITE=1 cannot be used together."
  exit 1
fi

RUN_MODE_FLAGS=()
if [[ "${RESUME}" == "1" ]]; then
  if [[ -z "${LATEST_CKPT}" ]]; then
    echo "ERROR: RESUME=1 but no numeric checkpoint was found under ${CHECKPOINT_DIR}."
    echo "Refusing to start from scratch silently."
    exit 1
  fi
  echo "Resuming from checkpoint step: ${LATEST_CKPT}"
  RUN_MODE_FLAGS=(--resume)
elif [[ "${OVERWRITE}" == "1" ]]; then
  echo "WARNING: OVERWRITE=1 will delete any existing checkpoint directory: ${CHECKPOINT_DIR}"
  RUN_MODE_FLAGS=(--overwrite)
elif [[ -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: Checkpoint directory already exists: ${CHECKPOINT_DIR}"
  echo "Use RESUME=1 to continue from an existing checkpoint, or OVERWRITE=1 to intentionally start over."
  exit 1
fi

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
  "${RUN_MODE_FLAGS[@]}"
