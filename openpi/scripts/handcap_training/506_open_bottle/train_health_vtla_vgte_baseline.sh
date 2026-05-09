#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_506_open_bottle_tactile_force_guide_vtla_vgte}"
EXP_NAME="${EXP_NAME:-506_open_bottle_handcap_pi05_4gpu_vtla_vgte_baseline}"
DATA_ROOT="Data/506_open_bottle_lerobot"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=16}"

echo "=========================================="
echo "Starting OpenPI PI05 ordinary VTLA/VGTE baseline training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: ${DATA_ROOT}"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Save interval: ${SAVE_INTERVAL}"
echo "Num workers: ${NUM_WORKERS}"
echo "OMP/MKL/OPENBLAS/NUMEXPR threads: ${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}/${NUMEXPR_NUM_THREADS}"
echo "Resume: ${RESUME}"
echo "Overwrite: ${OVERWRITE}"
echo "=========================================="

if [[ ! -d "${DATA_ROOT}/data" || ! -d "${DATA_ROOT}/meta" || ! -d "${DATA_ROOT}/videos" ]]; then
  echo "ERROR: Expected LeRobot dataset folders data/meta/videos under: ${DATA_ROOT}"
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
