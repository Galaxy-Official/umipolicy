#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# 自动保存终端日志到 logs/
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_506_open_bottle_tactile_force_guide}"
EXP_NAME="${EXP_NAME:-506_open_bottle_handcap_pi05_4gpu_tactile_force_guide}"

# ==============================================================================
# H200 (141GB) x4 & 80-Core 900GB RAM throughput-oriented defaults
# ==============================================================================
# Keep the global batch large enough to use the GPUs, but tune by samples/sec,
# not by memory percentage.
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
# Too many workers can overwhelm video/parquet random I/O and cause long
# epoch-boundary stalls. Start lower and sweep 8/16/24/32.
NUM_WORKERS="${NUM_WORKERS:-16}"
# H200 has enough memory to prefer data parallelism first. FSDP saves memory but
# often costs throughput through extra cross-GPU communication.
FSDP_DEVICES="${FSDP_DEVICES:-1}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

export XLA_PYTHON_CLIENT_PREALLOCATE="true"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
# 开启张量核心 TF32 计算加速，并增加 XLA 编译并发度
export TF_ENABLE_ONEDNN_OPTS=1
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16"
# ==============================================================================

echo "=========================================="
echo "Starting OpenPI PI05 Vision + Tactile + Force Guide training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: Data/506_open_bottle_lerobot"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Num Workers: ${NUM_WORKERS}"
echo "Resume: ${RESUME}"
echo "Overwrite: ${OVERWRITE}"
echo "=========================================="

CHECKPOINT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"
LATEST_CKPT="$(
  find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' 2>/dev/null     | awk -F/ '{print $NF}'     | sort -n     | tail -n 1     || true
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

# ==============================================================================
# DISK SPACE RESERVATION & WATCHDOG
# ==============================================================================
RESERVE_FILE="${CHECKPOINT_DIR}/reserved_50G_buffer.bin"
mkdir -p "${CHECKPOINT_DIR}"

# Check free space in KB (50GB = 52428800 KB). If more than 50GB free, reserve it.
FREE_KB=$(df -k "${CHECKPOINT_DIR}" | awk 'NR==2 {print $4}')
if [[ "${FREE_KB:-0}" =~ ^[0-9]+$ ]] && [ "${FREE_KB:-0}" -gt 52428800 ]; then
    echo "Reserving 50GB of disk space to prevent exhaustion by other processes..."
    if command -v fallocate >/dev/null 2>&1; then
        fallocate -l 50G "$RESERVE_FILE" || true
    else
        echo "WARNING: 'fallocate' not found (maybe on Mac?). Skipping 50GB reservation to avoid slow writes."
    fi
fi

# Watchdog to release space if disk gets too full (e.g. < 10GB free)
(
    while true; do
        CURRENT_FREE=$(df -k "${CHECKPOINT_DIR}" | awk 'NR==2 {print $4}')
        if [[ "${CURRENT_FREE:-0}" =~ ^[0-9]+$ ]] && [ "${CURRENT_FREE:-0}" -lt 10485760 ]; then # 10GB
            if [ -f "$RESERVE_FILE" ]; then
                echo "WARNING: Disk space low. Releasing reserved 50GB to allow training to continue."
                rm -f "$RESERVE_FILE"
            fi
        fi
        sleep 60
    done
) &
WATCHDOG_PID=$!

# Ensure watchdog and reserve file are cleaned up on exit
cleanup() {
    kill $WATCHDOG_PID 2>/dev/null || true
    rm -f "$RESERVE_FILE"
}
trap cleanup EXIT
# ==============================================================================

python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

python scripts/train.py   "${CONFIG_NAME}"   --exp-name "${EXP_NAME}"   --batch-size "${BATCH_SIZE}"   --num-train-steps "${NUM_TRAIN_STEPS}"   --save-interval "${SAVE_INTERVAL}"   --num-workers "${NUM_WORKERS}"   --no-wandb-enabled   --fsdp-devices "${FSDP_DEVICES}"   "${RUN_MODE_FLAGS[@]}"
