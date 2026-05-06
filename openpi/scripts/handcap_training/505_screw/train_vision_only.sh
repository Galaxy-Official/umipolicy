#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# 自动保存终端日志到 logs/
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_505_screw}"
EXP_NAME="${EXP_NAME:-505_screw_handcap_pi05_4gpu_vision_only}"

# ==============================================================================
# H200 (141GB) x4 & 80-Core 900GB RAM throughput-oriented defaults
# ==============================================================================
# Keep the global batch large enough to use the GPUs, but tune by samples/sec,
# not by memory percentage.
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
# Too many workers can overwhelm video/parquet random I/O and cause long
# epoch-boundary stalls. Start lower and sweep 8/16/24/32.
NUM_WORKERS="${NUM_WORKERS:-16}"
# H200 has enough memory to prefer data parallelism first. FSDP saves memory but
# often costs throughput through extra cross-GPU communication.
FSDP_DEVICES="${FSDP_DEVICES:-1}"
RESUME="${RESUME:-0}"

export XLA_PYTHON_CLIENT_PREALLOCATE="true"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
# 开启张量核心 TF32 计算加速，并增加 XLA 编译并发度
export TF_ENABLE_ONEDNN_OPTS=1
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16"
# ==============================================================================

echo "=========================================="
echo "Starting OpenPI PI05 Vision Only training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: Data/505_screw_lerobot"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Num Workers: ${NUM_WORKERS}"
echo "Resume: ${RESUME}"
echo "=========================================="

python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

RUN_MODE_FLAG="--overwrite"
if [[ "${RESUME}" == "1" ]]; then
  RUN_MODE_FLAG="--resume"
fi

python scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --no-wandb-enabled \
  --fsdp-devices "${FSDP_DEVICES}" \
  "${RUN_MODE_FLAG}"
