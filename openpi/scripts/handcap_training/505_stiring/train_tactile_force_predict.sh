#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# 自动保存终端日志到 logs/
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_512_stiring_tactile_force_predict}"
EXP_NAME="${EXP_NAME:-512_stiring_handcap_pi05_4gpu_tactile_force_predict}"
DATA_ROOT="${DATA_ROOT:-Data/512_stiring_lerobot}"

# ==============================================================================
# H200 (141GB) x4 & 80-Core 900GB RAM 极致资源榨干配置
# ==============================================================================
# 批量大小：由于 H200 有 141GB 显存，256 太过保守，直接拉升至 512（每张卡分担 128）
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
# 数据加载线程：80核CPU，保留 8 核给系统/调度，使用 72 核满载预处理
NUM_WORKERS="${NUM_WORKERS:-64}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"

export XLA_PYTHON_CLIENT_PREALLOCATE="true"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
# 开启张量核心 TF32 计算加速，并增加 XLA 编译并发度
export TF_ENABLE_ONEDNN_OPTS=1
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16"
# ==============================================================================

echo "=========================================="
echo "Starting OpenPI PI05 Vision + Tactile + Force Predict training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: ${DATA_ROOT}"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Num Workers: ${NUM_WORKERS}"
echo "=========================================="

if [[ ! -d "${DATA_ROOT}/data" || ! -d "${DATA_ROOT}/meta" || ! -d "${DATA_ROOT}/videos" ]]; then
  echo "ERROR: Expected LeRobot dataset folders data/meta/videos under: ${DATA_ROOT}"
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
  --overwrite
