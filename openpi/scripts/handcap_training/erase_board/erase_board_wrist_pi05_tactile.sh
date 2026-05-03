#!/usr/bin/env bash
set -euo pipefail

# 退回 openpi 根目录
cd "$(dirname "$0")/../../.."

# 自动保存终端日志到 logs/
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_erase_board_wrist_tactile}"
EXP_NAME="${EXP_NAME:-erase_board_wrist_0429_handcap_pi05_4gpu_tactile}"

# 充分利用 141GB H200 显存，大幅提升 Batch Size
BATCH_SIZE="${BATCH_SIZE:-256}"
# 调整总训练步数
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
# 充分利用 80 核 CPU 和 900GB 内存，极大加速数据加载
NUM_WORKERS="${NUM_WORKERS:-64}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"

# XLA 内存优化：针对大显存设备预分配，防止碎片化
export XLA_PYTHON_CLIENT_PREALLOCATE="true"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"

echo "=========================================="
echo "Starting OpenPI PI05 (Vision + Tactile) training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: Data/429_erase_board_lerobot"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
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
