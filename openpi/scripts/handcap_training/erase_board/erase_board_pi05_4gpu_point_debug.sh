#!/usr/bin/env bash
set -euo pipefail

# 退回 openpi 根目录
cd "$(dirname "$0")/../../.."

# 自动保存终端日志到 logs/
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

# 默认 4 卡；也可以外部覆盖：
# CUDA_VISIBLE_DEVICES=0,1 bash scripts/handcap/erase_board_pi05_4gpu_point_debug.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# 需要和 handcap_config.py 中注册的 TrainConfig.name 完全一致
CONFIG_NAME="${CONFIG_NAME:-pi05_erase_board_wrist_point_debug}"

# 强烈建议 point 测试使用全新的 exp name，避免恢复旧 checkpoint
EXP_NAME="${EXP_NAME:-erase_board_0429_pi05_point_debug_${TIMESTAMP}}"

# smoke test：先用小 batch 和少量 step，确认数据流、point encoder、forward/backward 能跑通
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"

# 是否计算 norm stats。第一次跑建议 true；如果已经算过可设为 false
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-true}"

# 是否 resume。point debug 默认 false，避免 Orbax sharding / 旧 checkpoint 影响判断
RESUME="${RESUME:-false}"

# XLA 内存设置。debug 小 batch 不需要吃满显存，避免影响同机其他任务
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"

echo "=========================================="
echo "Starting OpenPI PI05 (Vision + PointCloud) smoke test"
echo "Config: ${CONFIG_NAME}"
echo "Expected dataset: Data/429_erase_board_lerobot"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Save interval: ${SAVE_INTERVAL}"
echo "Num workers: ${NUM_WORKERS}"
echo "Compute norm stats: ${COMPUTE_NORM_STATS}"
echo "Resume: ${RESUME}"
echo "=========================================="

if [[ "${COMPUTE_NORM_STATS}" == "true" ]]; then
  echo "[1/2] Computing normalization stats..."
  python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
else
  echo "[1/2] Skipping normalization stats."
fi

echo "[2/2] Starting training..."

TRAIN_CMD=(
  python scripts/train.py
  "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --batch-size "${BATCH_SIZE}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --save-interval "${SAVE_INTERVAL}"
  --num-workers "${NUM_WORKERS}"
  --no-wandb-enabled
  --fsdp-devices "${FSDP_DEVICES}"
)

if [[ "${RESUME}" == "true" ]]; then
  TRAIN_CMD+=(--resume)
fi

echo "Command:"
printf ' %q' "${TRAIN_CMD[@]}"
echo

"${TRAIN_CMD[@]}"