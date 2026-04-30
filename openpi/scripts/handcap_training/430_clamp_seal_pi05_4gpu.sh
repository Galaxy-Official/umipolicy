#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# 自动保存终端日志到 logs/，方便训练中断后排查。
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

# 四卡训练。若机器卡号不同，可运行前覆盖 CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG_NAME="${CONFIG_NAME:-pi05_430_clamp_seal}"
EXP_NAME="${EXP_NAME:-430_clamp_seal_0430_handcap_pi05_4gpu}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"

echo "=========================================="
echo "Starting OpenPI PI05 training"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: Data/430_clamp_seal_lerobot"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "=========================================="

# 第一次训练新数据集前必须计算 norm stats。
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
