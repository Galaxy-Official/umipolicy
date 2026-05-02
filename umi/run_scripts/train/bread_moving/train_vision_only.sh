#!/usr/bin/env bash

set -eo pipefail

cd "$(dirname "$0")/../../.."

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "${TRAIN_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

# Start background monitors
nvidia-smi dmon -s mu -d 5 > "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}_gpu_util.log" &
GPU_MON_PID=$!

top -b -d 5 > "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}_cpu_util.log" &
CPU_MON_PID=$!

# Ensure monitors are stopped when the script exits
cleanup() {
    echo "Stopping resource monitors..."
    kill $GPU_MON_PID $CPU_MON_PID 2>/dev/null || true
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1

DATASET_PATH="${DATASET_PATH:-data/501breadmoving_umi.zarr}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/501breadmoving_vision_only}"
VISION_BACKBONE="${VISION_BACKBONE:-vit_base_patch16_224}"
VISION_CKPT="${VISION_CKPT:-ckpt/vit_b_16.pth}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
LOGGING_MODE="${LOGGING_MODE:-offline}"

echo "=========================================="
echo "Starting UMI Vision Only Diffusion Policy training"
echo "Dataset path: ${DATASET_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Vision backbone: ${VISION_BACKBONE}"
echo "Vision checkpoint: ${VISION_CKPT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="

python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    hydra.run.dir="${OUTPUT_PATH}" \
    +task.use_tactile=False \
    policy.obs_encoder.model_name="${VISION_BACKBONE}" \
    policy.obs_encoder.checkpoint_path="${VISION_CKPT}" \
    policy.obs_encoder.feature_aggregation=null \
    task.dataset.dataset_path="${DATASET_PATH}" \
    training.device="cuda:0" \
    dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    dataloader.persistent_workers=True \
    val_dataloader.batch_size="${BATCH_SIZE}" \
    val_dataloader.num_workers="${NUM_WORKERS}" \
    val_dataloader.persistent_workers=True \
    training.num_epochs="${NUM_EPOCHS}" \
    logging.mode="${LOGGING_MODE}"
