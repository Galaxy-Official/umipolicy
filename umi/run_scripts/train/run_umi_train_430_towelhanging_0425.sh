#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/../.."

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "${TRAIN_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

# Use physical GPU 1. Inside this process, training.device=cuda:0 maps to it.
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

DATASET_PATH="${DATASET_PATH:-data/430towelhanging_umi.zarr}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/430towelhanging_train_0425}"
VISION_BACKBONE="${VISION_BACKBONE:-vit_base_patch16_224}"
VISION_CKPT="${VISION_CKPT:-ckpt/vit_b_16.pth}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
LOGGING_MODE="${LOGGING_MODE:-offline}"

echo "=========================================="
echo "Starting UMI Diffusion Policy training"
echo "Dataset path: ${DATASET_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, training.device=cuda:0"
echo "Vision backbone: ${VISION_BACKBONE}"
echo "Vision checkpoint: ${VISION_CKPT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Logging mode: ${LOGGING_MODE}"
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
    training.num_epochs="${NUM_EPOCHS}" \
    logging.mode="${LOGGING_MODE}"
