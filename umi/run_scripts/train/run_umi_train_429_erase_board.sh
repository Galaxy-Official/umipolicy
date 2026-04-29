#!/usr/bin/env bash

set -eo pipefail

# Visible GPU list. Inside this process, training.device=cuda:0 means the first
# GPU in CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_PATH="${DATASET_PATH:-data/429_erase_board.zarr}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/erase_board_train_0429}"
VISION_BACKBONE="${VISION_BACKBONE:-vit_base_patch16_224}"
VISION_CKPT="${VISION_CKPT:-ckpt/vit_b_16.pth}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
LOGGING_MODE="${LOGGING_MODE:-offline}"

echo "=========================================="
echo "Starting UMI Diffusion Policy training"
echo "Dataset path: ${DATASET_PATH}"
echo "Vision backbone: ${VISION_BACKBONE}"
echo "Vision checkpoint: ${VISION_CKPT}"
echo "Output path: ${OUTPUT_PATH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="

export HYDRA_FULL_ERROR=1

python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    hydra.run.dir="${OUTPUT_PATH}" \
    policy.obs_encoder.model_name="${VISION_BACKBONE}" \
    policy.obs_encoder.checkpoint_path="${VISION_CKPT}" \
    policy.obs_encoder.feature_aggregation=null \
    task.dataset.dataset_path="${DATASET_PATH}" \
    training.device="cuda:0" \
    dataloader.batch_size="${BATCH_SIZE}" \
    training.num_epochs="${NUM_EPOCHS}" \
    logging.mode="${LOGGING_MODE}"
