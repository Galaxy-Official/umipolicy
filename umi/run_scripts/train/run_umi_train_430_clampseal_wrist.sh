#!/usr/bin/env bash

set -eo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_scripts/train/run_umi_train_430_clampseal_wrist.sh [options]

Options:
  --dataset_path PATH      UMI zarr dataset path.
  --output_path PATH       Training output directory.
  --vision_backbone NAME   Wrist RGB timm backbone.
  --vision_ckpt PATH       Wrist RGB checkpoint path.
  --batch_size N           Training batch size.
  --num_epochs N           Number of epochs.
  --logging_mode MODE      wandb mode, e.g. offline or online.
  -h, --help               Show this help.

Environment variables with the same uppercase names are also supported.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_path|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_path|--output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --vision_backbone|--vision-backbone|--wrist_backbone|--wrist-backbone)
            VISION_BACKBONE="$2"
            shift 2
            ;;
        --vision_ckpt|--vision-ckpt|--wrist_ckpt|--wrist-ckpt)
            VISION_CKPT="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs|--num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --logging_mode|--logging-mode)
            LOGGING_MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "${TRAIN_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

# Visible GPU list. Inside this process, training.device=cuda:0 means the first
# GPU in CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_PATH="${DATASET_PATH:-data/430clampseal.zarr}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/430clampseal_wrist_train}"
VISION_BACKBONE="${VISION_BACKBONE:-vit_base_patch16_224}"
VISION_CKPT="${VISION_CKPT:-ckpt/vit_b_16.pth}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
LOGGING_MODE="${LOGGING_MODE:-offline}"

echo "=========================================="
echo "Starting UMI wrist-only Diffusion Policy training"
echo "Dataset path: ${DATASET_PATH}"
echo "use_tactile: False"
echo "Vision backbone: ${VISION_BACKBONE}"
echo "Vision checkpoint: ${VISION_CKPT}"
echo "Output path: ${OUTPUT_PATH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="

export HYDRA_FULL_ERROR=1

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
