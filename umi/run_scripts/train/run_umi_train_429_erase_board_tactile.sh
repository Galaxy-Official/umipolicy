#!/usr/bin/env bash

set -eo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_scripts/train/run_umi_train_429_erase_board_tactile.sh [options]

Options:
  --dataset_path PATH          UMI zarr dataset path.
  --output_path PATH           Training output directory.
  --wrist_backbone NAME        Wrist RGB timm backbone.
  --wrist_ckpt PATH            Wrist RGB checkpoint path.
  --tactile_backbone NAME      Tactile timm backbone.
  --tactile_ckpt PATH          Tactile checkpoint path. Defaults to wrist ckpt
                               only when the tactile backbone matches wrist.
  --batch_size N               Training batch size.
  --num_epochs N               Number of epochs.
  --logging_mode MODE          wandb mode, e.g. offline or online.
  -h, --help                   Show this help.

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
        --wrist_backbone|--wrist-backbone|--vision_backbone|--vision-backbone)
            WRIST_BACKBONE="$2"
            shift 2
            ;;
        --wrist_ckpt|--wrist-ckpt|--vision_ckpt|--vision-ckpt)
            WRIST_CKPT="$2"
            shift 2
            ;;
        --tactile_backbone|--tactile-backbone)
            TACTILE_BACKBONE="$2"
            shift 2
            ;;
        --tactile_ckpt|--tactile-ckpt)
            TACTILE_CKPT="$2"
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

DATASET_PATH="${DATASET_PATH:-data/429_erase_board.zarr}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/erase_board_train_0429_tactile}"
WRIST_BACKBONE="${WRIST_BACKBONE:-vit_base_patch16_224}"
WRIST_CKPT="${WRIST_CKPT:-ckpt/vit_b_16.pth}"
TACTILE_BACKBONE="${TACTILE_BACKBONE:-vit_base_patch16_224}"
TACTILE_CKPT="${TACTILE_CKPT:-null}"
WRIST_FEATURE_AGGREGATION="${WRIST_FEATURE_AGGREGATION:-null}"
TACTILE_FEATURE_AGGREGATION="${TACTILE_FEATURE_AGGREGATION:-null}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
LOGGING_MODE="${LOGGING_MODE:-offline}"

echo "=========================================="
echo "Starting UMI tactile Diffusion Policy training"
echo "Dataset path: ${DATASET_PATH}"
echo "Wrist backbone: ${WRIST_BACKBONE}"
echo "Wrist checkpoint: ${WRIST_CKPT}"
echo "Tactile backbone: ${TACTILE_BACKBONE}"
echo "Tactile checkpoint: ${TACTILE_CKPT}"
echo "Output path: ${OUTPUT_PATH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="

export HYDRA_FULL_ERROR=1

python train.py --config-name=train_diffusion_unet_timm_umi_tactile_workspace \
    hydra.run.dir="${OUTPUT_PATH}" \
    task.use_tactile=True \
    wrist_backbone="${WRIST_BACKBONE}" \
    wrist_checkpoint_path="${WRIST_CKPT}" \
    tactile_backbone="${TACTILE_BACKBONE}" \
    tactile_checkpoint_path="${TACTILE_CKPT}" \
    policy.obs_encoder.feature_aggregation="${WRIST_FEATURE_AGGREGATION}" \
    policy.obs_encoder.tactile_feature_aggregation="${TACTILE_FEATURE_AGGREGATION}" \
    task.dataset.dataset_path="${DATASET_PATH}" \
    training.device="cuda:0" \
    dataloader.batch_size="${BATCH_SIZE}" \
    training.num_epochs="${NUM_EPOCHS}" \
    logging.mode="${LOGGING_MODE}"
