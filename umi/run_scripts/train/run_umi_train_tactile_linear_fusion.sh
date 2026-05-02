#!/usr/bin/env bash

set -eo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_scripts/train/run_umi_train_tactile_linear_fusion.sh [options]

Options:
  --dataset_path PATH          UMI zarr dataset path.
  --output_path PATH           Training output directory.
  --force_predict              Predict 2D force together with action.
  --force_guide                Use measured force to weight vision/tactile fusion.
  --wrist_backbone NAME        Wrist RGB timm backbone.
  --wrist_ckpt PATH            Wrist RGB checkpoint path.
  --tactile_backbone NAME      Tactile timm backbone.
  --tactile_ckpt PATH          Tactile checkpoint path.
  --batch_size N               Training batch size.
  --num_epochs N               Number of epochs.
  --logging_mode MODE          wandb mode, e.g. offline or online.
  -h, --help                   Show this help.
EOF
}

FORCE_PREDICT="${FORCE_PREDICT:-False}"
FORCE_GUIDE="${FORCE_GUIDE:-False}"
FUSION_METHOD="${FUSION_METHOD:-linear}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_path|--dataset-path)
            DATASET_PATH="$2"; shift 2 ;;
        --output_path|--output-path)
            OUTPUT_PATH="$2"; shift 2 ;;
        --force_predict|--force-predict)
            FORCE_PREDICT=True; shift ;;
        --force_guide|--force-guide)
            FORCE_GUIDE=True; shift ;;
        --wrist_backbone|--wrist-backbone|--vision_backbone|--vision-backbone)
            WRIST_BACKBONE="$2"; shift 2 ;;
        --wrist_ckpt|--wrist-ckpt|--vision_ckpt|--vision-ckpt)
            WRIST_CKPT="$2"; shift 2 ;;
        --tactile_backbone|--tactile-backbone)
            TACTILE_BACKBONE="$2"; shift 2 ;;
        --tactile_ckpt|--tactile-ckpt)
            TACTILE_CKPT="$2"; shift 2 ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --num_epochs|--num-epochs)
            NUM_EPOCHS="$2"; shift 2 ;;
        --logging_mode|--logging-mode)
            LOGGING_MODE="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "${TRAIN_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1

DATASET_PATH="${DATASET_PATH:-data/umi_tactile_force.zarr}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/umi_tactile_${FUSION_METHOD}_fusion}"
WRIST_BACKBONE="${WRIST_BACKBONE:-vit_base_patch16_224}"
WRIST_CKPT="${WRIST_CKPT:-ckpt/vit_b_16.pth}"
TACTILE_BACKBONE="${TACTILE_BACKBONE:-vit_base_patch16_224}"
TACTILE_CKPT="${TACTILE_CKPT:-null}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
LOGGING_MODE="${LOGGING_MODE:-offline}"

echo "=========================================="
echo "Starting UMI tactile Diffusion Policy training"
echo "Fusion method: ${FUSION_METHOD}"
echo "Force predict: ${FORCE_PREDICT}"
echo "Force guide: ${FORCE_GUIDE}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Wrist backbone: ${WRIST_BACKBONE}"
echo "Tactile backbone: ${TACTILE_BACKBONE}"
echo "=========================================="

python train.py --config-name=train_diffusion_unet_timm_umi_tactile_fusion_workspace \
    hydra.run.dir="${OUTPUT_PATH}" \
    fusion_method="${FUSION_METHOD}" \
    use_tactile=True \
    force_predict="${FORCE_PREDICT}" \
    force_guide="${FORCE_GUIDE}" \
    wrist_backbone="${WRIST_BACKBONE}" \
    wrist_checkpoint_path="${WRIST_CKPT}" \
    tactile_backbone="${TACTILE_BACKBONE}" \
    tactile_checkpoint_path="${TACTILE_CKPT}" \
    task.dataset.dataset_path="${DATASET_PATH}" \
    training.device="cuda:0" \
    dataloader.batch_size="${BATCH_SIZE}" \
    val_dataloader.batch_size="${BATCH_SIZE}" \
    training.num_epochs="${NUM_EPOCHS}" \
    logging.mode="${LOGGING_MODE}"
