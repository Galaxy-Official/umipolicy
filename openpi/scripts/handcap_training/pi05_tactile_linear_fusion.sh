#!/bin/bash
set -e

usage() {
  cat <<'EOF'
Usage:
  scripts/handcap_training/pi05_tactile_linear_fusion.sh [options]

Options:
  --config-name NAME       OpenPI train config name.
  --force-predict          Use the registered force config, action output is 12D.
  --force-guide            Use the registered force config, force guides tactile fusion.
  --batch-size N           Training batch size.
  --num-train-steps N      Number of training steps.
  --save-interval N        Checkpoint save interval.
  --num-workers N          Data loader workers.
  --fsdp-devices N         FSDP device count.
  --overwrite              Overwrite checkpoint directory.
  -h, --help               Show this help.
EOF
}

FUSION_METHOD="${FUSION_METHOD:-linear}"
DEFAULT_CONFIG="pi05_simple_sorting_tactile_${FUSION_METHOD}_fusion"
FORCE_CONFIG="pi05_simple_sorting_tactile_${FUSION_METHOD}_force_fusion"
CONFIG_NAME="${CONFIG_NAME:-$DEFAULT_CONFIG}"
CONFIG_EXPLICIT="false"
FORCE_PREDICT="${FORCE_PREDICT:-false}"
FORCE_GUIDE="${FORCE_GUIDE:-false}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
OVERWRITE="${OVERWRITE:-true}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-name)
      CONFIG_NAME="$2"; CONFIG_EXPLICIT="true"; shift 2 ;;
    --force-predict)
      FORCE_PREDICT="true"; shift ;;
    --force-guide)
      FORCE_GUIDE="true"; shift ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --num-train-steps)
      NUM_TRAIN_STEPS="$2"; shift 2 ;;
    --save-interval)
      SAVE_INTERVAL="$2"; shift 2 ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --fsdp-devices)
      FSDP_DEVICES="$2"; shift 2 ;;
    --overwrite)
      OVERWRITE="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$CONFIG_EXPLICIT" == "false" && ( "$FORCE_PREDICT" == "true" || "$FORCE_GUIDE" == "true" ) ]]; then
  CONFIG_NAME="$FORCE_CONFIG"
  FORCE_PREDICT="true"
  FORCE_GUIDE="true"
fi

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "=========================================="
echo "Starting OpenPI pi05 tactile training"
echo "Config: ${CONFIG_NAME}"
echo "Fusion method: ${FUSION_METHOD}"
echo "Force predict: ${FORCE_PREDICT}"
echo "Force guide: ${FORCE_GUIDE}"
echo "=========================================="

python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

TRAIN_ARGS=(
  scripts/train.py
  "${CONFIG_NAME}"
  --batch-size "${BATCH_SIZE}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --save-interval "${SAVE_INTERVAL}"
  --num-workers "${NUM_WORKERS}"
  --no-wandb-enabled
  --fsdp-devices "${FSDP_DEVICES}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi

python "${TRAIN_ARGS[@]}"
