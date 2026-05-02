#!/usr/bin/env bash

set -eo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_scripts/inference/run_eval_flexiv_tactile_linear_fusion.sh [options]

Options:
  --ckpt PATH                  Checkpoint path or output directory.
  --output PATH                Directory to save evaluation recording.
  --force_predict              Tell runtime the checkpoint predicts force.
  --force_guide                Use measured force to guide fusion.
  --frequency HZ               Control frequency.
  --steps_per_inference N      Receding-horizon steps per inference.
  --camera_config PATH         Handcap camera config json.
  -h, --help                   Show this help.
EOF
}

FORCE_PREDICT="${FORCE_PREDICT:-False}"
FORCE_GUIDE="${FORCE_GUIDE:-False}"
FUSION_METHOD="${FUSION_METHOD:-linear}"
FREQUENCY="${FREQUENCY:-5}"
STEPS_PER_INFERENCE="${STEPS_PER_INFERENCE:-4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt|--input)
            CKPT_PATH="$2"; shift 2 ;;
        --output)
            OUTPUT_DIR="$2"; shift 2 ;;
        --force_predict|--force-predict)
            FORCE_PREDICT=True; shift ;;
        --force_guide|--force-guide)
            FORCE_GUIDE=True; shift ;;
        --frequency|-f)
            FREQUENCY="$2"; shift 2 ;;
        --steps_per_inference|--steps-per-inference|-si)
            STEPS_PER_INFERENCE="$2"; shift 2 ;;
        --camera_config|--camera-config)
            CAMERA_CONFIG="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

export FLEXIV_ROBOT_IP="${FLEXIV_ROBOT_IP:-192.168.2.100}"
export FLEXIV_LOCAL_IP="${FLEXIV_LOCAL_IP:-192.168.2.102}"
export FLEXIV_ROBOT_SN="${FLEXIV_ROBOT_SN:-Rizon4-062339}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"
export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[-0.0009,-0.2370,-0.0133,2.1459,-0.0058,0.8170,-0.0012]}"

CKPT_PATH="${CKPT_PATH:-ckpt/umi_tactile_${FUSION_METHOD}_fusion.ckpt}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR:-data_local/umi_tactile_${FUSION_METHOD}_fusion/eval_${TIMESTAMP}}"

echo "=========================================="
echo "Starting Flexiv tactile policy inference"
echo "Fusion method: ${FUSION_METHOD}"
echo "Force predict: ${FORCE_PREDICT}"
echo "Force guide: ${FORCE_GUIDE}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

ARGS=(
    scripts_real/eval_flexiv.py
    --input "${CKPT_PATH}"
    --output "${OUTPUT_DIR}"
    --robot_ip "${FLEXIV_ROBOT_IP}"
    --local_ip "${FLEXIV_LOCAL_IP}"
    --frequency "${FREQUENCY}"
    --steps_per_inference "${STEPS_PER_INFERENCE}"
    --use_tactile
    --fusion_method "${FUSION_METHOD}"
)

if [[ -n "${CAMERA_CONFIG:-}" ]]; then
    ARGS+=(--camera_config "${CAMERA_CONFIG}")
fi
if [[ "${FORCE_PREDICT}" == "True" ]]; then
    ARGS+=(--force_predict)
fi
if [[ "${FORCE_GUIDE}" == "True" ]]; then
    ARGS+=(--force_guide)
fi

python "${ARGS[@]}"
