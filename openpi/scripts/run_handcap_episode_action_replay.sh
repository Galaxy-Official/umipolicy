#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: Neither python nor python3 was found. Activate the openpi env or set PYTHON_BIN=/path/to/python."
    exit 1
  fi
fi

# Defaults are for 513_screw 350-data checkpoint at 20000 steps.
# Override any of these from the shell, e.g.:
#   CONFIG_NAME=pi05_510_erase_board_350 CHECKPOINT_DIR=ckpt/lihong/510_erase_board_350_vision_20000 bash scripts/run_handcap_episode_action_replay.sh
CONFIG_NAME="${CONFIG_NAME:-pi05_513_screw_350}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-20000}"
EXP_NAME="${EXP_NAME:-513_screw_350_handcap_pi05_4gpu_vision_only}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}}"

EPISODE_INDEX="${EPISODE_INDEX:-0}"
START_OFFSET="${START_OFFSET:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
PLOT_HORIZON="${PLOT_HORIZON:-20}"
VIDEO_FPS="${VIDEO_FPS:-10}"
PYTORCH_DEVICE="${PYTORCH_DEVICE:-}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/episode_replay/${CONFIG_NAME}_${CHECKPOINT_STEP}_ep${EPISODE_INDEX}}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-${OUTPUT_DIR}/episode_replay.mp4}"

echo "=========================================="
echo "HandCap offline episode action replay"
echo "Config: ${CONFIG_NAME}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Episode: ${EPISODE_INDEX}"
echo "Start offset: ${START_OFFSET}"
echo "Max frames: ${MAX_FRAMES}"
echo "Frame stride: ${FRAME_STRIDE}"
echo "Plot horizon: ${PLOT_HORIZON}"
echo "Video FPS: ${VIDEO_FPS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Output video: ${OUTPUT_VIDEO}"
echo "Python: ${PYTHON_BIN}"
echo "=========================================="

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: checkpoint directory not found: ${CHECKPOINT_DIR}"
  echo "Set CHECKPOINT_DIR=/path/to/checkpoint or adjust CONFIG_NAME/EXP_NAME/CHECKPOINT_STEP."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  scripts/replay_handcap_episode_action_video.py
  --config-name "${CONFIG_NAME}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --episode-index "${EPISODE_INDEX}"
  --output-dir "${OUTPUT_DIR}"
  --output-video "${OUTPUT_VIDEO}"
  --start-offset "${START_OFFSET}"
  --max-frames "${MAX_FRAMES}"
  --frame-stride "${FRAME_STRIDE}"
  --plot-horizon "${PLOT_HORIZON}"
  --video-fps "${VIDEO_FPS}"
)

if [[ -n "${PYTORCH_DEVICE}" ]]; then
  CMD+=(--pytorch-device "${PYTORCH_DEVICE}")
fi

if [[ -n "${DEFAULT_PROMPT}" ]]; then
  CMD+=(--default-prompt "${DEFAULT_PROMPT}")
fi

"${CMD[@]}"
