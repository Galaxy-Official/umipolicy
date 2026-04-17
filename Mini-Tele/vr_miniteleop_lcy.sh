#!/usr/bin/env bash
set -euo pipefail

TASK_NAME=${TASK_NAME:-fry-test}
QUEST_STREAM_BIND_HOST=${QUEST_STREAM_BIND_HOST:-0.0.0.0}
QUEST_STREAM_PORT=${QUEST_STREAM_PORT:-8090}
QUEST_STREAM_CAMERA_KEY=${QUEST_STREAM_CAMERA_KEY:-observation.images.side}
QUEST_STREAM_FPS=${QUEST_STREAM_FPS:-15}
QUEST_STREAM_JPEG_QUALITY=${QUEST_STREAM_JPEG_QUALITY:-80}
QUEST_STREAM_ADVERTISE_HOST=${QUEST_STREAM_ADVERTISE_HOST:-auto}

export LEROBOT_HOME="${LEROBOT_HOME:-$HOME/${TASK_NAME}/$(date +"%m-%d-%H-%M-%S")}"

python lerobot/scripts/MiniTeleop_record.py \
  --robot-path lerobot/configs/robot/koch_flexiv.yaml \
  --warmup-time-s 5 \
  --episode-time-s 120 \
  --reset-time-s 10 \
  --num-episodes 25 \
  --push-to-hub 0 \
  --single-task "Fry." \
  --fps 30 \
  --quest-stream 1 \
  --quest-stream-bind-host "${QUEST_STREAM_BIND_HOST}" \
  --quest-stream-port "${QUEST_STREAM_PORT}" \
  --quest-stream-camera-key "${QUEST_STREAM_CAMERA_KEY}" \
  --quest-stream-fps "${QUEST_STREAM_FPS}" \
  --quest-stream-jpeg-quality "${QUEST_STREAM_JPEG_QUALITY}" \
  --quest-stream-advertise-host "${QUEST_STREAM_ADVERTISE_HOST}"
