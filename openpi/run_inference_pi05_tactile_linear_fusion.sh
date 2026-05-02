#!/bin/bash

set -eo pipefail

export FLEXIV_ROBOT_IP="${FLEXIV_ROBOT_IP:-192.168.2.100}"
export FLEXIV_LOCAL_IP="${FLEXIV_LOCAL_IP:-192.168.2.102}"
export FLEXIV_ROBOT_SN="${FLEXIV_ROBOT_SN:-Rizon4-062339}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"
export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[-0.0009,-0.2370,-0.0133,1.9935,-0.0058,0.6496,-0.0012]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

FUSION_METHOD="${FUSION_METHOD:-linear}"
FORCE_PREDICT="${FORCE_PREDICT:-false}"
FORCE_GUIDE="${FORCE_GUIDE:-false}"

if [[ "${FORCE_PREDICT}" == "true" || "${FORCE_GUIDE}" == "true" ]]; then
  POLICY_CONFIG="${POLICY_CONFIG:-pi05_simple_sorting_tactile_${FUSION_METHOD}_force_fusion}"
  FORCE_PREDICT="true"
  FORCE_GUIDE="true"
else
  POLICY_CONFIG="${POLICY_CONFIG:-pi05_simple_sorting_tactile_${FUSION_METHOD}_fusion}"
fi

POLICY_DIR="${POLICY_DIR:-/home/rhos/umipolicy/openpi/ckpt/pi05_tactile_${FUSION_METHOD}_fusion}"
PROMPT="${PROMPT:-simple sorting task}"

CTRL_FREQ="${CTRL_FREQ:-5}"
STEPS_PER_INFERENCE="${STEPS_PER_INFERENCE:-20}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"

ARGS=(
  start_handcap_remote_inference.sh
  --policy-config "${POLICY_CONFIG}"
  --policy-dir "${POLICY_DIR}"
  --robot-ip "${FLEXIV_ROBOT_IP}"
  --local-ip "${FLEXIV_LOCAL_IP}"
  --prompt "${PROMPT}"
  --ctrl-freq "${CTRL_FREQ}"
  --steps-per-inference "${STEPS_PER_INFERENCE}"
  --obs-horizon "${OBS_HORIZON}"
  --action-latency "${ACTION_LATENCY}"
  --init-qpos "${FLEXIV_INIT_POSE}"
  --startup-timeout 1800
  --use-tactile
)

if [[ "${FORCE_PREDICT}" == "true" ]]; then
  ARGS+=(--force-predict)
fi
if [[ "${FORCE_GUIDE}" == "true" ]]; then
  ARGS+=(--force-guide)
fi

bash "${ARGS[@]}"
