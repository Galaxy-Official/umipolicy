#!/bin/bash
# PI05 真机远程推理便捷启动脚本。
# 机械臂、MVS 相机、执行频率、action 执行步数逻辑与 run_inference_pi0.sh 保持一致。

export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_LOCAL_IP="192.168.2.102"
export FLEXIV_ROBOT_SN="${FLEXIV_ROBOT_SN:-Rizon4-062339}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"

POLICY_CONFIG="${POLICY_CONFIG:-pi05_simple_sorting}"
POLICY_DIR="${POLICY_DIR:-~/umipolicy/openpi/ckpt/99999}"
PROMPT="${PROMPT:-simple sorting task}"

CTRL_FREQ="${CTRL_FREQ:-20}"
STEPS_PER_INFERENCE="${STEPS_PER_INFERENCE:-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"

bash start_handcap_remote_inference.sh \
  --policy-config "${POLICY_CONFIG}" \
  --policy-dir "${POLICY_DIR}" \
  --robot-ip "${FLEXIV_ROBOT_IP}" \
  --local-ip "${FLEXIV_LOCAL_IP}" \
  --prompt "${PROMPT}" \
  --ctrl-freq "${CTRL_FREQ}" \
  --steps-per-inference "${STEPS_PER_INFERENCE}" \
  --obs-horizon "${OBS_HORIZON}" \
  --action-latency "${ACTION_LATENCY}" \
  --startup-timeout 1800 \
  --no-use-tactile
