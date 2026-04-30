#!/bin/bash
# PI05 真机远程推理便捷启动脚本。
# 机械臂、MVS 相机、执行频率、action 执行步数逻辑与 run_inference_pi0.sh 保持一致。

export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_LOCAL_IP="192.168.2.102"
export FLEXIV_ROBOT_SN="${FLEXIV_ROBOT_SN:-Rizon4-062339}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"
export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[-0.0009,-0.2370,-0.0133,1.9935,-0.0058,0.6496,-0.0012]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

POLICY_CONFIG="pi05_yellow_to_pink"
POLICY_DIR="/home/rhos/umipolicy/openpi/ckpt/70000"
PROMPT="yellow to pink task"

CTRL_FREQ="5"
STEPS_PER_INFERENCE="20"
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
  --init-qpos "${FLEXIV_INIT_POSE}" \
  --startup-timeout 1800 \
  --no-use-tactile
