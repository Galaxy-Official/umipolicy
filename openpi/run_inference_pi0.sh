#!/bin/bash
# 这是一个便捷启动脚本，你可以随时修改里面的参数

export FLEXIV_ROBOT_IP="${FLEXIV_ROBOT_IP:-192.168.2.100}"
export FLEXIV_LOCAL_IP="${FLEXIV_LOCAL_IP:-192.168.2.102}"
export FLEXIV_ROBOT_SN="${FLEXIV_ROBOT_SN:-Rizon4-062339}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"

bash start_handcap_remote_inference.sh \
  --policy-config pi0_simple_sorting \
  --policy-dir ~/umipolicy/openpi/ckpt/99999 \
  --robot-ip "${FLEXIV_ROBOT_IP}" \
  --local-ip "${FLEXIV_LOCAL_IP}" \
  --prompt "simple sorting task" \
  --startup-timeout 1800 \
  --no-use-tactile
