#!/bin/bash
# # (Conda is managed by the user's terminal environment)

# 获取脚本所在的目录 (支持从任何路径执行)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_DIR="${SCRIPT_DIR}/../.."

# 设置 PYTHONPATH，确保 lerobot 可被 import
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

# --- 环境变量配置 (兼容 RDK 1.0+) ---
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_ROBOT_SN="Rizon4-062339"
export FLEXIV_GRIPPER_NAME="Flexiv-GN01"
export FLEXIV_LOCAL_IP="192.168.2.102"

# Replay 默认直接发送 EEF Cartesian pose，不再通过 reachable() 解 IK。
export FLEXIV_DIRECT_EEF_CONTROL="${FLEXIV_DIRECT_EEF_CONTROL:-1}"
export FLEXIV_ARM_MAX_LINEAR_VEL="${FLEXIV_ARM_MAX_LINEAR_VEL:-0.05}"
export FLEXIV_ARM_MAX_ANGULAR_VEL="${FLEXIV_ARM_MAX_ANGULAR_VEL:-0.2}"
export FLEXIV_ARM_MAX_LINEAR_ACC="${FLEXIV_ARM_MAX_LINEAR_ACC:-0.1}"
export FLEXIV_ARM_MAX_ANGULAR_ACC="${FLEXIV_ARM_MAX_ANGULAR_ACC:-0.3}"
export FLEXIV_GRIPPER_MOVE_VELOCITY="${FLEXIV_GRIPPER_MOVE_VELOCITY:-0.03}"
export FLEXIV_GRIPPER_MOVE_FORCE="${FLEXIV_GRIPPER_MOVE_FORCE:-20}"

# simple sorting 409
export FLEXIV_INIT_POSE="[-0.0009,-0.1701,-0.0133,2.0214,-0.0058,0.6921,-0.0012]"



echo "====================================================="
echo " Starting LeRobot v3.0 Flexiv Replay Pipeline"
echo "====================================================="
echo "Direct EEF control: ${FLEXIV_DIRECT_EEF_CONTROL}"
echo "Cartesian linear vel/acc:  ${FLEXIV_ARM_MAX_LINEAR_VEL} / ${FLEXIV_ARM_MAX_LINEAR_ACC}"
echo "Cartesian angular vel/acc: ${FLEXIV_ARM_MAX_ANGULAR_VEL} / ${FLEXIV_ARM_MAX_ANGULAR_ACC}"
echo "Gripper vel/force: ${FLEXIV_GRIPPER_MOVE_VELOCITY} / ${FLEXIV_GRIPPER_MOVE_FORCE}"

python "${SRC_DIR}/lerobot/scripts/lerobot_replay_train.py" \
    --data_root "Data/replay/508_open_bottle_lerobot_health100" \
    --episode_index 0 \
    --task_name "eval_lerobot_replay"
