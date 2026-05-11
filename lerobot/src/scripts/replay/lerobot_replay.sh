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

# Keep arm translation conservative, but let the wrist joints rotate faster.
export FLEXIV_BASE_MAX_VEL="${FLEXIV_BASE_MAX_VEL:-0.1}"
export FLEXIV_BASE_MAX_ACC="${FLEXIV_BASE_MAX_ACC:-0.1}"
export FLEXIV_WRIST_MAX_VEL="${FLEXIV_WRIST_MAX_VEL:-0.3}"
export FLEXIV_WRIST_MAX_ACC="${FLEXIV_WRIST_MAX_ACC:-0.3}"
export FLEXIV_WRIST_JOINT_COUNT="${FLEXIV_WRIST_JOINT_COUNT:-3}"

# simple sorting 409
export FLEXIV_INIT_POSE="[-0.0009,-0.1701,-0.0133,2.0214,-0.0058,0.6921,-0.0012]"



echo "====================================================="
echo " Starting LeRobot v3.0 Flexiv Replay Pipeline"
echo "====================================================="
echo "Base joint limits:  vel=${FLEXIV_BASE_MAX_VEL}, acc=${FLEXIV_BASE_MAX_ACC}"
echo "Wrist joint limits: vel=${FLEXIV_WRIST_MAX_VEL}, acc=${FLEXIV_WRIST_MAX_ACC}, count=${FLEXIV_WRIST_JOINT_COUNT}"

python "${SRC_DIR}/lerobot/scripts/lerobot_replay_train.py" \
    --data_root "Data/replay/508_open_bottle_lerobot_health100" \
    --episode_index 0 \
    --task_name "eval_lerobot_replay"
