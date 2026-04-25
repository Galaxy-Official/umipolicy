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
export FLEXIV_INIT_POSE="[0.0000,-0.3455,-0.0001,1.9271,0.0003,0.6979,0.0000]"

echo "====================================================="
echo " Starting LeRobot v3.0 Flexiv Replay Pipeline"
echo "====================================================="

python "${SRC_DIR}/lerobot/scripts/lerobot_replay_train.py" \
    --data_root "Data/replay/simple_sorting_409/handcap" \
    --episode_index 0 \
    --task_name "eval_lerobot_replay"
