#!/bin/bash
# 启动 Flexiv 机器人真实轨迹重放 (Replay) 脚本

set -e

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
export FLEXIV_INIT_POSE="[-0.0000,-0.6987,-0.0001,1.5711,0.0003,0.6979,0.0000]"

echo "====================================================="
echo " Starting LeRobot v3.0 Flexiv Replay Pipeline"
echo "====================================================="

# 切换到 src 目录执行
cd "${SRC_DIR}"

python -m lerobot.scripts.lerobot_real_replay_train \
    --task_name "test_replay"
