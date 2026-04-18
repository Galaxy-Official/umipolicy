#!/bin/bash
# 启动 Flexiv 与 Koch 主从段遥操记录流程 (LeRobot 3.0标准)

# 确保脚本发生错误时立刻停止
set -e

# 获取脚本所在的目录 (支持从任何路径执行)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_DIR="${SCRIPT_DIR}/../.."

# 设置 PYTHONPATH，确保 lerobot 可被 import
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

# --- 环境变量配置 ---
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_LOCAL_IP="192.168.2.102"
export FLEXIV_INIT_POSE="[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"

# --- 采集参数配置 ---
REPO_ID="umipolicy/handcap_flexiv_demo"
TELEOP_TYPE="koch"                          # 可选: koch, so100
TELEOP_PORT="/dev/ttyUSB0"                  # 主臂串口路径 (Linux: /dev/ttyUSB0)
EPISODES=15                                 # 连续录入组数
EPISODE_TIME_S=60                           # 每组总时长 (秒)
FPS=30
TASK_NAME="teleop grasp target"

# --- 相机模式 ---
# 设为 true 开启触觉相机 (webcam), false 仅使用 MVS 工业相机
USE_TACTILE=false

echo "====================================================="
echo " Starting LeRobot v3.0 Flexiv Teleoperation Pipeline "
echo "====================================================="
echo " Leader:       ${TELEOP_TYPE} on ${TELEOP_PORT}"
echo " Repository:   ${REPO_ID}"
echo " Config:       ${EPISODES} episodes x ${EPISODE_TIME_S}s @ ${FPS} Hz"
echo " Tactile cams: ${USE_TACTILE}"
echo "====================================================="
echo ""

# 使用 python -m 模块调用方式（与推理脚本保持一致）
cd "${SRC_DIR}"

TACTILE_FLAG=""
if [ "${USE_TACTILE}" = "true" ]; then
    TACTILE_FLAG="--use_tactile"
fi

python -m lerobot.scripts.lerobot_flexiv_teleop_record \
    --repo-id "${REPO_ID}" \
    --teleop "${TELEOP_TYPE}" \
    --teleop_port "${TELEOP_PORT}" \
    --num_episodes ${EPISODES} \
    --episode_time_s ${EPISODE_TIME_S} \
    --fps ${FPS} \
    --single-task "${TASK_NAME}" \
    ${TACTILE_FLAG}
