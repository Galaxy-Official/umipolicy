#!/bin/bash
# 启动 Flexiv 与 Koch 主从段遥操记录流程 (LeRobot 3.0标准)

# 确保脚本发生错误时立刻停止
set -e

# 获取脚本所在的目录 (支持从任何路径执行)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- 环境变量配置 ---
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_INIT_POSE="[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"

# --- 采集参数配置 ---
REPO_ID="umipolicy/handcap_flexiv_demo"
TELEOP_TYPE="koch"                          # 可选: koch, so100
TELEOP_PORT="/dev/tty.usbserial-110"        # 主臂串口路径
EPISODES=15                                 # 连续录入组数
EPISODE_TIME_S=60                           # 每组总时长 (秒)
FPS=30
TASK_NAME="teleop grasp target"

echo "====================================================="
echo " Starting LeRobot v3.0 Flexiv Teleoperation Pipeline "
echo "====================================================="
echo " Leader: ${TELEOP_TYPE} on ${TELEOP_PORT}"
echo " Repository: ${REPO_ID}"
echo " Configuration: ${EPISODES} episodes x ${EPISODE_TIME_S} sec @ ${FPS} Hz"
echo "====================================================="
echo ""

# 调用刚编写好的Python脚本
python ${SCRIPT_DIR}/../../lerobot/scripts/lerobot_flexiv_teleop_record.py \
    --repo-id "${REPO_ID}" \
    --teleop "${TELEOP_TYPE}" \
    --teleop_port "${TELEOP_PORT}" \
    --num_episodes ${EPISODES} \
    --episode_time_s ${EPISODE_TIME_S} \
    --fps ${FPS} \
    --single-task "${TASK_NAME}"
