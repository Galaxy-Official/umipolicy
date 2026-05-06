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
export FLEXIV_ROBOT_SN="Rizon4-062339"
export FLEXIV_GRIPPER_NAME="Flexiv-GN01"
export FLEXIV_LOCAL_IP="192.168.2.102"
export FLEXIV_INIT_POSE="[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"

# --- 采集参数配置 ---
ROOT_DIR="Data/teleop"
TASK_NAME="box and block into paper box"
REPO_ID="${TASK_NAME}_$(date +"%Y%m%d_%H%M%S")"
TELEOP_TYPE="koch"                          # 可选: koch, so100
TELEOP_PORT="/dev/ttyUSB0"                  # 主臂串口路径 (Linux: /dev/ttyUSB0)
EPISODES=50                                 # 连续录入组数
EPISODE_TIME_S=50                           # 每组总时长 (秒)
FPS=20

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

# 确保串口权限 (需要 sudo 密码)
if [ -e "${TELEOP_PORT}" ]; then
    echo "Granting permissions to ${TELEOP_PORT}..."
    sudo chmod 777 "${TELEOP_PORT}"
else
    echo "Warning: ${TELEOP_PORT} does not exist!"
fi

TACTILE_FLAG=""
if [ "${USE_TACTILE}" = "true" ]; then
    TACTILE_FLAG="--use_tactile"
fi

START_TIME=$(date +%s)

function on_exit {
    # Ensure this only runs once
    if [ -z "${EXIT_PROCESSED}" ]; then
        export EXIT_PROCESSED=1
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Compute hours, minutes, seconds
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))
        FORMATTED_TIME=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

        # Append to summary file
        SUMMARY_FILE="${ROOT_DIR}/${REPO_ID}/collection_summary.txt"
        if [ -f "$SUMMARY_FILE" ]; then
            if ! grep -q "Total Duration (seconds)" "$SUMMARY_FILE"; then
                echo "Total Duration (seconds): ${DURATION}" >> "$SUMMARY_FILE"
                echo "Total Duration (formatted): ${FORMATTED_TIME}" >> "$SUMMARY_FILE"
            fi
            
            echo ""
            echo "====================================================="
            echo "             Collection Finished!                    "
            echo "====================================================="
            cat "$SUMMARY_FILE"
            echo "====================================================="
        fi
    fi
}

# Trap EXIT and SIGINT so the summary is always printed, even if Ctrl+C is pressed
trap on_exit EXIT SIGINT

python -m lerobot.scripts.lerobot_flexiv_teleop_record \
    --repo-id "${REPO_ID}" \
    --root "${ROOT_DIR}" \
    --teleop "${TELEOP_TYPE}" \
    --teleop_port "${TELEOP_PORT}" \
    --num_episodes ${EPISODES} \
    --episode_time_s ${EPISODE_TIME_S} \
    --fps ${FPS} \
    --single-task "${TASK_NAME}" \
    ${TACTILE_FLAG}

