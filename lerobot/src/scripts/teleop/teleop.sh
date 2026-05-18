#!/bin/bash
# 启动 Flexiv 与 Koch 主从段遥操记录流程 (LeRobot 3.0标准)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_DIR="${SCRIPT_DIR}/../.."

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_ROBOT_SN="Rizon4-062339"
export FLEXIV_GRIPPER_NAME="Flexiv-GN01"
export FLEXIV_LOCAL_IP="192.168.2.102"
export FLEXIV_INIT_POSE="[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"

ROOT_DIR="Data/teleop/yibo"
TASK_NAME="box_and_block_into_paper_box"
REPO_ID="${TASK_NAME}_$(date +"%Y%m%d_%H%M%S")"
TELEOP_TYPE="koch"
TELEOP_PORT="/dev/ttyUSB0"
EPISODES=50
EPISODE_TIME_S=50
FPS=20
USE_TACTILE=false

SAFETY_FILE="${SRC_DIR}/lerobot/common/robot_devices/robots/flexiv_safety.py"
SAFETY_BACKUP="${SAFETY_FILE}.bak"

echo "====================================================="
echo " Starting LeRobot v3.0 Flexiv Teleoperation Pipeline "
echo "====================================================="
echo " Leader:       ${TELEOP_TYPE} on ${TELEOP_PORT}"
echo " Repository:   ${REPO_ID}"
echo " Config:       ${EPISODES} episodes x ${EPISODE_TIME_S}s @ ${FPS} Hz"
echo " Tactile cams: ${USE_TACTILE}"
echo "====================================================="
echo ""

cd "${SRC_DIR}"

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

# 备份并禁用安全边界
cp "${SAFETY_FILE}" "${SAFETY_BACKUP}"
sed -i 's/X_MIN = [0-9.-]*/X_MIN = -10.0/' "${SAFETY_FILE}"
sed -i 's/X_MAX = [0-9.]*/X_MAX = 10.0/' "${SAFETY_FILE}"
sed -i 's/Y_MIN = -[0-9.]*/Y_MIN = -10.0/' "${SAFETY_FILE}"
sed -i 's/Y_MAX = [0-9.]*/Y_MAX = 10.0/' "${SAFETY_FILE}"
sed -i 's/Z_MIN_BASE = [0-9.]*/Z_MIN_BASE = -10.0/' "${SAFETY_FILE}"
sed -i 's/Z_MAX = [0-9.]*/Z_MAX = 10.0/' "${SAFETY_FILE}"
echo "Safety boundaries disabled."

START_TIME=$(date +%s)

function on_exit {
    if [ -z "${EXIT_PROCESSED}" ]; then
        export EXIT_PROCESSED=1

        # 还原安全边界
        if [ -f "${SAFETY_BACKUP}" ]; then
            cp "${SAFETY_BACKUP}" "${SAFETY_FILE}"
            rm "${SAFETY_BACKUP}"
            echo "Safety boundaries restored."
        fi

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))
        FORMATTED_TIME=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

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