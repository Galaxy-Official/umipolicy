#!/bin/bash
# PI05 真机远程推理 + 轨迹质量指标便捷启动脚本。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 用户自定义配置区
# ==============================================================================
# 1. 相机配置：wrist 相机用于推理，RealSense 用于和 wrist 左右拼接录像。

# export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[0.1352,0.0841,-0.0087,2.1653,-0.0565,0.8688,0.2226]}"
export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[0.1443,-0.0036,0.0202,2.1655,-0.0559,0.8107,0.2559]}"
# export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[0.2082,-0.1915,0.1613,1.7303,-0.0706,0.4346,0.4882]}"
# export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[0.0152,-0.2409,-0.0362,1.7667,-0.0382,0.4920,0.0718]}"


CAMERA_CONFIG_PATH="${CAMERA_CONFIG_PATH:-src/perception/configs/camera/handcap_realsense_no_tactile.json}"

# 2. 外置旁观录像相机（仅录像，不参与推理）。
# 填入 /dev/video 编号（如 4）。若不录像则留空。RealSense wrist 录像由推理内部记录。
export RECORDING_INDEX="${RECORDING_INDEX:-}"

# stiring
# 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
# POLICY_CONFIG="${POLICY_CONFIG:-pi05_514_stiring_350}"

# # 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
# POLICY_DIR="${POLICY_DIR:-ckpt/lihong/514_stiring_350_vision_30000}"

# # 5. 任务 Prompt 提示词（输入给模型的语言指令）
# PROMPT="${PROMPT:-pick up the red stick to stir the granules in a white box.}"

# screw
# # 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
POLICY_CONFIG="${POLICY_CONFIG:-pi05_513_screw_350}"

# 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
# POLICY_DIR="${POLICY_DIR:-ckpt/lihong/513_screw_vision_90000}"
POLICY_DIR="${POLICY_DIR:-ckpt/lihong/513_screw_350_vision_70000}"

# # 5. 任务 Prompt 提示词（输入给模型的语言指令）
PROMPT="${PROMPT:-pick up the red socket wrench to loose the screw on the board.}"

# towel hanging
# 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
# POLICY_CONFIG="${POLICY_CONFIG:-pi05_430_towel_hanging}"

# # 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
# POLICY_DIR="${POLICY_DIR:-ckpt/430_towel_hanging_pi05}"

# # 5. 任务 Prompt 提示词（输入给模型的语言指令）
# PROMPT="${PROMPT:-Pick up the towel and hang it on the pink rack.}"

#  erase board
# 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
# POLICY_CONFIG="${POLICY_CONFIG:-pi05_erase_board_wrist}"

# # 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
# POLICY_DIR="${POLICY_DIR:-ckpt/lihong/pi05_erase_board_wrist_110000}"

# # 5. 任务 Prompt 提示词（输入给模型的语言指令）
# PROMPT="${PROMPT:-Pick up the blackboard eraser and wipe off the blackboard}"

# #  clamp seal
# # 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
# POLICY_CONFIG="${POLICY_CONFIG:-pi05_430_clamp_seal}"

# # 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
# POLICY_DIR="${POLICY_DIR:-ckpt/430_clamp_seal_pi05_60000}"

# # 5. 任务 Prompt 提示词（输入给模型的语言指令）
# PROMPT="${PROMPT:-Pick up the clip and clip the bag.}"

#  bread moving
# 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
# POLICY_CONFIG="${POLICY_CONFIG:-pi05_bread_moving}"

# # 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
# POLICY_DIR="${POLICY_DIR:-ckpt/501_bread_moving_pi05_199999}"

# # 5. 任务 Prompt 提示词（输入给模型的语言指令）
# PROMPT="${PROMPT:-Pick up the bread in the basket and put it in the blue bowl.}"
# ==============================================================================

# 每次启动前清理旧的 server/client。只清理端口不够：
# 上一次 serve_policy.py 如果还在加载模型但尚未监听 8000，就不会被 lsof 找到。
echo "Cleaning up stale OpenPI inference processes..."
pkill -TERM -f "scripts/handcap_flexiv_remote_metrics.py" 2>/dev/null || true
pkill -TERM -f "scripts/serve_policy.py --port=8000" 2>/dev/null || true
sleep 1
pkill -KILL -f "scripts/handcap_flexiv_remote_metrics.py" 2>/dev/null || true
pkill -KILL -f "scripts/serve_policy.py --port=8000" 2>/dev/null || true

echo "Cleaning up port 8000..."
lsof -ti:8000 | xargs -r kill -9 || true

RECORD_PID=""
RECORD_FILE=""
RECORD_ROOT="${RECORD_ROOT:-real_robot_inference_recording}"
if [[ "$RECORD_ROOT" == /* ]]; then
  RECORD_ROOT_PATH="$RECORD_ROOT"
else
  RECORD_ROOT_PATH="${SCRIPT_DIR}/${RECORD_ROOT}"
fi
CLEANED_UP=0
cleanup_recording() {
  if [[ "$CLEANED_UP" == "1" ]]; then
    return
  fi
  CLEANED_UP=1

  if [[ -n "$RECORD_PID" ]]; then
    echo "Stopping external recording process $RECORD_PID..."
    kill -TERM "$RECORD_PID" 2>/dev/null || true
    wait "$RECORD_PID" 2>/dev/null || true
  fi

  # 等待内部脚本清理和打印日志完毕
  sleep 1.5

  if [[ ! ( -n "$RECORD_FILE" && -f "$RECORD_FILE" ) ]]; then
    return
  fi

  # 内部真机推理 episode 数据由空格停止时逐轮确认，这里只处理外置录像。
  echo ""
  read -p "❓ 外置相机整段录像是否需要保留? 输入 y 保留，直接回车或其他键删除 [y/N]: " keep_data </dev/tty
  
  case "$keep_data" in
    y|Y|yes|Yes ) 
      echo "✅ 已保留外置录像。"
      if [[ -n "$RECORD_FILE" && -f "$RECORD_FILE" ]]; then
        SPEEDUP_SCRIPT="${SCRIPT_DIR}/../speedup_video.py"
        if [[ -f "$SPEEDUP_SCRIPT" ]]; then
          echo "Applying 3x speedup to $RECORD_FILE..."
          python "$SPEEDUP_SCRIPT" "$RECORD_FILE" --speed 3
        else
          echo "Skipping 3x speedup because $SPEEDUP_SCRIPT was not found."
        fi
      fi
      ;;
    * ) 
      echo "🗑️  不保留，正在清理外置录像..."
      if [[ -n "$RECORD_FILE" && -f "$RECORD_FILE" ]]; then
        rm -f "$RECORD_FILE"
        echo "已删除外部录像: $RECORD_FILE"
      fi
      ;;
  esac
}
trap cleanup_recording EXIT INT TERM

if [[ -n "$RECORDING_INDEX" ]]; then
  mkdir -p "recordings/${POLICY_CONFIG}"
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  RECORD_FILE="recordings/${POLICY_CONFIG}/${TIMESTAMP}.mp4"
  echo "Starting background recording on external camera $RECORDING_INDEX to $RECORD_FILE"
  python scripts/record_video.py "$RECORDING_INDEX" "$RECORD_FILE" &
  RECORD_PID=$!
fi

export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_LOCAL_IP="192.168.2.102"
export FLEXIV_ROBOT_SN="${FLEXIV_ROBOT_SN:-Rizon4-062339}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"

export FLEXIV_ACTION_FRAME="${FLEXIV_ACTION_FRAME:-flange}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export FLEXIV_ARM_MAX_LINEAR_VEL="${FLEXIV_ARM_MAX_LINEAR_VEL:-0.1}"
export FLEXIV_ARM_MAX_ANGULAR_VEL="${FLEXIV_ARM_MAX_ANGULAR_VEL:-0.4}"
export FLEXIV_ARM_MAX_LINEAR_ACC="${FLEXIV_ARM_MAX_LINEAR_ACC:-0.2}"
export FLEXIV_ARM_MAX_ANGULAR_ACC="${FLEXIV_ARM_MAX_ANGULAR_ACC:-0.5}"
export FLEXIV_GRIPPER_MOVE_VELOCITY="${FLEXIV_GRIPPER_MOVE_VELOCITY:-0.05}"
export FLEXIV_GRIPPER_MOVE_FORCE="${FLEXIV_GRIPPER_MOVE_FORCE:-20}"
export FLEXIV_ENABLE_SAFETY_CLIP="${FLEXIV_ENABLE_SAFETY_CLIP:-1}"
RECORD_HZ="${RECORD_HZ:-20}"

echo "Flexiv Cartesian EEF control:"
echo "  action frame: ${FLEXIV_ACTION_FRAME}"
echo "  linear vel/acc:  ${FLEXIV_ARM_MAX_LINEAR_VEL} / ${FLEXIV_ARM_MAX_LINEAR_ACC}"
echo "  angular vel/acc: ${FLEXIV_ARM_MAX_ANGULAR_VEL} / ${FLEXIV_ARM_MAX_ANGULAR_ACC}"
echo "  gripper vel/force: ${FLEXIV_GRIPPER_MOVE_VELOCITY} / ${FLEXIV_GRIPPER_MOVE_FORCE}"
echo "  safety clip: ${FLEXIV_ENABLE_SAFETY_CLIP}"
echo "  camera config: ${CAMERA_CONFIG_PATH}"
echo "  record hz: ${RECORD_HZ}"
echo "  record root: ${RECORD_ROOT_PATH}"
echo "Runtime controls: SPACE=start/stop one episode, r=reset to FLEXIV_INIT_POSE, Ctrl+C=exit session"

CTRL_FREQ="10"
STEPS_PER_INFERENCE="20"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"
TASK_NAME="${TASK_NAME:-handcap_flexiv_mvs_metrics}"
METRIC_MONITOR_HZ="${METRIC_MONITOR_HZ:-50}"
METRIC_T_REF="${METRIC_T_REF:-10.0}"

bash start_handcap_remote_inference_metrics.sh \
  --policy-config "${POLICY_CONFIG}" \
  --policy-dir "${POLICY_DIR}" \
  --robot-ip "${FLEXIV_ROBOT_IP}" \
  --local-ip "${FLEXIV_LOCAL_IP}" \
  --prompt "${PROMPT}" \
  --task-name "${TASK_NAME}" \
  --ctrl-freq "${CTRL_FREQ}" \
  --record-hz "${RECORD_HZ}" \
  --steps-per-inference "${STEPS_PER_INFERENCE}" \
  --obs-horizon "${OBS_HORIZON}" \
  --camera-config-path "${CAMERA_CONFIG_PATH}" \
  --action-latency "${ACTION_LATENCY}" \
  --metric-monitor-hz "${METRIC_MONITOR_HZ}" \
  --metric-t-ref "${METRIC_T_REF}" \
  --init-qpos "${FLEXIV_INIT_POSE}" \
  --record-root "${RECORD_ROOT}" \
  --interactive-control \
  --startup-timeout 1800 \
  --no-use-tactile
