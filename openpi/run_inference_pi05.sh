#!/bin/bash
# PI05 真机远程推理 + 轨迹质量指标便捷启动脚本。
# 原 run_inference_pi05.sh 不变；本脚本额外记录并计算 J, J_e, J_m, J_c。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 用户自定义配置区
# ==============================================================================
# 1. 外置旁观录像相机（仅录像，不参与推理）。
# 填入 /dev/video 编号（如 4）。若不录像则留空。
export RECORDING_INDEX="${RECORDING_INDEX:-0}"

# screw
# 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
POLICY_CONFIG="${POLICY_CONFIG:-pi05_513_screw}"

# 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
POLICY_DIR="${POLICY_DIR:-ckpt/lihong/512_stiring_vision_only_50000}"

# 5. 任务 Prompt 提示词（输入给模型的语言指令）
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

# 每次启动前清理占用 8000 端口的僵尸进程
echo "Cleaning up port 8000..."
lsof -ti:8000 | xargs -r kill -9 || true

RECORD_PID=""
RECORD_FILE=""
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

  # 无论是否使用外置录像，都询问是否保留当前推理的所有数据
  echo ""
  read -p "❓ 刚刚的推理数据 (录像和轨迹代价函数) 是否需要保留? 输入 y 保留，直接回车或其他键删除 [y/N]: " keep_data </dev/tty
  
  case "$keep_data" in
    y|Y|yes|Yes ) 
      echo "✅ 已保留数据。"
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
      echo "🗑️  不保留，正在清理本次产生的数据..."
      if [[ -n "$RECORD_FILE" && -f "$RECORD_FILE" ]]; then
        rm -f "$RECORD_FILE"
        echo "已删除外部录像: $RECORD_FILE"
      fi
      
      # 删除最新生成的 realworld_replay_recording 目录
      LATEST_DIR=$(ls -td "${SCRIPT_DIR}/realworld_replay_recording/${TASK_NAME}/"* 2>/dev/null | head -n 1)
      if [[ -n "$LATEST_DIR" ]]; then
        rm -rf "$LATEST_DIR"
        echo "已删除推理内部记录: $LATEST_DIR"
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
export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[-0.0083,-0.1166,-0.0274,2.0942,-0.0049,0.7096,0.0272]}"
export FLEXIV_ACTION_FRAME="${FLEXIV_ACTION_FRAME:-flange}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export FLEXIV_ARM_MAX_LINEAR_VEL="${FLEXIV_ARM_MAX_LINEAR_VEL:-0.05}"
export FLEXIV_ARM_MAX_ANGULAR_VEL="${FLEXIV_ARM_MAX_ANGULAR_VEL:-0.2}"
export FLEXIV_ARM_MAX_LINEAR_ACC="${FLEXIV_ARM_MAX_LINEAR_ACC:-0.1}"
export FLEXIV_ARM_MAX_ANGULAR_ACC="${FLEXIV_ARM_MAX_ANGULAR_ACC:-0.3}"
export FLEXIV_GRIPPER_MOVE_VELOCITY="${FLEXIV_GRIPPER_MOVE_VELOCITY:-0.03}"
export FLEXIV_GRIPPER_MOVE_FORCE="${FLEXIV_GRIPPER_MOVE_FORCE:-20}"
export FLEXIV_ENABLE_SAFETY_CLIP="${FLEXIV_ENABLE_SAFETY_CLIP:-1}"

echo "Flexiv Cartesian EEF control:"
echo "  action frame: ${FLEXIV_ACTION_FRAME}"
echo "  linear vel/acc:  ${FLEXIV_ARM_MAX_LINEAR_VEL} / ${FLEXIV_ARM_MAX_LINEAR_ACC}"
echo "  angular vel/acc: ${FLEXIV_ARM_MAX_ANGULAR_VEL} / ${FLEXIV_ARM_MAX_ANGULAR_ACC}"
echo "  gripper vel/force: ${FLEXIV_GRIPPER_MOVE_VELOCITY} / ${FLEXIV_GRIPPER_MOVE_FORCE}"
echo "  safety clip: ${FLEXIV_ENABLE_SAFETY_CLIP}"

CTRL_FREQ="5"
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
  --steps-per-inference "${STEPS_PER_INFERENCE}" \
  --obs-horizon "${OBS_HORIZON}" \
  --action-latency "${ACTION_LATENCY}" \
  --metric-monitor-hz "${METRIC_MONITOR_HZ}" \
  --metric-t-ref "${METRIC_T_REF}" \
  --init-qpos "${FLEXIV_INIT_POSE}" \
  --startup-timeout 1800 \
  --no-use-tactile
