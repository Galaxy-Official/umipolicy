#!/bin/bash
# PI05 506 open bottle 真机远程推理脚本 - multi-health distill + wrist + tactile + force.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 用户自定义配置区
# ==============================================================================
# 1. 外置旁观录像相机（仅录像，不参与推理）。
# 填入 /dev/video 编号（如 0）。若不录像则留空。
export RECORDING_INDEX="${RECORDING_INDEX:-0}"

# 2. 左侧和右侧触觉相机编号（仅参与 tactile 推理）。
# 填入 /dev/video 编号（如 4 和 2）。
export TACTILE_LEFT_INDEX="${TACTILE_LEFT_INDEX:-4}"
export TACTILE_RIGHT_INDEX="${TACTILE_RIGHT_INDEX:-2}"
export TACTILE_CAPTURE_WIDTH="${TACTILE_CAPTURE_WIDTH:-640}"
export TACTILE_CAPTURE_HEIGHT="${TACTILE_CAPTURE_HEIGHT:-480}"

# 3. 任务策略名称：必须与 health distill 训练脚本里的 CONFIG_NAME 对齐。
POLICY_CONFIG="${POLICY_CONFIG:-pi05_506_open_bottle_health_distill_tactile_wrist_force}"

# 4. 策略权重路径：这里要求目录下直接有 params/ 和 assets/。
POLICY_DIR="${POLICY_DIR:-ckpt/508_open_bottle_multihealth_pi05_health_distill_tactile_wrist_force_50000}"

# 5. 任务 Prompt 提示词（输入给模型的语言指令）。
PROMPT="${PROMPT:-Open the bottle.}"

# 6. 是否使用模型预测的力觉来动态控制夹爪夹持力（最小保护为 1.0N）。
export INFER_FORCE_CONTROL="${INFER_FORCE_CONTROL:-true}"

# 7. 与 health distill config 的 repo_id / asset_id 对齐。
NORM_ASSET_ID="${NORM_ASSET_ID:-506_open_bottle_health_distill}"
SKIP_NORM_STATS_CHECK="${SKIP_NORM_STATS_CHECK:-0}"
# ==============================================================================

echo "Cleaning up port 8000..."
lsof -ti:8000 | xargs -r kill -9 || true

if [[ "${SKIP_NORM_STATS_CHECK}" != "1" && "${POLICY_DIR}" != gs://* && "${POLICY_DIR}" != s3://* ]]; then
  NORM_STATS_PATH="${POLICY_DIR}/assets/${NORM_ASSET_ID}/norm_stats.json"
  if [[ ! -f "${NORM_STATS_PATH}" ]]; then
    echo "ERROR: Norm stats not found for this health-distill policy."
    echo "Expected: ${NORM_STATS_PATH}"
    echo ""
    echo "This script uses POLICY_CONFIG=${POLICY_CONFIG}, whose asset id is ${NORM_ASSET_ID}."
    echo "Make sure the exported checkpoint keeps the training checkpoint's assets/ directory,"
    echo "or copy norm_stats.json into the expected path above."
    exit 1
  fi
fi

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

  sleep 1.5

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
export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[0.0529,-0.1326,-0.0259,1.8405,-0.1027,0.4303,0.1024]}"
# The dataset/model action pose is expressed in the robot flange frame. Flexiv's
# Cartesian API still accepts target TCP poses, so the runtime converts
# target_flange -> target_tcp before SendCartesianMotionForce().
export FLEXIV_ACTION_FRAME="${FLEXIV_ACTION_FRAME:-flange}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

CTRL_FREQ="${CTRL_FREQ:-5}"
STEPS_PER_INFERENCE="${STEPS_PER_INFERENCE:-20}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"
TASK_NAME="${TASK_NAME:-506_open_bottle_health_distill_tactile_wrist_force}"
METRIC_MONITOR_HZ="${METRIC_MONITOR_HZ:-50}"
METRIC_T_REF="${METRIC_T_REF:-10.0}"

INFER_FORCE_ARG=""
if [[ "$INFER_FORCE_CONTROL" == "true" ]]; then
  INFER_FORCE_ARG="--infer-force-control"
fi

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
  --use-tactile \
  --left-video-index "${TACTILE_LEFT_INDEX}" \
  --right-video-index "${TACTILE_RIGHT_INDEX}" \
  --tactile-capture-width "${TACTILE_CAPTURE_WIDTH}" \
  --tactile-capture-height "${TACTILE_CAPTURE_HEIGHT}" \
  --force-predict \
  ${INFER_FORCE_ARG}
