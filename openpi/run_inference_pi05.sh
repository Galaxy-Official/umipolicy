#!/bin/bash
# PI05 真机远程推理便捷启动脚本。
# 机械臂、MVS 相机、执行频率、action 执行步数逻辑与 run_inference_pi0.sh 保持一致。

# ==============================================================================
# 用户自定义配置区
# ==============================================================================
# 1. 外置旁观录像相机（仅录像，不参与推理）。
# 填入 /dev/video 编号（如 4）。若不录像则留空。
export RECORDING_INDEX="${RECORDING_INDEX:-14}"

# 3. 任务策略名称（需与 handcap_config.py 中的 registered name 对应）
POLICY_CONFIG="${POLICY_CONFIG:-pi05_bread_moving}"

# 4. 策略权重路径（保存模型的 ckpt 文件夹相对路径）
POLICY_DIR="${POLICY_DIR:-ckpt/430_clamp_seal_pi05_60000}"

# 5. 任务 Prompt 提示词（输入给模型的语言指令）
PROMPT="${PROMPT:-Pick up the bread and put it in the bowl on the right.}"
# ==============================================================================

# 每次启动前清理占用 8000 端口的僵尸进程
echo "Cleaning up port 8000..."
lsof -ti:8000 | xargs -r kill -9 || true

RECORD_PID=""
RECORD_FILE=""
cleanup_recording() {
  if [[ -n "$RECORD_PID" ]]; then
    echo "Stopping external recording process $RECORD_PID..."
    kill -TERM "$RECORD_PID" 2>/dev/null || true
    wait "$RECORD_PID" 2>/dev/null || true
    if [[ -f "$RECORD_FILE" ]]; then
      echo "Applying 3x speedup to $RECORD_FILE..."
      python /Users/macbookpro/Desktop/workspace/umipolicy/speedup_video.py "$RECORD_FILE" --speed 3
    fi
  fi
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
export FLEXIV_INIT_POSE="${FLEXIV_INIT_POSE:-[-0.0009,-0.2370,-0.0133,1.9935,-0.0058,0.6496,-0.0012]}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

CTRL_FREQ="5"
STEPS_PER_INFERENCE="20"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"

bash start_handcap_remote_inference.sh \
  --policy-config "${POLICY_CONFIG}" \
  --policy-dir "${POLICY_DIR}" \
  --robot-ip "${FLEXIV_ROBOT_IP}" \
  --local-ip "${FLEXIV_LOCAL_IP}" \
  --prompt "${PROMPT}" \
  --ctrl-freq "${CTRL_FREQ}" \
  --steps-per-inference "${STEPS_PER_INFERENCE}" \
  --obs-horizon "${OBS_HORIZON}" \
  --action-latency "${ACTION_LATENCY}" \
  --init-qpos "${FLEXIV_INIT_POSE}" \
  --startup-timeout 1800 \
  --no-use-tactile
