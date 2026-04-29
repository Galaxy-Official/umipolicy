#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export FLEXIV_ROBOT_IP="${FLEXIV_ROBOT_IP:-192.168.2.100}"
export FLEXIV_LOCAL_IP="${FLEXIV_LOCAL_IP:-192.168.2.102}"
export FLEXIV_ROBOT_SN="${FLEXIV_ROBOT_SN:-Rizon4-062339}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"

CONFIG_NAME="${CONFIG_NAME:-flexiv}"
SERVER_PORT="${SERVER_PORT:-29536}"
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
SAVE_ROOT="${SAVE_ROOT:-./train_out}"
LINGBOT_VA_BASE_CKPT="${LINGBOT_VA_BASE_CKPT:-${LINGBOT_VA_CKPT:-$REPO_ROOT/ckpt/lingbot-va-base}}"
LINGBOT_VA_TRANSFORMER_CKPT="${LINGBOT_VA_TRANSFORMER_CKPT:-}"

PROMPT="${PROMPT:-simple sorting task}"
CTRL_FREQ="${CTRL_FREQ:-20}"
STEPS_PER_INFERENCE="${STEPS_PER_INFERENCE:-4}"
OBS_HORIZON="${OBS_HORIZON:-1}"
CAMERA_FPS="${CAMERA_FPS:-20}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"
ACTION_POSE_MODE="${ACTION_POSE_MODE:-relative}"
USE_TACTILE="${USE_TACTILE:-false}"
INIT_QPOS="${INIT_QPOS:-[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]}"
CAMERA_CONFIG_PATH="${CAMERA_CONFIG_PATH:-}"
DRY_RUN="${DRY_RUN:-false}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
LINGBOT_ENABLE_OFFLOAD="${LINGBOT_ENABLE_OFFLOAD:-true}"
LINGBOT_ATTN_WINDOW="${LINGBOT_ATTN_WINDOW:-16}"
LINGBOT_GUIDANCE_SCALE="${LINGBOT_GUIDANCE_SCALE:-1.0}"
LINGBOT_ACTION_GUIDANCE_SCALE="${LINGBOT_ACTION_GUIDANCE_SCALE:-1.0}"
LINGBOT_NUM_INFERENCE_STEPS="${LINGBOT_NUM_INFERENCE_STEPS:-4}"
LINGBOT_ACTION_NUM_INFERENCE_STEPS="${LINGBOT_ACTION_NUM_INFERENCE_STEPS:-4}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/flexiv_lingbot_real/$TIMESTAMP}"
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"
PID_FILE="$LOG_DIR/pids.env"
CMD_FILE="$LOG_DIR/run_command.txt"

join_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q' "$arg") "
  done
  printf '%s\n' "${out% }"
}

is_health_ready() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
request = f"GET /healthz HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
with socket.create_connection((host, port), timeout=1) as sock:
    sock.settimeout(1)
    sock.sendall(request.encode("ascii"))
    response = sock.recv(128)
if b"200" not in response.split(b"\r\n", 1)[0]:
    raise SystemExit(1)
PY
}

is_process_running() {
  local pid="$1"
  local state=""
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 1
  fi
  state="$(ps -p "$pid" -o stat= 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z* ]]
}

wait_for_server() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local pid="$4"
  local start_time=$SECONDS
  local deadline=$((SECONDS + timeout))
  local next_status=$((SECONDS + 20))

  while (( SECONDS < deadline )); do
    if is_health_ready "$host" "$port"; then
      return 0
    fi
    if ! is_process_running "$pid"; then
      wait "$pid" || true
      return 2
    fi
    if (( SECONDS >= next_status )); then
      echo "Still waiting for LingBot-VA server on ${host}:${port} ... ($((SECONDS - start_time))s elapsed)"
      if [[ -s "$SERVER_LOG" ]]; then
        echo "==== server.log (tail) ===="
        tail -n 30 "$SERVER_LOG" || true
        echo "==========================="
      fi
      next_status=$((SECONDS + 20))
    fi
    sleep 1
  done
  return 1
}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF
export LINGBOT_VA_BASE_CKPT
export LINGBOT_VA_TRANSFORMER_CKPT

for required_path in \
  "$LINGBOT_VA_BASE_CKPT/vae/config.json" \
  "$LINGBOT_VA_BASE_CKPT/tokenizer" \
  "$LINGBOT_VA_BASE_CKPT/text_encoder/config.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Cannot find required LingBot-VA model file or directory: $required_path" >&2
    echo "Set LINGBOT_VA_BASE_CKPT to the pretrained model root that contains vae, tokenizer, text_encoder, and transformer." >&2
    exit 1
  fi
done

if [[ -z "$LINGBOT_VA_TRANSFORMER_CKPT" ]]; then
  LINGBOT_VA_TRANSFORMER_CKPT="$LINGBOT_VA_BASE_CKPT/transformer"
fi

for required_path in \
  "$LINGBOT_VA_TRANSFORMER_CKPT/config.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Cannot find required LingBot-VA transformer file: $required_path" >&2
    echo "For pretrained inference, use LINGBOT_VA_TRANSFORMER_CKPT=$LINGBOT_VA_BASE_CKPT/transformer." >&2
    echo "For your trained model, set LINGBOT_VA_TRANSFORMER_CKPT to a checkpoint transformer directory." >&2
    exit 1
  fi
done

shopt -s nullglob
TRANSFORMER_WEIGHT_FILES=(
  "$LINGBOT_VA_TRANSFORMER_CKPT"/diffusion_pytorch_model*.safetensors
  "$LINGBOT_VA_TRANSFORMER_CKPT"/model*.safetensors
  "$LINGBOT_VA_TRANSFORMER_CKPT"/*.bin
)
shopt -u nullglob
if [[ "${#TRANSFORMER_WEIGHT_FILES[@]}" -eq 0 ]]; then
  echo "Cannot find LingBot-VA transformer weights in: $LINGBOT_VA_TRANSFORMER_CKPT" >&2
  echo "Expected diffusion_pytorch_model*.safetensors, model*.safetensors, or *.bin." >&2
  echo "Your pretrained download is incomplete; re-download lingbot-va-base or run git lfs pull in that directory." >&2
  exit 1
fi

echo "LingBot-VA base checkpoint: $LINGBOT_VA_BASE_CKPT"
echo "LingBot-VA transformer checkpoint: $LINGBOT_VA_TRANSFORMER_CKPT"
echo "Found ${#TRANSFORMER_WEIGHT_FILES[@]} transformer weight file(s)."

SERVER_CMD=(
  python -m torch.distributed.run
  --nproc_per_node="$NGPU"
  --local-ranks-filter="$LOG_RANK"
  --master_port "$MASTER_PORT"
  --tee 3
  -m wan_va.wan_va_server
  --config-name "$CONFIG_NAME"
  --port "$SERVER_PORT"
  --save_root "$SAVE_ROOT"
  --base-ckpt-path "$LINGBOT_VA_BASE_CKPT"
  --transformer-ckpt-path "$LINGBOT_VA_TRANSFORMER_CKPT"
)

if [[ "$LINGBOT_ENABLE_OFFLOAD" == "true" ]]; then
  SERVER_CMD+=(--enable-offload)
else
  SERVER_CMD+=(--no-enable-offload)
fi

SERVER_CMD+=(
  --attn-window "$LINGBOT_ATTN_WINDOW"
  --guidance-scale "$LINGBOT_GUIDANCE_SCALE"
  --action-guidance-scale "$LINGBOT_ACTION_GUIDANCE_SCALE"
)

if [[ "$USE_TACTILE" == "true" ]]; then
  SERVER_CMD+=(--use-tactile)
else
  SERVER_CMD+=(--no-use-tactile)
fi

if [[ -n "$LINGBOT_NUM_INFERENCE_STEPS" ]]; then
  SERVER_CMD+=(--num-inference-steps "$LINGBOT_NUM_INFERENCE_STEPS")
fi

if [[ -n "$LINGBOT_ACTION_NUM_INFERENCE_STEPS" ]]; then
  SERVER_CMD+=(--action-num-inference-steps "$LINGBOT_ACTION_NUM_INFERENCE_STEPS")
fi

CLIENT_CMD=(
  python scripts/flexiv_lingbot_remote.py
  --host 127.0.0.1
  --port "$SERVER_PORT"
  --prompt "$PROMPT"
  --robot-ip "$FLEXIV_ROBOT_IP"
  --local-ip "$FLEXIV_LOCAL_IP"
  --init-qpos "$INIT_QPOS"
  --ctrl-freq "$CTRL_FREQ"
  --steps-per-inference "$STEPS_PER_INFERENCE"
  --obs-horizon "$OBS_HORIZON"
  --camera-fps "$CAMERA_FPS"
  --action-latency "$ACTION_LATENCY"
  --action-pose-mode "$ACTION_POSE_MODE"
)

if [[ -n "$CAMERA_CONFIG_PATH" ]]; then
  CLIENT_CMD+=(--camera-config-path "$CAMERA_CONFIG_PATH")
fi

if [[ "$USE_TACTILE" == "true" ]]; then
  CLIENT_CMD+=(--use-tactile)
fi

if [[ "$DRY_RUN" == "true" ]]; then
  CLIENT_CMD+=(--dry-run)
fi

CLIENT_LD_PRELOAD="${LD_PRELOAD:-}"
SYSTEM_LIBSTDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYSTEM_LIBSTDCXX" ]] && grep -a -q "GLIBCXX_3.4.30" "$SYSTEM_LIBSTDCXX"; then
  CLIENT_LD_PRELOAD="$SYSTEM_LIBSTDCXX${CLIENT_LD_PRELOAD:+:$CLIENT_LD_PRELOAD}"
fi

{
  echo "# Generated at $(date --iso-8601=seconds)"
  echo "Server command:"
  join_cmd "${SERVER_CMD[@]}"
  echo
  echo "Client command:"
  if [[ -n "$CLIENT_LD_PRELOAD" ]]; then
    echo "LD_PRELOAD=$(printf '%q' "$CLIENT_LD_PRELOAD") $(join_cmd "${CLIENT_CMD[@]}")"
  else
    join_cmd "${CLIENT_CMD[@]}"
  fi
} > "$CMD_FILE"

SERVER_PID=""
CLIENT_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "$CLIENT_PID" >/dev/null 2>&1; then
    kill "$CLIENT_PID" >/dev/null 2>&1 || true
    wait "$CLIENT_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting LingBot-VA policy server..."
echo "Server log: $SERVER_LOG"
echo "Server command: $(join_cmd "${SERVER_CMD[@]}")"
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE="$TORCHFT_LIGHTHOUSE" \
  "${SERVER_CMD[@]}" \
  > >(tee "$SERVER_LOG") \
  2> >(tee -a "$SERVER_LOG" >&2) &
SERVER_PID=$!

echo "Waiting for server on 127.0.0.1:$SERVER_PORT ..."
WAIT_STATUS=0
wait_for_server 127.0.0.1 "$SERVER_PORT" "$STARTUP_TIMEOUT" "$SERVER_PID" || WAIT_STATUS=$?
if [[ "$WAIT_STATUS" != "0" ]]; then
  if [[ "$WAIT_STATUS" == "2" ]]; then
    echo "Server process exited before opening 127.0.0.1:$SERVER_PORT." >&2
  else
    echo "Server did not become ready within ${STARTUP_TIMEOUT}s." >&2
  fi
  echo "==== server.log (tail) ====" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "Starting Flexiv real-robot LingBot client..."
if [[ -n "$CLIENT_LD_PRELOAD" ]]; then
  echo "Client LD_PRELOAD: $CLIENT_LD_PRELOAD"
  LD_PRELOAD="$CLIENT_LD_PRELOAD" PYTHONUNBUFFERED=1 "${CLIENT_CMD[@]}" \
    > >(tee "$CLIENT_LOG") \
    2> >(tee -a "$CLIENT_LOG" >&2) &
else
  PYTHONUNBUFFERED=1 "${CLIENT_CMD[@]}" \
    > >(tee "$CLIENT_LOG") \
    2> >(tee -a "$CLIENT_LOG" >&2) &
fi
CLIENT_PID=$!

{
  echo "SERVER_PID=$SERVER_PID"
  echo "CLIENT_PID=$CLIENT_PID"
} > "$PID_FILE"

cat <<EOF
LingBot-VA real inference started.

Logs:
  Server: $SERVER_LOG
  Client: $CLIENT_LOG

PID file:
  $PID_FILE

Press Ctrl+C to stop both processes.
EOF

EXITED_PROCESS=""
EXIT_STATUS=0
while true; do
  if ! is_process_running "$SERVER_PID"; then
    wait "$SERVER_PID" || EXIT_STATUS=$?
    EXITED_PROCESS="server"
    break
  fi
  if ! is_process_running "$CLIENT_PID"; then
    wait "$CLIENT_PID" || EXIT_STATUS=$?
    EXITED_PROCESS="client"
    break
  fi
  sleep 1
done

if [[ "$EXIT_STATUS" == "0" ]]; then
  echo "$EXITED_PROCESS process exited normally. Stopping the remaining process..."
else
  echo "$EXITED_PROCESS process exited with status $EXIT_STATUS. Stopping the remaining process..." >&2
fi

echo "==== server.log (tail) ===="
tail -n 80 "$SERVER_LOG" || true
echo "==== client.log (tail) ===="
tail -n 80 "$CLIENT_LOG" || true
