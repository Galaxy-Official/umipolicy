#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./start_handcap_remote_inference.sh \
    --policy-config <handcap_config_name> \
    --policy-dir <checkpoint_dir> \
    --robot-ip <flexiv_ip> \
    --prompt "<task_prompt>" \
    [--server-port 8000] \
    [--task-name handcap_flexiv_mvs] \
    [--ctrl-freq 30] \
    [--obs-horizon 2] \
    [--camera-config-path <path>] \
    [--use-tactile | --no-use-tactile] \
    [--action-latency 0.0] \
    [--init-qpos "<python_tuple_literal>"] \
    [--record-root <dir>] \
    [--server-default-prompt "<prompt>"] \
    [--startup-timeout 120] \
    [--log-dir <dir>] \
    [--dry-run]

Required:
  --policy-config       Handcap config name registered in openpi training config.
  --policy-dir          Checkpoint directory to serve.
  --robot-ip            Flexiv robot IP.
  --prompt              Task prompt sent to the policy server.
EOF
}

require_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "Missing value for $flag" >&2
    usage
    exit 1
  fi
}

join_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q' "$arg") "
  done
  printf '%s\n' "${out% }"
}

is_port_open() {
  local host="$1"
  local port="$2"

  python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import base64
import os
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

key = base64.b64encode(os.urandom(16)).decode("ascii")
request = (
    f"GET / HTTP/1.1\r\n"
    f"Host: {host}:{port}\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "\r\n"
)

with socket.create_connection((host, port), timeout=1) as sock:
    sock.settimeout(1)
    sock.sendall(request.encode("ascii"))
    response = sock.recv(1024)

status_line = response.split(b"\r\n", 1)[0]
if b" 101 " not in status_line:
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

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local pid="${4-}"
  local log_file="${5-}"
  local start_time=$SECONDS
  local deadline=$((SECONDS + timeout))
  local next_status=$((SECONDS + 20))

  while (( SECONDS < deadline )); do
    if is_port_open "$host" "$port"; then
      return 0
    fi

    if [[ -n "$pid" ]] && ! is_process_running "$pid"; then
      wait "$pid" || true
      return 2
    fi

    if (( SECONDS >= next_status )); then
      echo "Still waiting for server on ${host}:${port} ... ($((SECONDS - start_time))s elapsed)"
      if [[ -n "$log_file" && -s "$log_file" ]]; then
        echo "==== server.log (tail) ===="
        tail -n 20 "$log_file" || true
        echo "==========================="
      fi
      next_status=$((SECONDS + 20))
    fi

    sleep 1
  done
  return 1
}

POLICY_CONFIG=""
POLICY_DIR=""
ROBOT_IP="192.168.1.100"
PROMPT="simple sorting task"
SERVER_PORT="8000"
TASK_NAME="handcap_flexiv_mvs"
CTRL_FREQ="30"
OBS_HORIZON="2"
CAMERA_CONFIG_PATH=""
USE_TACTILE="true"
ACTION_LATENCY="0.0"
INIT_QPOS=""
RECORD_ROOT=""
SERVER_DEFAULT_PROMPT=""
STARTUP_TIMEOUT="600"
LOG_DIR=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-config)
      require_value "$1" "${2-}"
      POLICY_CONFIG="$2"
      shift 2
      ;;
    --policy-dir)
      require_value "$1" "${2-}"
      POLICY_DIR="$2"
      shift 2
      ;;
    --robot-ip)
      require_value "$1" "${2-}"
      ROBOT_IP="$2"
      shift 2
      ;;
    --prompt)
      require_value "$1" "${2-}"
      PROMPT="$2"
      shift 2
      ;;
    --server-port)
      require_value "$1" "${2-}"
      SERVER_PORT="$2"
      shift 2
      ;;
    --task-name)
      require_value "$1" "${2-}"
      TASK_NAME="$2"
      shift 2
      ;;
    --ctrl-freq)
      require_value "$1" "${2-}"
      CTRL_FREQ="$2"
      shift 2
      ;;
    --obs-horizon)
      require_value "$1" "${2-}"
      OBS_HORIZON="$2"
      shift 2
      ;;
    --camera-config-path)
      require_value "$1" "${2-}"
      CAMERA_CONFIG_PATH="$2"
      shift 2
      ;;
    --use-tactile)
      USE_TACTILE="true"
      shift
      ;;
    --no-use-tactile)
      USE_TACTILE="false"
      shift
      ;;
    --action-latency)
      require_value "$1" "${2-}"
      ACTION_LATENCY="$2"
      shift 2
      ;;
    --init-qpos)
      require_value "$1" "${2-}"
      INIT_QPOS="$2"
      shift 2
      ;;
    --record-root)
      require_value "$1" "${2-}"
      RECORD_ROOT="$2"
      shift 2
      ;;
    --server-default-prompt)
      require_value "$1" "${2-}"
      SERVER_DEFAULT_PROMPT="$2"
      shift 2
      ;;
    --startup-timeout)
      require_value "$1" "${2-}"
      STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --log-dir)
      require_value "$1" "${2-}"
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$POLICY_CONFIG" || -z "$POLICY_DIR" || -z "$ROBOT_IP" || -z "$PROMPT" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ "$POLICY_DIR" == "~/"* ]]; then
  POLICY_DIR="$HOME/${POLICY_DIR#~/}"
fi

if [[ "$POLICY_DIR" != gs://* && "$POLICY_DIR" != s3://* && ! -d "$POLICY_DIR" ]]; then
  echo "Cannot find policy checkpoint directory: $POLICY_DIR" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/scripts/serve_policy.py" ]]; then
  echo "Cannot find $REPO_ROOT/scripts/serve_policy.py" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/scripts/handcap_flexiv_remote.py" ]]; then
  echo "Cannot find $REPO_ROOT/scripts/handcap_flexiv_remote.py" >&2
  exit 1
fi

if [[ ! -f "/home/rhos/miniconda3/bin/activate" ]]; then
  echo "Cannot find conda activate script at /home/rhos/miniconda3/bin/activate" >&2
  exit 1
fi

source /home/rhos/miniconda3/bin/activate
conda activate umi

# uv check removed

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in the current environment." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not available in the current environment." >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$REPO_ROOT/logs/handcap_remote_inference/$TIMESTAMP"
fi
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"
PID_FILE="$LOG_DIR/pids.env"
CMD_FILE="$LOG_DIR/run_command.txt"

SERVER_CMD=(
  python
  scripts/serve_policy.py
  --port="$SERVER_PORT"
  policy:checkpoint
  --policy.config="$POLICY_CONFIG"
  --policy.dir="$POLICY_DIR"
)

if [[ -n "$SERVER_DEFAULT_PROMPT" ]]; then
  SERVER_CMD=("${SERVER_CMD[@]:0:3}" --default-prompt="$SERVER_DEFAULT_PROMPT" "${SERVER_CMD[@]:3}")
fi

CLIENT_CMD=(
  python
  scripts/handcap_flexiv_remote.py
  --host 127.0.0.1
  --port "$SERVER_PORT"
  --robot-ip "$ROBOT_IP"
  --prompt "$PROMPT"
  --task-name "$TASK_NAME"
  --ctrl-freq "$CTRL_FREQ"
  --obs-horizon "$OBS_HORIZON"
  --action-latency "$ACTION_LATENCY"
)

if [[ -n "$CAMERA_CONFIG_PATH" ]]; then
  CLIENT_CMD+=(--camera-config-path "$CAMERA_CONFIG_PATH")
fi

if [[ "$USE_TACTILE" == "true" ]]; then
  CLIENT_CMD+=(--use-tactile)
else
  CLIENT_CMD+=(--no-use-tactile)
fi

if [[ -n "$INIT_QPOS" ]]; then
  CLIENT_CMD+=(--init-qpos "$INIT_QPOS")
fi

if [[ -n "$RECORD_ROOT" ]]; then
  CLIENT_CMD+=(--record-root "$RECORD_ROOT")
fi

if [[ "$DRY_RUN" == "true" ]]; then
  CLIENT_CMD+=(--dry-run)
fi

{
  echo "# Generated at $(date --iso-8601=seconds)"
  echo "Server command:"
  join_cmd "${SERVER_CMD[@]}"
  echo
  echo "Client command:"
  join_cmd "${CLIENT_CMD[@]}"
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

echo "Starting OpenPI policy server..."
echo "Server log: $SERVER_LOG"
echo "Server command: $(join_cmd "${SERVER_CMD[@]}")"
if is_port_open "127.0.0.1" "$SERVER_PORT"; then
  echo "Port 127.0.0.1:$SERVER_PORT is already open. Stop the old server or use --server-port." >&2
  exit 1
fi
PYTHONUNBUFFERED=1 "${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for server on 127.0.0.1:$SERVER_PORT ..."
WAIT_STATUS=0
wait_for_port "127.0.0.1" "$SERVER_PORT" "$STARTUP_TIMEOUT" "$SERVER_PID" "$SERVER_LOG" || WAIT_STATUS=$?
if [[ "$WAIT_STATUS" != "0" ]]; then
  if [[ "$WAIT_STATUS" == "2" ]]; then
    echo "Server process exited before opening 127.0.0.1:$SERVER_PORT." >&2
  else
    echo "Server did not become ready within ${STARTUP_TIMEOUT}s." >&2
  fi
  if [[ -f "$SERVER_LOG" ]]; then
    echo "==== server.log (tail) ====" >&2
    tail -n 50 "$SERVER_LOG" >&2 || true
  fi
  exit 1
fi

echo "Starting Flexiv real-robot client..."
"${CLIENT_CMD[@]}" >"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!

{
  echo "SERVER_PID=$SERVER_PID"
  echo "CLIENT_PID=$CLIENT_PID"
} > "$PID_FILE"

cat <<EOF
Remote inference started.

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
    if wait "$SERVER_PID"; then
      EXIT_STATUS=0
    else
      EXIT_STATUS=$?
    fi
    EXITED_PROCESS="server"
    break
  fi

  if ! is_process_running "$CLIENT_PID"; then
    if wait "$CLIENT_PID"; then
      EXIT_STATUS=0
    else
      EXIT_STATUS=$?
    fi
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

if [[ -f "$SERVER_LOG" ]]; then
  echo "==== server.log (tail) ===="
  tail -n 80 "$SERVER_LOG" || true
fi

if [[ -f "$CLIENT_LOG" ]]; then
  echo "==== client.log (tail) ===="
  tail -n 80 "$CLIENT_LOG" || true
fi
