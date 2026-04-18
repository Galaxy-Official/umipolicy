#!/usr/bin/env python

import argparse
import asyncio
import json
import signal
import time
import uuid
from contextlib import suppress
from pathlib import Path

from aiohttp import WSMsgType, web
from loguru import logger

from lerobot.remote_teleop.common import (
    CommandReply,
    LeaderState,
    LeaderToFollowerMapper,
    OperatorCommand,
    RobotTelemetry,
    SessionReply,
    encode_jpeg,
    now_ms,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging


class AsyncMjpegStreamStore:
    def __init__(self, camera_names: list[str], jpeg_quality: int = 75):
        self.camera_names = list(camera_names)
        self.jpeg_quality = jpeg_quality
        self._frames = {name: None for name in self.camera_names}
        self._frame_ids = {name: 0 for name in self.camera_names}
        self._conditions = {name: asyncio.Condition() for name in self.camera_names}

    async def publish_frame(self, camera_name: str, frame) -> None:
        if camera_name not in self._frames:
            return

        jpeg_bytes = encode_jpeg(frame, self.jpeg_quality)
        condition = self._conditions[camera_name]
        async with condition:
            self._frames[camera_name] = jpeg_bytes
            self._frame_ids[camera_name] += 1
            condition.notify_all()

    async def mjpeg_handler(self, request: web.Request) -> web.StreamResponse:
        camera_name = request.match_info["camera_name"]
        if camera_name not in self._frames:
            raise web.HTTPNotFound(text=f"Unknown camera: {camera_name}")

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        await response.prepare(request)

        last_frame_id = 0
        condition = self._conditions[camera_name]
        try:
            while True:
                async with condition:
                    while self._frame_ids[camera_name] == last_frame_id:
                        await condition.wait()
                    frame_bytes = self._frames[camera_name]
                    last_frame_id = self._frame_ids[camera_name]

                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode()
                    + frame_bytes
                    + b"\r\n"
                )
        except (asyncio.CancelledError, ConnectionResetError, RuntimeError):
            pass

        return response


class RemoteTeleopRobotServer:
    def __init__(
        self,
        robot_path: str,
        bind_host: str,
        port: int,
        control_fps: int,
        observation_fps: int,
        hold_timeout_ms: int,
        stop_timeout_ms: int,
        camera_names: list[str],
        jpeg_quality: int,
        robot_id: str,
        save_dir: str,
    ):
        self.robot_path = robot_path
        self.bind_host = bind_host
        self.port = port
        self.control_fps = control_fps
        self.observation_fps = observation_fps
        self.hold_timeout_ms = hold_timeout_ms
        self.stop_timeout_ms = stop_timeout_ms
        self.camera_names = camera_names
        self.jpeg_quality = jpeg_quality
        self.robot_id = robot_id
        self.save_dir = Path(save_dir)

        self.robot = None
        self.mapper = None
        self.leader_robot_path = None

        self.stream_store = AsyncMjpegStreamStore(self.camera_names, jpeg_quality=self.jpeg_quality)
        self.dashboard_clients: set[web.WebSocketResponse] = set()
        self.operator_ws: web.WebSocketResponse | None = None

        self.active_session_id = ""
        self.active_operator_id = ""
        self.last_leader_state: LeaderState | None = None
        self.last_leader_monotonic = 0.0
        self.last_observation_monotonic = 0.0
        self.last_status_message = "idle"
        self.safety_state = "idle"
        self.session_paused = False
        self.last_telemetry: RobotTelemetry | None = None

        self.app = web.Application()
        self.app.add_routes(
            [
                web.get("/", self.dashboard_handler),
                web.get("/healthz", self.healthz_handler),
                web.get("/api/operator/ws", self.operator_ws_handler),
                web.get("/api/dashboard/ws", self.dashboard_ws_handler),
                web.get("/video/{camera_name}.mjpg", self.stream_store.mjpeg_handler),
            ]
        )
        self.app.on_startup.append(self.on_startup)
        self.app.on_cleanup.append(self.on_cleanup)
        self.control_task = None

    async def on_startup(self, app: web.Application):
        robot_cfg = init_hydra_config(self.robot_path)
        if "leader_arm" not in robot_cfg or "robot_path" not in robot_cfg.leader_arm:
            raise ValueError("Remote teleop server requires `leader_arm.robot_path` in the robot config.")

        self.leader_robot_path = robot_cfg.leader_arm.robot_path
        self.robot = make_robot(robot_cfg)
        self.robot.leader_arm = {}
        self.robot.connect()
        self.robot.set_robot_home_position()
        self.mapper = LeaderToFollowerMapper(self.leader_robot_path)
        self.control_task = asyncio.create_task(self.control_loop(), name="remote-teleop-control-loop")

        logger.info(f"Remote robot server ready at http://{self.bind_host}:{self.port}")
        logger.info(
            "Dashboard URLs: "
            + ", ".join(
                [f"http://{self.bind_host}:{self.port}/video/{camera}.mjpg" for camera in self.camera_names]
            )
        )

    async def on_cleanup(self, app: web.Application):
        if self.control_task is not None:
            self.control_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.control_task

        for ws in list(self.dashboard_clients):
            await ws.close()

        if self.operator_ws is not None:
            await self.operator_ws.close()
            self.operator_ws = None

        if self.mapper is not None:
            self.mapper.close()
            self.mapper = None

        if self.robot is not None and getattr(self.robot, "is_connected", False):
            self.robot.disconnect()
            self.robot = None

    async def dashboard_handler(self, request: web.Request) -> web.Response:
        available_cameras = "".join(
            f'<div class="camera"><h2>{camera}</h2><img src="/video/{camera}.mjpg" alt="{camera}"></div>'
            for camera in self.camera_names
        )
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Remote Teleop Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #08111c;
      --panel: rgba(7, 20, 32, 0.88);
      --border: rgba(120, 168, 215, 0.28);
      --text: #eff8ff;
      --muted: #98afc4;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top, #18304b, #08111c 56%);
      color: var(--text);
      font-family: Helvetica, Arial, sans-serif;
      display: grid;
      grid-template-columns: minmax(340px, 420px) 1fr;
      gap: 20px;
      padding: 20px;
      box-sizing: border-box;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      backdrop-filter: blur(14px);
    }}
    .telemetry {{
      white-space: pre-wrap;
      font-family: Menlo, Consolas, monospace;
      line-height: 1.55;
      color: var(--muted);
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    button {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      background: rgba(18, 42, 65, 0.92);
      color: var(--text);
      font-weight: 600;
      cursor: pointer;
    }}
    button:hover {{
      background: rgba(35, 70, 101, 0.96);
    }}
    .command-status {{
      margin-top: 12px;
      min-height: 1.4em;
      color: #b7d0e7;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .camera {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
    }}
    img {{
      width: 100%;
      border-radius: 12px;
      object-fit: contain;
      background: #000;
    }}
  </style>
</head>
<body>
  <section class="panel">
    <h1>Remote Teleop Dashboard</h1>
    <div class="telemetry" id="telemetry">Waiting for telemetry...</div>
    <div class="controls">
      <button type="button" data-command="save">Save Snapshot</button>
      <button type="button" data-command="pause">Pause Teleop</button>
      <button type="button" data-command="resume">Resume Teleop</button>
      <button type="button" data-command="end">End Teleop</button>
    </div>
    <div class="command-status" id="command-status"></div>
  </section>
  <section class="grid">
    {available_cameras}
  </section>
  <script>
    const telemetry = document.getElementById("telemetry");
    const commandStatus = document.getElementById("command-status");
    const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(wsScheme + "://" + window.location.host + "/api/dashboard/ws");
    ws.onmessage = (event) => {{
      const payload = JSON.parse(event.data);
      if (payload.type === "telemetry") {{
        telemetry.textContent = JSON.stringify(payload, null, 2);
        return;
      }}
      if (payload.type === "command_reply") {{
        commandStatus.textContent = payload.reason || (payload.accepted ? "Command accepted" : "Command rejected");
      }}
    }};
    ws.onclose = () => {{
      telemetry.textContent = "Dashboard websocket disconnected.";
    }};
    document.querySelectorAll("button[data-command]").forEach((button) => {{
      button.addEventListener("click", () => {{
        if (ws.readyState !== WebSocket.OPEN) {{
          commandStatus.textContent = "Dashboard websocket is not connected.";
          return;
        }}
        const command = button.getAttribute("data-command");
        ws.send(JSON.stringify({{ type: "dashboard_command", command }}));
        commandStatus.textContent = "Sent command: " + command;
      }});
    }});
  </script>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    async def healthz_handler(self, request: web.Request) -> web.Response:
        payload = {
            "status": "ok",
            "robot_id": self.robot_id,
            "session_id": self.active_session_id,
            "operator_id": self.active_operator_id,
            "safety_state": self.safety_state,
            "camera_names": self.camera_names,
            "has_operator": self.operator_ws is not None,
        }
        return web.json_response(payload)

    async def operator_ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)

        if self.operator_ws is not None and not self.operator_ws.closed:
            await ws.send_json(
                SessionReply(
                    accepted=False,
                    session_id=self.active_session_id,
                    reason="operator already connected",
                    robot_id=self.robot_id,
                ).to_dict()
            )
            await ws.close()
            return ws

        self.operator_ws = ws
        logger.info("Operator websocket connected")

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue

                payload = json.loads(msg.data)
                msg_type = payload.get("type")

                if msg_type == "claim_session":
                    await self.handle_claim_session(ws, payload)
                elif msg_type == "leader_state":
                    await self.handle_leader_state(payload)
                elif msg_type == "operator_command":
                    await self.handle_operator_command(ws, payload)
                elif msg_type == "release_session":
                    await self.release_session(reason=str(payload.get("reason", "operator released session")))
                elif msg_type == "ping":
                    await ws.send_json({"type": "pong", "ts": now_ms()})
        finally:
            if self.operator_ws is ws:
                self.operator_ws = None
            await self.release_session(reason="operator disconnected")
            logger.info("Operator websocket disconnected")

        return ws

    async def dashboard_ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        self.dashboard_clients.add(ws)

        if self.last_telemetry is not None:
            await ws.send_json(self.last_telemetry.to_dict())

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    if payload.get("type") == "dashboard_command":
                        await self.handle_dashboard_command(ws, payload)
                        continue
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.dashboard_clients.discard(ws)

        return ws

    async def handle_claim_session(self, ws: web.WebSocketResponse, payload: dict):
        operator_id = str(payload.get("operator_id", "")).strip()
        if not operator_id:
            await ws.send_json(
                SessionReply(
                    accepted=False,
                    session_id="",
                    reason="operator_id is required",
                    robot_id=self.robot_id,
                ).to_dict()
            )
            return

        if self.active_session_id and self.active_operator_id != operator_id:
            await ws.send_json(
                SessionReply(
                    accepted=False,
                    session_id=self.active_session_id,
                    reason=f"robot is busy with operator {self.active_operator_id}",
                    robot_id=self.robot_id,
                ).to_dict()
            )
            return

        if not self.active_session_id:
            self.active_session_id = uuid.uuid4().hex
            self.active_operator_id = operator_id
            self.session_paused = False
            self.last_status_message = "session claimed"
            logger.info(f"Session {self.active_session_id} claimed by {self.active_operator_id}")

        await ws.send_json(
            SessionReply(
                accepted=True,
                session_id=self.active_session_id,
                reason="session accepted",
                robot_id=self.robot_id,
            ).to_dict()
        )

    async def handle_leader_state(self, payload: dict):
        leader_state = LeaderState.from_dict(payload)
        if not self.active_session_id or leader_state.session_id != self.active_session_id:
            return
        if self.session_paused:
            return

        self.last_leader_state = leader_state
        self.last_leader_monotonic = time.monotonic()

    async def handle_operator_command(self, ws: web.WebSocketResponse, payload: dict):
        command = OperatorCommand.from_dict(payload)
        if not self.active_session_id or command.session_id != self.active_session_id:
            await ws.send_json(
                CommandReply(
                    accepted=False,
                    command=command.command,
                    session_id=command.session_id,
                    reason="session is not active",
                ).to_dict()
            )
            return
        if command.operator_id != self.active_operator_id:
            await ws.send_json(
                CommandReply(
                    accepted=False,
                    command=command.command,
                    session_id=command.session_id,
                    reason="operator does not own the active session",
                ).to_dict()
            )
            return

        accepted, reason = await self.execute_dashboard_command(command.command, source="operator")
        await ws.send_json(
            CommandReply(
                accepted=accepted,
                command=command.command,
                session_id=command.session_id,
                reason=reason,
            ).to_dict()
        )

    async def handle_dashboard_command(self, ws: web.WebSocketResponse, payload: dict):
        command = str(payload.get("command", "")).strip().lower()
        accepted, reason = await self.execute_dashboard_command(command, source="dashboard")
        await ws.send_json(
            CommandReply(
                accepted=accepted,
                command=command,
                session_id=self.active_session_id,
                reason=reason,
            ).to_dict()
        )

    async def execute_dashboard_command(self, command: str, source: str) -> tuple[bool, str]:
        if command == "save":
            snapshot_path = self.save_snapshot()
            reason = f"snapshot saved to {snapshot_path}"
            self.last_status_message = reason
            logger.info(f"Dashboard save command from {source}: {snapshot_path}")
            return True, reason

        if not self.active_session_id:
            return False, "no active teleop session"

        if command == "pause":
            if self.session_paused:
                return True, "teleop already paused"
            self.session_paused = True
            self.last_leader_state = None
            self.last_leader_monotonic = 0.0
            self.safety_state = "paused"
            self.last_status_message = "teleop paused by operator"
            logger.info(f"Session {self.active_session_id} paused from {source}")
            return True, "teleop paused"

        if command == "resume":
            if not self.session_paused:
                return True, "teleop already active"
            self.session_paused = False
            self.last_leader_state = None
            self.last_leader_monotonic = 0.0
            self.safety_state = "waiting_input"
            self.last_status_message = "teleop resumed, waiting for fresh leader state"
            logger.info(f"Session {self.active_session_id} resumed from {source}")
            return True, "teleop resumed"

        if command == "end":
            await self.release_session(reason=f"teleop ended from {source}")
            logger.info(f"Session ended from {source}")
            return True, "teleop session ended"

        return False, "unsupported command"

    def save_snapshot(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        session_stub = self.active_session_id or "no-session"
        snapshot_path = self.save_dir / f"{session_stub}-{now_ms()}.json"
        snapshot = {
            "saved_at_unix_ms": now_ms(),
            "session_id": self.active_session_id,
            "operator_id": self.active_operator_id,
            "robot_id": self.robot_id,
            "session_paused": self.session_paused,
            "safety_state": self.safety_state,
            "status_message": self.last_status_message,
            "last_leader_state": self.last_leader_state.to_dict() if self.last_leader_state is not None else None,
            "last_telemetry": self.last_telemetry.to_dict() if self.last_telemetry is not None else None,
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
        return snapshot_path

    async def release_session(self, reason: str):
        if not self.active_session_id:
            return

        logger.info(f"Releasing session {self.active_session_id}: {reason}")
        self.active_session_id = ""
        self.active_operator_id = ""
        self.last_leader_state = None
        self.last_leader_monotonic = 0.0
        self.safety_state = "idle"
        self.session_paused = False
        self.last_status_message = reason

    async def broadcast_telemetry(self, telemetry: RobotTelemetry):
        self.last_telemetry = telemetry
        payload = telemetry.to_dict()

        recipients = set(self.dashboard_clients)
        if self.operator_ws is not None and not self.operator_ws.closed:
            recipients.add(self.operator_ws)

        for ws in list(recipients):
            if ws.closed:
                self.dashboard_clients.discard(ws)
                continue
            try:
                await ws.send_json(payload)
            except ConnectionResetError:
                self.dashboard_clients.discard(ws)

    async def collect_observation_and_publish(self) -> RobotTelemetry:
        observation = self.robot.capture_observation()
        state = observation["observation.state"].tolist()
        for camera_name in self.camera_names:
            camera_key = f"observation.images.{camera_name}"
            if camera_key in observation:
                await self.stream_store.publish_frame(camera_name, observation[camera_key])

        last_cmd_age_ms = -1
        if self.last_leader_monotonic > 0:
            last_cmd_age_ms = int((time.monotonic() - self.last_leader_monotonic) * 1000)

        return RobotTelemetry(
            session_id=self.active_session_id,
            robot_joint_pos=[float(v) for v in state[:-1]],
            robot_gripper=float(state[-1]),
            robot_fault=bool(self.robot.flexiv.isFault()),
            robot_mode="nrt_joint_position",
            last_cmd_age_ms=last_cmd_age_ms,
            link_state="connected" if self.operator_ws is not None else "idle",
            safety_state=self.safety_state,
            status_message=self.last_status_message,
            updated_at_unix_ms=now_ms(),
            camera_urls={
                name: f"/video/{name}.mjpg"
                for name in self.camera_names
            },
        )

    async def control_loop(self):
        control_period_s = 1.0 / self.control_fps
        observation_period_s = 1.0 / self.observation_fps

        while True:
            loop_started = time.perf_counter()
            age_ms = -1
            if self.last_leader_monotonic > 0:
                age_ms = int((time.monotonic() - self.last_leader_monotonic) * 1000)

            if self.active_session_id and self.session_paused:
                self.safety_state = "paused"
                self.last_status_message = "teleop paused by operator"
            elif self.active_session_id and self.last_leader_state is not None:
                if self.last_leader_state.estop_pressed:
                    self.safety_state = "estop"
                    self.last_status_message = "remote estop pressed"
                elif age_ms <= self.hold_timeout_ms and self.last_leader_state.source_ok:
                    try:
                        mapped_target = self.mapper.map_leader_state(self.last_leader_state)
                        applied = self.robot.apply_teleop_target(
                            mapped_target.joint_target,
                            mapped_target.gripper_width,
                            mapped_target.eef_pos,
                        )
                        self.safety_state = "active" if applied else "blocked"
                        self.last_status_message = (
                            "teleop command applied" if applied else "teleop target rejected"
                        )
                    except Exception as exc:
                        self.safety_state = "fault"
                        self.last_status_message = f"teleop mapping/apply failed: {exc}"
                        logger.exception("Remote teleop control loop failed")
                elif age_ms <= self.stop_timeout_ms:
                    self.safety_state = "hold"
                    self.last_status_message = f"holding after control timeout ({age_ms} ms)"
                else:
                    self.safety_state = "safe_stop"
                    self.last_status_message = f"safe stop after control timeout ({age_ms} ms)"
            elif self.active_session_id:
                self.safety_state = "waiting_input"
                self.last_status_message = "waiting for fresh leader state"
            else:
                self.safety_state = "idle"
                self.last_status_message = "waiting for operator session"

            if time.monotonic() - self.last_observation_monotonic >= observation_period_s:
                try:
                    telemetry = await self.collect_observation_and_publish()
                    await self.broadcast_telemetry(telemetry)
                except Exception as exc:
                    logger.exception("Observation publish failed")
                    telemetry = RobotTelemetry(
                        session_id=self.active_session_id,
                        robot_fault=True,
                        link_state="degraded",
                        safety_state="fault",
                        status_message=f"observation failure: {exc}",
                    )
                    await self.broadcast_telemetry(telemetry)
                self.last_observation_monotonic = time.monotonic()

            elapsed = time.perf_counter() - loop_started
            await asyncio.sleep(max(0.0, control_period_s - elapsed))


def build_parser():
    parser = argparse.ArgumentParser(description="Run the remote teleoperation robot-side server.")
    parser.add_argument("--robot-path", type=str, default="lerobot/configs/robot/koch_flexiv.yaml")
    parser.add_argument("--bind-host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--control-fps", type=int, default=30)
    parser.add_argument("--observation-fps", type=int, default=10)
    parser.add_argument("--hold-timeout-ms", type=int, default=300)
    parser.add_argument("--stop-timeout-ms", type=int, default=1000)
    parser.add_argument("--camera-names", nargs="*", default=["side", "head", "wrist"])
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--robot-id", type=str, default="flexiv-site")
    parser.add_argument("--save-dir", type=str, default="remote_teleop_server_snapshots")
    return parser


async def main_async():
    args = build_parser().parse_args()
    server = RemoteTeleopRobotServer(
        robot_path=args.robot_path,
        bind_host=args.bind_host,
        port=args.port,
        control_fps=args.control_fps,
        observation_fps=args.observation_fps,
        hold_timeout_ms=args.hold_timeout_ms,
        stop_timeout_ms=args.stop_timeout_ms,
        camera_names=args.camera_names,
        jpeg_quality=args.jpeg_quality,
        robot_id=args.robot_id,
        save_dir=args.save_dir,
    )
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, host=args.bind_host, port=args.port)
    await site.start()

    stop_event = asyncio.Event()

    def _stop():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, _stop)

    await stop_event.wait()
    await runner.cleanup()


def main():
    init_logging()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
