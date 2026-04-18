#!/usr/bin/env python

import argparse
import asyncio
import json
from contextlib import suppress
from pathlib import Path

from aiohttp import ClientSession, WSMsgType
from loguru import logger

from lerobot.remote_teleop.common import (
    CommandReply,
    LocalLeaderDevice,
    OperatorCommand,
    SessionClaim,
    SessionReply,
    now_ms,
)
from lerobot.common.utils.utils import init_logging


class RemoteOperatorClient:
    def __init__(
        self,
        server_url: str,
        operator_id: str,
        robot_id: str,
        leader_path: str,
        fps: int,
        status_interval_s: float,
        save_dir: str,
        interactive: bool,
    ):
        self.server_url = server_url.rstrip("/")
        self.operator_id = operator_id
        self.robot_id = robot_id
        self.leader_path = leader_path
        self.fps = fps
        self.status_interval_s = status_interval_s
        self.save_dir = Path(save_dir)
        self.interactive = interactive

        self.session_id = ""
        self.last_status_log_t = 0.0
        self.latest_telemetry = None
        self.latest_leader_state = None
        self.teleop_paused = False
        self.shutdown_requested = False
        self.leader_device = LocalLeaderDevice(self.leader_path)

    @property
    def operator_ws_url(self) -> str:
        if self.server_url.startswith("https://"):
            return "wss://" + self.server_url[len("https://") :] + "/api/operator/ws"
        if self.server_url.startswith("http://"):
            return "ws://" + self.server_url[len("http://") :] + "/api/operator/ws"
        return self.server_url + "/api/operator/ws"

    async def run(self):
        self.leader_device.connect()
        try:
            async with ClientSession() as session:
                async with session.ws_connect(self.operator_ws_url, heartbeat=15) as ws:
                    claim = SessionClaim(operator_id=self.operator_id, robot_id=self.robot_id)
                    await ws.send_json(claim.to_dict())

                    claim_reply = await ws.receive_json()
                    reply = SessionReply.from_dict(claim_reply)
                    if not reply.accepted:
                        raise RuntimeError(f"Session rejected: {reply.reason}")

                    self.session_id = reply.session_id
                    logger.info(f"Remote teleop session accepted: {self.session_id}")
                    if self.interactive:
                        self.print_command_help()

                    tasks = [
                        asyncio.create_task(self.sender_loop(ws), name="remote-operator-sender"),
                        asyncio.create_task(self.receiver_loop(ws), name="remote-operator-receiver"),
                    ]
                    if self.interactive:
                        tasks.append(
                            asyncio.create_task(self.command_loop(ws), name="remote-operator-command-loop")
                        )

                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                    for task in done:
                        task.result()
        finally:
            self.leader_device.disconnect()

    async def sender_loop(self, ws):
        period_s = 1.0 / self.fps
        seq = 0
        while True:
            if ws.closed or self.shutdown_requested:
                return

            started = asyncio.get_running_loop().time()
            if self.teleop_paused:
                await asyncio.sleep(period_s)
                continue
            try:
                leader_state = self.leader_device.read_leader_state(
                    session_id=self.session_id,
                    operator_id=self.operator_id,
                    seq=seq,
                )
            except Exception as exc:
                logger.exception("Failed reading local leader arm")
                await ws.send_json(
                    {
                        "type": "leader_state",
                        "session_id": self.session_id,
                        "operator_id": self.operator_id,
                        "seq": seq,
                        "sent_at_unix_ms": 0,
                        "leader_joint_pos": [],
                        "leader_gripper": 0.0,
                        "estop_pressed": False,
                        "source_ok": False,
                        "error": str(exc),
                    }
                )
            else:
                self.latest_leader_state = leader_state.to_dict()
                await ws.send_json(leader_state.to_dict())
            seq += 1
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, period_s - elapsed))

    async def receiver_loop(self, ws):
        while True:
            msg = await ws.receive()
            if msg.type == WSMsgType.CLOSED:
                return
            if msg.type == WSMsgType.ERROR:
                raise ws.exception()
            if msg.type != WSMsgType.TEXT:
                continue

            payload = msg.json()
            if payload.get("type") == "command_reply":
                reply = CommandReply.from_dict(payload)
                if not reply.accepted and reply.command == "pause":
                    self.teleop_paused = False
                if not reply.accepted and reply.command == "resume":
                    self.teleop_paused = True
                level = logger.info if reply.accepted else logger.warning
                level(f"Remote command `{reply.command}`: {reply.reason}")
                continue
            if payload.get("type") != "telemetry":
                continue

            self.latest_telemetry = payload
            now = asyncio.get_running_loop().time()
            if now - self.last_status_log_t >= self.status_interval_s:
                self.last_status_log_t = now
                logger.info(
                    "Remote telemetry: "
                    f"safety={payload.get('safety_state')} "
                    f"link={payload.get('link_state')} "
                    f"cmd_age_ms={payload.get('last_cmd_age_ms')} "
                    f"fault={payload.get('robot_fault')} "
                    f"msg={payload.get('status_message')}"
                )

    async def command_loop(self, ws):
        while True:
            try:
                raw_command = await asyncio.to_thread(input, "remote-teleop> ")
            except EOFError:
                raw_command = "quit"

            command = raw_command.strip().lower()
            if not command:
                continue

            if command in {"help", "h", "?", "帮助"}:
                self.print_command_help()
                continue
            if command in {"save", "s", "保存"}:
                try:
                    saved_path = self.save_snapshot()
                except Exception as exc:
                    logger.exception(f"Failed to save remote teleop snapshot: {exc}")
                else:
                    logger.info(f"Saved remote teleop snapshot to {saved_path}")
                continue
            if command in {"pause", "p", "暂停"}:
                if self.teleop_paused:
                    logger.info("Remote teleop is already paused")
                    continue
                self.teleop_paused = True
                await ws.send_json(
                    OperatorCommand(
                        session_id=self.session_id,
                        operator_id=self.operator_id,
                        command="pause",
                    ).to_dict()
                )
                logger.info("Pause command sent")
                continue
            if command in {"resume", "r", "恢复"}:
                if not self.teleop_paused:
                    logger.info("Remote teleop is not paused")
                    continue
                self.teleop_paused = False
                await ws.send_json(
                    OperatorCommand(
                        session_id=self.session_id,
                        operator_id=self.operator_id,
                        command="resume",
                    ).to_dict()
                )
                logger.info("Resume command sent")
                continue
            if command in {"quit", "q", "exit", "end", "结束"}:
                await self.shutdown(ws, "operator ended session")
                return

            logger.warning(f"Unknown command: {raw_command.strip()}")
            self.print_command_help()

    async def shutdown(self, ws, reason: str):
        if self.shutdown_requested:
            return

        self.shutdown_requested = True
        if not ws.closed:
            with suppress(Exception):
                await ws.send_json(
                    {
                        "type": "release_session",
                        "session_id": self.session_id,
                        "operator_id": self.operator_id,
                        "reason": reason,
                    }
                )
            with suppress(Exception):
                await ws.close()

    def save_snapshot(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        session_stub = self.session_id or "no-session"
        snapshot_path = self.save_dir / f"{session_stub}-{now_ms()}.json"
        snapshot = {
            "saved_at_unix_ms": now_ms(),
            "session_id": self.session_id,
            "operator_id": self.operator_id,
            "robot_id": self.robot_id,
            "server_url": self.server_url,
            "teleop_paused": self.teleop_paused,
            "latest_leader_state": self.latest_leader_state,
            "latest_telemetry": self.latest_telemetry,
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
        return snapshot_path

    @staticmethod
    def print_command_help():
        logger.info("Client commands: save(s/保存), pause(p/暂停), resume(r/恢复), quit(q/结束), help")


def build_parser():
    parser = argparse.ArgumentParser(description="Run the operator-side remote teleop client.")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8765")
    parser.add_argument("--operator-id", type=str, default="operator-1")
    parser.add_argument("--robot-id", type=str, default="flexiv-site")
    parser.add_argument("--leader-path", type=str, default="lerobot/configs/robot/koch_leader.yaml")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--status-interval-s", type=float, default=2.0)
    parser.add_argument("--save-dir", type=str, default="remote_teleop_snapshots")
    parser.add_argument("--no-interactive", action="store_true")
    return parser


async def main_async():
    args = build_parser().parse_args()
    client = RemoteOperatorClient(
        server_url=args.server_url,
        operator_id=args.operator_id,
        robot_id=args.robot_id,
        leader_path=args.leader_path,
        fps=args.fps,
        status_interval_s=args.status_interval_s,
        save_dir=args.save_dir,
        interactive=not args.no_interactive,
    )
    await client.run()


def main():
    init_logging()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
