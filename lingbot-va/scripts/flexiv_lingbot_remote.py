#!/usr/bin/env python3
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import scipy.spatial.transform as st

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy


LOGGER = logging.getLogger(__name__)
INFER_PROGRESS_INTERVAL_S = 10.0


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_camera_config_path() -> Path:
    override = os.environ.get("FLEXIV_CAMERA_CONFIG_PATH")
    if override:
        return Path(override)
    camera_dir = _workspace_root() / "lerobot" / "src" / "perception" / "configs" / "camera"
    return camera_dir / "flexiv_camera.json"


def _bootstrap_perception_src() -> None:
    perception_src = _workspace_root() / "lerobot" / "src"
    if str(perception_src) not in sys.path:
        sys.path.insert(0, str(perception_src))


def _load_base_camera() -> Any:
    _bootstrap_perception_src()
    from perception.cameras.base_camera import BaseCamera

    return BaseCamera


def pos_rot_to_mat(pos: np.ndarray, rot: st.Rotation) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = np.asarray(pos, dtype=np.float64)
    mat[:3, :3] = rot.as_matrix()
    return mat


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    return pos_rot_to_mat(pose[:3], st.Rotation.from_rotvec(pose[3:6]))


def mat_to_rotvec_pose(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    pos = mat[..., :3, 3]
    rot = st.Rotation.from_matrix(mat[..., :3, :3].reshape(-1, 3, 3))
    rotvec = rot.as_rotvec().reshape(mat.shape[:-2] + (3,))
    return np.concatenate([pos, rotvec], axis=-1)


def pose_to_pos_rot(pose: np.ndarray) -> tuple[np.ndarray, st.Rotation]:
    pose = np.asarray(pose, dtype=np.float64)
    return pose[:3], st.Rotation.from_rotvec(pose[3:6])


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def pose10d_to_rotvec_pose(d10: np.ndarray) -> np.ndarray:
    d10 = np.asarray(d10, dtype=np.float64)
    mat = np.zeros(d10.shape[:-1] + (4, 4), dtype=np.float64)
    mat[..., :3, 3] = d10[..., :3]
    mat[..., :3, :3] = _rot6d_to_mat(d10[..., 3:9])
    mat[..., 3, 3] = 1.0
    return mat_to_rotvec_pose(mat)


def decode_lingbot_action_chunk(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 3:
        # LingBot Flexiv server returns [C, F, action_per_frame].
        if action.shape[0] == 10:
            action = np.moveaxis(action, 0, -1).reshape(-1, 10)
        elif action.shape[-1] == 10:
            action = action.reshape(-1, 10)
        else:
            raise ValueError(f"Cannot decode LingBot action shape {action.shape}")
    elif action.ndim == 2:
        if action.shape[-1] != 10 and action.shape[0] == 10:
            action = action.T
    else:
        raise ValueError(f"Expected action rank 2 or 3, got shape {action.shape}")

    if action.shape[-1] != 10:
        raise ValueError(f"Expected action dim 10, got shape {action.shape}")

    pose_rotvec = pose10d_to_rotvec_pose(action[:, :9])
    return np.concatenate([pose_rotvec, action[:, 9:10]], axis=-1)


def relative_action_to_absolute(raw_action_rotvec: np.ndarray, current_pose_rotvec: np.ndarray) -> np.ndarray:
    return mat_to_rotvec_pose(pose_to_mat(current_pose_rotvec[:6]) @ pose_to_mat(raw_action_rotvec[:6]))


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _make_frame_obs(frame: dict[str, Any], use_tactile: bool) -> dict[str, np.ndarray]:
    if frame["wrist_img"] is None:
        raise ValueError("Latest wrist image is missing")
    wrist_rgb = _bgr_to_rgb(frame["wrist_img"])

    obs = {
        "observation.images.wrist": np.ascontiguousarray(wrist_rgb),
    }

    if use_tactile:
        if frame["left_tactile_img"] is None or frame["right_tactile_img"] is None:
            raise ValueError("Tactile images are required when use_tactile=True")
        left_rgb = _bgr_to_rgb(frame["left_tactile_img"])
        right_rgb = _bgr_to_rgb(frame["right_tactile_img"])
        obs.update(
            {
                "observation.tactiles.left": np.ascontiguousarray(left_rgb),
                "observation.tactiles.right": np.ascontiguousarray(right_rgb),
            }
        )

    return obs


def make_lingbot_observation(frames: list[dict[str, Any]], prompt: str, use_tactile: bool) -> dict[str, Any]:
    return {
        "obs": [_make_frame_obs(frame, use_tactile) for frame in frames],
        "prompt": prompt,
    }


def infer_with_progress(
    executor: ThreadPoolExecutor,
    policy: WebsocketClientPolicy,
    observation: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    start_time = time.time()
    LOGGER.info("%s: sending request to LingBot-VA server", label)
    future = executor.submit(policy.infer, observation)
    while True:
        try:
            result = future.result(timeout=INFER_PROGRESS_INTERVAL_S)
            LOGGER.info("%s: server returned after %.1fs", label, time.time() - start_time)
            return result
        except TimeoutError:
            LOGGER.info("%s: still waiting for server response (%.1fs elapsed)", label, time.time() - start_time)


class ObservationThread(threading.Thread):
    def __init__(
        self,
        cam_wrist,
        cam_tactile_left,
        cam_tactile_right,
        env,
        *,
        maxlen: int,
        camera_fps: float,
    ) -> None:
        super().__init__(daemon=True)
        self.cam_wrist = cam_wrist
        self.cam_tactile_left = cam_tactile_left
        self.cam_tactile_right = cam_tactile_right
        self.env = env
        self.maxlen = maxlen
        self.camera_dt = 1.0 / camera_fps if camera_fps > 0 else 0.0
        self.running = True
        self.lock = threading.Lock()
        self.queue: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while self.running:
                start = time.monotonic()
                _, wrist_img, cam_cap_time = self.cam_wrist.read()
                left_tactile_img, right_tactile_img = None, None
                if self.cam_tactile_left is not None:
                    left_tactile_img, _ = self.cam_tactile_left.get_data()
                if self.cam_tactile_right is not None:
                    right_tactile_img, _ = self.cam_tactile_right.get_data()

                eepose = self.env.get_ee_pose()
                gripper_width = self.env.get_gripper_width()
                with self.lock:
                    self.queue.append(
                        {
                            "wrist_img": wrist_img.copy() if wrist_img is not None else None,
                            "left_tactile_img": left_tactile_img.copy() if left_tactile_img is not None else None,
                            "right_tactile_img": right_tactile_img.copy() if right_tactile_img is not None else None,
                            "cam_cap_time": cam_cap_time,
                            "eepose": np.asarray(eepose, dtype=np.float64),
                            "gripper_width": float(gripper_width),
                        }
                    )
                    self.queue = self.queue[-self.maxlen :]

                if self.camera_dt > 0:
                    target = start + self.camera_dt
                    while True:
                        remaining = target - time.monotonic()
                        if remaining <= 0:
                            break
                        time.sleep(min(remaining, 0.002))
        except Exception as exc:
            self.error = exc
            self.running = False

    def get_obs(self, n: int) -> list[dict[str, Any]] | None:
        if self.error is not None:
            raise RuntimeError("Observation thread failed") from self.error
        with self.lock:
            if len(self.queue) < n:
                return None
            return list(self.queue[-n:])

    def stop(self) -> None:
        self.running = False


class FlexivRealEnv:
    def __init__(
        self,
        init_qpos: list[float],
        *,
        robot_ip: str,
        local_ip: str,
        dry_run: bool,
    ) -> None:
        import flexivrdk

        self._mode = flexivrdk.Mode
        self._dry_run = dry_run
        self._init_qpos = list(init_qpos)
        robot_sn = os.environ.get("FLEXIV_ROBOT_SN", "Rizon4-062339")
        gripper_name = os.environ.get("FLEXIV_GRIPPER_NAME", "Flexiv-GN01")

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((robot_ip, 1))
            actual_local_ip = sock.getsockname()[0]
            sock.close()
        except Exception:
            actual_local_ip = local_ip

        LOGGER.info("Initializing Flexiv Robot SN=%s, robot_ip=%s, local_ip=%s", robot_sn, robot_ip, actual_local_ip)
        self._robot = flexivrdk.Robot(robot_sn, [actual_local_ip])
        self._gripper = flexivrdk.Gripper(self._robot)
        self._model = flexivrdk.Model(self._robot)

        try:
            LOGGER.info("Enabling gripper [%s].", gripper_name)
            self._gripper.Enable(gripper_name)
        except Exception as exc:
            LOGGER.warning("Failed to enable gripper [%s]: %s", gripper_name, exc)

        if self._robot.fault():
            self._robot.ClearFault()
            time.sleep(2)
        self._robot.Enable()
        while not self._robot.operational():
            time.sleep(1)

        self._switch_cartesian_mode()
        if not dry_run:
            self._gripper.Move(self._gripper.params().max_width, 0.1, 20)
            time.sleep(1)

    def _switch_cartesian_mode(self) -> None:
        LOGGER.info("Switching to NRT_CARTESIAN_MOTION_FORCE mode.")
        self._robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
        self._robot.SetForceControlAxis([False, False, False, False, False, False])

    def reset(self) -> None:
        LOGGER.info("Switching to NRT_JOINT_POSITION for reset.")
        self._robot.SwitchMode(self._mode.NRT_JOINT_POSITION)
        dof = len(self._init_qpos)
        if not self._dry_run:
            LOGGER.info("Resetting robot to initial joint positions: %s", self._init_qpos)
            self._robot.SendJointPosition(self._init_qpos, [0.0] * dof, [0.1] * dof, [0.1] * dof)
            self._gripper.Move(self._gripper.params().max_width, 0.1, 20)
            time.sleep(10)
        self._switch_cartesian_mode()

    def get_ee_pose(self) -> np.ndarray:
        tcp_pose = self._robot.states().tcp_pose.copy()
        pos = np.asarray(tcp_pose[:3], dtype=np.float64)
        qw, qx, qy, qz = tcp_pose[3:]
        rot = st.Rotation.from_quat([qx, qy, qz, qw])
        return mat_to_rotvec_pose(pos_rot_to_mat(pos, rot))

    def get_gripper_width(self) -> float:
        return float(self._gripper.states().width)

    def exec_actions(self, actions: np.ndarray, timestamps: np.ndarray) -> None:
        receive_time = time.time()
        new_actions = actions[timestamps > receive_time]
        new_timestamps = timestamps[timestamps > receive_time]

        for i, action in enumerate(new_actions):
            target_pose = action[:6]
            target_width = float(action[6])
            pos, rot = pose_to_pos_rot(target_pose)
            quat = rot.as_quat()
            target_tcp = [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(quat[3]),
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
            ]

            if self._dry_run:
                LOGGER.info("Dry run: action %d/%d target_tcp=%s gripper=%.4f", i + 1, len(new_actions), target_tcp, target_width)
            else:
                self._robot.SendCartesianMotionForce(target_tcp, [0.0] * 6, [0.0] * 6, 0.15, 0.5, 0.3, 1.0)
                max_width = self._gripper.params().max_width
                safe_width = min(max(target_width, 0.001), max_width - 0.001)
                self._gripper.Move(safe_width, 0.1, 20)

            dt = new_timestamps[i] - time.time()
            if dt > 0:
                time.sleep(dt)


def init_cameras(camera_config_path: Path, use_tactile: bool):
    BaseCamera = _load_base_camera()
    config_path = camera_config_path
    if not use_tactile:
        no_tactile_candidates = [
            camera_config_path.with_name(f"{camera_config_path.stem}_no_tactile{camera_config_path.suffix}"),
            camera_config_path.with_name("flexiv_camera_no_tactile.json"),
        ]
        for no_tactile_config_path in no_tactile_candidates:
            if no_tactile_config_path.exists():
                config_path = no_tactile_config_path
                LOGGER.info("use_tactile=False: using wrist-only camera config %s", config_path)
                cam_dict = BaseCamera.create_cameras_from_config(config_path=str(config_path))
                return cam_dict["wrist"], None, None

    cam_dict = BaseCamera.create_cameras_from_config(config_path=str(config_path))
    if not use_tactile:
        return cam_dict["wrist"], None, None
    return cam_dict["wrist"], cam_dict["left_tactile"], cam_dict["right_tactile"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29536)
    parser.add_argument("--prompt", default="simple sorting task")
    parser.add_argument("--robot-ip", default=os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100"))
    parser.add_argument("--local-ip", default=os.environ.get("FLEXIV_LOCAL_IP", "192.168.2.102"))
    parser.add_argument("--init-qpos", default="[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]")
    parser.add_argument("--camera-config-path", type=Path, default=_default_camera_config_path())
    parser.add_argument("--obs-horizon", type=int, default=1)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--ctrl-freq", type=float, default=20.0)
    parser.add_argument("--steps-per-inference", type=int, default=4)
    parser.add_argument("--action-latency", type=float, default=0.0)
    parser.add_argument("--action-pose-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--use-tactile", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_qpos = list(ast.literal_eval(args.init_qpos))
    robot_dt = 1.0 / args.ctrl_freq

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    LOGGER.info(
        "Control config: ctrl_freq=%.3fHz, steps_per_inference=%d, camera_fps=%.3fHz, action_pose_mode=%s",
        args.ctrl_freq,
        args.steps_per_inference,
        args.camera_fps,
        args.action_pose_mode,
    )

    LOGGER.info("Connecting to LingBot-VA server at ws://%s:%d", args.host, args.port)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    LOGGER.info("Server metadata: %s", policy.get_server_metadata())

    executor = ThreadPoolExecutor(max_workers=1)
    infer_with_progress(executor, policy, {"reset": True, "prompt": args.prompt}, "reset")

    camera_config_path = args.camera_config_path
    if not camera_config_path.is_absolute():
        camera_config_path = (_workspace_root() / camera_config_path).resolve()
    LOGGER.info("Initializing cameras from %s (use_tactile=%s)", camera_config_path, args.use_tactile)
    cam_wrist, cam_left, cam_right = init_cameras(camera_config_path, args.use_tactile)

    LOGGER.info("Initializing Flexiv environment at %s", args.robot_ip)
    env = FlexivRealEnv(init_qpos, robot_ip=args.robot_ip, local_ip=args.local_ip, dry_run=args.dry_run)
    env.reset()

    obs_thread = ObservationThread(
        cam_wrist,
        cam_left,
        cam_right,
        env,
        maxlen=args.obs_horizon,
        camera_fps=args.camera_fps,
    )
    obs_thread.start()
    LOGGER.info("Waiting for observation queue to fill (obs_horizon=%d)", args.obs_horizon)
    while obs_thread.get_obs(args.obs_horizon) is None:
        time.sleep(0.01)
    LOGGER.info("Observation queue filled. Starting LingBot-VA inference loop.")

    step = 0
    try:
        while True:
            loop_start = time.time()
            frames = obs_thread.get_obs(args.obs_horizon)
            if frames is None:
                time.sleep(0.01)
                continue

            latest_frame = frames[-1]
            observation = make_lingbot_observation(frames, args.prompt, args.use_tactile)
            result = infer_with_progress(executor, policy, observation, f"step={step + 1}")
            action_chunk = np.asarray(result["action"], dtype=np.float64)
            decoded_actions = decode_lingbot_action_chunk(action_chunk)
            policy_chunk_size = len(decoded_actions)

            current_eepose = latest_frame["eepose"]
            physical_actions = []
            for action in decoded_actions:
                if args.action_pose_mode == "relative":
                    target_pose = relative_action_to_absolute(action[:6], current_eepose)
                else:
                    target_pose = action[:6]
                physical_actions.append(np.concatenate([target_pose, action[6:7]], axis=-1))
            physical_actions = np.asarray(physical_actions, dtype=np.float64)
            if args.steps_per_inference > 0:
                physical_actions = physical_actions[: args.steps_per_inference]

            action_timestamps = (
                (1 + np.arange(len(physical_actions), dtype=np.float64)) * robot_dt + time.time() - args.action_latency
            )
            env.exec_actions(physical_actions, action_timestamps)

            step += 1
            inference_latency = time.time() - loop_start
            first_target = physical_actions[0]
            LOGGER.info(
                "[step=%d] infer=%.1fms policy_chunk=%d exec_steps=%d eef_xyz=(%.3f, %.3f, %.3f) "
                "gripper=%.4f target_xyz=(%.3f, %.3f, %.3f) target_gripper=%.4f server_ms=%s",
                step,
                inference_latency * 1000.0,
                policy_chunk_size,
                len(physical_actions),
                float(current_eepose[0]),
                float(current_eepose[1]),
                float(current_eepose[2]),
                float(latest_frame["gripper_width"]),
                float(first_target[0]),
                float(first_target[1]),
                float(first_target[2]),
                float(first_target[6]),
                result.get("server_timing", {}).get("infer_ms"),
            )
    finally:
        obs_thread.stop()
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
