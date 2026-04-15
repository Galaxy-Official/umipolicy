import dataclasses
import json
import logging
import os
from pathlib import Path
import signal as signal_module
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import scipy.spatial.transform as st

from openpi_client import image_tools


_EXTERNAL_LEROBOT_SRC = Path(__file__).resolve().parents[2] / "lerobot" / "src"
if str(_EXTERNAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_LEROBOT_SRC))

from lerobot.datasets.pose_utils import mat_to_certain_pose_type
from lerobot.datasets.pose_utils import mat_to_pose
from lerobot.datasets.pose_utils import pose_to_mat
from lerobot.datasets.pose_utils import pose_to_pos_rot
from lerobot.datasets.pose_utils import pos_rot_to_mat


LOGGER = logging.getLogger(__name__)
_ACTIVE_RECORDER = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_camera_config_path() -> Path:
    return _workspace_root() / "lerobot" / "src" / "perception" / "configs" / "camera" / "handcap_camera.json"


def _bootstrap_lerobot_src() -> Path:
    if str(_EXTERNAL_LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(_EXTERNAL_LEROBOT_SRC))
    return _EXTERNAL_LEROBOT_SRC


def _load_base_camera() -> Any:
    try:
        from perception.cameras.base_camera import BaseCamera
    except ImportError:
        _bootstrap_lerobot_src()
        from perception.cameras.base_camera import BaseCamera

    return BaseCamera


def encode_state_to_handcap_state(eepose_rotvec: np.ndarray, gripper_width: float) -> np.ndarray:
    pose10d = mat_to_certain_pose_type(pose_to_mat(np.asarray(eepose_rotvec, dtype=np.float64)), "10d")
    return np.concatenate([pose10d, np.array([gripper_width], dtype=np.float64)], axis=-1)


def decode_handcap_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk with rank 2, got shape {actions.shape}")
    if actions.shape[-1] != 10:
        raise ValueError(f"Expected handcap action dim 10, got {actions.shape[-1]}")

    pose_mat = _pose10d_to_mat(actions[:, :9])
    pose_rotvec = mat_to_certain_pose_type(pose_mat, "rotvec")
    return np.concatenate([pose_rotvec, actions[:, 9:10]], axis=-1)


def _pose10d_to_mat(d10: np.ndarray) -> np.ndarray:
    pos = d10[..., :3]
    rot_mat = _rot6d_to_mat(d10[..., 3:])
    mat = np.zeros(d10.shape[:-1] + (4, 4), dtype=d10.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot_mat
    mat[..., 3, 3] = 1
    return mat


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _prepare_camera_image(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = image_tools.resize_with_pad(rgb, 224, 224)
    return image_tools.convert_to_uint8(rgb)


def _prepare_tactile_image(image_bgr: np.ndarray, use_tactile: bool) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = image_tools.resize_with_pad(rgb, 224, 224)
    rgb = image_tools.convert_to_uint8(rgb)
    if use_tactile:
        return rgb
    return np.zeros_like(rgb)


def _make_policy_observation(latest_frame: dict[str, Any], prompt: str, use_tactile: bool) -> dict[str, Any]:
    if latest_frame["wrist_img"] is None:
        raise ValueError("Latest wrist image is missing")
    if latest_frame["left_tactile_img"] is None:
        raise ValueError("Latest left tactile image is missing")
    if latest_frame["right_tactile_img"] is None:
        raise ValueError("Latest right tactile image is missing")

    state = encode_state_to_handcap_state(latest_frame["eepose"], float(latest_frame["gripper_width"]))
    return {
        "observation.images.wrist": _prepare_camera_image(latest_frame["wrist_img"]),
        "observation.tactiles.left": _prepare_tactile_image(latest_frame["left_tactile_img"], use_tactile),
        "observation.tactiles.right": _prepare_tactile_image(latest_frame["right_tactile_img"], use_tactile),
        "observation.state": state,
        "prompt": prompt,
    }


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    obs_horizon: int = 2
    ctrl_freq: int = 30
    action_latency: float = 0.0
    task_name: str = "handcap_flexiv_mvs"
    prompt: str = ""
    robot_ip: str = "192.168.2.100"
    init_qpos: tuple[float, ...] = (-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0)
    camera_config_path: Path = dataclasses.field(default_factory=_default_camera_config_path)
    record_root: str = "realworld_replay_recording"
    use_tactile: bool = True
    dry_run: bool = False
    save_states_json: bool = True


class Recorder:
    def __init__(
        self,
        output_dir: Path,
        ctrl_freq: int,
        wrist_shape: tuple[int, int],
        left_shape: tuple[int, int],
        right_shape: tuple[int, int],
        *,
        save_states_json: bool,
    ) -> None:
        self.output_dir = output_dir
        self.save_states_json = save_states_json
        self.states_data: list[dict[str, Any]] = []
        self._closed = False

        output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.wrist_video = cv2.VideoWriter(str(output_dir / "view1_wrist.mp4"), fourcc, ctrl_freq, wrist_shape)
        self.tactile_left_video = cv2.VideoWriter(
            str(output_dir / "view2_tactile_left.mp4"), fourcc, ctrl_freq, left_shape
        )
        self.tactile_right_video = cv2.VideoWriter(
            str(output_dir / "view3_tactile_right.mp4"), fourcc, ctrl_freq, right_shape
        )

    def record_frame(self, latest_frame: dict[str, Any]) -> None:
        self.wrist_video.write(latest_frame["wrist_img"])
        self.tactile_left_video.write(latest_frame["left_tactile_img"])
        self.tactile_right_video.write(latest_frame["right_tactile_img"])

    def append_state(
        self,
        *,
        step: int,
        inference_latency: float,
        latest_frame: dict[str, Any],
        action_chunk: np.ndarray,
        server_timing: dict[str, Any] | None,
        policy_timing: dict[str, Any] | None,
    ) -> None:
        state_entry = {
            "step": step,
            "inference_latency": float(inference_latency),
            "robot0_eef_pos_x": float(latest_frame["eepose"][0]),
            "robot0_eef_pos_y": float(latest_frame["eepose"][1]),
            "robot0_eef_pos_z": float(latest_frame["eepose"][2]),
            "robot0_eef_rot_x": float(latest_frame["eepose"][3]),
            "robot0_eef_rot_y": float(latest_frame["eepose"][4]),
            "robot0_eef_rot_z": float(latest_frame["eepose"][5]),
            "gripper_width": float(latest_frame["gripper_width"]),
            "action_chunk_size": int(action_chunk.shape[0]),
        }
        if server_timing is not None:
            for key, value in server_timing.items():
                state_entry[f"server_{key}"] = float(value)
        if policy_timing is not None:
            for key, value in policy_timing.items():
                state_entry[f"policy_{key}"] = float(value)
        self.states_data.append(state_entry)

    def flush(self) -> None:
        if self._closed:
            return
        self._closed = True

        self.wrist_video.release()
        self.tactile_left_video.release()
        self.tactile_right_video.release()

        if self.states_data:
            if self.save_states_json:
                with open(self.output_dir / "states.json", "w", encoding="utf-8") as f:
                    json.dump(self.states_data, f)
            try:
                import polars as pl

                pl.DataFrame(self.states_data).write_parquet(str(self.output_dir / "states.parquet"))
            except ModuleNotFoundError:
                try:
                    import pandas as pd

                    pd.DataFrame(self.states_data).to_parquet(str(self.output_dir / "states.parquet"))
                except Exception as exc:  # pragma: no cover - environment-specific parquet backend failure
                    LOGGER.warning("Failed to write states.parquet with pandas fallback: %s", exc)
            except Exception as exc:  # pragma: no cover - environment-specific parquet writer failure
                LOGGER.warning("Failed to write states.parquet with polars: %s", exc)

        if hasattr(os, "sync"):
            os.sync()


def _signal_handler(sig, frame) -> None:
    del sig, frame
    LOGGER.info("Detected interrupt signal, flushing recorder.")
    global _ACTIVE_RECORDER
    if _ACTIVE_RECORDER is not None:
        _ACTIVE_RECORDER.flush()
    raise SystemExit(0)


class ObservationThread(threading.Thread):
    def __init__(self, cam_wrist, cam_tactile_left, cam_tactile_right, env, maxlen: int = 2):
        super().__init__(daemon=True)
        self.cam_wrist = cam_wrist
        self.cam_tactile_left = cam_tactile_left
        self.cam_tactile_right = cam_tactile_right
        self.env = env
        self.maxlen = maxlen
        self.running = True
        self.lock = threading.Lock()
        self.queue: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while self.running:
                _, wrist_img, cam_cap_time = self.cam_wrist.read()
                left_tactile_img, _ = self.cam_tactile_left.get_data()
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
        except Exception as exc:  # pragma: no cover - hardware thread failure path
            self.error = exc
            self.running = False

    def get_obs(self, n: int = 2) -> list[dict[str, Any]] | None:
        if self.error is not None:
            raise RuntimeError("Observation thread failed") from self.error
        with self.lock:
            if len(self.queue) < n:
                return None
            return list(self.queue[-n:])

    def stop(self) -> None:
        self.running = False


class FlexivRealEnv:
    tx_flange_tip = np.identity(4)
    tx_flange_tip[:3, 3] = np.array([0.0, 0.0, 0.185], dtype=np.float64)
    tx_tip_flange = np.linalg.inv(tx_flange_tip)

    @staticmethod
    def tip_to_flange_pose(tip_pose: np.ndarray) -> np.ndarray:
        return mat_to_pose(pose_to_mat(tip_pose) @ FlexivRealEnv.tx_tip_flange)

    def __init__(
        self,
        init_qpos: list[float] | tuple[float, ...],
        *,
        robot_ip: str,
        dry_run: bool,
    ) -> None:
        import flexivrdk

        self._flexivrdk = flexivrdk
        self._robot = flexivrdk.Robot(robot_ip)
        self._model = flexivrdk.Model(self._robot)
        self._gripper = flexivrdk.Gripper(self._robot)
        self._mode = flexivrdk.Mode
        self._dry_run = dry_run

        if self._robot.fault():
            self._robot.ClearFault()
            time.sleep(2)

        self._robot.Enable()
        while not self._robot.operational():
            time.sleep(1)

        if not self._dry_run:
            self._gripper.Move(0.12, 0.1, 10)

        self._robot.SwitchMode(self._mode.NRT_JOINT_POSITION)
        dof = self._robot.info().DoF
        self.target_vel = [0.0] * dof
        self.max_vel = [0.5] * dof
        self.max_acc = [2.0] * dof

        if init_qpos is not None:
            self._robot.SendJointPosition(list(init_qpos), self.target_vel, self.max_vel, self.max_acc)
            time.sleep(3)
        time.sleep(1)

    def get_ee_pose(self) -> np.ndarray:
        flange_pose_raw = self._robot.states().tcp_pose.copy()
        pos = flange_pose_raw[:3]
        qw, qx, qy, qz = flange_pose_raw[3:]
        rot = st.Rotation.from_quat([qx, qy, qz, qw])
        tip_pose_mat = pos_rot_to_mat(np.array(pos, dtype=np.float64), rot) @ self.tx_flange_tip
        return mat_to_certain_pose_type(tip_pose_mat, "rotvec")

    def get_gripper_width(self) -> float:
        states = self._flexivrdk.GripperStates()
        self._gripper.GetGripperStates(states)
        return float(states.width)

    def exec_actions(self, actions: np.ndarray, timestamps: np.ndarray) -> None:
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        for i in range(len(new_actions)):
            tip_pose = new_actions[i, :6]
            target_width = float(new_actions[i, 6])

            flange_pose = self.tip_to_flange_pose(tip_pose)
            pos, rot = pose_to_pos_rot(flange_pose)
            quat_xyzw = rot.as_quat(scalar_first=False)
            flexiv_target_pose = [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(quat_xyzw[3]),
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ]

            current_q = self._robot.states().q.copy()
            reachable, target_q = self._model.reachable(flexiv_target_pose, current_q, True)
            if not reachable:
                LOGGER.warning("Target pose is unreachable: %s", flexiv_target_pose)
                continue

            if self._dry_run:
                LOGGER.info("Dry run: skipping joint/gripper command for action %d/%d", i + 1, len(new_actions))
            else:
                self._robot.SendJointPosition(target_q, self.target_vel, self.max_vel, self.max_acc)
                self._gripper.Move(max(target_width - 0.01, 0.005), 0.1, 10)

            dt = new_timestamps[i] - time.time()
            if dt > 0:
                time.sleep(dt)


def _make_output_dir(args: Args) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return _repo_root() / args.record_root / args.task_name / timestamp


def _init_cameras(camera_config_path: Path) -> tuple[Any, Any, Any]:
    BaseCamera = _load_base_camera()
    cam_dict = BaseCamera.create_cameras_from_config(config_path=str(camera_config_path))
    return cam_dict["wrist"], cam_dict["left_tactile"], cam_dict["right_tactile"]


def _init_recorder(
    output_dir: Path,
    ctrl_freq: int,
    cam_wrist,
    cam_tactile_left,
    cam_tactile_right,
    *,
    save_states_json: bool,
) -> Recorder:
    _, init_wrist_img, _ = cam_wrist.read()
    left_tactile_img, _ = cam_tactile_left.get_data()
    right_tactile_img, _ = cam_tactile_right.get_data()

    if init_wrist_img is None or left_tactile_img is None or right_tactile_img is None:
        raise RuntimeError("Failed to fetch initial frames for recorder setup")

    return Recorder(
        output_dir=output_dir,
        ctrl_freq=ctrl_freq,
        wrist_shape=(init_wrist_img.shape[1], init_wrist_img.shape[0]),
        left_shape=(left_tactile_img.shape[1], left_tactile_img.shape[0]),
        right_shape=(right_tactile_img.shape[1], right_tactile_img.shape[0]),
        save_states_json=save_states_json,
    )


def main(args: Args) -> None:
    from openpi_client import websocket_client_policy

    LOGGER.info("Connecting to remote policy server at ws://%s:%d", args.host, args.port)
    prompt = args.prompt or args.task_name
    robot_dt = 1.0 / args.ctrl_freq

    camera_config_path = args.camera_config_path
    if not camera_config_path.is_absolute():
        camera_config_path = (_repo_root() / camera_config_path).resolve()
    if not camera_config_path.exists():
        camera_config_path = _default_camera_config_path()

    output_dir = _make_output_dir(args)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    LOGGER.info("Server metadata: %s", policy.get_server_metadata())

    LOGGER.info("Initializing cameras from %s", camera_config_path)
    cam_wrist, cam_tactile_left, cam_tactile_right = _init_cameras(camera_config_path)

    LOGGER.info("Initializing recorder in %s", output_dir)
    recorder = _init_recorder(
        output_dir,
        args.ctrl_freq,
        cam_wrist,
        cam_tactile_left,
        cam_tactile_right,
        save_states_json=args.save_states_json,
    )

    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder
    signal_module.signal(signal_module.SIGINT, _signal_handler)
    signal_module.signal(signal_module.SIGTERM, _signal_handler)

    LOGGER.info("Initializing Flexiv environment at %s", args.robot_ip)
    env = FlexivRealEnv(args.init_qpos, robot_ip=args.robot_ip, dry_run=args.dry_run)

    obs_thread = ObservationThread(cam_wrist, cam_tactile_left, cam_tactile_right, env, maxlen=args.obs_horizon)
    obs_thread.start()

    LOGGER.info("Waiting for observation queue to fill (obs_horizon=%d)", args.obs_horizon)
    while obs_thread.get_obs(args.obs_horizon) is None:
        time.sleep(0.01)
    LOGGER.info("Observation queue filled. Starting inference loop.")

    step = 0
    try:
        while True:
            loop_start = time.time()
            frames = obs_thread.get_obs(args.obs_horizon)
            if frames is None:
                time.sleep(0.01)
                continue

            latest_frame = frames[-1]
            observation = _make_policy_observation(latest_frame, prompt, args.use_tactile)
            action_result = policy.infer(observation)
            action_chunk = np.asarray(action_result["actions"], dtype=np.float64)
            decoded_actions = decode_handcap_action_chunk(action_chunk)
            action_timestamps = (
                (1 + np.arange(len(decoded_actions), dtype=np.float64)) * robot_dt + time.time() - args.action_latency
            )

            env.exec_actions(decoded_actions, action_timestamps)

            inference_latency = time.time() - loop_start
            recorder.record_frame(latest_frame)
            step += 1
            recorder.append_state(
                step=step,
                inference_latency=inference_latency,
                latest_frame=latest_frame,
                action_chunk=action_chunk,
                server_timing=action_result.get("server_timing"),
                policy_timing=action_result.get("policy_timing"),
            )

            if step % 10 == 0:
                LOGGER.info(
                    "[%d] Inference cycle %.1f ms | chunk=%d",
                    step,
                    inference_latency * 1000.0,
                    action_chunk.shape[0],
                )
    finally:
        obs_thread.stop()
        recorder.flush()


if __name__ == "__main__":
    import tyro

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
