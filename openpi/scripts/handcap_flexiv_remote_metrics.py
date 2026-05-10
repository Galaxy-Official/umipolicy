import dataclasses
import json
import logging
import os
from pathlib import Path
import signal as signal_module
import socket
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
WRIST_RECORD_SHAPE = (640, 480)


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


from scipy.spatial.transform import Rotation

def encode_state_to_handcap_state(eepose_rotvec: np.ndarray, gripper_width: float) -> np.ndarray:
    # Because the dataset's __getitem__ processes all states into RELATIVE states using process_to_relative_rot6d,
    # the model was trained expecting the current observation.state to be the origin (Identity matrix).
    # Therefore, we MUST feed it the identity pose (pos=0, rot=identity) during inference!
    identity_pos = np.array([0.0, 0.0, 0.0])
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    pose10d = np.concatenate([identity_pos, identity_rot6d])
    return np.concatenate([pose10d, np.array([gripper_width], dtype=np.float64)], axis=-1)

def get_real_umi_inference_action(raw_action_rotvec: np.ndarray, abs_pose_rotvec: np.ndarray) -> np.ndarray:
    # raw_action_rotvec is shape (6,) -> [pos, rotvec]
    # abs_pose_rotvec is shape (7,) -> [pos, rotvec, gripper] or (6,) -> [pos, rotvec]
    
    current_pose_mat = pose_to_mat(abs_pose_rotvec[:6])
    relative_action_mat = pose_to_mat(raw_action_rotvec[:6])
    
    # target_pose = current_pose @ relative_action
    target_pose_mat = current_pose_mat @ relative_action_mat
    
    target_pose_rotvec = mat_to_certain_pose_type(target_pose_mat, "rotvec")
    return target_pose_rotvec

def resize_with_black_padding(image, target_h=480, target_w=640, is_depth=False):
    """
    Resize image to fit within (target_h, target_w) while preserving aspect ratio,
    then pad with black borders to reach exactly (target_h, target_w).
    """
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
        
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    
    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=-1)
        
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    if c == 3:
        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    else:
        canvas = np.zeros((target_h, target_w, 1), dtype=image.dtype)
        
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return canvas


def decode_handcap_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk with rank 2, got shape {actions.shape}")
    if actions.shape[-1] < 10:
        raise ValueError(f"Expected handcap action dim >= 10, got {actions.shape[-1]}")
    actions = actions[:, :10]

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


def _prepare_tactile_image(image_bgr: np.ndarray | None, use_tactile: bool) -> np.ndarray:
    if image_bgr is None:
        if use_tactile:
            raise ValueError("Tactile image is missing while use_tactile=True")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (224, 224):
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = image_tools.convert_to_uint8(rgb)
    if use_tactile:
        return rgb
    return np.zeros_like(rgb)


def _prepare_force(force: np.ndarray | None) -> np.ndarray:
    if force is None:
        return np.zeros((2,), dtype=np.float32)
    force_arr = np.asarray(force, dtype=np.float32).reshape(-1)
    if force_arr.shape[-1] < 2:
        force_arr = np.pad(force_arr, (0, 2 - force_arr.shape[-1]))
    return np.clip(force_arr[:2], 0.0, 10.0)


def _prepare_video_frame(image: np.ndarray | None, frame_shape: tuple[int, int]) -> np.ndarray | None:
    if image is None:
        return None

    frame = np.asarray(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        LOGGER.warning("Skipping video frame with unsupported shape: %s", frame.shape)
        return None

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if (frame.shape[1], frame.shape[0]) != frame_shape:
        frame = cv2.resize(frame, frame_shape, interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(frame)


def _make_policy_observation(latest_frame: dict[str, Any], prompt: str, use_tactile: bool) -> dict[str, Any]:
    if latest_frame["wrist_img"] is None:
        raise ValueError("Latest wrist image is missing")
    if use_tactile and latest_frame["left_tactile_img"] is None:
        raise ValueError("Latest left tactile image is missing")
    if use_tactile and latest_frame["right_tactile_img"] is None:
        raise ValueError("Latest right tactile image is missing")

    wrist_image = _prepare_camera_image(latest_frame["wrist_img"])
    left_tactile = _prepare_tactile_image(latest_frame["left_tactile_img"], use_tactile)
    right_tactile = _prepare_tactile_image(latest_frame["right_tactile_img"], use_tactile)
    state = encode_state_to_handcap_state(latest_frame["eepose"], float(latest_frame["gripper_width"]))
    force = _prepare_force(latest_frame.get("force"))

    return {
        "observation/wrist_image": wrist_image,
        "observation/left_tactile": left_tactile,
        "observation/right_tactile": right_tactile,
        "observation/state": state,
        "observation/force": force,
        # Keep LeRobot-style names too, in case this script is used with a server
        # configuration that includes HandcapRepackTransform.
        "observation.images.wrist": wrist_image,
        "observation.tactiles.left": left_tactile,
        "observation.tactiles.right": right_tactile,
        "observation.state": state,
        "observation.force": force,
        "prompt": prompt,
    }


@dataclasses.dataclass
class TrajectoryCostConfig:
    monitor_hz: float = 50.0
    w_e: float = 0.30
    w_m: float = 0.55
    w_c: float = 0.15
    t_ref: float = 10.0
    d_min: float = 0.02
    v_min: float = 0.005
    a_ref: float = 0.30
    j_ref: float = 1.00
    a_max: float = 0.30
    f_ref: float = 10.0
    f_max: float = 30.0
    tau_ref: float = 5.0
    lambda_l: float = 1.0
    lambda_s: float = 1.0
    lambda_v: float = 1.0
    lambda_a: float = 1.0
    lambda_j: float = 1.0
    lambda_p: float = 1.0
    lambda_q: float = 1.0
    lambda_f: float = 1.0
    lambda_tau: float = 1.0


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


def _vector(value: Any, min_size: int) -> np.ndarray | None:
    arr = np.asarray(_float_list(value), dtype=np.float64).reshape(-1)
    if arr.size < min_size or not np.all(np.isfinite(arr)):
        return None
    return arr


def _first_vector(min_size: int, *values: Any) -> np.ndarray | None:
    for value in values:
        vec = _vector(value, min_size)
        if vec is not None:
            return vec
    return None


def _rms_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        norms = np.abs(arr)
    else:
        norms = np.linalg.norm(arr, axis=1)
    return float(np.sqrt(np.mean(np.square(norms)))) if norms.size else 0.0


def _p95_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        norms = np.abs(arr)
    else:
        norms = np.linalg.norm(arr, axis=1)
    return float(np.percentile(norms, 95)) if norms.size else 0.0


def _quality_band(cost: float) -> str:
    if cost < 0.5:
        return "good"
    if cost < 1.0:
        return "acceptable"
    if cost < 2.0:
        return "fair"
    return "poor"


def _select_time(rows: list[dict[str, Any]]) -> tuple[np.ndarray, str]:
    for key in ("robot_time", "wall_time"):
        t = np.asarray([_finite_float(row.get(key), np.nan) for row in rows], dtype=np.float64)
        if np.all(np.isfinite(t)) and t[-1] > t[0]:
            return t, key
    t = np.arange(len(rows), dtype=np.float64)
    return t, "sample_index"


def compute_trajectory_cost(
    samples: list[dict[str, Any]],
    config: TrajectoryCostConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        pose = _vector(sample.get("tcp_pose"), 7)
        vel = _vector(sample.get("tcp_vel"), 3)
        wall_time = _finite_float(sample.get("wall_time"))
        if pose is None or wall_time is None:
            continue
        rows.append(
            {
                "wall_time": wall_time,
                "robot_time": _finite_float(sample.get("robot_time")),
                "p": pose[:3],
                "v": vel[:3] if vel is not None else None,
                "dq": _first_vector(1, sample.get("dq"), sample.get("dtheta")),
                "dq_max": _vector(sample.get("dq_max"), 1),
                "f": _first_vector(
                    3,
                    sample.get("ext_wrench_in_world"),
                    sample.get("ext_wrench_in_tcp"),
                    sample.get("ft_sensor_raw"),
                ),
                "tau": _vector(sample.get("tau_ext"), 1),
                "target_tcp": _vector(sample.get("target_tcp"), 7),
            }
        )

    if len(rows) < 3:
        return {
            "valid": False,
            "reason": "Need at least 3 valid robot state samples with tcp_pose.",
            "num_samples": len(samples),
            "num_valid_samples": len(rows),
            "config": dataclasses.asdict(config),
        }

    t, time_source = _select_time(rows)
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    if keep.sum() < 3:
        return {
            "valid": False,
            "reason": "Robot state timestamps are not increasing.",
            "num_samples": len(samples),
            "num_valid_samples": int(keep.sum()),
            "time_source": time_source,
            "config": dataclasses.asdict(config),
        }

    t = t[keep]
    kept_rows = [row for row, flag in zip(rows, keep) if flag]
    p = np.vstack([row["p"] for row in kept_rows])
    dt = np.diff(t)

    v_rows = [row["v"] for row in kept_rows]
    if all(v is not None for v in v_rows):
        v = np.vstack(v_rows)
        velocity_source = "tcp_vel"
    else:
        segment_v = np.diff(p, axis=0) / dt[:, None]
        v = np.zeros_like(p)
        v[0] = segment_v[0]
        v[-1] = segment_v[-1]
        if len(p) > 2:
            v[1:-1] = 0.5 * (segment_v[:-1] + segment_v[1:])
        velocity_source = "pose_diff"
    speed = np.linalg.norm(v, axis=1)

    T = float(t[-1] - t[0])
    L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    D = float(np.linalg.norm(p[-1] - p[0]))
    path_ratio = L / max(D, config.d_min)
    path_excess = max(path_ratio - 1.0, 0.0)
    r_stop = float(np.mean(speed < config.v_min))

    a = np.diff(v, axis=0) / dt[:, None]
    j = np.diff(a, axis=0) / dt[1:, None] if len(a) > 1 else np.empty((0, 3), dtype=np.float64)
    mu_v = float(np.mean(speed))
    sigma_v = float(np.std(speed))
    a_rms = _rms_norm(a)
    j_rms = _rms_norm(j)
    a_p95 = _p95_norm(a)

    dq_ratio_max = 0.0
    dq_available = False
    for row in kept_rows:
        dq = row["dq"]
        dq_max = row["dq_max"]
        if dq is None or dq_max is None:
            continue
        width = min(dq.size, dq_max.size)
        if width == 0:
            continue
        denom = np.maximum(np.abs(dq_max[:width]), 1e-9)
        dq_ratio_max = max(dq_ratio_max, float(np.max(np.abs(dq[:width]) / denom)))
        dq_available = True

    f_rows = [row["f"][:3] for row in kept_rows if row["f"] is not None]
    f = np.vstack(f_rows) if f_rows else np.empty((0, 3), dtype=np.float64)
    tau_rows = [row["tau"] for row in kept_rows if row["tau"] is not None]
    tau = np.vstack(tau_rows) if tau_rows else np.empty((0, 0), dtype=np.float64)
    f_rms = _rms_norm(f)
    f_p95 = _p95_norm(f)
    tau_rms = _rms_norm(tau)

    target_errors = []
    for row in kept_rows:
        target_tcp = row["target_tcp"]
        if target_tcp is not None:
            target_errors.append(float(np.linalg.norm(row["p"] - target_tcp[:3])))
    target_pos_error_rms = (
        float(np.sqrt(np.mean(np.square(target_errors)))) if target_errors else None
    )

    eps = 1e-9
    J_e = (
        T / max(config.t_ref, eps)
        + config.lambda_l * path_excess
        + config.lambda_s * r_stop
    )
    J_m = (
        config.lambda_v * sigma_v / (mu_v + eps)
        + config.lambda_a * a_rms / max(config.a_ref, eps)
        + config.lambda_j * j_rms / max(config.j_ref, eps)
        + config.lambda_p * a_p95 / max(config.a_max, eps)
        + config.lambda_q * dq_ratio_max
    )
    J_c = (
        f_rms / max(config.f_ref, eps)
        + config.lambda_f * f_p95 / max(config.f_max, eps)
        + config.lambda_tau * tau_rms / max(config.tau_ref, eps)
    )
    J = config.w_e * J_e + config.w_m * J_m + config.w_c * J_c

    return {
        "valid": True,
        "J": float(J),
        "J_e": float(J_e),
        "J_m": float(J_m),
        "J_c": float(J_c),
        "quality_band": _quality_band(float(J)),
        "time_source": time_source,
        "velocity_source": velocity_source,
        "num_samples": len(samples),
        "num_valid_samples": len(kept_rows),
        "sample_hz_est": float((len(t) - 1) / T) if T > 0 else 0.0,
        "components": {
            "T": T,
            "L": L,
            "D": D,
            "path_ratio": float(path_ratio),
            "path_excess": float(path_excess),
            "r_stop": r_stop,
            "mu_v": mu_v,
            "sigma_v": sigma_v,
            "a_rms": a_rms,
            "j_rms": j_rms,
            "a_p95": a_p95,
            "dq_ratio_max": dq_ratio_max,
            "dq_available": dq_available,
            "f_rms": f_rms,
            "f_p95": f_p95,
            "tau_rms": tau_rms,
            "target_pos_error_rms": target_pos_error_rms,
        },
        "config": dataclasses.asdict(config),
    }


class TrajectoryMetricMonitor(threading.Thread):
    def __init__(self, env: "FlexivRealEnv", config: TrajectoryCostConfig):
        super().__init__(daemon=True)
        self.env = env
        self.config = config
        self.running = True
        self.lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def run(self) -> None:
        period = 1.0 / max(self.config.monitor_hz, 1e-6)
        while self.running:
            start = time.time()
            try:
                sample = self.env.get_metric_snapshot()
                with self.lock:
                    self.samples.append(sample)
            except Exception as exc:  # pragma: no cover - hardware state path
                if self.error is None:
                    LOGGER.warning("Trajectory metric monitor failed to sample robot state: %s", exc)
                self.error = exc
            elapsed = time.time() - start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self.running = False

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.samples)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    obs_horizon: int = 2
    ctrl_freq: float = 20.0
    steps_per_inference: int = 4
    action_latency: float = 0.0
    task_name: str = "handcap_flexiv_mvs"
    prompt: str = ""
    robot_ip: str = "192.168.2.100"
    local_ip: str = dataclasses.field(default_factory=lambda: os.environ.get("FLEXIV_LOCAL_IP", "192.168.2.102"))
    init_qpos: tuple[float, ...] = (-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0)
    camera_config_path: Path = dataclasses.field(default_factory=_default_camera_config_path)
    record_root: str = "realworld_replay_recording"
    use_tactile: bool = True
    force_predict: bool = False
    force_guide: bool = False
    infer_force_control: bool = False
    dry_run: bool = False
    save_states_json: bool = True
    metric_monitor_hz: float = 50.0
    metric_t_ref: float = 10.0
    metric_d_min: float = 0.02
    metric_v_min: float = 0.005
    metric_a_ref: float = 0.30
    metric_j_ref: float = 1.00
    metric_a_max: float = 0.30
    metric_f_ref: float = 10.0
    metric_f_max: float = 30.0
    metric_tau_ref: float = 5.0
    metric_lambda_l: float = 1.0
    metric_lambda_s: float = 1.0
    metric_lambda_v: float = 1.0
    metric_lambda_a: float = 1.0
    metric_lambda_j: float = 1.0
    metric_lambda_p: float = 1.0
    metric_lambda_q: float = 1.0
    metric_lambda_f: float = 1.0
    metric_lambda_tau: float = 1.0


def _make_trajectory_cost_config(args: Args) -> TrajectoryCostConfig:
    return TrajectoryCostConfig(
        monitor_hz=args.metric_monitor_hz,
        t_ref=args.metric_t_ref,
        d_min=args.metric_d_min,
        v_min=args.metric_v_min,
        a_ref=args.metric_a_ref,
        j_ref=args.metric_j_ref,
        a_max=args.metric_a_max,
        f_ref=args.metric_f_ref,
        f_max=args.metric_f_max,
        tau_ref=args.metric_tau_ref,
        lambda_l=args.metric_lambda_l,
        lambda_s=args.metric_lambda_s,
        lambda_v=args.metric_lambda_v,
        lambda_a=args.metric_lambda_a,
        lambda_j=args.metric_lambda_j,
        lambda_p=args.metric_lambda_p,
        lambda_q=args.metric_lambda_q,
        lambda_f=args.metric_lambda_f,
        lambda_tau=args.metric_lambda_tau,
    )


class Recorder:
    def __init__(
        self,
        output_dir: Path,
        ctrl_freq: int,
        wrist_shape: tuple[int, int],
        left_shape: tuple[int, int] | None,
        right_shape: tuple[int, int] | None,
        *,
        save_states_json: bool,
    ) -> None:
        self.output_dir = output_dir
        self.save_states_json = save_states_json
        self.states_data: list[dict[str, Any]] = []
        self._closed = False
        self.metric_monitor: TrajectoryMetricMonitor | None = None
        self.metric_config: TrajectoryCostConfig | None = None

        output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.wrist_shape = wrist_shape
        self.left_shape = left_shape
        self.right_shape = right_shape
        self.wrist_video = cv2.VideoWriter(str(output_dir / "view1_wrist.mp4"), fourcc, ctrl_freq, wrist_shape)
        if not self.wrist_video.isOpened():
            LOGGER.warning("Failed to open wrist video writer; only state logs will be saved.")
            self.wrist_video = None
        self.tactile_left_video = None
        self.tactile_right_video = None
        if left_shape is not None:
            self.tactile_left_video = cv2.VideoWriter(
                str(output_dir / "view2_tactile_left.mp4"), fourcc, ctrl_freq, left_shape
            )
            if not self.tactile_left_video.isOpened():
                LOGGER.warning("Failed to open left tactile video writer.")
                self.tactile_left_video = None
        if right_shape is not None:
            self.tactile_right_video = cv2.VideoWriter(
                str(output_dir / "view3_tactile_right.mp4"), fourcc, ctrl_freq, right_shape
            )
            if not self.tactile_right_video.isOpened():
                LOGGER.warning("Failed to open right tactile video writer.")
                self.tactile_right_video = None

    def attach_metric_monitor(
        self,
        metric_monitor: TrajectoryMetricMonitor,
        metric_config: TrajectoryCostConfig,
    ) -> None:
        self.metric_monitor = metric_monitor
        self.metric_config = metric_config

    def record_frame(self, latest_frame: dict[str, Any]) -> None:
        if self.wrist_video is not None:
            wrist_frame = _prepare_video_frame(latest_frame["wrist_img"], self.wrist_shape)
            if wrist_frame is not None:
                self.wrist_video.write(wrist_frame)
        if self.tactile_left_video is not None and latest_frame["left_tactile_img"] is not None:
            left_frame = _prepare_video_frame(latest_frame["left_tactile_img"], self.left_shape)
            if left_frame is not None:
                self.tactile_left_video.write(left_frame)
        if self.tactile_right_video is not None and latest_frame["right_tactile_img"] is not None:
            right_frame = _prepare_video_frame(latest_frame["right_tactile_img"], self.right_shape)
            if right_frame is not None:
                self.tactile_right_video.write(right_frame)

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
        if "force" in latest_frame:
            force = _prepare_force(latest_frame.get("force"))
            state_entry["force_left"] = float(force[0])
            state_entry["force_right"] = float(force[1])
        if action_chunk.shape[-1] >= 12:
            state_entry["pred_force_left"] = float(action_chunk[0, 10])
            state_entry["pred_force_right"] = float(action_chunk[0, 11])
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

        if self.wrist_video is not None:
            self.wrist_video.release()
        if self.tactile_left_video is not None:
            self.tactile_left_video.release()
        if self.tactile_right_video is not None:
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

        if self.metric_monitor is not None and self.metric_config is not None:
            metric_samples = self.metric_monitor.snapshot()
            metric_summary = compute_trajectory_cost(metric_samples, self.metric_config)
            with open(self.output_dir / "trajectory_cost.json", "w", encoding="utf-8") as f:
                json.dump(metric_summary, f, indent=2)
            if self.save_states_json:
                with open(self.output_dir / "trajectory_metric_samples.json", "w", encoding="utf-8") as f:
                    json.dump(metric_samples, f)
            if metric_samples:
                try:
                    import polars as pl

                    pl.DataFrame(metric_samples).write_parquet(str(self.output_dir / "trajectory_metric_samples.parquet"))
                except ModuleNotFoundError:
                    try:
                        import pandas as pd

                        pd.DataFrame(metric_samples).to_parquet(
                            str(self.output_dir / "trajectory_metric_samples.parquet")
                        )
                    except Exception as exc:  # pragma: no cover - environment-specific parquet backend failure
                        LOGGER.warning("Failed to write trajectory_metric_samples.parquet with pandas fallback: %s", exc)
                except Exception as exc:  # pragma: no cover - environment-specific parquet writer failure
                    LOGGER.warning("Failed to write trajectory_metric_samples.parquet with polars: %s", exc)
            LOGGER.info(
                "Trajectory cost saved to %s: J=%s, J_e=%s, J_m=%s, J_c=%s",
                self.output_dir / "trajectory_cost.json",
                metric_summary.get("J"),
                metric_summary.get("J_e"),
                metric_summary.get("J_m"),
                metric_summary.get("J_c"),
            )

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
                left_tactile_img, right_tactile_img = None, None
                if self.cam_tactile_left is not None:
                    left_tactile_img, _ = self.cam_tactile_left.get_data()
                if self.cam_tactile_right is not None:
                    right_tactile_img, _ = self.cam_tactile_right.get_data()
                eepose = self.env.get_ee_pose()
                gripper_width = self.env.get_gripper_width()
                force = self.env.get_force()

                if wrist_img is not None:
                    if wrist_img.shape[0] != 768 or wrist_img.shape[1] != 768:
                        wrist_img = cv2.resize(wrist_img, (768, 768), interpolation=cv2.INTER_AREA)
                    wrist_img = resize_with_black_padding(wrist_img, target_h=480, target_w=640)

                with self.lock:
                    self.queue.append(
                        {
                            "wrist_img": wrist_img.copy() if wrist_img is not None else None,
                            "left_tactile_img": left_tactile_img.copy() if left_tactile_img is not None else None,
                            "right_tactile_img": right_tactile_img.copy() if right_tactile_img is not None else None,
                            "cam_cap_time": cam_cap_time,
                            "eepose": np.asarray(eepose, dtype=np.float64),
                            "gripper_width": float(gripper_width),
                            "force": np.asarray(force, dtype=np.float32),
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
    def __init__(
        self,
        init_qpos: list[float] | tuple[float, ...],
        *,
        robot_ip: str,
        local_ip: str,
        dry_run: bool,
    ) -> None:
        import flexivrdk

        self._flexivrdk = flexivrdk
        self._mode = flexivrdk.Mode
        self._dry_run = dry_run
        self._init_qpos = list(init_qpos) if init_qpos is not None else None

        robot_sn = os.environ.get("FLEXIV_ROBOT_SN", "Rizon4-062339")
        gripper_name = os.environ.get("FLEXIV_GRIPPER_NAME", "Flexiv-GN01")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((robot_ip, 1))
            actual_local_ip = sock.getsockname()[0]
            sock.close()
        except Exception:
            actual_local_ip = local_ip

        LOGGER.info(
            "Initializing Flexiv Robot SN=%s, robot_ip=%s, local_ip=%s",
            robot_sn,
            robot_ip,
            actual_local_ip,
        )
        self._robot = flexivrdk.Robot(robot_sn, [actual_local_ip])

        LOGGER.info("Initializing Gripper API.")
        self._gripper = flexivrdk.Gripper(self._robot)
        LOGGER.info("Initializing Model API.")
        self._model = flexivrdk.Model(self._robot)
        self._target_lock = threading.Lock()
        self._latest_target_tcp: list[float] | None = None
        self._latest_target_time: float | None = None
        try:
            self._dq_max = _float_list(getattr(self._robot.info(), "dq_max", None))
        except Exception:
            self._dq_max = []

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

        LOGGER.info("Switching to NRT_CARTESIAN_MOTION_FORCE mode.")
        self._robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
        self._robot.SetForceControlAxis([False, False, False, False, False, False])

        if not self._dry_run:
            max_width = self._gripper.params().max_width
            self._gripper.Move(max_width, 0.1, 20)
            time.sleep(1)

    def reset(self) -> None:
        if self._init_qpos is None:
            return

        LOGGER.info("Switching to NRT_JOINT_POSITION for reset.")
        self._robot.SwitchMode(self._mode.NRT_JOINT_POSITION)
        dof = len(self._init_qpos)
        if not self._dry_run:
            LOGGER.info("Resetting robot to initial joint positions: %s", self._init_qpos)
            self._robot.SendJointPosition(self._init_qpos, [0.0] * dof, [0.1] * dof, [0.1] * dof)
            max_width = self._gripper.params().max_width
            self._gripper.Move(max_width, 0.1, 20)
            time.sleep(10)

        LOGGER.info("Switching back to NRT_CARTESIAN_MOTION_FORCE mode.")
        self._robot.SwitchMode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
        self._robot.SetForceControlAxis([False, False, False, False, False, False])

    def get_ee_pose(self) -> np.ndarray:
        flange_pose_raw = self._robot.states().tcp_pose.copy()
        pos = flange_pose_raw[:3]
        qw, qx, qy, qz = flange_pose_raw[3:]
        rot = st.Rotation.from_quat([qx, qy, qz, qw])
        return mat_to_certain_pose_type(pos_rot_to_mat(np.array(pos, dtype=np.float64), rot), "rotvec")

    def get_gripper_width(self) -> float:
        return float(self._gripper.states().width)

    def get_force(self) -> np.ndarray:
        try:
            states = self._gripper.states()
            if isinstance(states, dict):
                left = states.get("left_force", states.get("force", 0.0))
                right = states.get("right_force", states.get("force", left))
            else:
                left = getattr(states, "left_force", getattr(states, "force", 0.0))
                right = getattr(states, "right_force", getattr(states, "force", left))
            return np.asarray([left, right], dtype=np.float32)
        except Exception:
            return np.zeros((2,), dtype=np.float32)

    def _set_latest_target(self, target_tcp: list[float], target_time: float) -> None:
        with self._target_lock:
            self._latest_target_tcp = list(target_tcp)
            self._latest_target_time = float(target_time)

    @staticmethod
    def _timestamp_to_seconds(timestamp: Any) -> float | None:
        values = _float_list(timestamp)
        if len(values) >= 2:
            return values[0] + values[1] * 1e-9
        if len(values) == 1:
            return values[0]
        return None

    def get_metric_snapshot(self) -> dict[str, Any]:
        states = self._robot.states()
        with self._target_lock:
            target_tcp = list(self._latest_target_tcp) if self._latest_target_tcp is not None else None
            target_time = self._latest_target_time

        return {
            "wall_time": time.time(),
            "robot_time": self._timestamp_to_seconds(getattr(states, "timestamp", None)),
            "tcp_pose": _float_list(getattr(states, "tcp_pose", None)),
            "tcp_vel": _float_list(getattr(states, "tcp_vel", None)),
            "flange_pose": _float_list(getattr(states, "flange_pose", None)),
            "flange_vel": _float_list(getattr(states, "flange_vel", None)),
            "q": _float_list(getattr(states, "q", None)),
            "theta": _float_list(getattr(states, "theta", None)),
            "dq": _float_list(getattr(states, "dq", None)),
            "dtheta": _float_list(getattr(states, "dtheta", None)),
            "tau": _float_list(getattr(states, "tau", None)),
            "tau_ext": _float_list(getattr(states, "tau_ext", None)),
            "ext_wrench_in_world": _float_list(getattr(states, "ext_wrench_in_world", None)),
            "ext_wrench_in_tcp": _float_list(getattr(states, "ext_wrench_in_tcp", None)),
            "ft_sensor_raw": _float_list(getattr(states, "ft_sensor_raw", None)),
            "dq_max": self._dq_max,
            "target_tcp": target_tcp,
            "target_time": target_time,
        }

    def exec_actions(self, actions: np.ndarray, timestamps: np.ndarray, target_force: float = 20.0) -> None:
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        for i in range(len(new_actions)):
            target_pose = new_actions[i, :6]
            target_width = float(new_actions[i, 6])

            pos, rot = pose_to_pos_rot(target_pose)
            quat_xyzw = rot.as_quat(scalar_first=False)
            target_tcp = [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(quat_xyzw[3]),
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ]
            self._set_latest_target(target_tcp, float(new_timestamps[i]))

            if self._dry_run:
                LOGGER.info("Dry run: skipping joint/gripper command for action %d/%d", i + 1, len(new_actions))
            else:
                self._robot.SendCartesianMotionForce(target_tcp, [0.0] * 6, [0.0] * 6, 0.15, 0.5, 0.3, 1.0)
                max_width = self._gripper.params().max_width
                safe_width = min(max(target_width, 0.001), max_width - 0.001)
                self._gripper.Move(safe_width, 0.1, target_force)

            dt = new_timestamps[i] - time.time()
            if dt > 0:
                time.sleep(dt)


def _make_output_dir(args: Args) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return _repo_root() / args.record_root / args.task_name / timestamp


def _init_cameras(camera_config_path: Path, use_tactile: bool) -> tuple[Any, Any | None, Any | None]:
    BaseCamera = _load_base_camera()
    config_path = camera_config_path

    if not use_tactile:
        no_tactile_config_path = camera_config_path.with_name("handcap_camera_no_tactile.json")
        if no_tactile_config_path.exists():
            config_path = no_tactile_config_path
            LOGGER.info("use_tactile=False: using wrist-only camera config %s", config_path)
            cam_dict = BaseCamera.create_cameras_from_config(config_path=str(config_path))
        else:
            LOGGER.info("use_tactile=False: using wrist camera only from %s", config_path)
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            wrist_config = config.get("wrist")
            if wrist_config is None:
                raise KeyError(f"Camera config must contain a 'wrist' camera: {config_path}")
            if wrist_config.get("type") != "mvs_cam":
                raise ValueError("use_tactile=False expects the wrist camera to be type 'mvs_cam'")
            _bootstrap_lerobot_src()
            from perception.cameras.mvs_cam import MVSCamera

            cam_dict = {
                "wrist": MVSCamera(
                    {
                        "serial": wrist_config.get("serial"),
                        "exposure": wrist_config.get("exposure"),
                    }
                )
            }
        if "wrist" not in cam_dict:
            raise KeyError(f"Camera config must contain a 'wrist' camera: {config_path}")
        return cam_dict["wrist"], None, None

    cam_dict = BaseCamera.create_cameras_from_config(config_path=str(config_path))
    if "wrist" not in cam_dict:
        raise KeyError(f"Camera config must contain a 'wrist' camera: {config_path}")
    return cam_dict["wrist"], cam_dict["left_tactile"], cam_dict["right_tactile"]


def _init_recorder(
    output_dir: Path,
    ctrl_freq: int,
    cam_wrist,
    cam_tactile_left,
    cam_tactile_right,
    *,
    use_tactile: bool,
    save_states_json: bool,
) -> Recorder:
    _, init_wrist_img, _ = cam_wrist.read()

    if init_wrist_img is None:
        raise RuntimeError("Failed to fetch initial wrist frame for recorder setup")

    left_shape = None
    right_shape = None
    if use_tactile:
        if cam_tactile_left is None or cam_tactile_right is None:
            raise RuntimeError("use_tactile=True requires left and right tactile cameras")
        left_tactile_img, _ = cam_tactile_left.get_data()
        right_tactile_img, _ = cam_tactile_right.get_data()
        if left_tactile_img is None or right_tactile_img is None:
            raise RuntimeError("Failed to fetch initial tactile frames for recorder setup")
        left_shape = (left_tactile_img.shape[1], left_tactile_img.shape[0])
        right_shape = (right_tactile_img.shape[1], right_tactile_img.shape[0])

    return Recorder(
        output_dir=output_dir,
        ctrl_freq=ctrl_freq,
        wrist_shape=WRIST_RECORD_SHAPE,
        left_shape=left_shape,
        right_shape=right_shape,
        save_states_json=save_states_json,
    )


def main(args: Args) -> None:
    from openpi_client import websocket_client_policy

    LOGGER.info("Connecting to remote policy server at ws://%s:%d", args.host, args.port)
    prompt = args.prompt or args.task_name
    robot_dt = 1.0 / args.ctrl_freq
    LOGGER.info(
        "Control config: ctrl_freq=%.3fHz, robot_dt=%.4fs, steps_per_inference=%d, action_latency=%.4fs, "
        "force_predict=%s, force_guide=%s, infer_force_control=%s",
        args.ctrl_freq,
        robot_dt,
        args.steps_per_inference,
        args.action_latency,
        args.force_predict,
        args.force_guide,
        args.infer_force_control,
    )
    metric_config = _make_trajectory_cost_config(args)
    LOGGER.info(
        "Trajectory cost config: monitor_hz=%.1f, T_ref=%.3f, weights=(%.2f, %.2f, %.2f)",
        metric_config.monitor_hz,
        metric_config.t_ref,
        metric_config.w_e,
        metric_config.w_m,
        metric_config.w_c,
    )

    camera_config_path = args.camera_config_path
    if not camera_config_path.is_absolute():
        camera_config_path = (_repo_root() / camera_config_path).resolve()
    if not camera_config_path.exists():
        camera_config_path = _default_camera_config_path()

    output_dir = _make_output_dir(args)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    LOGGER.info("Server metadata: %s", policy.get_server_metadata())

    LOGGER.info("Initializing cameras from %s (use_tactile=%s)", camera_config_path, args.use_tactile)
    cam_wrist, cam_tactile_left, cam_tactile_right = _init_cameras(camera_config_path, args.use_tactile)

    LOGGER.info("Initializing recorder in %s", output_dir)
    recorder = _init_recorder(
        output_dir,
        args.ctrl_freq,
        cam_wrist,
        cam_tactile_left,
        cam_tactile_right,
        use_tactile=args.use_tactile,
        save_states_json=args.save_states_json,
    )

    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder
    signal_module.signal(signal_module.SIGINT, _signal_handler)
    signal_module.signal(signal_module.SIGTERM, _signal_handler)

    LOGGER.info("Initializing Flexiv environment at %s", args.robot_ip)
    env = FlexivRealEnv(args.init_qpos, robot_ip=args.robot_ip, local_ip=args.local_ip, dry_run=args.dry_run)
    LOGGER.info("Moving to init pose: %s", args.init_qpos)
    env.reset()

    obs_thread = ObservationThread(cam_wrist, cam_tactile_left, cam_tactile_right, env, maxlen=args.obs_horizon)
    obs_thread.start()

    LOGGER.info("Waiting for observation queue to fill (obs_horizon=%d)", args.obs_horizon)
    while obs_thread.get_obs(args.obs_horizon) is None:
        time.sleep(0.01)
    LOGGER.info("Observation queue filled. Starting inference loop.")

    metric_monitor = TrajectoryMetricMonitor(env, metric_config)
    recorder.attach_metric_monitor(metric_monitor, metric_config)
    metric_started = False

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
            policy_chunk_size = len(decoded_actions)
            
            # Convert the decoded actions (which are SE(3) relative pose increments) back to absolute physical target poses
            physical_actions = []
            current_eepose = latest_frame["eepose"]
            for action in decoded_actions:
                abs_phys_pose = get_real_umi_inference_action(action[:6], current_eepose)
                physical_actions.append(np.concatenate([abs_phys_pose, action[6:7]], axis=-1))
            physical_actions = np.array(physical_actions)
            if args.steps_per_inference > 0:
                physical_actions = physical_actions[: args.steps_per_inference]
            
            action_timestamps = (
                (1 + np.arange(len(physical_actions), dtype=np.float64)) * robot_dt + time.time() - args.action_latency
            )

            force_pred = action_chunk[0, 10:12] if action_chunk.shape[-1] >= 12 else np.zeros((2,), dtype=np.float64)
            if args.infer_force_control and (args.force_predict or args.force_guide):
                target_force = max(1.0, float(np.mean(force_pred)))
            else:
                target_force = 20.0

            if not metric_started:
                metric_monitor.start()
                metric_started = True

            env.exec_actions(physical_actions, action_timestamps, target_force=target_force)

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

            first_target = physical_actions[0]
            force_obs = _prepare_force(latest_frame.get("force"))
            exec_horizon = len(physical_actions) * robot_dt
            LOGGER.info(
                "[step=%d] infer=%.1fms policy_chunk=%d exec_steps=%d eef_xyz=(%.3f, %.3f, %.3f) "
                "exec_horizon=%.2fs gripper=%.4f force=(%.2f, %.2f) target_xyz=(%.3f, %.3f, %.3f) "
                "target_gripper=%.4f pred_force=(%.2f, %.2f)",
                step,
                inference_latency * 1000.0,
                policy_chunk_size,
                len(physical_actions),
                float(current_eepose[0]),
                float(current_eepose[1]),
                float(current_eepose[2]),
                exec_horizon,
                float(latest_frame["gripper_width"]),
                float(force_obs[0]),
                float(force_obs[1]),
                float(first_target[0]),
                float(first_target[1]),
                float(first_target[2]),
                float(first_target[6]),
                float(force_pred[0]),
                float(force_pred[1]),
            )
            
            # Print metrics periodically during inference
            if step % 2 == 0:
                metric_samples_snapshot = metric_monitor.snapshot()
                metric_summary = compute_trajectory_cost(metric_samples_snapshot, metric_config)
                if metric_summary.get("valid"):
                    LOGGER.info(
                        "[step=%d] Current Trajectory Cost: J=%.3f (J_e=%.3f, J_m=%.3f, J_c=%.3f)",
                        step,
                        metric_summary.get("J", 0.0),
                        metric_summary.get("J_e", 0.0),
                        metric_summary.get("J_m", 0.0),
                        metric_summary.get("J_c", 0.0),
                    )
    finally:
        metric_monitor.stop()
        if metric_started:
            metric_monitor.join(timeout=2.0)
        if metric_monitor.error is not None:
            LOGGER.warning("Trajectory metric monitor stopped after sampling error: %s", metric_monitor.error)
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
