# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run Cosmos Policy on a real Flexiv robot.

Example:
    PYTHONPATH=. python -m cosmos_policy.experiments.robot.flexiv.run_flexiv_eval \
        --config cosmos_predict2_flexiv__inference_only \
        --ckpt_path outputs/cosmos_predict2_flexiv/checkpoints/iter_000050000 \
        --data_dir data/yellow_cosmos_428 \
        --task_description "put yellow block onto pink block" \
        --camera_config ../umi/handcap_camera.json \
        --num_open_loop_steps 6 \
        --steps_per_inference 6 \
        --control_frequency 10
"""

import ast
import os
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.flexiv.flexiv_utils import (
    absolute_state_to_relative_pose10d,
    relative_pose10d_actions_to_absolute,
)
from cosmos_policy.utils.utils import set_seed_everywhere


DEFAULT_INIT_QPOS = "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"
DEFAULT_MVS_CAPTURE_RESOLUTION = (768, 768)
DEFAULT_MVS_FPS = 20.0
DEFAULT_MVS_VIDEO_FOURCC = "mp4v"


@dataclass
class FlexivEvalConfig:
    # Model and data
    suite: str = "flexiv"
    config: str = "cosmos_predict2_flexiv__inference_only"
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    data_dir: str = ""
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    task_description: str = "manipulate object"
    output_dir: str = "cosmos_policy/experiments/robot/flexiv/logs"

    # Cosmos action inference
    chunk_size: int = 50
    num_open_loop_steps: int = 6
    num_denoising_steps_action: int = 10
    trained_with_image_aug: bool = True
    use_jpeg_compression: bool = False
    use_variance_scale: bool = False
    flip_images: bool = False
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    use_wrist_image: bool = True
    num_wrist_images: int = 2
    use_third_person_image: bool = True
    num_third_person_images: int = 1

    # Robot and camera
    robot_ip: str = "192.168.2.100"
    local_ip: str = "192.168.2.102"
    init_qpos: str = DEFAULT_INIT_QPOS
    init_joints: bool = True
    camera_config: str = ""
    umi_root: str = ""
    lerobot_root: str = ""

    # Runtime
    obs_horizon: int = 2
    camera_fps: float = DEFAULT_MVS_FPS
    control_frequency: float = 10.0
    steps_per_inference: int = 6
    max_duration: float = 60.0
    start_delay: float = 1.0
    frame_latency: float = 1.0 / 60.0
    action_exec_latency: float = 0.01
    command_latency: float = 0.01
    seed: int = 195
    randomize_seed: bool = False
    deterministic: bool = True
    use_mp4_roundtrip: bool = True


class Mp4RoundTrip:
    def __init__(self, resolution=DEFAULT_MVS_CAPTURE_RESOLUTION, fps=DEFAULT_MVS_FPS):
        self.resolution = tuple(resolution)
        self.fps = float(fps)
        self.fourcc = cv2.VideoWriter_fourcc(*DEFAULT_MVS_VIDEO_FOURCC)

    def __call__(self, frames):
        if len(frames) == 0:
            return []

        tmp_file = tempfile.NamedTemporaryFile(prefix="eval_flexiv_cosmos_", suffix=".mp4", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        writer = None
        cap = None
        try:
            writer = cv2.VideoWriter(tmp_path, self.fourcc, self.fps, self.resolution)
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open temporary mp4 writer: {tmp_path}")
            for frame in frames:
                writer.write(np.ascontiguousarray(frame))
            writer.release()
            writer = None

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open temporary mp4 reader: {tmp_path}")
            decoded = []
            for frame_idx in range(len(frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to decode mp4 round-trip frame {frame_idx}")
                decoded.append(frame)
            return decoded
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class ObservationThread(threading.Thread):
    def __init__(self, cam_wrist, env, maxlen=2, camera_fps=DEFAULT_MVS_FPS, precise_wait=None):
        super().__init__(daemon=True)
        self.cam_wrist = cam_wrist
        self.env = env
        self.queue = deque(maxlen=maxlen)
        self.running = True
        self.lock = threading.Lock()
        self.camera_dt = 1.0 / float(camera_fps) if camera_fps > 0 else 0.0
        self.precise_wait = precise_wait

    def run(self):
        while self.running:
            iter_start_time = time.monotonic()
            _, wrist_img, cam_cap_time = self.cam_wrist.read()
            eepose = self.env.get_ee_pose()
            gripper_width = self.env.get_gripper_width()
            with self.lock:
                self.queue.append(
                    {
                        "wrist_img": wrist_img.copy() if wrist_img is not None else wrist_img,
                        "cam_cap_time": cam_cap_time,
                        "eepose": eepose,
                        "gripper_width": gripper_width,
                    }
                )
            if self.camera_dt > 0 and self.precise_wait is not None:
                self.precise_wait(iter_start_time + self.camera_dt, time_func=time.monotonic)

    def get_obs(self, n=1):
        with self.lock:
            if len(self.queue) < n:
                return None
            return list(self.queue)[-n:]

    def stop(self):
        self.running = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _configure_external_paths(cfg: FlexivEvalConfig) -> dict[str, Any]:
    repo_root = _repo_root()
    umi_root = Path(cfg.umi_root).expanduser() if cfg.umi_root else repo_root.parent / "umi"
    lerobot_root = Path(cfg.lerobot_root).expanduser() if cfg.lerobot_root else repo_root.parent / "lerobot" / "src"
    sys.path.insert(0, str(umi_root.resolve()))
    sys.path.insert(0, str(lerobot_root.resolve()))

    from perception.cameras.base_camera import BaseCamera
    from scripts_real.flexiv_env.env import FlexivEnv
    from umi.common.precise_sleep import precise_wait

    return {"BaseCamera": BaseCamera, "FlexivEnv": FlexivEnv, "precise_wait": precise_wait}


def _resolve_paths(cfg: FlexivEvalConfig) -> None:
    if cfg.data_dir:
        if not cfg.dataset_stats_path:
            cfg.dataset_stats_path = str(Path(cfg.data_dir) / "dataset_statistics_relative_pose10d_state_action.json")
        if not cfg.t5_text_embeddings_path:
            cfg.t5_text_embeddings_path = str(Path(cfg.data_dir) / "t5_embeddings.pkl")

    if not cfg.ckpt_path:
        raise ValueError("Must provide --ckpt_path")
    if not cfg.dataset_stats_path:
        raise ValueError("Must provide --dataset_stats_path or --data_dir")
    if not cfg.t5_text_embeddings_path:
        raise ValueError("Must provide --t5_text_embeddings_path or --data_dir")
    if not cfg.camera_config:
        raise ValueError("Must provide --camera_config")


def _create_wrist_camera(base_camera_cls, camera_config: str):
    cam_dict = base_camera_cls.create_cameras_from_config(config_path=camera_config)
    if "wrist" not in cam_dict:
        raise KeyError(f"Camera config must contain a 'wrist' camera. Got keys: {list(cam_dict.keys())}")
    return cam_dict["wrist"]


def _format_wrist_image(frame: dict[str, Any], mp4_round_trip: Mp4RoundTrip | None) -> np.ndarray:
    wrist_bgr = frame["wrist_img"]
    if wrist_bgr is None:
        raise RuntimeError("Wrist camera returned None image.")
    if (wrist_bgr.shape[1], wrist_bgr.shape[0]) != DEFAULT_MVS_CAPTURE_RESOLUTION:
        wrist_bgr = cv2.resize(wrist_bgr, DEFAULT_MVS_CAPTURE_RESOLUTION, interpolation=cv2.INTER_AREA)
    if mp4_round_trip is not None:
        wrist_bgr = mp4_round_trip([wrist_bgr])[0]
    return cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _build_observation(frame: dict[str, Any], mp4_round_trip: Mp4RoundTrip | None):
    wrist_rgb = _format_wrist_image(frame, mp4_round_trip)
    base_state = np.concatenate(
        [
            np.asarray(frame["eepose"][:6], dtype=np.float32),
            np.asarray([frame["gripper_width"]], dtype=np.float32),
        ],
        axis=0,
    )
    proprio = absolute_state_to_relative_pose10d(base_state, base_state)
    observation = {
        "wrist_image": wrist_rgb,
        "primary_image": wrist_rgb,
        "left_wrist_image": wrist_rgb,
        "right_wrist_image": wrist_rgb,
        "proprio": proprio,
    }
    return observation, base_state, float(frame["cam_cap_time"])


@draccus.wrap()
def main(cfg: FlexivEvalConfig) -> None:
    _resolve_paths(cfg)
    external = _configure_external_paths(cfg)
    precise_wait = external["precise_wait"]

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)
    if cfg.chunk_size != cosmos_config.dataloader_train.dataset.chunk_size:
        raise ValueError(
            f"Mismatch between eval chunk_size={cfg.chunk_size} and train "
            f"chunk_size={cosmos_config.dataloader_train.dataset.chunk_size}"
        )

    init_qpos = ast.literal_eval(cfg.init_qpos)
    env = external["FlexivEnv"](
        init_qpos,
        obs_horizon=cfg.obs_horizon,
        robot_ip=cfg.robot_ip,
        local_ip=cfg.local_ip,
        use_gripper_width_mapping=False,
        pose_type="rotvec",
    )
    if cfg.init_joints:
        env.reset()

    cam_wrist = _create_wrist_camera(external["BaseCamera"], cfg.camera_config)
    obs_thread = ObservationThread(
        cam_wrist,
        env,
        maxlen=max(2, cfg.obs_horizon),
        camera_fps=cfg.camera_fps,
        precise_wait=precise_wait,
    )
    obs_thread.start()
    while obs_thread.get_obs(1) is None:
        time.sleep(0.01)

    mp4_round_trip = Mp4RoundTrip(fps=cfg.camera_fps) if cfg.use_mp4_roundtrip else None
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Ready. Press Enter to start policy execution. Use Ctrl+C to stop.")
    input()

    dt = 1.0 / cfg.control_frequency
    eval_t_start = time.time() + cfg.start_delay
    t_start = time.monotonic() + cfg.start_delay
    precise_wait(eval_t_start - cfg.frame_latency, time_func=time.time)
    print("Policy started.")

    iter_idx = 0
    executed_actions = []
    relative_actions = []
    base_states = []
    timestamps = []

    try:
        while (time.monotonic() - t_start) < cfg.max_duration:
            t_cycle_end = t_start + (iter_idx + cfg.steps_per_inference) * dt
            frames = obs_thread.get_obs(1)
            if frames is None:
                raise RuntimeError("Observation queue is not ready.")
            observation, base_state, obs_timestamp = _build_observation(frames[-1], mp4_round_trip)

            query_start = time.time()
            action_return_dict = get_action(
                cfg,
                model,
                dataset_stats,
                observation,
                cfg.task_description,
                seed=cfg.seed,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=False,
            )
            raw_actions = np.asarray(action_return_dict["actions"], dtype=np.float32)
            raw_actions = raw_actions[: cfg.num_open_loop_steps]
            target_poses = relative_pose10d_actions_to_absolute(raw_actions, base_state)
            print(f"Inference latency: {time.time() - query_start:.3f}s")

            action_timestamps = np.arange(len(target_poses), dtype=np.float64) * dt + obs_timestamp
            curr_time = time.time()
            is_new = action_timestamps > (curr_time + cfg.action_exec_latency)
            if np.sum(is_new) == 0:
                target_poses = target_poses[[-1]]
                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                raw_actions = raw_actions[[-1]]
            else:
                target_poses = target_poses[is_new]
                action_timestamps = action_timestamps[is_new]
                raw_actions = raw_actions[is_new]

            steps_to_exec = min(cfg.steps_per_inference, len(target_poses))
            target_poses = target_poses[:steps_to_exec]
            action_timestamps = action_timestamps[:steps_to_exec]
            raw_actions = raw_actions[:steps_to_exec]

            env.exec_actions(actions=target_poses, timestamps=action_timestamps)
            executed_actions.append(target_poses.copy())
            relative_actions.append(raw_actions.copy())
            base_states.append(base_state.copy())
            timestamps.append(action_timestamps.copy())

            iter_idx += cfg.steps_per_inference
            precise_wait(t_cycle_end - cfg.command_latency, time_func=time.monotonic)
    except KeyboardInterrupt:
        print("Stopping Flexiv eval.")
    finally:
        obs_thread.stop()
        if executed_actions:
            log_path = Path(cfg.output_dir) / f"flexiv_eval_{int(time.time())}.npz"
            np.savez_compressed(
                log_path,
                executed_actions=np.array(executed_actions, dtype=object),
                relative_actions=np.array(relative_actions, dtype=object),
                base_states=np.array(base_states, dtype=np.float32),
                timestamps=np.array(timestamps, dtype=object),
            )
            print(f"Saved eval log: {log_path}")


if __name__ == "__main__":
    main()
