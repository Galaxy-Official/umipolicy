import os
import sys
import time
import argparse
import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import signal as signal_module
import torch

from lerobot.scripts.umi_realworld.utils.pose_util import *
from lerobot.scripts.umi_realworld.real_inference_util import *
from lerobot.scripts.umi_realworld.env import FlexivEnv


states_data = None
output_dir = None


def signal_handler(sig, frame):
    global states_data, output_dir
    logger.info("\nDetected interrupt signal, saving data...")
    try:
        if states_data:
            states_array = np.array(states_data)
            save_path = str(output_dir / 'states.npy')
            np.save(save_path, states_array)
            logger.info(f"Success save frames, totally {len(states_data)} frames")

        if output_dir:
            logger.info(f"All data has been saved to: {output_dir}")

        os.sync()
    except Exception as e:
        logger.error(f"Error occurred while saving data: {str(e)}")
    finally:
        sys.exit(0)


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def _read_parquet_dir(parquet_dir: Path) -> pd.DataFrame:
    paths = sorted(parquet_dir.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {parquet_dir}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _load_replay_states(data_root: str | Path, episode_index: int) -> np.ndarray:
    """Load only proprioceptive states for replay; no video decoder is needed."""
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {info_path}. "
            f"Current working directory is {Path.cwd()}; check --data_root."
        )

    with info_path.open("r") as f:
        info = json.load(f)

    logger.info(f"Reading episode metadata from: {root / 'meta' / 'episodes'}")
    episodes = _read_parquet_dir(root / "meta" / "episodes")
    if "episode_index" in episodes.columns:
        selected = episodes[episodes["episode_index"].map(_as_int) == episode_index]
        if selected.empty:
            raise IndexError(f"Episode {episode_index} not found in {root / 'meta' / 'episodes'}")
        episode = selected.iloc[0]
    else:
        if episode_index >= len(episodes):
            raise IndexError(f"Episode index {episode_index} out of range; dataset has {len(episodes)} episodes.")
        episode = episodes.iloc[episode_index]

    data_path_template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    data_path = root / data_path_template.format(
        chunk_index=_as_int(episode["data/chunk_index"]),
        file_index=_as_int(episode["data/file_index"]),
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Episode data parquet not found: {data_path}")

    logger.info(f"Reading replay data parquet: {data_path}")
    data = pd.read_parquet(data_path)
    if "episode_index" in data.columns:
        data = data[data["episode_index"].map(_as_int) == episode_index]
    if data.empty:
        raise ValueError(f"No frames for episode {episode_index} in {data_path}")
    if "index" in data.columns:
        data = data.sort_values("index")
    elif "frame_index" in data.columns:
        data = data.sort_values("frame_index")

    state_key = next(
        (key for key in ("observation.state", "observation/state", "observation_state") if key in data.columns),
        None,
    )
    if state_key is None:
        raise KeyError(
            "Replay requires an observation state column, but none of "
            "observation.state / observation/state / observation_state exists. "
            f"Available columns: {list(data.columns)}"
        )

    states = np.stack([np.asarray(value, dtype=np.float32) for value in data[state_key].to_list()], axis=0)
    if states.ndim != 2 or states.shape[1] < 7:
        raise ValueError(f"Unexpected replay state shape {states.shape}; expected [T, >=7].")
    return states


def main(args: argparse.Namespace):
    global states_data, output_dir

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"recordings/{args.task_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    states_data = []

    signal_module.signal(signal_module.SIGINT, signal_handler)  # Ctrl+C
    signal_module.signal(signal_module.SIGTERM, signal_handler)

    logger.info(args)

    logger.info(f"Loading replay states from: {args.data_root}")
    episode_states = _load_replay_states(args.data_root, args.episode_index)
    num_frames = len(episode_states)
    logger.info(f"Episode {args.episode_index} has {num_frames} frames. Starting Initial-Offset Absolute Replay...")

    # get init robot pose
    init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[0, -40, 0, 90, 0, 40]"))
    env = FlexivEnv(init_qpos, obs_horizon=args.obs_horizon, use_gripper_width_mapping=False, pose_type="rotvec")
    env.reset()

    robot_dt = 1. / args.ctrl_freq
    action_latency = 0

    # --- 核心修复：计算基准偏移 ---
    # 获取数据集的第 0 帧作为基准
    dataset_init_item = episode_states[0]
    dataset_init_mat = certain_pose_type_to_mat(dataset_init_item[:6], pose_type="rotvec")
    
    # 获取机器人当前真实的物理起始位姿作为基准
    robot_init_pose = env.get_ee_pose()
    robot_init_mat = certain_pose_type_to_mat(robot_init_pose, pose_type="rotvec")

    for frame_idx in range(num_frames - 1):
        s = time.time()
        
        # 1. 获取下一帧的目标状态
        target_dataset_pose10d = episode_states[frame_idx + 1]
        
        # 2. 将目标状态转换为 4x4 矩阵
        target_dataset_mat = certain_pose_type_to_mat(target_dataset_pose10d[:6], pose_type="rotvec")
        
        # 3. 计算从【数据集第 0 帧】到【当前目标帧】的完美纯净相对变换矩阵
        # 这样可以彻底避免因为物理机械臂跟不上而导致每次 Delta 累加被“吃掉”从而缩小轨迹的问题
        T_rel = np.linalg.inv(dataset_init_mat) @ target_dataset_mat
        
        # 4. 把这个完美的相对变换，叠加到【机械臂的初始位姿】上，得到机械臂应到达的绝对坐标
        T_robot_target = robot_init_mat @ T_rel
        
        # 5. 组装发给 env.exec_actions 的格式 (7D: 3Pos + 3Rotvec + 1Gripper)
        target_robot_pose6d = mat_to_certain_pose_type(T_robot_target, pose_type="rotvec")
        target_gripper = target_dataset_pose10d[6:7]
        this_target_poses = np.concatenate([target_robot_pose6d, target_gripper], axis=-1)[np.newaxis, :]
        
        # Execute absolute action on robot
        action_timestamps = np.array([time.time() + robot_dt - action_latency])
        
        env.exec_actions(
            actions=this_target_poses,
            timestamps=action_timestamps
        )
        
        # Track latency and wait if necessary to maintain roughly dataset FPS
        elapsed = time.time() - s
        if elapsed < robot_dt:
            time.sleep(robot_dt - elapsed)
            
        # Record actual state for saving
        states_data.append(env.get_ee_pose())

    # Save at end
    if states_data:
        states_array = np.array(states_data)
        np.save(str(output_dir / 'states.npy'), states_array)
        
    logger.info(f"Replay finished. Data saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the LeRobot dataset",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        required=True,
        help="The index of the episode to replay",
    )
    parser.add_argument(
        "--obs_horizon",
        type=int,
        default=2
    )
    parser.add_argument(
        "--ctrl_freq",
        action="store",
        type=int,
        help="The control frequency of the robot",
        default=20,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the task",
        default="replay",
    )
    
    args = parser.parse_args()
    main(args)
