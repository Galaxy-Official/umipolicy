import os
import sys
import time
import argparse
import datetime
import numpy as np
from pathlib import Path
from loguru import logger
import signal as signal_module
import scipy.spatial.transform as st
import torch

from lerobot.scripts.umi_realworld.utils.pose_util import *
from lerobot.scripts.umi_realworld.real_inference_util import *
from lerobot.scripts.umi_realworld.env import FlexivEnv
from lerobot.datasets.lerobot_dataset_handcap import LeRobotDatasetHandcap, process_to_relative_rot6d


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


def _to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _signed_twist_deg(rot: st.Rotation, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    quat_xyzw = rot.as_quat()
    vec = quat_xyzw[:3]
    proj = np.dot(vec, axis)
    w = quat_xyzw[3]
    norm = max(np.hypot(proj, w), 1e-12)
    return float(np.degrees(2.0 * np.arctan2(proj / norm, w / norm)))


def log_replay_rotation_summary(dataset, num_frames: int, dataset_init_mat: np.ndarray) -> None:
    total_angles = []
    twist_xyz = []
    inv_dataset_init = np.linalg.inv(dataset_init_mat)

    for idx in range(num_frames):
        pose = _to_numpy(dataset.get_raw_item(idx)["observation.state"])[:6]
        target_mat = certain_pose_type_to_mat(pose, pose_type="rotvec")
        rel_mat = inv_dataset_init @ target_mat
        rel_rot = st.Rotation.from_matrix(rel_mat[:3, :3])
        total_angles.append(float(np.degrees(rel_rot.magnitude())))
        twist_xyz.append(
            [
                _signed_twist_deg(rel_rot, np.array([1.0, 0.0, 0.0])),
                _signed_twist_deg(rel_rot, np.array([0.0, 1.0, 0.0])),
                _signed_twist_deg(rel_rot, np.array([0.0, 0.0, 1.0])),
            ]
        )

    total_angles = np.asarray(total_angles)
    twist_xyz = np.unwrap(np.radians(np.asarray(twist_xyz)), axis=0)
    twist_xyz = np.degrees(twist_xyz)
    max_z_idx = int(np.argmax(np.abs(twist_xyz[:, 2])))
    max_total_idx = int(np.argmax(total_angles))
    logger.info(
        "Replay target rotation summary: final_total={:.2f}deg, "
        "final_twist_xyz=({:.2f}, {:.2f}, {:.2f})deg, "
        "max_abs_twist_z={:.2f}deg at frame {}, max_total={:.2f}deg at frame {}",
        total_angles[-1],
        twist_xyz[-1, 0],
        twist_xyz[-1, 1],
        twist_xyz[-1, 2],
        twist_xyz[max_z_idx, 2],
        max_z_idx,
        total_angles[max_total_idx],
        max_total_idx,
    )


def main(args: argparse.Namespace):
    global states_data, output_dir

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"recordings/{args.task_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    states_data = []

    signal_module.signal(signal_module.SIGINT, signal_handler)  # Ctrl+C
    signal_module.signal(signal_module.SIGTERM, signal_handler)

    logger.info(args)

    # get init robot pose
    init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[0, -40, 0, 90, 0, 40]"))
    direct_eef_control = os.environ.get("FLEXIV_DIRECT_EEF_CONTROL", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    env = FlexivEnv(
        init_qpos,
        obs_horizon=args.obs_horizon,
        use_gripper_width_mapping=False,
        pose_type="rotvec",
        direct_eef_control=direct_eef_control,
    )
    env.reset()

    robot_dt = 1. / args.ctrl_freq
    action_latency = 0

    logger.info(f"Loading dataset from: {args.data_root}")
    # Load LeRobot Handcap dataset
    dataset = LeRobotDatasetHandcap(
        repo_id="local_replay",
        root=args.data_root,
        episodes=[args.episode_index],
    )
    
    num_frames = len(dataset)
    logger.info(f"Episode {args.episode_index} has {num_frames} frames. Starting Initial-Offset Absolute Replay...")

    # --- 核心修复：计算基准偏移 ---
    # 获取数据集的第 0 帧作为基准
    dataset_init_item = _to_numpy(dataset.get_raw_item(0)["observation.state"])
    dataset_init_mat = certain_pose_type_to_mat(dataset_init_item[:6], pose_type="rotvec")
    log_replay_rotation_summary(dataset, num_frames, dataset_init_mat)
    
    # 获取机器人当前真实的物理起始位姿作为基准
    robot_init_pose = env.get_ee_pose()
    robot_init_mat = certain_pose_type_to_mat(robot_init_pose, pose_type="rotvec")

    for frame_idx in range(num_frames - 1):
        s = time.time()
        
        # 1. 获取下一帧的目标状态
        next_item = dataset.get_raw_item(frame_idx + 1)
        target_dataset_pose10d = _to_numpy(next_item["observation.state"]).astype(np.float32)
        
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
