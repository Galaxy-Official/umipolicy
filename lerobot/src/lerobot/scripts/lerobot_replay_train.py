import os
import sys
import time
import argparse
import datetime
import numpy as np
from pathlib import Path
from loguru import logger
import signal as signal_module
import torch

from lerobot.scripts.umi_realworld.utils.pose_util import *
from lerobot.scripts.umi_realworld.real_inference_util import *
from lerobot.scripts.umi_realworld.env import FlexivEnv
from lerobot.datasets.lerobot_dataset_handcap import LeRobotDatasetHandcap


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
    env = FlexivEnv(init_qpos, obs_horizon=args.obs_horizon, use_gripper_width_mapping=False, pose_type="rotvec")
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
    logger.info(f"Episode {args.episode_index} has {num_frames} frames. Starting replay...")

    for frame_idx in range(num_frames - 1):
        s = time.time()
        
        # Get frame from dataset
        item = dataset[frame_idx]
        
        # Get processed action from dataset
        # In LeRobotDatasetHandcap, item["action"] is already converted to relative rot6d + gripper width
        raw_action = item["action"]
        
        # If raw_action is 1D, we add a sequence dimension for compatibility
        if raw_action.ndim == 1:
            raw_action = raw_action.unsqueeze(0)

        # Get absolute pose from current real robot state
        abs_pose = []
        abs_eepose = env.get_ee_pose()
        abs_pose.append(abs_eepose)

        gripper_width = env.get_gripper_width()
        
        # in_abs_pose expects shape [1, 7] (or [7] if 1 step)
        # It concatenates [abs_pose, gripper_width]
        in_abs_pose = np.concatenate([abs_pose, np.array([[gripper_width]])], axis=-1)

        # Convert back to absolute target poses
        this_target_poses = get_real_umi_inference_action(raw_action.cpu().numpy(), in_abs_pose, "relative")
        
        print(f"Step {frame_idx}: Computed target poses:")
        print(this_target_poses)

        # Execute actions on robot
        action_timestamps = (1 + np.arange(len(this_target_poses), dtype=np.float64)) * robot_dt + time.time() - action_latency
        
        env.exec_actions(
            actions=this_target_poses,
            timestamps=action_timestamps
        )
        print(f"Submitted {len(this_target_poses)} steps of actions.")
        print('Action latency:', time.time() - s)

        states_data.append(abs_eepose)

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
