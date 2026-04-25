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
    logger.info(f"Episode {args.episode_index} has {num_frames} frames. Starting relative delta-control replay...")

    for frame_idx in range(num_frames - 1):
        s = time.time()
        
        # 1. 获取当前帧和下一帧的未处理绝对状态 (10D: [xyz, rotvec, gripper, 0,0,0])
        refer_item = dataset.get_raw_item(frame_idx)
        next_item = dataset.get_raw_item(frame_idx + 1)
        
        obs_state_tensor = refer_item["observation.state"].clone().detach().to(torch.float32)
        # 我们把下一帧当作目标 action
        action_tensor = next_item["observation.state"].clone().detach().to(torch.float32)
        
        # --- DEBUG 打印数据集原始夹爪值 ---
        dataset_gripper = action_tensor[6].item()
        # ---------------------------------
        
        # 2. 调用数据集内的标准方法，将绝对 10D 姿态转成 Relative 10d 姿态（基于 rot6d）
        # 返回的 raw_action 包含了相对的 9 维轨迹差量和绝对的 1 维夹爪目标
        _, raw_action = process_to_relative_rot6d(obs_state_tensor, action_tensor)
        
        # 补充 sequence 维度，适配后续推理
        if raw_action.ndim == 1:
            raw_action = raw_action.unsqueeze(0)

        # 3. 获取机器人的真实绝对位姿
        abs_eepose = env.get_ee_pose()
        
        # Form in_abs_pose 传入当前机器人的夹爪宽度只是为了凑足 UMI 推理的 7D 格式
        # get_real_umi_inference_action 内部实际返回的夹爪依然会使用 raw_action 里的绝对夹爪目标
        current_gripper = np.array([[env.get_gripper_width()]])
        in_abs_pose = np.concatenate([[abs_eepose], current_gripper], axis=-1)

        # 4. 根据当前的物理真实位姿和 Relative 动作，解算出最终发给机械臂的绝对动作
        this_target_poses = get_real_umi_inference_action(raw_action.numpy(), in_abs_pose, "relative")
        
        print(f"Frame {frame_idx} | Dataset Raw Gripper: {dataset_gripper:.4f} | Target Sent to Robot: {this_target_poses[0][6]:.4f}")
        
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
