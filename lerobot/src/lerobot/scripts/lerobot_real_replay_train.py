import os
import sys
import cv2
import torch
import time
import argparse
import datetime
import numpy as np
from pathlib import Path
from loguru import logger
import signal as signal_module
from torchvision import transforms
from lerobot.scripts.umi_realworld.utils.pose_util import *
from lerobot.scripts.umi_realworld.real_inference_util import *
from lerobot.scripts.umi_realworld.env import FlexivEnv
from lerobot.datasets.pose_utils import quat_to_rot

from scipy.spatial.transform import Rotation

tactile_transforms = {
    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'touch': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]),
    }

def compute_transform(raw_pose):
    
    pose_mat = pose_to_mat(quat_to_rot(raw_pose))
    
    # Apply base frame transformation: PyBullet URDF base to RDK Physical base

    # Apply end-effector local transformation
    additional_transform_matrix = np.eye(4)
    
    additional_R_y = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    additional_R_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    transform_mat = additional_R_y @ additional_R_z
    additional_transform_matrix[:3, :3] = transform_mat
    pose_mat_out = pose_mat @ additional_transform_matrix
    
    return mat_to_certain_pose_type(pose_mat_out, pose_type="rotvec")

def read_pose_from_raw_data(raw_root, eposide_index):
    eposide_pose_dir = os.path.join(raw_root, f"pose_{eposide_index}")
    pose_files = os.listdir(eposide_pose_dir)
    pose_files = [f for f in pose_files if f.endswith('.npz')]
    pose_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_pose = []

    for pose_f in pose_files:
        pose_path = os.path.join(eposide_pose_dir, pose_f)
        pose_data = np.load(pose_path)
        all_pose.extend(pose_data["pose"])
    return all_pose



inference_time_s = 60
fps = 5
device = "cuda"

pre_action = np.array(
    [0, -40, 0, 90, 0, 40]
)  # TODO: initialize with the initial action
action = None
robot = None
# LEAP
ROBOT_IP = "192.168.2.100"
LOCAL_IP = "192.168.2.102"

# 700

# ROBOT_IP = "192.168.2.100"
# LOCAL_IP = "192.168.2.103"


view1_video = None
tactile1_video = None
tactile2_video = None

states_data = None
output_dir = None


def signal_handler(sig, frame):
    global view1_video, states_data, output_dir
    logger.info("\nDetected interrupt signal, saving data...")
    try:
        if view1_video is not None:
            view1_video.release()
        
        if states_data:
            states_array = np.array(states_data)
            save_path = str(output_dir / 'states.npy')
            np.save(save_path, states_array)
            logger.info(f"Success save frames, totally {len(states_data)} frames")

        logger.info(f"All data has been saved to: {output_dir}")

        os.sync()
        
    except Exception as e:
        logger.error(f"Error occurred while saving data: {str(e)}")
    finally:
        sys.exit(0)


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)



def main(args: argparse.Namespace):
    
    raw_data_path = "/home/rhos/umipolicy/lerobot/src/Data/replay/112501_2026032614"
    episode_index = 3

    raw_pose = read_pose_from_raw_data(raw_data_path, episode_index)

    robot_dt = 1./ 20
    action_latency = 0
    global view1_video, view2_video, view3_video, view4_video, view5_video, states_data, output_dir

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"recordings/{args.task_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    states_data = []
    
    signal_module.signal(signal_module.SIGINT, signal_handler)  # Ctrl+C
    signal_module.signal(signal_module.SIGTERM, signal_handler)  # 终止信号

    
    # get init robot pose
    init_qpos=eval(os.environ.get("FLEXIV_INIT_POSE", "Set the pose in environ"))
    env = FlexivEnv(init_qpos, obs_horizon=args.obs_horizon, use_gripper_width_mapping=False,pose_type="rotvec")

    logger.info(args)
    
    
    env.reset()
    

    for frame_idx in range(len(raw_pose) -1):
            # get abs pose
            abs_pose = []
            abs_eepose = env.get_ee_pose()
            abs_pose.append(abs_eepose)
            
            # get reference start pose
            refer_eepose = compute_transform(raw_pose[frame_idx])
            refer_obs_data = {
                'robot0_eef_pos': [],
                'robot0_eef_rot_axis_angle': [],
                'robot0_gripper_width': []
            }
    
            refer_mat = pose_to_mat(refer_eepose)
            refer_ee_vec = mat_to_certain_pose_type(refer_mat, env.pose_type)
            refer_obs_data['robot0_eef_pos'].append(refer_ee_vec[:3])
            refer_obs_data['robot0_eef_rot_axis_angle'].append(refer_ee_vec[3:])
            refer_obs_data['robot0_gripper_width'].append(np.array([env.get_gripper_width()]))
            
            episode_start_pose = np.concatenate([refer_obs_data['robot0_eef_pos'], refer_obs_data['robot0_eef_rot_axis_angle']], axis=-1)[-1]
            start_pose_mat = certain_pose_type_to_mat(episode_start_pose, pose_type="rotvec")
            
            s = time.time()
            obs_data = {
                    'robot0_eef_pos': [],
                    'robot0_eef_rot_axis_angle': [],
                    'robot0_gripper_width': []
                }
            
            
            eepose = compute_transform(raw_pose[frame_idx+1])
            print("next:", eepose - refer_eepose)

            # ee_mat = pose_to_mat(eepose)
            # ee_vec = mat_to_certain_pose_type(ee_mat, env.pose_type)

            obs_data['robot0_eef_pos'].append(eepose[:3] )
            obs_data['robot0_eef_rot_axis_angle'].append( eepose[3:] )
            obs_data['robot0_gripper_width'].append( np.array([env.get_gripper_width()]) )
                
            for key in obs_data.keys():
                obs_data[key] = np.stack(obs_data[key], axis=0)
                
            print("Obs latency", time.time()-s)
            
            pos_mat = certain_pose_type_to_mat(np.concatenate([obs_data['robot0_eef_pos'], obs_data['robot0_eef_rot_axis_angle']], axis=-1),pose_type="rotvec")

            print("convert", mat_to_certain_pose_type(pos_mat, "rotvec")- mat_to_certain_pose_type(start_pose_mat, "rotvec"))
            
            real_bos_pose_mat = convert_pose_mat_rep(pose_mat=pos_mat,
                                                        base_pose_mat=start_pose_mat,
                                                        pose_rep="relative",
                                                        backward=False)
            
            rel_obs_pose = mat_to_certain_pose_type(real_bos_pose_mat, "10d")
            print("rel", mat_to_certain_pose_type(real_bos_pose_mat, "rotvec"))
            

            raw_action =to_torch(np.concatenate([rel_obs_pose, obs_data['robot0_gripper_width']], axis=-1))
            in_abs_pose = np.concatenate([abs_pose, obs_data['robot0_gripper_width']], axis=-1)
            this_target_poses = get_real_umi_inference_action(raw_action.cpu().numpy(), in_abs_pose, "relative")
            print(this_target_poses)
            # input()
            action_timestamps = (1+np.arange(len(this_target_poses), dtype=np.float64)
                ) * robot_dt + time.time() - action_latency

            env.exec_actions(
                actions=this_target_poses,
                timestamps=action_timestamps
            )
            print(f"Submitted {len(this_target_poses)} steps of actions.")
            
            print('Action latency:', time.time() - s)
            

if states_data:
    states_array = np.array(states_data)
    np.save(str(output_dir / 'states.npy'), states_array)
    
logger.info(f"数据已保存至: {output_dir}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obs_horizon",
        type=int,
        default=2 
    )
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Random seed",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_left_cmd_topic",
        default="/master/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_right_cmd_topic",
        default="/master/joint_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_topic",
        action="store",
        type=str,
        help="puppet_arm_left_topic",
        default="/puppet/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_topic",
        action="store",
        type=str,
        help="puppet_arm_right_topic",
        default="/puppet/joint_right",
        required=False,
    )
    parser.add_argument(
        "--robot_base_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/odom_raw",
        required=False,
    )
    parser.add_argument(
        "--robot_base_cmd_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/cmd_vel",
        required=False,
    )
    parser.add_argument(
        "--use_robot_base",
        action="store_true",
        help="Whether to use the robot base to move around",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--ctrl_freq",
        action="store",
        type=int,
        help="The control frequency of the robot",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=64,
        required=False,
    )
    
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the perception task for decision making",
        default="",
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        help="The maximum change allowed for each joint per timestep",
        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
        required=False,
    )

    parser.add_argument(
        "--use_actions_interpolation",
        action="store_true",
        help="Whether to interpolate the actions if the difference is too large",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--disable_puppet_arm",
        action="store_true",
        help="Whether to disable the puppet arm. This is useful for safely debugging",
        default=False,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="robot/configs/base.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,  # 预训练的模型
        required=False,
        default=None,
        help="Name or path to the pretrained model",
    )
    parser.add_argument(
        "--lang_embeddings_path",
        type=str,  # 换成自己的路径
        required=False,
        default=None,
        help="Path to the pre-encoded language instruction embeddings",
    )
    parser.add_argument(
        "--cooperate",
        type=bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--buffer_length",
        type=int,
        help="Buffer length, if buffer length is 0, then no buffer is used",
        default=1,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        help="Logging level",
        default="INFO",
        choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="Whether to show the image",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--debug_step",
        type=int,
        default=1000000,
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        choices=["rdt", "act", "diffusion", "pi0"],
        default="rdt",
        help="Type of policy to use for inference",
    )
    parser.add_argument(
        "--task_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--policy.vision_backbone",
        type=str,
        choices=["dinov3-vit7b16", "dinov3-convnext-tiny"],  # add more if needed
        default=None,
        help="Vision backbone for the policy model",
    )
    parser.add_argument(
        "--use_handcap",
        type=bool,
        default=True
    )
    args = parser.parse_args()
    main(args)
