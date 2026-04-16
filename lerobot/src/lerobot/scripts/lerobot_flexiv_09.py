import os
import sys
import cv2
import time
import argparse
import datetime
import traceback
import threading
import numpy as np
from collections import deque
from pathlib import Path
import signal as signal_module
from loguru import logger

# PyTorch & Torchvision
import torch
from torchvision import transforms

# Lerobot imports
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device

# Flexiv imports (Old RDK 0.9 via flexiv_simple_env)
from lerobot.datasets.pose_utils import *
from lerobot.scripts.umi_realworld.real_inference_util import *
from perception.cameras.base_camera import BaseCamera
from lerobot.scripts.umi_realworld.flexiv_simple_env import SimpleFlexivEnv


# ---------------------------------------------------------------------
# Handcap Tactile and Image Transformations
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Asynchronous Camera Observation Thread (Handcap Mode)
# ---------------------------------------------------------------------
class ObservationThread(threading.Thread):
    def __init__(self, cam_wrist, cam_tactile_left, cam_tactile_right, env, maxlen=2):
        super().__init__()
        self.cam_wrist = cam_wrist
        self.cam_tactile_left = cam_tactile_left
        self.cam_tactile_right = cam_tactile_right
        self.env = env
        self.queue = deque(maxlen=maxlen)
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()
        
    def run(self):
        while self.running:
            cam_state, wrist_img, cam_cap_time = self.cam_wrist.read()
            left_tactile_img, _ = self.cam_tactile_left.get_data()
            right_tactile_img, _ = self.cam_tactile_right.get_data()
            
            # Use SimpleFlexivEnv (FlexivInterface inside)
            eepose = self.env.robot.get_ee_pose()
            gripper_width = self.env.robot.get_gripper_width()
            
            with self.lock:
                self.queue.append({
                    'wrist_img': wrist_img.copy() if wrist_img is not None else wrist_img,
                    'left_tactile_img': left_tactile_img.copy() if left_tactile_img is not None else left_tactile_img,
                    'right_tactile_img': right_tactile_img.copy() if right_tactile_img is not None else right_tactile_img,
                    'cam_cap_time': cam_cap_time,
                    'eepose': eepose,
                    'gripper_width': gripper_width
                })

    def get_obs(self, n=2):
        with self.lock:
            if len(self.queue) < n:
                return None
            return list(self.queue)[-n:]

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------
# Global vars for signal handling
# ---------------------------------------------------------------------
wrist_video = None
tactile_left_video = None
tactile_right_video = None
states_data = None
output_dir = None


def signal_handler(sig, frame):
    global wrist_video, tactile_left_video, tactile_right_video, states_data, output_dir
    logger.info("\nDetected interrupt signal, saving data...")
    try:
        if wrist_video is not None:
            wrist_video.release()
        if tactile_left_video is not None:
            tactile_left_video.release()
        if tactile_right_video is not None:
            tactile_right_video.release()

        if states_data and output_dir:
            import pandas as pd
            df = pd.DataFrame(states_data)
            df.to_parquet(str(output_dir / 'states.parquet'))
            import json
            with open(str(output_dir / 'states.json'), 'w') as f:
                json.dump(states_data, f)
            logger.info(f"Saved {len(states_data)} frames of states data to {output_dir}")
        os.sync()
    except Exception as e:
        logger.error(f"Error when save data: {str(e)}")
    finally:
        sys.exit(0)


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


# ---------------------------------------------------------------------
# MAIN SCRIPT: Replicating lerobot_record functionality paired with handcap (RDK 0.9)
# ---------------------------------------------------------------------
def main(args):
    global wrist_video, tactile_left_video, tactile_right_video, states_data, output_dir
    
    robot_dt = 1. / args.ctrl_freq
    action_latency = 0
    obs_horizon = args.obs_horizon

    # Output directory structuring
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent.parent.parent.parent / "realworld_replay_recording" / args.task_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 1. Initialize Cameras (copied MVS logic via base_camera.json config)
    logger.info("Initializing Handcap Cameras (MVS & Webcams)...")
    handcap_camera_cfg = Path(__file__).resolve().parent.parent.parent / "perception/configs/camera/handcap_camera.json"
    cam_dict = BaseCamera.create_cameras_from_config(config_path=str(handcap_camera_cfg))
    
    cam_wrist = cam_dict["wrist"]
    cam_tactile_left = cam_dict["left_tactile"]
    cam_tactile_right = cam_dict["right_tactile"]

    # Grab initial frames to set up VideoWriter metadata
    cam_state, init_agent_img, cam_cap_time = cam_wrist.read()
    wrist_size = (init_agent_img.shape[1], init_agent_img.shape[0])
    
    left_tactile_img, _ = cam_tactile_left.get_data()
    left_size = (left_tactile_img.shape[1], left_tactile_img.shape[0])

    right_tactile_img, _ = cam_tactile_right.get_data()
    right_size = (right_tactile_img.shape[1], right_tactile_img.shape[0])

    wrist_video = cv2.VideoWriter(str(output_dir / 'view1_wrist.mp4'), fourcc, args.ctrl_freq, wrist_size)
    tactile_left_video = cv2.VideoWriter(str(output_dir / 'view2_tactile_left.mp4'), fourcc, args.ctrl_freq, left_size)
    tactile_right_video = cv2.VideoWriter(str(output_dir / 'view3_tactile_right.mp4'), fourcc, args.ctrl_freq, right_size)

    states_data = []
    
    # Set up interrupt hooks
    signal_module.signal(signal_module.SIGINT, signal_handler)  
    signal_module.signal(signal_module.SIGTERM, signal_handler) 

    # 2. Init Robot (Old Flexiv RDK 0.9 Setup)
    logger.info("Initializing SimpleFlexivEnv (RDK 0.9)...")
    init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"))
    
    # SimpleFlexivEnv internally handles flexivrdk.Robot, clearing faults, and moving home
    env = SimpleFlexivEnv(init_qpos, obs_horizon=obs_horizon, use_gripper_width_mapping=False, pose_type="rotvec")

    # 3. Load Hugging Face Policy (LeRobot standard approach adapted for Handcap)
    logger.info("Initializing Hugging Face Policy using LeRobot factory mechanics...")
    cli_overrides = [] # Modify if extracting from command line or custom args
    policy_config = PreTrainedConfig.from_pretrained(args.pretrained_model_name_or_path, cli_overrides=cli_overrides)
    policy_config.pretrained_path = args.pretrained_model_name_or_path
    
    policy = make_policy(policy_config)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_config.pretrained_path,
        dataset_stats=None, 
        preprocessor_overrides={"device_processor": {"device": policy_config.device}},
    )
    
    env.reset()
    policy.reset()
    if preprocessor: preprocessor.reset()
    if postprocessor: postprocessor.reset()

    # 4. Start Handcap observation thread
    obs_thread = ObservationThread(cam_wrist, cam_tactile_left, cam_tactile_right, env, maxlen=obs_horizon)
    obs_thread.start()
    
    logger.info("Waiting for observation queue to fill...")
    while True:
        if obs_thread.get_obs(obs_horizon) is not None:
            break
        time.sleep(0.01)
    logger.info("Observation queue filled. Starting real robot inference control loop!")

    current_step = 0
    try:
        while True:
            s = time.time()
            
            obs_data = {
                'observation.images.wrist': [],
                'observation.tactiles.left':[],
                'observation.tactiles.right':[],
                'robot0_eef_pos': [],
                'robot0_eef_rot_axis_angle': [],
                'robot0_gripper_width': []
            }
            
            abs_pose = []
            frames = obs_thread.get_obs(obs_horizon)
            for i_hor in range(obs_horizon):
                frame_data = frames[i_hor]
                wrist_img = frame_data['wrist_img']
                left_tactile_img = frame_data['left_tactile_img']
                right_tactile_img = frame_data['right_tactile_img']
                
                wrist_frame_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                left_tactile_frame_rgb = cv2.cvtColor(left_tactile_img, cv2.COLOR_BGR2RGB)
                right_tactile_frame_rgb = cv2.cvtColor(right_tactile_img, cv2.COLOR_BGR2RGB)
                
                # Write live video
                if i_hor == obs_horizon - 1:
                    wrist_video.write(wrist_frame_rgb)
                    tactile_left_video.write(left_tactile_frame_rgb)
                    tactile_right_video.write(right_tactile_frame_rgb)
                
                # Preprocess for model
                wrist_frame = wrist_frame_rgb / 255.0
                left_tactile_frame = left_tactile_frame_rgb / 255.0
                right_tactile_frame = right_tactile_frame_rgb / 255.0
                
                obs_data['observation.images.wrist'].append(wrist_frame.transpose(2, 0, 1))
                obs_data['observation.tactiles.left'].append(tactile_transforms["touch"](torch.from_numpy(left_tactile_frame.transpose(2, 0, 1))))
                obs_data['observation.tactiles.right'].append(tactile_transforms["touch"](torch.from_numpy(right_tactile_frame.transpose(2, 0, 1))))
                
                eepose = frame_data['eepose']
                abs_pose.append(eepose)
                obs_data['robot0_eef_pos'].append(eepose[:3])
                obs_data['robot0_eef_rot_axis_angle'].append(eepose[3:])
                obs_data['robot0_gripper_width'].append(np.array([frame_data['gripper_width']]))
                
            for key in obs_data.keys():
                obs_data[key] = np.stack(obs_data[key], axis=0)
            
            episode_start_pose = np.concatenate([obs_data['robot0_eef_pos'], obs_data['robot0_eef_rot_axis_angle']], axis=-1)[-1]
            abs_pose = np.stack(abs_pose, axis=0)
            
            with torch.inference_mode():
                
                # Handcap relative pose transformation
                pos_mat = certain_pose_type_to_mat(np.concatenate([obs_data['robot0_eef_pos'], obs_data['robot0_eef_rot_axis_angle']], axis=-1), pose_type="rotvec")
                start_pose_mat = certain_pose_type_to_mat(episode_start_pose, pose_type="rotvec")
                
                real_obs_pose_mat = convert_pose_mat_rep(pose_mat=pos_mat, base_pose_mat=start_pose_mat, pose_rep="relative", backward=False)
                rel_obs_pose = mat_to_certain_pose_type(real_obs_pose_mat, "10d")
                
                # Assemble unified LeRobot dictionary observation
                observation = {
                    "observation.state": to_torch(np.concatenate([rel_obs_pose, obs_data['robot0_gripper_width']], axis=-1)),
                    "observation.images.wrist": to_torch(obs_data["observation.images.wrist"]),
                    "observation.tactiles.left": to_torch(obs_data['observation.tactiles.left']),
                    "observation.tactiles.right": to_torch(obs_data['observation.tactiles.right']),
                }

                # LeRobot Action Prediction Hook
                try:
                    action_values_dict = predict_action(
                        observation=observation,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=None,
                        robot_type=""
                    )
                    if isinstance(action_values_dict, dict) and "action" in action_values_dict:
                        raw_action = action_values_dict["action"].squeeze(0).cpu().numpy()
                    else:
                        # Fallback simple tensor assumption
                        raw_action = action_values_dict.squeeze(0).cpu().numpy()
                except Exception as eval_e:
                    logger.warning(f"Failed using LeRobot predict action. Falling back to simple policy forward. Error: {eval_e}")
                    tensor_out = policy.select_action(observation)
                    raw_action = tensor_out.squeeze(0).cpu().numpy()

            # Handcap backwards logic: Action back to Original Absolute Coordinates
            abs_pose = np.concatenate([abs_pose, obs_data['robot0_gripper_width']], axis=-1)
            abs_pose = np.array([abs_pose[-1] for _ in range(len(raw_action))])
            this_target_poses = get_real_umi_inference_action(raw_action, abs_pose, "relative")
            
            # Formulate Action Timings
            action_timestamps = (1 + np.arange(len(this_target_poses), dtype=np.float64)) * robot_dt + time.time() - action_latency
            current_step += 1
            
            # Send Actions
            env.exec_actions(actions=this_target_poses, timestamps=action_timestamps)
            
            # Data Saving / Profiling
            inference_latency = time.time() - s
            latest_obs = frames[-1]
            states_data.append({
                'step': current_step,
                'inference_latency': inference_latency,
                'robot0_eef_pos_x': float(latest_obs['eepose'][0]),
                'robot0_eef_pos_y': float(latest_obs['eepose'][1]),
                'robot0_eef_pos_z': float(latest_obs['eepose'][2]),
                'robot0_eef_rot_x': float(latest_obs['eepose'][3]),
                'robot0_eef_rot_y': float(latest_obs['eepose'][4]),
                'robot0_eef_rot_z': float(latest_obs['eepose'][5]),
                'gripper_width': float(latest_obs['gripper_width'])
            })
            
            if current_step % 10 == 0:
                print(f"[{current_step}] Inference Cycle (ms): {inference_latency*1000:.1f}")

    except Exception as e:
        logger.error(f"Error when processing: {str(e)}\n{traceback.format_exc()}")
        signal_handler(None, None)
    finally:
        logger.info("\nProgram finished normally, saving data...")
        obs_thread.stop()
        wrist_video.release()
        tactile_left_video.release()
        tactile_right_video.release()
        
        if states_data and output_dir:
            import pandas as pd
            pd.DataFrame(states_data).to_parquet(str(output_dir / 'states.parquet'))
        logger.info(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--ctrl_freq", action="store", type=int, default=30)
    parser.add_argument("--task_name", type=str, default="handcap_flexiv_mvs")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="HF Model or path to the pretrained policy")
    
    args = parser.parse_args()
    main(args)
