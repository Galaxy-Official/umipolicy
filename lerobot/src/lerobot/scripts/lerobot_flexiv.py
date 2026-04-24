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
import torchvision.utils as vutils

# Lerobot imports
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.processor.rename_processor import rename_stats
from lerobot.datasets.feature_utils import build_dataset_frame

# Flexiv imports (New RDK)
import flexivrdk
import scipy.spatial.transform as st
from lerobot.datasets.pose_utils import *
from lerobot.scripts.umi_realworld.real_inference_util import *
from lerobot.scripts.umi_realworld.utils.precise_sleep import precise_wait
from lerobot.scripts.umi_realworld.env import FlexivEnv
from perception.cameras.base_camera import BaseCamera


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


def compute_real_transform(raw_pose_10d):
    """
    Applies local coordinate compensation to a 10D action array: [x, y, z, r1, r2, r3, r4, r5, r6, width]
    """
    from lerobot.datasets.pose_utils import certain_pose_type_to_mat, mat_to_certain_pose_type
    
    pose_mat = certain_pose_type_to_mat(raw_pose_10d[:9], pose_type="10d")
    
    # Apply position compensation (107.68, 39, -76.86) mm
    # ---------------------------------------------------------
    offset_mm = np.array([107.68, 39.0, -76.86])
    offset_m = offset_mm / 1000.0
    
    # 构建纯平移的齐次矩阵
    offset_transform = np.eye(4)
    offset_transform[:3, 3] = offset_m
    
    # 如果该补偿是在"转换后的局部坐标系(Local Frame)"下定义的，请使用右乘：
    pose_mat_out = pose_mat @ offset_transform
    
    # 注意：如果您的补偿值是相对于"世界/基坐标系(Base Frame)"的绝对偏移，
    # 请将上面这行改为左乘：pose_mat_out = offset_transform @ pose_mat
    
    out_pose_9d = mat_to_certain_pose_type(pose_mat_out, pose_type="10d")
    
    return np.concatenate([out_pose_9d, [raw_pose_10d[9]]])

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
            left_tactile_img, right_tactile_img = None, None
            if self.cam_tactile_left is not None:
                left_tactile_img, _ = self.cam_tactile_left.get_data()
            if self.cam_tactile_right is not None:
                right_tactile_img, _ = self.cam_tactile_right.get_data()
            
            eepose = self.env.get_ee_pose()
            gripper_width = self.env.get_gripper_width()
            
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
# MAIN SCRIPT: Replicating lerobot_record functionality paired with handcap
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
    if args.use_tactile:
        handcap_camera_cfg = Path(__file__).resolve().parent.parent.parent / "perception/configs/camera/handcap_camera.json"
    else:
        handcap_camera_cfg = Path(__file__).resolve().parent.parent.parent / "perception/configs/camera/handcap_camera_no_tactile.json"
    cam_dict = BaseCamera.create_cameras_from_config(config_path=str(handcap_camera_cfg))
    
    cam_wrist = cam_dict["wrist"]
    cam_tactile_left = cam_dict.get("left_tactile")
    cam_tactile_right = cam_dict.get("right_tactile")

    # Grab initial frames to set up VideoWriter metadata
    cam_state, init_agent_img, cam_cap_time = cam_wrist.read()
    wrist_size = (init_agent_img.shape[1], init_agent_img.shape[0])
    
    wrist_video = cv2.VideoWriter(str(output_dir / 'view1_wrist.mp4'), fourcc, args.ctrl_freq, wrist_size)
    if args.use_tactile:
        left_tactile_img, _ = cam_tactile_left.get_data()
        left_size = (left_tactile_img.shape[1], left_tactile_img.shape[0])
        right_tactile_img, _ = cam_tactile_right.get_data()
        right_size = (right_tactile_img.shape[1], right_tactile_img.shape[0])
        tactile_left_video = cv2.VideoWriter(str(output_dir / 'view2_tactile_left.mp4'), fourcc, args.ctrl_freq, left_size)
        tactile_right_video = cv2.VideoWriter(str(output_dir / 'view3_tactile_right.mp4'), fourcc, args.ctrl_freq, right_size)

    states_data = []
    
    # Set up interrupt hooks
    signal_module.signal(signal_module.SIGINT, signal_handler)  
    signal_module.signal(signal_module.SIGTERM, signal_handler) 

    # 2. Init Robot (New Flexiv RDK Setup)
    logger.info("Initializing FlexivEnv...")
    robot_ip = os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100")
    local_ip = os.environ.get("FLEXIV_LOCAL_IP", "192.168.2.102")
    init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"))  # Give a safe default
    
    env = FlexivEnv(init_qpos, obs_horizon=obs_horizon, robot_ip=robot_ip, local_ip=local_ip, use_gripper_width_mapping=False, pose_type="rotvec")

    
    # Check robot fault (Handled automatically in FlexivEnv init)

    # 3. Load Hugging Face Policy (LeRobot standard approach adapted for Handcap)
    logger.info("Initializing Hugging Face Policy using LeRobot factory mechanics...")
    cli_overrides = [] # Modify if extracting from command line or custom args
    policy_config = PreTrainedConfig.from_pretrained(args.pretrained_model_name_or_path, cli_overrides=cli_overrides)
    policy_config.pretrained_path = args.pretrained_model_name_or_path
    
    from lerobot.policies.factory import get_policy_class
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(args.pretrained_model_name_or_path, config=policy_config)
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

    eval_t_start = time.time()
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
                
                if not args.use_tactile:
                    left_tactile_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    right_tactile_img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                wrist_frame_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                left_tactile_frame_rgb = cv2.cvtColor(left_tactile_img, cv2.COLOR_BGR2RGB)
                right_tactile_frame_rgb = cv2.cvtColor(right_tactile_img, cv2.COLOR_BGR2RGB)
                
                # Write live video
                if i_hor == obs_horizon - 1:
                    wrist_video.write(wrist_frame_rgb)
                    if args.use_tactile:
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
                    "observation.state": np.concatenate([rel_obs_pose, obs_data['robot0_gripper_width']], axis=-1).astype(np.float32),
                    "observation.images.wrist": obs_data["observation.images.wrist"],
                }
                if 'observation.tactiles.left' in obs_data:
                    observation["observation.tactiles.left"] = obs_data['observation.tactiles.left']
                if 'observation.tactiles.right' in obs_data:
                    observation["observation.tactiles.right"] = obs_data['observation.tactiles.right']

                # Handcap explicit Tensor mapping (Manual caching since Handcap uses relative poses that invalidate internal LeRobot queues)
                device = get_safe_torch_device(policy.config.device)
                for k, v in observation.items():
                    if isinstance(v, np.ndarray):
                        observation[k] = torch.from_numpy(v).to(device).unsqueeze(0).to(torch.float32)
                    elif isinstance(v, torch.Tensor):
                        observation[k] = v.to(device).unsqueeze(0).to(torch.float32)

                # Resize images to match expected training resolution from config
                if hasattr(policy.config, "image_features") and policy.config.image_features:
                    for img_key, img_feat in policy.config.image_features.items():
                        if img_key in observation:
                            expected_shape = img_feat.shape  # (C, H, W)
                            expected_h, expected_w = expected_shape[1], expected_shape[2]
                            actual_h, actual_w = observation[img_key].shape[-2], observation[img_key].shape[-1]
                            if current_step == 0:
                                logger.info(f"[Shape Debug] '{img_key}': config expects ({expected_shape}), actual tensor shape = {tuple(observation[img_key].shape)}, image HxW = {actual_h}x{actual_w}")
                            if actual_h != expected_h or actual_w != expected_w:
                                if current_step == 0:
                                    logger.info(f"[Shape Debug] Resizing '{img_key}' from {actual_h}x{actual_w} -> {expected_h}x{expected_w}")
                                # Reshape for F.interpolate: merge batch+sequence dims -> (N, C, H, W)
                                orig_shape = observation[img_key].shape  # (B, S, C, H, W) or (B, C, H, W)
                                if observation[img_key].ndim == 5:
                                    B, S, C, H, W = orig_shape
                                    flat = observation[img_key].reshape(B * S, C, H, W)
                                    flat = torch.nn.functional.interpolate(flat, size=(expected_h, expected_w), mode='bilinear', align_corners=False)
                                    observation[img_key] = flat.reshape(B, S, C, expected_h, expected_w)
                                elif observation[img_key].ndim == 4:
                                    observation[img_key] = torch.nn.functional.interpolate(
                                        observation[img_key], size=(expected_h, expected_w), mode='bilinear', align_corners=False
                                    )

                # Execute Dataset Stat Normalizations
                observation = preprocessor(observation)
                
                # Build OBS_IMAGES with correct shape: (B, S, N, C, H, W)
                # where N=num_cameras. generate_actions uses einops "b s n ... -> (b s n) ..."
                # which flattens to (B*S*N, C, H, W) for the CNN backbone.
                if hasattr(policy.config, "image_features") and policy.config.image_features:
                    img_list = [observation[key] for key in policy.config.image_features]
                    # Each img is (B=1, S=2, C, H, W) = 5D. Stack at dim=2 to add camera dim N.
                    observation['observation.images'] = torch.stack(img_list, dim=2)
                    if current_step == 0:
                        logger.info(f"[Shape Debug] OBS_IMAGES shape: {observation['observation.images'].shape}")
                        logger.info(f"[Shape Debug] OBS_STATE shape: {observation['observation.state'].shape}")

                try:
                    # Direct generate_actions call - bypasses select_action queue mechanism
                    # which expects single-timestep input. Matches robopolicy pattern.
                    action_chunk = policy.diffusion.generate_actions(observation)
                    raw_action = action_chunk.squeeze(0).cpu().numpy()
                except Exception as eval_e:
                    logger.error(f"Error when processing model generation: {str(eval_e)}\n{traceback.format_exc()}")
                    signal_handler(None, None)


            # Handcap backwards logic: Action back to Original Absolute Coordinates
            abs_pose = np.concatenate([abs_pose, obs_data['robot0_gripper_width']], axis=-1)
            # Apply user-defined real-world transformation compensation to the raw relative 10D actions
            for idx in range(len(raw_action)):
                raw_action[idx] = compute_real_transform(raw_action[idx])
            
            abs_pose = np.array([abs_pose[-1] for _ in range(len(raw_action))])
            this_target_poses = get_real_umi_inference_action(raw_action, abs_pose, "relative")
            
            # Formulate Action Timings and enforce Time Budget (Latency Compensation)
            # Use hardware timestamp 'cam_cap_time' as the base observation timestamp for this cycle
            cam_cap_time = frames[-1]['cam_cap_time']
            action_timestamps = (np.arange(len(this_target_poses), dtype=np.float64)) * robot_dt + cam_cap_time
            action_exec_latency = 0.01
            curr_time = time.time()
            is_new = action_timestamps > (curr_time + action_exec_latency)
            
            if np.sum(is_new) == 0:
                # Exceeded time budget, inference was too slow.
                # Force execution of the last action at the next available multiple of dt
                this_target_poses = this_target_poses[[-1]]
                next_step_idx = int(np.ceil((curr_time - eval_t_start) / robot_dt))
                action_timestamp = eval_t_start + (next_step_idx) * robot_dt
                print(f'Over budget! Delay: {action_timestamp - curr_time:.3f}s')
                action_timestamps = np.array([action_timestamp])
            else:
                # Keep only future actions
                this_target_poses = this_target_poses[is_new]
                action_timestamps = action_timestamps[is_new]

            # Receding Horizon Control (RHC): Execute only a subset of predicted steps
            steps_to_exec = min(args.steps_per_inference, len(this_target_poses))
            this_target_poses = this_target_poses[:steps_to_exec]
            action_timestamps = action_timestamps[:steps_to_exec]

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

            # Rate Limiting (Precise Wait) to prevent loop from spinning faster than robot_dt
            # Subtract 0.005s as a small frame capture latency tolerance
            t_cycle_end = eval_t_start + current_step * robot_dt
            precise_wait(t_cycle_end - 0.005, time_func=time.time)
            
            # Advance the global clock by the number of steps we just executed
            current_step += steps_to_exec

    except Exception as e:
        logger.error(f"Error when processing: {str(e)}\n{traceback.format_exc()}")
        signal_handler(None, None)
    finally:
        logger.info("\nProgram finished normally, saving data...")
        obs_thread.stop()
        if wrist_video is not None:
            wrist_video.release()
        if tactile_left_video is not None:
            tactile_left_video.release()
        if tactile_right_video is not None:
            tactile_right_video.release()
        
        if states_data and output_dir:
            import pandas as pd
            pd.DataFrame(states_data).to_parquet(str(output_dir / 'states.parquet'))
        logger.info(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--ctrl_freq", action="store", type=int, default=30)
    parser.add_argument("--steps_per_inference", type=int, default=1, help="Number of action steps to execute per inference loop (RHC)")
    parser.add_argument("--task_name", type=str, default="handcap_flexiv_mvs")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="HF Model or path to the pretrained policy")
    parser.add_argument("--use_tactile", action="store_true", help="Use tactile cameras")
    
    args = parser.parse_args()
    main(args)
