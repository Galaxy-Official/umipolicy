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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib_py"))
import flexivrdk
import scipy.spatial.transform as st
from lerobot.datasets.pose_utils import *
from lerobot.scripts.umi_realworld.real_inference_util import *
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

def self_exam(log):
    # Setup from old script checking flexiv robot faults
    pass

# ---------------------------------------------------------------------
# NewFlexivEnv (New RDK 1.8 API)
# ---------------------------------------------------------------------
class NewFlexivEnv:
    tx_flange_tip = np.identity(4)
    tx_flange_tip[:3, 3] = np.array([0, 0, 0.185])  # measured physically matching legacy
    tx_tip_flange = np.linalg.inv(tx_flange_tip)

    @staticmethod
    def tip_to_flange_pose(tip_pose):
        return mat_to_pose(pose_to_mat(tip_pose) @ NewFlexivEnv.tx_tip_flange)

    def __init__(self, init_qpos, obs_horizon=2, robot_ip="192.168.2.100", local_ip="192.168.2.102", use_gripper_width_mapping=False, pose_type="rotvec"):
        self.obs_horizon = obs_horizon
        self.pose_type = pose_type
        
        # New RDK Setup
        self.robot = flexivrdk.Robot(robot_ip, local_ip)
        self.model = flexivrdk.Model(self.robot)
        self.gripper = flexivrdk.Gripper(self.robot)
        self.mode = flexivrdk.Mode
        
        # Clear faults and enable
        if self.robot.fault():
            self.robot.ClearFault()
            time.sleep(2)
        self.robot.Enable()
        while not self.robot.operational():
            time.sleep(1)
            
        # Initial Gripper
        self.gripper.Move(0.12, 0.1, 10)
                
        # Switch to Joint Position Mode
        self.robot.SwitchMode(self.mode.NRT_JOINT_POSITION)
        
        # Define constraints
        self.target_vel = [0.0] * self.robot.info().DoF
        self.max_vel = [0.5] * self.robot.info().DoF # Slower, safer max vel for inference
        self.max_acc = [2.0] * self.robot.info().DoF
        
        if init_qpos is not None:
            self.robot.SendJointPosition(init_qpos, self.target_vel, self.max_vel, self.max_acc)
            time.sleep(3)
        time.sleep(1)

    def reset(self):
        pass # Optional resetting logic internally

    class _MockRobot:
        def __init__(self, parent):
            self.parent = parent
        def get_ee_pose(self):
            return self.parent.get_ee_pose()
        def get_gripper_width(self):
            return self.parent.get_gripper_width()

    @property
    def robot_mock(self):
        # We supply this property so `self.env.robot.get_ee_pose()` backwards-compatibility continues to work
        if not hasattr(self, "_robot_inst"):
            self._robot_inst = self._MockRobot(self)
        return self._robot_inst
    
    # Reroute property to alias back properly to `env.robot.xxx` downstream
    def __getattr__(self, name):
        if name == "robot_mock":
            return self.robot_mock
        if name == "robot" and not hasattr(self.__dict__, "robot"):
            return self.robot_mock
        return self.__dict__.get(name) or super().__getattribute__(name)

    def get_ee_pose(self):
        # Flange pose natively reported from RDK
        flange_pose_raw = self.robot.states().tcp_pose.copy()
        
        pos = flange_pose_raw[:3]
        # Flexiv native TCP pose returned is [x, y, z, qw, qx, qy, qz]
        qw, qx, qy, qz = flange_pose_raw[3:]
        rot = st.Rotation.from_quat([qx, qy, qz, qw])
        
        # We need to add the offset manually to output the expected tooltip pose (like legacy FlexivInterface)
        tip_pose_mat = pos_rot_to_mat(np.array(pos), rot) @ NewFlexivEnv.tx_flange_tip
        umi_tip_pose = mat_to_pose(tip_pose_mat)
        return umi_tip_pose

    def get_gripper_width(self):
        states = flexivrdk.GripperStates()
        self.gripper.GetGripperStates(states)
        return states.width

    def exec_actions(self, actions, timestamps):
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        
        for i in range(len(new_actions)):
            tip_pose = new_actions[i, 0:6]
            target_width = new_actions[i, 6]
            
            # Convert tooltip pose back to flange pose
            flange_pose = NewFlexivEnv.tip_to_flange_pose(tip_pose)
            pos, rot = pose_to_pos_rot(flange_pose)
            quat_xyzw = rot.as_quat(scalar_first=False)
            
            # Convert to Flexiv convention [x, y, z, qw, qx, qy, qz]
            flexiv_target_pose = [
                pos[0], pos[1], pos[2],
                quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
            ]
            
            # Use new model IK to find joints
            current_q = self.robot.states().q.copy()
            reachable, target_q = self.model.reachable(flexiv_target_pose, current_q, True)
            
            if reachable:
                self.robot.SendJointPosition(target_q, self.target_vel, self.max_vel, self.max_acc)
            else:
                logger.warning(f"Target pose {flexiv_target_pose} is unreachable!")
            
            # move gripper
            # Account for Gripper offset identically to legacy
            self.gripper.Move(max(target_width - 0.01, 0.005), 0.1, 10)
            
            dt = new_timestamps[i] - time.time()
            if dt > 0:
                time.sleep(dt)


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
    logger.info("Initializing NewFlexivEnv...")
    robot_ip = os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100")
    local_ip = os.environ.get("FLEXIV_LOCAL_IP", "192.168.2.102")
    init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"))  # Give a safe default
    
    env = NewFlexivEnv(init_qpos, obs_horizon=obs_horizon, robot_ip=robot_ip, local_ip=local_ip, use_gripper_width_mapping=False, pose_type="rotvec")
    env.robot = env.robot_mock # Patch up observation thread calls

    
    # Check robot fault
    self_exam(flexivrdk.Log())

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
                    "observation.state": to_torch(np.concatenate([rel_obs_pose, obs_data['robot0_gripper_width']], axis=-1)),
                    "observation.images.wrist": to_torch(obs_data["observation.images.wrist"]),
                    "observation.tactiles.left": to_torch(obs_data['observation.tactiles.left']),
                    "observation.tactiles.right": to_torch(obs_data['observation.tactiles.right']),
                }

                # LeRobot Action Prediction Hook
                # Note: `action_values_dict` mapping heavily depends on LeRobot's architecture. 
                # Assumes policy returns {"action": Tensor}
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
                    # Direct inference if predict_action isn't perfectly mapped
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
    parser.add_argument("--use_tactile", action="store_true", help="Use tactile cameras")
    
    args = parser.parse_args()
    main(args)
