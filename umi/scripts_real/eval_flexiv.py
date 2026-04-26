import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import time
import threading
from collections import deque
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json

from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
ENABLE_GUI = os.environ.get('DISPLAY', '') != ''
if ENABLE_GUI:
    try:
        from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode, Key
    except Exception as e:
        print(f"Warning: Failed to load pynput ({e}). GUI disabled.")
        ENABLE_GUI = False

if not ENABLE_GUI:
    print("Running in headless mode. OpenCV display and keyboard 'S' stop are disabled. Use Ctrl+C to stop.")
    class KeyCode:
        def __init__(self, char):
            self.char = char
    class DummyKey:
        pass
    Key = DummyKey()
    class KeystrokeCounter:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def get_press_events(self):
            return []
from umi.real_world.real_inference_util import (get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.real_world.spacemouse_shared_memory import Spacemouse

# 引入 Flexiv 环境 (本仓内的独立剥离版)
from scripts_real.flexiv_env.env import FlexivEnv
from scripts_real.flexiv_env.pose_util import certain_pose_type_to_mat, mat_to_certain_pose_type

# 引入相机 (使用 lerobot/src/perception 中的 BaseCamera)
sys.path.append(os.path.join(ROOT_DIR, "../lerobot/src"))
from perception.cameras.base_camera import BaseCamera


OmegaConf.register_new_resolver("eval", eval, replace=True)

# ---------------------------------------------------------------------
# Asynchronous Camera Observation Thread (MVS Mode)
# ---------------------------------------------------------------------
class ObservationThread(threading.Thread):
    def __init__(self, cam_wrist, env, maxlen=2):
        super().__init__()
        self.cam_wrist = cam_wrist
        self.env = env
        self.queue = deque(maxlen=maxlen)
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()
        
    def run(self):
        while self.running:
            cam_state, wrist_img, cam_cap_time = self.cam_wrist.read()
            
            eepose = self.env.get_ee_pose()
            gripper_width = self.env.get_gripper_width()
            
            with self.lock:
                self.queue.append({
                    'wrist_img': wrist_img.copy() if wrist_img is not None else wrist_img,
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


def resize_with_black_padding(image, target_h=480, target_w=640):
    """等比例缩放并用黑色填充边缘，避免形变"""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return canvas


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', default='192.168.2.100')
@click.option('--local_ip', default='192.168.2.102')
@click.option('--camera_config', default=None, help='Path to camera config json (e.g. handcap_camera.json)')
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, local_ip, camera_config,
    init_joints, steps_per_inference, max_duration,
    frequency, command_latency):
    
    max_gripper_width = 0.09
    gripper_speed = 0.2

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    obs_horizon = cfg.task.shape_meta.obs.camera0_rgb.horizon

    # 1. Initialize Cameras
    print("Initializing MVS Cameras...")
    if camera_config is None:
        camera_config = os.path.join(ROOT_DIR, "../lerobot/src/perception/configs/camera/handcap_camera.json")
    cam_dict = BaseCamera.create_cameras_from_config(config_path=camera_config)
    cam_wrist = cam_dict["wrist"]

    with SharedMemoryManager() as shm_manager:
        # with Spacemouse(shm_manager=shm_manager) as sm, \
        with KeystrokeCounter() as key_counter:
             
            # 2. Init Flexiv Robot
            print("Initializing FlexivEnv...")
            init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"))
            env = FlexivEnv(init_qpos, obs_horizon=obs_horizon, robot_ip=robot_ip, local_ip=local_ip, use_gripper_width_mapping=False, pose_type="rotvec")
            
            print(f"Moving to init pose: {init_qpos}")
            env.reset()

            # 3. Start Camera Observation Thread
            obs_thread = ObservationThread(cam_wrist, env, maxlen=obs_horizon)
            obs_thread.start()
            
            print("Waiting for observation queue to fill...")
            while True:
                if obs_thread.get_obs(obs_horizon) is not None:
                    break
                time.sleep(0.01)

            cv2.setNumThreads(2)
            
            # creating model
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr

            device = torch.device('cuda')
            policy.eval().to(device)

            print('Ready!')
            
            # Helper to fetch formatted obs
            def get_formatted_obs():
                frames = obs_thread.get_obs(obs_horizon)
                env_obs = {
                    'camera0_rgb': [],
                    'robot0_eef_pos': [],
                    'robot0_eef_rot_axis_angle': [],
                    'robot0_gripper_width': [],
                    'timestamp': []
                }
                for frame in frames:
                    wrist_img = frame['wrist_img']
                    # Resize logic matching UMI / Handcap
                    if wrist_img.shape[0] != 768 or wrist_img.shape[1] != 768:
                        wrist_img = cv2.resize(wrist_img, (768, 768), interpolation=cv2.INTER_AREA)
                    wrist_img = resize_with_black_padding(wrist_img, target_h=obs_res[1], target_w=obs_res[0])
                    
                    wrist_img_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                    env_obs['camera0_rgb'].append(wrist_img_rgb)
                    env_obs['robot0_eef_pos'].append(frame['eepose'][:3])
                    env_obs['robot0_eef_rot_axis_angle'].append(frame['eepose'][3:])
                    env_obs['robot0_gripper_width'].append(frame['gripper_width'])
                    env_obs['timestamp'].append(frame['cam_cap_time'])

                for k in env_obs:
                    env_obs[k] = np.stack(env_obs[k])
                env_obs['robot0_gripper_width'] = env_obs['robot0_gripper_width'][:, np.newaxis]
                return env_obs

            # Warming up
            print("Warming up policy inference")
            with torch.no_grad():
                policy.reset()
                obs = get_formatted_obs()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                del result
            print("Warmup finished!")

            while True:
                # ========= human control loop ==========
                ''' 暂时先注释掉 SpaceMouse 遥控循环
                print("Human in control!")
                eepose = env.get_ee_pose()
                target_pose = eepose.copy()
                gripper_target_pos = env.get_gripper_width()
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = get_formatted_obs()

                    # visualize
                    vis_img = obs['camera0_rgb'][-1]
                    cv2.putText(
                        vis_img,
                        "Human Control",
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=2,
                        color=(0,0,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            print("Exiting...")
                            obs_thread.stop()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            print("Handing control to policy!")
                            start_policy = True
                    
                    if start_policy:
                        break

                    precise_wait(t_sample)
                    
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (0.5 / frequency)
                    drot_xyz = sm_state[3:] * (1.5 / frequency)

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    target_pose[2] = np.maximum(target_pose[2], 0.055)
                    
                    dpos = 0
                    if sm.is_button_pressed(0):
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)

                    action = np.zeros((7,))
                    action[:6] = target_pose
                    action[-1] = gripper_target_pos     

                    # execute teleop command
                    env.exec_actions(
                        actions=np.array([action]), 
                        timestamps=np.array([t_command_target-time.monotonic()+time.time()])
                    )
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                '''
                
                # ========== policy control loop ==============
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Policy Started!")
                    iter_idx = 0
                    
                    while True:
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = get_formatted_obs()
                        obs_timestamps = obs['timestamp']
                        
                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print(f'Inference latency: {time.time() - s:.3f}s')
                        
                        this_target_poses = action

                        # action timestamps
                        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        
                        if np.sum(is_new) == 0:
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # Apply Receding Horizon
                        steps_to_exec = min(steps_per_inference, len(this_target_poses))
                        this_target_poses = this_target_poses[:steps_to_exec]
                        action_timestamps = action_timestamps[:steps_to_exec]

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )

                        # visualize
                        if ENABLE_GUI:
                            vis_img = obs['camera0_rgb'][-1]
                            cv2.putText(
                                vis_img,
                                f'Policy Mode, Time: {time.monotonic() - t_start:.1f}',
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=2,
                                color=(0,255,0)
                            )
                            cv2.imshow('default', vis_img[...,::-1])
                            _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                print('Stopped by user.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        
                        if stop_episode:
                            break

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    break
                
                print("Stopped. Returning to human control.")

if __name__ == '__main__':
    main()
