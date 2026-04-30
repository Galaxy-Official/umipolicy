import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import time
import threading
import tempfile
from collections import deque
from multiprocessing.managers import SharedMemoryManager

import click

if 'QT_QPA_FONTDIR' not in os.environ:
    for font_dir in (
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        '/usr/share/fonts/truetype',
        '/usr/share/fonts/truetype/dejavu',
        '/System/Library/Fonts',
        '/Library/Fonts',
    ):
        if os.path.isdir(font_dir):
            os.environ['QT_QPA_FONTDIR'] = font_dir
            break

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
# 暂时屏蔽 SpaceMouse 导入，因为 Python 3.12 已经弃用了相关 C API 导致 spnav 报错
# from umi.real_world.spacemouse_shared_memory import Spacemouse

# 引入 Flexiv 环境 (本仓内的独立剥离版)
from scripts_real.flexiv_env.env import FlexivEnv
from scripts_real.flexiv_env.pose_util import certain_pose_type_to_mat, mat_to_certain_pose_type

# 引入相机 (使用 lerobot/src/perception 中的 BaseCamera)
sys.path.append(os.path.join(ROOT_DIR, "../lerobot/src"))
from perception.cameras.base_camera import BaseCamera


OmegaConf.register_new_resolver("eval", eval, replace=True)

MVS_CAPTURE_RESOLUTION = (768, 768)
MVS_VIDEO_FOURCC = "mp4v"
DEFAULT_MVS_FPS = 20.0


class Mp4RoundTrip:
    """Match the handcap offline path: VideoWriter(mp4v) then VideoCapture decode."""

    def __init__(self, resolution=MVS_CAPTURE_RESOLUTION, fps=DEFAULT_MVS_FPS):
        self.resolution = tuple(resolution)
        self.fps = float(fps)
        self.fourcc = cv2.VideoWriter_fourcc(*MVS_VIDEO_FOURCC)

    def __call__(self, frames):
        if len(frames) == 0:
            return []

        tmp_file = tempfile.NamedTemporaryFile(
            prefix="eval_flexiv_mvs_", suffix=".mp4", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        writer = None
        cap = None
        try:
            writer = cv2.VideoWriter(
                tmp_path, self.fourcc, self.fps, self.resolution)
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
                    raise RuntimeError(
                        f"Failed to decode mp4 round-trip frame {frame_idx}")
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


# ---------------------------------------------------------------------
# Asynchronous Camera Observation Thread (MVS Mode)
# ---------------------------------------------------------------------
class ObservationThread(threading.Thread):
    def __init__(self, cam_wrist, env, maxlen=2, camera_fps=DEFAULT_MVS_FPS):
        super().__init__()
        self.cam_wrist = cam_wrist
        self.env = env
        self.queue = deque(maxlen=maxlen)
        self.running = True
        self.daemon = True
        self.lock = threading.Lock()
        self.camera_fps = float(camera_fps)
        self.camera_dt = 1.0 / self.camera_fps if self.camera_fps > 0 else 0.0
        
    def run(self):
        while self.running:
            iter_start_time = time.monotonic()
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

            if self.camera_dt > 0:
                precise_wait(iter_start_time + self.camera_dt, time_func=time.monotonic)

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
@click.option('--use_tactile', is_flag=True, default=False, help="Whether to load tactile cameras.")
@click.option('--data_capture_fps', '--camera_fps', default=DEFAULT_MVS_FPS, type=float, help="MVS wrist capture FPS, matching handcap_rgb.py --fps.")
@click.option('--gripper-width-offset', '--gripper_width_offset', default=0.0, type=float, help="Offset added to policy-predicted gripper width before execution, in gripper command units.")
@click.option('--gripper-width-min', '--gripper_width_min', default=0.1, type=float, help="Minimum safe gripper width after applying offset.")
@click.option('--gripper-width-max', '--gripper_width_max', default=0.9, type=float, help="Maximum safe gripper width after applying offset.")
@click.option('--arm-max-linear-vel', '--arm_max_linear_vel', default=0.05, type=float, help="Flexiv max linear velocity for Cartesian policy execution.")
@click.option('--arm-max-angular-vel', '--arm_max_angular_vel', default=0.2, type=float, help="Flexiv max angular velocity for Cartesian policy execution.")
@click.option('--arm-max-linear-acc', '--arm_max_linear_acc', default=0.1, type=float, help="Flexiv max linear acceleration for Cartesian policy execution.")
@click.option('--arm-max-angular-acc', '--arm_max_angular_acc', default=0.3, type=float, help="Flexiv max angular acceleration for Cartesian policy execution.")
@click.option('--gripper-move-velocity', '--gripper_move_velocity', default=0.03, type=float, help="Flexiv gripper Move velocity.")
def main(input, output, robot_ip, local_ip, camera_config,
    init_joints, steps_per_inference, max_duration,
    frequency, command_latency, use_tactile, data_capture_fps,
    gripper_width_offset, gripper_width_min, gripper_width_max,
    arm_max_linear_vel, arm_max_angular_vel, arm_max_linear_acc,
    arm_max_angular_acc, gripper_move_velocity):
    
    if gripper_width_min > gripper_width_max:
        raise click.ClickException(
            f"Invalid gripper safety range: min {gripper_width_min} > max {gripper_width_max}")

    max_gripper_width = gripper_width_max
    gripper_speed = 0.2
    print(
        f"Policy gripper width offset: {gripper_width_offset:+.4f}; "
        f"safety clip=[{gripper_width_min:.4f}, {gripper_width_max:.4f}]")
    print(
        "Flexiv execution limits: "
        f"linear_vel={arm_max_linear_vel}, angular_vel={arm_max_angular_vel}, "
        f"linear_acc={arm_max_linear_acc}, angular_acc={arm_max_angular_acc}, "
        f"gripper_velocity={gripper_move_velocity}")

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
    camera_shape_meta = cfg.task.shape_meta.obs.camera0_rgb
    obs_horizon = int(camera_shape_meta.horizon)
    camera_down_sample_steps = int(camera_shape_meta.down_sample_steps)
    obs_history_len = (obs_horizon - 1) * camera_down_sample_steps + 1

    # 1. Initialize Cameras
    print("Initializing MVS Cameras...")
    if camera_config is None:
        camera_config = os.path.join(ROOT_DIR, "../lerobot/src/perception/configs/camera/handcap_camera.json")
    
    import tempfile
    with open(camera_config, "r") as f:
        cam_config_data = json.load(f)
        
    if not use_tactile:
        cam_config_data.pop("left_tactile", None)
        cam_config_data.pop("right_tactile", None)
        
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tmp_file:
        json.dump(cam_config_data, tmp_file)
        tmp_path = tmp_file.name
        
    cam_dict = BaseCamera.create_cameras_from_config(config_path=tmp_path)
    os.remove(tmp_path)
    
    cam_wrist = cam_dict["wrist"]
    cam_left_tactile = cam_dict.get("left_tactile")
    cam_right_tactile = cam_dict.get("right_tactile")

    with SharedMemoryManager() as shm_manager:
        # with Spacemouse(shm_manager=shm_manager) as sm, \
        with KeystrokeCounter() as key_counter:
             
            # 2. Init Flexiv Robot
            print("Initializing FlexivEnv...")
            init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"))
            env = FlexivEnv(
                init_qpos, obs_horizon=obs_horizon, robot_ip=robot_ip,
                local_ip=local_ip, use_gripper_width_mapping=False,
                pose_type="rotvec", arm_max_linear_vel=arm_max_linear_vel,
                arm_max_angular_vel=arm_max_angular_vel,
                arm_max_linear_acc=arm_max_linear_acc,
                arm_max_angular_acc=arm_max_angular_acc,
                gripper_move_velocity=gripper_move_velocity)
            
            print(f"Moving to init pose: {init_qpos}")
            env.reset()

            # 3. Start Camera Observation Thread
            obs_thread = ObservationThread(
                cam_wrist, env, maxlen=obs_history_len,
                camera_fps=data_capture_fps)
            obs_thread.start()
            
            print("Waiting for observation queue to fill...")
            while True:
                if obs_thread.get_obs(obs_history_len) is not None:
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
            mp4_round_trip = Mp4RoundTrip(
                resolution=MVS_CAPTURE_RESOLUTION, fps=data_capture_fps)
            print(
                "MVS capture matched to handcap data collection: "
                f"{data_capture_fps:g}Hz, down_sample_steps={camera_down_sample_steps}, "
                f"history_len={obs_history_len}, codec={MVS_VIDEO_FOURCC}")
            
            # Helper to fetch formatted obs
            def get_formatted_obs():
                raw_frames = obs_thread.get_obs(obs_history_len)
                if raw_frames is None:
                    raise RuntimeError("Observation queue is not ready.")
                selected_idxs = list(range(
                    0, obs_history_len, camera_down_sample_steps))
                frames = [raw_frames[idx] for idx in selected_idxs]
                env_obs = {
                    'camera0_rgb': [],
                    'robot0_eef_pos': [],
                    'robot0_eef_rot_axis_angle': [],
                    'robot0_gripper_width': [],
                    'timestamp': []
                }

                wrist_frames_bgr = []
                for frame in raw_frames:
                    wrist_img = frame['wrist_img']
                    if (wrist_img.shape[1], wrist_img.shape[0]) != MVS_CAPTURE_RESOLUTION:
                        wrist_img = cv2.resize(
                            wrist_img, MVS_CAPTURE_RESOLUTION,
                            interpolation=cv2.INTER_AREA)
                    wrist_frames_bgr.append(wrist_img)

                decoded_wrist_frames = mp4_round_trip(wrist_frames_bgr)
                selected_wrist_frames = [
                    decoded_wrist_frames[idx] for idx in selected_idxs]
                for frame, wrist_img in zip(frames, selected_wrist_frames):
                    wrist_img_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                    wrist_img_rgb = resize_with_black_padding(
                        wrist_img_rgb, target_h=obs_res[1], target_w=obs_res[0])
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
                episode_start_pose = [env.get_ee_pose()]
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose)
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
                    
                    episode_start_pose = [env.get_ee_pose()]
                    
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
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr).copy()
                            gripper_idxs = np.arange(6, action.shape[-1], 7)
                            action[..., gripper_idxs] = np.clip(
                                action[..., gripper_idxs] + gripper_width_offset,
                                gripper_width_min,
                                gripper_width_max)
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
