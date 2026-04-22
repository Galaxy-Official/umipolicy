import os
import sys
import cv2
import time
import torch
import numpy as np
from lerobot.datasets.pose_utils import *

from lerobot.umi.real_world.flexiv import *
import flexivrdk
import yaml

from lerobot.datasets.pose_utils import *
from lerobot.umi.real_world.flexiv_controller import FlexivInterface



class Camera:
    def __init__(self, 
        camera_id = None,
        v4l_path = None,
        fps=30,
        resolution=(1920, 1080)
    ):
        if camera_id is None and v4l_path is None:
            raise ValueError("Either camera_id or v4l_path must be provided")
        if camera_id is not None:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera with id {camera_id}")
        if v4l_path is not None:
            self.cap = cv2.VideoCapture(v4l_path, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera")
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        w, h = resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    
    def get_image(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read frame")
        return frame

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()


class SimpleFlexivEnv:
    def __init__(self, init_qpos, obs_horizon=2, robot_downsample=1, camera_downsample=1, resolution=(1920, 1080), use_gripper_width_mapping=False, use_transform=True, pose_type="10d"):
        self.robot = FlexivInterface(
            robot_ip="192.168.2.100",local_ip="192.168.2.103", use_gripper_width_mapping=use_gripper_width_mapping,
            device="cuda",
            move_home=False
        )
        self.init_qpos = init_qpos

        self.robot.send_gripper_state(0.09, 0.1, 10)

        self.obs_horizon = obs_horizon
        self.robot_downsample = robot_downsample
        self.camera_downsample = camera_downsample
        self.resolution = resolution
        self.use_transform = use_transform
        # self.rizon = Rizon4(headless=True)

        self.pose_type = pose_type

    def move_home(self):
        self.robot.move_to_home()

    def reset(self):
        print("Resetting robot")
        self.robot.send_joint_position(self.init_qpos)
        self.robot.send_gripper_state(0.09, 0.1, 10)
        time.sleep(15)

    def calculate_ik(self, obs_data):
        obs = {
            'camera0_rgb': [],
            'joint_angle': [],
            'robot0_gripper_width': []
        }
        eef_pos = obs_data['robot0_eef_pos']
        eef_rot = obs_data['robot0_eef_rot_axis_angle']
        for i in range(eef_pos.shape[0]):
            pos = eef_pos[i]
            rot = eef_rot[i]
            # print(rot.shape)
            flange_pose = np.concatenate([pos, rot])
            tcp_pose = mat_to_pose(pose_to_mat(flange_pose) @ FlexivInterface.tx_flange_tip)
            tcp_pose[2] = max(tcp_pose[2], 0.02)
            flange_pose = mat_to_pose(pose_to_mat(tcp_pose) @ FlexivInterface.tx_tip_flange)
            
            # 转换为位置和四元数
            pos, rot = pose_to_pos_rot(flange_pose)
            quat = rot.as_quat()
            flange_pose = np.concatenate([pos, quat])
            
            # 计算IK
            next_joints = self.rizon.calc_ik(flange_pose)
            
            assert next_joints is not None and np.all(np.isfinite(next_joints)), "IK failed"
            obs['joint_angle'].append(next_joints)
            # print(type(obs_data['camera0_rgb']))
            obs['camera0_rgb'].append(obs_data['camera0_rgb'][i])
            obs['robot0_gripper_width'].append(obs_data['robot0_gripper_width'][i])
        obs['camera0_rgb']=np.array(obs['camera0_rgb'])
        obs['robot0_gripper_width']=np.array(obs['robot0_gripper_width'])
        obs['joint_angle'] = np.array(obs['joint_angle'])
        return obs


    def exec_actions(self, actions, timestamps):
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        
        for i in range(len(new_actions)):
            # user_input = input()
            # if user_input.lower():
            #     print('continue')
            tip_pose = new_actions[i, 0:  6]   # since we have only 1 robot
            target_width = new_actions[i, 6]
            # move robot
            flange_pose = FlexivInterface.tip_to_flange_pose(tip_pose)

            self.robot.send_flange_pose(flange_pose)

            # move gripper
            self.robot.send_gripper_state(target_width, 0.1, 10)
            dt = new_timestamps[i] - time.time()
            if dt>0:
                time.sleep(dt)


class TactileFlexivEnv(SimpleFlexivEnv):
    def __init__(self, init_qpos, obs_horizon=2, robot_downsample=1, camera_downsample=1, resolution=(1920, 1080),
                 use_gripper_width_mapping=True, use_transform=True):
        super().__init__(init_qpos, obs_horizon, robot_downsample, camera_downsample, resolution,
                         use_gripper_width_mapping, use_transform)
        
        self.tactile_cameras = {}
        self.tactile_sensors = {}

        for side in ["left", "right"]:
            config_path = f"9DTact/shape_reconstruction/shape_config_{side}.yaml"
            with open(config_path, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            cfg["calibration_root_dir"] = "9DTact/shape_reconstruction/" + cfg["calibration_root_dir"]
            self.tactile_sensors[side] = None # TactileSensor(cfg)
            self.tactile_cameras[side] = Camera(camera_id = cfg["camera_setting"]["camera_channel"],resolution=(640, 480))
            
        self.tactile_is_ready = False
        self.initialize_sensor()



    def get_obs(self):
        obs_data = {
            'camera0_rgb': [],
            'robot0_eef_pos': [],
            'robot0_eef_rot_axis_angle': [],
            'robot0_gripper_width': [],
            'tactile_combined': []
        }

        for i_hor in range(self.obs_horizon):
            if i_hor % self.camera_downsample == 0:
                obs_data['camera0_rgb'].append(self.transform(self.camera.get_image()))

            # obs_data['timestamp'] = xx

            if i_hor % self.robot_downsample == 0:
                eepose = self.robot.get_ee_pose()
                obs_data['robot0_eef_pos'].append( eepose[:3] )
                obs_data['robot0_eef_rot_axis_angle'].append( eepose[3:] )
                obs_data['robot0_gripper_width'].append( np.array([self.robot.get_gripper_width()]) )
                obs_data["tactile_combined"].append(self.get_processed_tactile())
        
        for key in obs_data.keys():
            obs_data[key] = np.stack(obs_data[key], axis=0)

        return obs_data
    

    def initialize_sensor(self):
        """initialize the reference image with current capture"""
        for side, cap in self.tactile_cameras.items():
            for _ in range(5):
                cap.get_image()
            
            raw_img = cap.get_image()
            self.tactile_sensors[side].update_ref(raw_img)

        self.tactile_is_ready = True
        print("Tactile sensors are ready")


    def get_processed_tactile(self):
        assert self.tactile_is_ready

        _tact = {}
        for side, cap in self.tactile_cameras.items():
            raw_img = cap.get_image()
            img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            img = self.tactile_sensors[side].raw_image_2_xy_repr(img_GRAY)
            img = np.round(img).astype(np.uint8)
            img = self.tactile_sensors[side].get_rectify_crop_image(img)
            _tact[side] = img

        # img_left = frame_left.to_ndarray(format='rgb24')
        # img_right = frame_right.to_ndarray(format='rgb24')
        img_left = cv2.rotate(_tact["left"], cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_right = cv2.rotate(_tact["right"], cv2.ROTATE_90_CLOCKWISE)
        img = cv2.hconcat([img_left, img_right])
        return img  # tactile_combined

        

class FooFlexivEnv:
    def __init__(self, obs_horizon=2, robot_downsample=1, camera_downsample=1, resolution=(1920, 1080), *args, **kwargs):
        self.obs_horizon = obs_horizon
        self.robot_downsample = robot_downsample
        self.camera_downsample = camera_downsample
        self.resolution = resolution

    def get_obs(self):
        obs_data = {
            'camera0_rgb': [],
            'robot0_eef_pos': [],
            'robot0_eef_rot_axis_angle': [],
            'robot0_gripper_width': []
        }

        for i_hor in range(self.obs_horizon):
            if i_hor % self.camera_downsample == 0:
                obs_data['camera0_rgb'].append( np.zeros([224, 224, 3], dtype=np.float32) )

            # obs_data['timestamp'] = xx

            if i_hor % self.robot_downsample == 0:
                obs_data['robot0_eef_pos'].append( np.zeros([3]) )
                obs_data['robot0_eef_rot_axis_angle'].append( np.zeros([3]) )
                obs_data['robot0_gripper_width'].append( np.array([0.05]) )
        
        for key in obs_data.keys():
            obs_data[key] = np.stack(obs_data[key], axis=0)

        return obs_data



    def exec_actions(self, actions, timestamps):
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        for i in range(len(new_actions)):
            tip_pose = new_actions[i, 0:  6]   # since we have only 1 robot
            target_width = new_actions[i, 6]
            print("Pose", tip_pose, target_width)
