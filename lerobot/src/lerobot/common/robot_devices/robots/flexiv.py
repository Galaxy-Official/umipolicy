#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from dataclasses import dataclass, field, replace
from loguru import logger
import torch
import numpy as np
from typing import List
import os
import flexivrdk
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.MiniTeleop_utils import LeadArmReader


@dataclass
class FlexivRobotConfig:
    robot_type: str | None = "flexiv"
    leader_arm: dict[str, str] = field(default_factory=lambda: {})
    ips: dict[str, str] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    # TODO(aliberts): add feature with max_relative target
    # TODO(aliberts): add comment on max_relative target
    max_relative_target: list[float] | float | None = None
    gripper_open_degree: float | None = None


class FlexivRobot():
    """Wrapper of stretch_body.robot.Robot"""

    def __init__(self, config: FlexivRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = FlexivRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.robot_type = self.config.robot_type
        self.leader_arm = self.config.leader_arm
        self.ips = self.config.ips
        self.cameras = self.config.cameras
        self.is_connected = False
        self.teleop = None
        self.logs = {}
        try:
            import spdlog
            self.log = spdlog.ConsoleLogger("Flexiv")
        except ImportError:
            import logging
            self.log = logging.getLogger("Flexiv")

        self.state_keys = None
        self.action_keys = None
        self.DOF = 7
        self.target_vel = [0.0] * self.DOF
        self.target_acc = [0.0] * self.DOF
        self.MAX_VEL = [0.3] * self.DOF
        self.MAX_ACC = [0.3] * self.DOF

        robot_sn = os.environ.get("FLEXIV_ROBOT_SN", self.ips.get("robot_ip", ""))
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((self.ips["robot_ip"], 1))
            actual_local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            actual_local_ip = self.ips.get("local_ip", "")

        self.flexiv = flexivrdk.Robot(robot_sn, [actual_local_ip])
        self.gripper = flexivrdk.Gripper(self.flexiv)
        
        gripper_name = os.environ.get("FLEXIV_GRIPPER_NAME", "Flexiv-GN01")
        try:
            self.gripper.Enable(gripper_name)
        except Exception as e:
            self.log.warn(f"Failed to enable gripper [{gripper_name}]: {e}")

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "dtype": "video",
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
            # Add depth features if depth is enabled
            if hasattr(cam, 'use_depth') and cam.use_depth:
                depth_key = f"observation.depth.{cam_key}"
                cam_ft[depth_key] = {
                    "dtype": "video",
                    "shape": (cam.height, cam.width),
                    "names": ["height", "width"],
                    "info": None,
                }
        return cam_ft
    
    @property
    def motor_features(self) -> dict:
        # BUG: action shape = DOF+1
        DOF = self.DOF
        return {
            "action": {
                "dtype": "float32",
                "shape": (DOF+1,),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (DOF+1,),
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        # connect flexiv
        try:
            self.is_connected = not self.flexiv.fault()

            if not self.is_connected:
                self.log.warn("Fault occurred on robot server, trying to clear ...")
                self.flexiv.ClearFault()
                time.sleep(2)
                if self.flexiv.fault():
                    self.log.error("Fault cannot be cleared, exiting ...")
                    raise ConnectionError()
                self.log.info("Fault on robot server is cleared")

            self.log.info("Enabling robot ...")
            self.flexiv.Enable()

            while not self.flexiv.operational():
                time.sleep(1)

            self.log.info("Robot is now operational")

        except Exception as e:
            # Print exception error message
            self.log.error(str(e))

        # connect leader arm
        if self.leader_arm and "robot_path" in self.leader_arm:
            robot_overrides = ["~cameras", "~follower_arms"]
            self.teleop = LeadArmReader(
                self.leader_arm["robot_path"], robot_overrides, visualize=self.leader_arm["visualize"]
            )
        else:
            self.teleop = None

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print(
                "Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.home()
        # self.home_for_twist() # by zhiyuan_hong for specific situations

    def home(self) -> None:
        """Move the robot to its home position smoothly and safely"""
        self.log.info("Moving to home pose smoothly...")
        self.flexiv.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        
        # Standard Flexiv Home Pose
        home_qpos = [0.0, -0.6981317, 0.0, 1.5707963, 0.0, 0.6981317, 0.0]
        self.flexiv.SendJointPosition(home_qpos, [0]*7, [0.3]*7, [0.3]*7)
        
        # Wait until joints are close to home
        import numpy as np
        while True:
            curr_q = np.array(self.flexiv.states().q)
            if np.max(np.abs(curr_q - home_qpos)) < 0.05:
                break
            time.sleep(0.5)
        self.log.info("Reached home pose.")
            
        self.log.info("Zeroing Force/Torque sensors...")
        self.flexiv.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self.flexiv.ExecutePrimitive("ZeroFTSensor", dict())
        # Safe wait instead of brittle primitive_states parsing
        time.sleep(5)
        self.log.info("Sensor zeroing complete.")

        self.flexiv.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)

    def home_for_twist(self) -> None:
        # by zhiyuan_hong
        # Only for specific situations
        print("=============================================================")
        print("||      Warning: This home position is only for Zhiyuan    ||")
        print("||      Please do not use it in other situations.          ||")
        print("=============================================================")
        mode = flexivrdk.Mode
        new_home = [0.682659924030304,-0.11099029332399368,0.13164113461971283,-0.0045234146527945995,-0.00043452114914543927,0.9999896287918091,0.0003461818560026586]
        new_home[2] += 0.1
        self.flexiv.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        self.flexiv.SendCartesianMotionForce(new_home, maxLinearVel=0.1, maxAngularVel=0.4)
        time.sleep(3)
        self.flexiv.SwitchMode(mode.NRT_JOINT_POSITION)



    def set_robot_home_position(self) -> None:
        # self.gripper.Move(0.01, 0.1, 20)
        max_width = self.gripper.params().max_width
        self.gripper.Move(max_width, 0.1, 20)
        self.home()
        # self.home_for_twist()  # by zhiyuan_hong for specific situations

        # # Only for specific situations
        # print("========================================================")
        # print("||      Warning: This home position is only for Ziyu    ||")
        # print("||      Please do not use it in other situations.       ||")
        # print("========================================================")
        # new_home = [3.0889e-07, -6.9813e-01,  3.9452e-06,  1.5708e+00,  1.3448e-06, 6.9814e-01,  1.5726e+00,  1.0000e-01]
        # now_home_action = new_home[:self.DOF]
        # self.flexiv.sendJointPosition(
        #     now_home_action, self.target_vel, self.target_acc, self.MAX_VEL, self.MAX_ACC)
        # time.sleep(5)




    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None:
            print("teleop is None")
            raise ConnectionError()

        before_read_t = time.perf_counter()
        state = self.get_state()
        # state, cam_pose = self.get_state_and_cam()

        # action = self.teleop.gamepad_controller.get_state()
        action, gripper_width = self.teleop.get_follow_arm_joints()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()

        eef_position = self.teleop.get_lead_arm_eef()
        self.apply_teleop_target(action, gripper_width, eef_position[0])

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # if self.state_keys is None:
        #     self.state_keys = list(state)

        if not record_data:
            return

        action = torch.as_tensor(action)[:self.DOF]
        gripper_width = torch.tensor([gripper_width])
        action = torch.cat((action, gripper_width))

        # Capture images from cameras
        images = {}
        depth_maps = {}

        for name in self.cameras:
            before_camread_t = time.perf_counter()
            camera_data = self.cameras[name].async_read()

            # Handle cameras with depth enabled
            if hasattr(self.cameras[name], 'use_depth') and self.cameras[name].use_depth:
                if isinstance(camera_data, tuple):
                    color_image, depth_map = camera_data
                    images[name] = torch.from_numpy(color_image)
                    depth_maps[name] = torch.from_numpy(depth_map)
                else:
                    images[name] = torch.from_numpy(camera_data)
            else:
                images[name] = torch.from_numpy(camera_data)
                
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter(
            ) - before_camread_t

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        # obs_dict["observation.wrist_cam_pose"] = cam_pose
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
            if name in depth_maps:
                obs_dict[f"observation.depth.{name}"] = depth_maps[name]


        return obs_dict, action_dict

    def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

    def get_state_and_cam(self):
        import scipy.spatial.transform as st
        states = self.flexiv.states()
        arm_state = states.q
        arm_state = torch.as_tensor(list(arm_state))[:self.DOF]
        
        gripper_width = torch.tensor([self.get_gripper_state()])
        
        state = torch.cat((arm_state, gripper_width))
        cam_pose = getattr(states, "cam_pose", states.tcp_pose)
        pos, quat = cam_pose[:3], cam_pose[3:]
        print("quat", quat)
        pos = np.array(pos)
        rot = st.Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        rot = rot.as_matrix()
        T = np.eye(4)
        T[0, 3] = pos[0]
        T[1, 3] = pos[1]
        T[2, 3] = pos[2]
        T[:3, :3] = rot
        print("T", T)
        return state, T




    def get_state(self):
        states = self.flexiv.states()
        arm_state = states.q
        arm_state = torch.as_tensor(list(arm_state))[:self.DOF]
        
        gripper_width = torch.tensor([self.get_gripper_state()])

        # print(f"arm: {arm_state.shape}")
        # print(f"gripper_width: {gripper_width.shape}")
        
        state = torch.cat((arm_state, gripper_width))
        return state
    
    def readCamPose(self):
        import scipy.spatial.transform as st
        states = self.flexiv.states()
        cam_pose = getattr(states, "cam_pose", states.tcp_pose)
        pos, quat = cam_pose[:3], cam_pose[3:]
        print("quat", quat)
        pos = np.array(pos)
        rot = st.Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        rot = rot.as_matrix()
        T = np.eye(4)
        T[0, 3] = pos[0]
        T[1, 3] = pos[1]
        T[2, 3] = pos[2]
        T[:3, :3] = rot
        return T

    def get_gripper_state(self) -> float:
        return round(self.gripper.states().width, 2)

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        # before_read_t = time.perf_counter()
        state = self.get_state()
        # self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # if self.state_keys is None:
        #     self.state_keys = list(state)

        # Capture images from cameras

        images = {}
        depth_maps = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            camera_data = self.cameras[name].async_read()

            # Handle cameras with depth enabled
            if hasattr(self.cameras[name], 'use_depth') and self.cameras[name].use_depth:
                if isinstance(camera_data, tuple):
                    color_image, depth_map = camera_data
                    images[name] = torch.from_numpy(color_image)
                    depth_maps[name] = torch.from_numpy(depth_map)
                else:
                    images[name] = torch.from_numpy(camera_data)
            else:
                images[name] = torch.from_numpy(camera_data)
                
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter(
            ) - before_camread_t

        # Populate output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
            if name in depth_maps:
                obs_dict[f"observation.depth.{name}"] = depth_maps[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        # if self.teleop is None:
        #     self.teleop = GamePadTeleop(robot_instance=False)
        #     self.teleop.startup(robot=self)

        # if self.action_keys is None:
        #     dummy_action = self.teleop.gamepad_controller.get_state()
        #     self.action_keys = list(dummy_action.keys())

        action = action[:self.DOF]
        gripper_width = action[-1]

        before_write_t = time.perf_counter()

        eef_position = self.teleop.get_lead_arm_eef()
        self.apply_teleop_target(action, gripper_width, eef_position[0])

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # TODO(aliberts): return action_sent when motion is limited
        return action

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def stop(self) -> None:
        # Placeholder used by disconnect and remote runtimes. Flexiv RDK itself does
        # not require a local background teleop thread to be stopped here.
        return None

    def is_eef_within_workspace(self, eef_pos) -> bool:
        if eef_pos is None:
            return True

        # Relaxed bounds to allow free movement of Koch arm
        lowcost_lower_bound = np.array([-0.5, -0.5, -0.5])
        lowcost_upper_bound = np.array([0.5, 0.5, 0.5])
        return not (np.any(eef_pos > lowcost_upper_bound) or np.any(eef_pos < lowcost_lower_bound))

    def apply_teleop_target(self, action, gripper_width, eef_pos=None) -> bool:
        action_valid = (action is not None) and (gripper_width is not None) and self.is_eef_within_workspace(
            eef_pos
        )

        if eef_pos is not None and not self.is_eef_within_workspace(eef_pos):
            logger.warning(f"Out of bound! {eef_pos}")

        if self.flexiv.fault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if not action_valid:
            return False

        act_dof = torch.tensor(action)
        try:
            action = act_dof[:self.DOF].tolist()
            self.flexiv.SendJointPosition(
                action, self.target_vel, self.MAX_VEL, self.MAX_ACC
            )
            self.gripper.Move(float(gripper_width), 0.1, 20)
        except Exception as e:
            self.log.error(str(e))
            return False

        return True

    # def teleop_safety_stop(self) -> None:
    #     if self.teleop is not None:
    #         self.teleop._safety_stop(robot=self)

    def disconnect(self) -> None:
        self.stop()
        if self.teleop is not None:
            if hasattr(self.teleop, "gamepad_controller") and self.teleop.gamepad_controller is not None:
                self.teleop.gamepad_controller.stop()
            if hasattr(self.teleop, "disconnect"):
                self.teleop.disconnect()
            elif hasattr(self.teleop, "stop"):
                self.teleop.stop()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
