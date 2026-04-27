import os
import time
import socket
import numpy as np
import flexivrdk
import scipy.spatial.transform as st
from loguru import logger
from .pose_util import pos_rot_to_pose, pose_to_pos_rot

class FlexivEnv:
    def __init__(self, init_qpos, obs_horizon=2, robot_ip="192.168.2.100", local_ip="192.168.2.102", use_gripper_width_mapping=False, pose_type="rotvec"):
        self.obs_horizon = obs_horizon
        self.pose_type = pose_type
        self.init_qpos = init_qpos
        
        # New RDK 1.0+ Native Setup
        robot_sn = os.environ.get("FLEXIV_ROBOT_SN", "Rizon4-062339")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((robot_ip, 1))
            actual_local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            actual_local_ip = local_ip

        self.robot = flexivrdk.Robot(robot_sn, [actual_local_ip])
        logger.info("Initializing Gripper API...")
        self.gripper = flexivrdk.Gripper(self.robot)
        logger.info("Initializing Model API...")
        self.model = flexivrdk.Model(self.robot)
        
        gripper_name = os.environ.get("FLEXIV_GRIPPER_NAME", "Flexiv-GN01")
        try:
            logger.info(f"Enabling gripper [{gripper_name}]...")
            self.gripper.Enable(gripper_name)
        except Exception as e:
            logger.warning(f"Failed to enable gripper [{gripper_name}]: {e}")
            
        logger.info("Checking faults...")
        if self.robot.fault():
            self.robot.ClearFault()
            time.sleep(2)
        logger.info("Enabling robot...")
        self.robot.Enable()
        logger.info("Waiting for operational status...")
        while not self.robot.operational():
            time.sleep(1)
            
        logger.info("Switching to NRT_CARTESIAN_MOTION_FORCE mode...")
        self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SetForceControlAxis([False, False, False, False, False, False])
        
        max_width = self.gripper.params().max_width
        self.gripper.Move(max_width, 0.1, 20)
        time.sleep(1)

    def get_ee_pose(self):
        pose = self.robot.states().tcp_pose
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
        rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
        return pos_rot_to_pose(np.array(pose[:3]), rot)

    def get_gripper_width(self):
        return self.gripper.states().width

    def reset(self):
        logger.info("Resetting robot to initial joint positions...")
        self.robot.SendJointPosition(self.init_qpos, [0]*7, [0.1]*7, [0.1]*7)
        max_width = self.gripper.params().max_width
        self.gripper.Move(max_width, 0.1, 20)
        time.sleep(10) # Reduced from 15 to 10 for inference

    def exec_actions(self, actions, timestamps):
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        
        for i in range(len(new_actions)):
            tip_pose = new_actions[i, 0:6]
            target_width = new_actions[i, 6]
            
            # Format target TCP pose to [x, y, z, qw, qx, qy, qz]
            pos, rot = pose_to_pos_rot(tip_pose)
            quat = rot.as_quat(scalar_first=False) # x,y,z,w
            target_tcp = [pos[0], pos[1], pos[2], quat[3], quat[0], quat[1], quat[2]]
            
            # --- Safety Boundary Clip ---
            from .flexiv_safety import clip_target_pose_7d
            target_tcp = clip_target_pose_7d(target_tcp)
            
            # Directly send Cartesian pose (Internal IK handled by RDK)
            self.robot.SendCartesianMotionForce(target_tcp)
            
            max_w = self.gripper.params().max_width
            safe_width = min(max(target_width, 0.001), max_w - 0.001)
            self.gripper.Move(safe_width, 0.1, 20)
            
            dt = new_timestamps[i] - time.time()
            # Removed time.sleep(dt) to make exec_actions non-blocking. 
            # Timing is handled by precise_wait in the main control loop.
