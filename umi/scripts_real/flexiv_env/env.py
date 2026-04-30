import os
import time
import socket
import numpy as np
import flexivrdk
import scipy.spatial.transform as st
from loguru import logger
from .pose_util import pos_rot_to_pose, pose_to_pos_rot

class FlexivEnv:
    def __init__(
            self, init_qpos, obs_horizon=2, robot_ip="192.168.2.100",
            local_ip="192.168.2.102", use_gripper_width_mapping=False,
            pose_type="rotvec", arm_max_linear_vel=0.05,
            arm_max_angular_vel=0.2, arm_max_linear_acc=0.1,
            arm_max_angular_acc=0.3, gripper_move_velocity=0.03,
            gripper_move_force=20):
        self.obs_horizon = obs_horizon
        self.pose_type = pose_type
        self.init_qpos = init_qpos
        self.arm_max_linear_vel = float(arm_max_linear_vel)
        self.arm_max_angular_vel = float(arm_max_angular_vel)
        self.arm_max_linear_acc = float(arm_max_linear_acc)
        self.arm_max_angular_acc = float(arm_max_angular_acc)
        self.gripper_move_velocity = float(gripper_move_velocity)
        self.gripper_move_force = float(gripper_move_force)
        logger.info(
            "Flexiv motion limits: max_linear_vel={} max_angular_vel={} "
            "max_linear_acc={} max_angular_acc={} gripper_vel={} gripper_force={}",
            self.arm_max_linear_vel,
            self.arm_max_angular_vel,
            self.arm_max_linear_acc,
            self.arm_max_angular_acc,
            self.gripper_move_velocity,
            self.gripper_move_force)
        
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
        self.gripper.Move(max_width, self.gripper_move_velocity, self.gripper_move_force)
        time.sleep(1)

    def get_ee_pose(self):
        pose = self.robot.states().tcp_pose
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
        rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
        return pos_rot_to_pose(np.array(pose[:3]), rot)

    def get_gripper_width(self):
        return self.gripper.states().width

    def reset(self):
        # Temporarily switch to joint mode for reset
        logger.info("Switching to NRT_JOINT_POSITION for reset...")
        self.robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        
        logger.info("Resetting robot to initial joint positions...")
        self.robot.SendJointPosition(self.init_qpos, [0]*7, [0.1]*7, [0.1]*7)
        
        max_width = self.gripper.params().max_width
        self.gripper.Move(max_width, self.gripper_move_velocity, self.gripper_move_force)
        time.sleep(10) 
        
        # Switch back to Cartesian mode for subsequent actions
        logger.info("Switching back to NRT_CARTESIAN_MOTION_FORCE...")
        self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SetForceControlAxis([False, False, False, False, False, False])

    def exec_actions(self, actions, timestamps):
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        
        if len(new_actions) > 0:
            # Execute the last target in the chunk to avoid constant re-planning stutter
            i = len(new_actions) - 1
            tip_pose = new_actions[i, 0:6]
            target_width = new_actions[i, 6]
            
            # Format target TCP pose to [x, y, z, qw, qx, qy, qz]
            pos, rot = pose_to_pos_rot(tip_pose)
            quat = rot.as_quat(scalar_first=False) # x,y,z,w
            target_tcp = [pos[0], pos[1], pos[2], quat[3], quat[0], quat[1], quat[2]]
            
            # --- Safety Boundary Clip ---
            from .flexiv_safety import clip_target_pose_7d
            target_tcp = clip_target_pose_7d(target_tcp)
            
            # Reduced velocity and acceleration scales for slower, smoother motion
            # Signature: SendCartesianMotionForce(pose, wrench, velocity, max_linear_vel, max_angular_vel, max_linear_acc, max_angular_acc)
            self.robot.SendCartesianMotionForce(
                target_tcp, [0]*6, [0]*6,
                self.arm_max_linear_vel,
                self.arm_max_angular_vel,
                self.arm_max_linear_acc,
                self.arm_max_angular_acc)
            
            max_w = self.gripper.params().max_width
            safe_width = min(max(target_width, 0.001), max_w - 0.001)
            self.gripper.Move(safe_width, self.gripper_move_velocity, self.gripper_move_force)
            
            dt = new_timestamps[i] - time.time()
            # Removed time.sleep(dt) to make exec_actions non-blocking. 
            # Timing is handled by precise_wait in the main control loop.
