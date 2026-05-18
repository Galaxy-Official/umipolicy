import os
import time
import socket
import numpy as np
import flexivrdk
import scipy.spatial.transform as st
from loguru import logger
from lerobot.datasets.pose_utils import pose_to_mat, pose_to_pos_rot, pos_rot_to_mat, pos_rot_to_pose


def pose7_to_mat(pose7):
    pose = np.asarray(pose7, dtype=np.float64)
    pos = pose[:3]
    qw, qx, qy, qz = pose[3:]
    return pos_rot_to_mat(pos, st.Rotation.from_quat([qx, qy, qz, qw]))


def mat_to_tcp_pose7(mat):
    pos = np.asarray(mat[:3, 3], dtype=np.float64)
    quat = st.Rotation.from_matrix(mat[:3, :3]).as_quat(scalar_first=False)
    return [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat[3]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
    ]

class FlexivEnv:
    def __init__(
        self,
        init_qpos,
        obs_horizon=2,
        robot_ip="192.168.2.100",
        local_ip="192.168.2.102",
        use_gripper_width_mapping=False,
        pose_type="rotvec",
        direct_eef_control=None,
        arm_max_linear_vel=0.05,
        arm_max_angular_vel=0.2,
        arm_max_linear_acc=0.1,
        arm_max_angular_acc=0.3,
        gripper_move_velocity=0.03,
        gripper_move_force=20,
    ):
        self.obs_horizon = obs_horizon
        self.pose_type = pose_type
        self.init_qpos = init_qpos
        if direct_eef_control is None:
            direct_eef_control = os.environ.get("FLEXIV_DIRECT_EEF_CONTROL", "0").lower() in (
                "1",
                "true",
                "yes",
            )
        self.direct_eef_control = bool(direct_eef_control)
        self.arm_max_linear_vel = float(os.environ.get("FLEXIV_ARM_MAX_LINEAR_VEL", arm_max_linear_vel))
        self.arm_max_angular_vel = float(os.environ.get("FLEXIV_ARM_MAX_ANGULAR_VEL", arm_max_angular_vel))
        self.arm_max_linear_acc = float(os.environ.get("FLEXIV_ARM_MAX_LINEAR_ACC", arm_max_linear_acc))
        self.arm_max_angular_acc = float(os.environ.get("FLEXIV_ARM_MAX_ANGULAR_ACC", arm_max_angular_acc))
        self.gripper_move_velocity = float(os.environ.get("FLEXIV_GRIPPER_MOVE_VELOCITY", gripper_move_velocity))
        self.gripper_move_force = float(os.environ.get("FLEXIV_GRIPPER_MOVE_FORCE", gripper_move_force))
        self.action_frame = os.environ.get("FLEXIV_ACTION_FRAME", "tcp").lower()
        if self.action_frame not in ("tcp", "flange"):
            logger.warning("Invalid FLEXIV_ACTION_FRAME={}, using tcp", self.action_frame)
            self.action_frame = "tcp"
        base_max_vel = float(os.environ.get("FLEXIV_BASE_MAX_VEL", "0.1"))
        base_max_acc = float(os.environ.get("FLEXIV_BASE_MAX_ACC", "0.1"))
        wrist_max_vel = float(os.environ.get("FLEXIV_WRIST_MAX_VEL", str(base_max_vel)))
        wrist_max_acc = float(os.environ.get("FLEXIV_WRIST_MAX_ACC", str(base_max_acc)))
        wrist_joint_count = int(os.environ.get("FLEXIV_WRIST_JOINT_COUNT", "3"))
        wrist_joint_count = max(0, min(7, wrist_joint_count))
        self.exec_max_vel = [base_max_vel] * 7
        self.exec_max_acc = [base_max_acc] * 7
        if wrist_joint_count:
            for joint_idx in range(7 - wrist_joint_count, 7):
                self.exec_max_vel[joint_idx] = wrist_max_vel
                self.exec_max_acc[joint_idx] = wrist_max_acc
        self.ik_free_orientation = os.environ.get("FLEXIV_IK_FREE_ORIENTATION", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        
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

        if self.direct_eef_control:
            logger.info("Switching to NRT_CARTESIAN_MOTION_FORCE mode...")
            self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            self.robot.SetForceControlAxis([False, False, False, False, False, False])
        else:
            logger.info("Switching to NRT_JOINT_POSITION mode...")
            self.robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        logger.info(f"Execution max velocity limits: {self.exec_max_vel}")
        logger.info(f"Execution max acceleration limits: {self.exec_max_acc}")
        logger.info(f"IK free orientation: {self.ik_free_orientation}")
        logger.info(
            "Direct EEF control: {} | action_frame={} | Cartesian limits: linear_vel={} angular_vel={} "
            "linear_acc={} angular_acc={} gripper_vel={} gripper_force={}",
            self.direct_eef_control,
            self.action_frame,
            self.arm_max_linear_vel,
            self.arm_max_angular_vel,
            self.arm_max_linear_acc,
            self.arm_max_angular_acc,
            self.gripper_move_velocity,
            self.gripper_move_force,
        )
        
        max_width = self.gripper.params().max_width
        self.gripper.Move(max_width, self.gripper_move_velocity, self.gripper_move_force)
        time.sleep(1)

    def get_ee_pose(self):
        states = self.robot.states()
        pose = states.flange_pose if self.action_frame == "flange" else states.tcp_pose
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
        rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
        return pos_rot_to_pose(np.array(pose[:3]), rot)

    def get_gripper_width(self):
        return self.gripper.states().width

    def action_pose_to_target_tcp(self, target_pose):
        target_pose_mat = pose_to_mat(target_pose)
        if self.action_frame == "tcp":
            return mat_to_tcp_pose7(target_pose_mat)

        states = self.robot.states()
        current_flange_mat = pose7_to_mat(states.flange_pose)
        current_tcp_mat = pose7_to_mat(states.tcp_pose)
        flange_to_tcp_mat = np.linalg.inv(current_flange_mat) @ current_tcp_mat
        target_tcp_mat = target_pose_mat @ flange_to_tcp_mat
        return mat_to_tcp_pose7(target_tcp_mat)

    def reset(self):
        if self.direct_eef_control:
            logger.info("Switching to NRT_JOINT_POSITION for reset...")
            self.robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        logger.info("Resetting robot to initial joint positions...")
        self.robot.SendJointPosition(self.init_qpos, [0]*7, [0.1]*7, [0.1]*7)
        max_width = self.gripper.params().max_width
        self.gripper.Move(max_width, self.gripper_move_velocity, self.gripper_move_force)
        time.sleep(10) # Reduced from 15 to 10 for inference
        if self.direct_eef_control:
            logger.info("Switching back to NRT_CARTESIAN_MOTION_FORCE...")
            self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            self.robot.SetForceControlAxis([False, False, False, False, False, False])

    def exec_actions(self, actions, timestamps):
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        
        for i in range(len(new_actions)):
            tip_pose = new_actions[i, 0:6]
            target_width = new_actions[i, 6]
            
            target_tcp = self.action_pose_to_target_tcp(tip_pose)
            
            # --- Safety Boundary Clip ---
            from lerobot.common.robot_devices.robots.flexiv_safety import clip_target_pose_7d
            target_tcp = clip_target_pose_7d(target_tcp)
            
            if self.direct_eef_control:
                self.robot.SendCartesianMotionForce(
                    target_tcp,
                    [0.0] * 6,
                    [0.0] * 6,
                    self.arm_max_linear_vel,
                    self.arm_max_angular_vel,
                    self.arm_max_linear_acc,
                    self.arm_max_angular_acc,
                )
            else:
                # Native IK calculation using RDK Model API
                result = self.model.reachable(target_tcp, self.robot.states().q, self.ik_free_orientation)
                if result[0]:
                    self.robot.SendJointPosition(result[1], [0]*7, self.exec_max_vel, self.exec_max_acc)
                else:
                    logger.warning(f"Pose {target_tcp} is not reachable!")
            
            max_w = self.gripper.params().max_width
            safe_width = min(max(target_width, 0.001), max_w - 0.001)
            self.gripper.Move(safe_width, self.gripper_move_velocity, self.gripper_move_force)
            
            dt = new_timestamps[i] - time.time()
            if dt > 0:
                time.sleep(dt)
