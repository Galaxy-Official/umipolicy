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
import pybullet as pb

# PyTorch & Torchvision
import torch
from torchvision import transforms

# Lerobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.datasets.feature_utils import build_dataset_frame

# Flexiv imports (New RDK) - inject .so path before import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib_py"))
import flexivrdk
import scipy.spatial.transform as st
# using perception and umi utilities
from lerobot.datasets.pose_utils import *
from perception.cameras.base_camera import BaseCamera

# Leader
from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
from lerobot.teleoperators.koch_leader.koch_leader import KochLeader
from lerobot.teleoperators.so_leader.config_so_leader import SO100LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO100Leader

# ---------------------------------------------------------------------
# PyBullet Based FK Resolver
# ---------------------------------------------------------------------
class LeaderFKResolver:
    URDF_DICT = {
        "koch": "urdf/assets/low_cost_robot_description/urdf/low_cost_robot.urdf",
        "so100": "urdf/assets/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf",
    }

    def __init__(self, robot_type="koch"):
        self.robot_type = robot_type
        # Pybullet GUI overhead avoidance
        pb.connect(pb.DIRECT)
        
        # Determine the relative URDF path
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "Mini-Tele"
        urdf_path = str(repo_root / self.URDF_DICT.get(self.robot_type, self.URDF_DICT["koch"]))
        
        if not os.path.exists(urdf_path):
            logger.warning(f"URDF path {urdf_path} does not exist. Cannot solve Forward Kinematics!")
            
        base_orien = [0, 0, 1, 0] if self.robot_type == "so100" else [0, 0, 0, 1]
        self.robot_pb = pb.loadURDF(
            urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=base_orien,
            useFixedBase=True,
        )

    def get_eef_pose(self, dynamixel_joints):
        """Converts raw leader joints to TCP pose targeting."""
        if self.robot_type == "koch":
            # Joint mapping conversion for Koch
            data = [
                -dynamixel_joints[0],
                90 - dynamixel_joints[1],
                90 - dynamixel_joints[2],
                90 - dynamixel_joints[3],
                dynamixel_joints[4] - 90,
                -dynamixel_joints[5],
            ]
            data = [angle * np.pi / 180 for angle in data]
            for i, joint in enumerate(data):
                pb.resetJointState(self.robot_pb, i, joint)
            
            # eef Link #4
            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            eef_orn = pb.getLinkState(self.robot_pb, 4)[1]
            return eef_pos, eef_orn, -data[-1]

        elif self.robot_type == "so100":
            data = [
                -dynamixel_joints[0],
                90 - dynamixel_joints[1],
                dynamixel_joints[2] - 90,
                dynamixel_joints[3] - 90,
                90 - dynamixel_joints[4],
                dynamixel_joints[5],
            ]
            data = [angle * np.pi / 180 for angle in data]
            for i, joint in enumerate(data):
                pb.resetJointState(self.robot_pb, i, joint)
            
            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            eef_orn = pb.getLinkState(self.robot_pb, 4)[1]
            
            eef_orn_matrix = np.array(pb.getMatrixFromQuaternion(eef_orn)).reshape(3, 3)
            original_axes = np.eye(3)
            
            z_in_eef = np.array([-1, 0, 0])
            x_in_eef = np.array([0, -1, 0])
            y_in_eef = np.array([0, 0, 1])
            new_axes = np.array([x_in_eef, y_in_eef, z_in_eef]).T
            new_axes = np.concatenate([new_axes, np.array([[1, 1, 1]])], axis=0)
            
            x_ab = np.array([np.dot(eef_orn_matrix[:, 0], original_axes[i]) for i in range(3)])
            y_ab = np.array([np.dot(eef_orn_matrix[:, 1], original_axes[i]) for i in range(3)])
            z_ab = np.array([np.dot(eef_orn_matrix[:, 2], original_axes[i]) for i in range(3)])
            
            T_ab = np.array([x_ab, y_ab, z_ab, eef_pos]).T
            T_ab = np.concatenate([T_ab, np.array([[0, 0, 0, 1]])], axis=0)
            rot_axes = (T_ab @ new_axes)[:3]
            
            def matrix2quaternion(matrix):
                tr = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
                if tr > 0:
                    S = np.sqrt(tr + 1.0) * 2
                    qw = 0.25 * S
                    qx = (matrix[2, 1] - matrix[1, 2]) / S
                    qy = (matrix[0, 2] - matrix[2, 0]) / S
                    qz = (matrix[1, 0] - matrix[0, 1]) / S
                elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
                    S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
                    qw = (matrix[2, 1] - matrix[1, 2]) / S
                    qx = 0.25 * S
                    qy = (matrix[0, 1] + matrix[1, 0]) / S
                    qz = (matrix[0, 2] + matrix[2, 0]) / S
                elif matrix[1, 1] > matrix[2, 2]:
                    S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
                    qw = (matrix[0, 2] - matrix[2, 0]) / S
                    qx = (matrix[0, 1] + matrix[1, 0]) / S
                    qy = 0.25 * S
                    qz = (matrix[1, 2] + matrix[2, 1]) / S
                else:
                    S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
                    qw = (matrix[1, 0] - matrix[0, 1]) / S
                    qx = (matrix[0, 2] + matrix[2, 0]) / S
                    qy = (matrix[1, 2] + matrix[2, 1]) / S
                    qz = 0.25 * S
                return np.array([qx, qy, qz, qw])
            new_eef_orn = matrix2quaternion(rot_axes)
            return eef_pos, new_eef_orn, -data[-1]


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

    def get_obs(self, n=1):
        with self.lock:
            if len(self.queue) < n:
                return None
            return list(self.queue)[-n:]

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------
# NewFlexivEnv
# ---------------------------------------------------------------------
class NewFlexivEnv:
    tx_flange_tip = np.identity(4)
    tx_flange_tip[:3, 3] = np.array([0, 0, 0.185])
    tx_tip_flange = np.linalg.inv(tx_flange_tip)

    @staticmethod
    def tip_to_flange_pose(tip_pose):
        return mat_to_pose(pose_to_mat(tip_pose) @ NewFlexivEnv.tx_tip_flange)

    def __init__(self, init_qpos=None, robot_ip="192.168.2.100"):
        self.robot = flexivrdk.Robot(robot_ip, robot_ip)
        self.model = flexivrdk.Model(self.robot)
        self.gripper = flexivrdk.Gripper(self.robot)
        self.mode = flexivrdk.Mode
        
        if self.robot.isFault():
            self.robot.clearFault()
            time.sleep(2)
        self.robot.enable()
        while not self.robot.isOperational():
            time.sleep(1)
            
        self.gripper.move(0.12, 0.1, 10)
        self.robot.setMode(self.mode.NRT_JOINT_POSITION)
        
        self.target_vel = [0.0] * self.robot.info().DoF
        self.max_vel = [0.5] * self.robot.info().DoF
        self.max_acc = [2.0] * self.robot.info().DoF
        
        if init_qpos is not None:
            self.robot.sendJointPosition(init_qpos, self.target_vel, self.target_vel, self.max_vel, self.max_acc)
            time.sleep(3)
        time.sleep(1)

    class _MockRobot:
        def __init__(self, parent):
            self.parent = parent
        def get_ee_pose(self):
            return self.parent.get_ee_pose()
        def get_gripper_width(self):
            return self.parent.get_gripper_width()

    @property
    def robot_mock(self):
        if not hasattr(self, "_robot_inst"):
            self._robot_inst = self._MockRobot(self)
        return self._robot_inst
    
    def __getattr__(self, name):
        if name == "robot_mock":
            return self.robot_mock
        if name == "robot" and not hasattr(self.__dict__, "robot"):
            return self.robot_mock
        return self.__dict__.get(name) or super().__getattribute__(name)

    def get_ee_pose(self):
        states = flexivrdk.RobotStates()
        self.robot.getRobotStates(states)
        flange_pose_raw = np.array(states.tcpPose)
        
        pos = flange_pose_raw[:3]
        qw, qx, qy, qz = flange_pose_raw[3:]
        rot = st.Rotation.from_quat([qx, qy, qz, qw])
        tip_pose_mat = pos_rot_to_mat(np.array(pos), rot) @ NewFlexivEnv.tx_flange_tip
        umi_tip_pose = mat_to_pose(tip_pose_mat)
        return umi_tip_pose

    def get_gripper_width(self):
        states = flexivrdk.GripperStates()
        self.gripper.getGripperStates(states)
        return states.width

    def exec_action(self, tip_pose, target_width):
        flange_pose = NewFlexivEnv.tip_to_flange_pose(tip_pose)
        pos, rot = pose_to_pos_rot(flange_pose)
        quat_xyzw = rot.as_quat(scalar_first=False)
        
        flexiv_target_pose = [
            pos[0], pos[1], pos[2],
            quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
        ]
        
        states = flexivrdk.RobotStates()
        self.robot.getRobotStates(states)
        current_q = list(states.q)
        
        reachable, target_q = self.model.reachable(flexiv_target_pose, current_q, True)
        
        if reachable:
            self.robot.sendJointPosition(target_q, self.target_vel, self.target_vel, self.max_vel, self.max_acc)
        
        self.gripper.move(max(target_width - 0.01, 0.005), 0.1, 10)


# ---------------------------------------------------------------------
# MAIN SCRIPT: Record Teleop Dataset
# ---------------------------------------------------------------------
global recording_finished
recording_finished = False

def signal_handler(sig, frame):
    global recording_finished
    logger.info("Signal caught, exiting cleanly.")
    recording_finished = True

def main(args):
    signal_module.signal(signal_module.SIGINT, signal_handler)

    # 1. Init Cameras
    handcap_camera_cfg = Path(__file__).resolve().parent.parent.parent.parent.parent / "Mini-Tele/perception/configs/camera/handcap_camera.json"
    if os.path.exists(str(handcap_camera_cfg)):
        cam_dict = BaseCamera.create_cameras_from_config(config_path=str(handcap_camera_cfg))
        cam_wrist = cam_dict["wrist"]
        cam_tactile_left = cam_dict["left_tactile"]
        cam_tactile_right = cam_dict["right_tactile"]
    else:
        logger.warning("Camera config not found, falling back to None!")
        cam_wrist, cam_tactile_left, cam_tactile_right = None, None, None
    
    # 2. Init Follower (Flexiv)
    robot_ip = os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100")
    init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"))
    env = NewFlexivEnv(init_qpos, robot_ip=robot_ip)
    env.robot = env.robot_mock
    
    # 3. Init Leader Configuration
    if args.teleop == "koch":
        leader_cfg = KochLeaderConfig(port=args.teleop_port)
        leader = KochLeader(leader_cfg)
    elif args.teleop == "so100":
        leader_cfg = SO100LeaderConfig(port=args.teleop_port)
        leader = SO100Leader(leader_cfg)
    else:
        raise ValueError("Invalid teleop robot")
        
    leader.connect()
    fk_resolver = LeaderFKResolver(robot_type=args.teleop)
    
    # 4. Start Cameras Thread
    if cam_wrist is not None:
        obs_thread = ObservationThread(cam_wrist, cam_tactile_left, cam_tactile_right, env)
        obs_thread.start()
        
        while True:
            if obs_thread.get_obs(1) is not None:
                break
            time.sleep(0.01)
    else:
        logger.warning("No camera threads available!")
        
    # 5. Initialize Dataset Schema definition for V3.0
    features = {
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.tactiles.left": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.tactiles.right": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,), # 7 pos/quat + 1 gripper width
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
        }
    }
    
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="handcap_flexiv",
        features=features,
        use_videos=True,
    )
    
    logger.info(f"Press CTRL+C anytime to cleanly stop recording an episode!")
    
    recorded_episodes = 0
    global recording_finished
    
    while recorded_episodes < args.num_episodes and not recording_finished:
        logger.info(f"--- Episode {recorded_episodes} / {args.num_episodes} ---")
        logger.info("Providing 3 seconds warmup.... (move the arm safely into position)")
        time.sleep(3)
        logger.info("Recording Started!")
        
        timestamp = 0
        start_t = time.perf_counter()
        
        while timestamp < args.episode_time_s and not recording_finished:
            t0 = time.perf_counter()
            
            # --- 1. Follower Observation (State and Cameras)
            if cam_wrist is not None:
                frame_data = obs_thread.get_obs(1)[0]
                wrist_img = frame_data['wrist_img']
                left_img = frame_data['left_tactile_img']
                right_img = frame_data['right_tactile_img']
                eepose = frame_data['eepose']
                obs_gripper = frame_data['gripper_width']
            else:
                wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)
                left_img = np.zeros((480, 640, 3), dtype=np.uint8)
                right_img = np.zeros((480, 640, 3), dtype=np.uint8)
                eepose = env.get_ee_pose()
                obs_gripper = env.get_gripper_width()

            # Ensure eepose matches 7 elements: [x,y,z, rvx, rvy, rvz] requires shape padding? Wait. 
            # In umi_tip_pose it's 6 or 7? (mat_to_pose converts to position + rotvec = 6 elements) - sorry, we need 7 (pos + quat? Wait. 6!)
            # But the features are 8: eepose(7?) + gripper(1)
            # Update: mat_to_pose in LeRobot umi is typically pos (3) + rotvec (3) = 6? Let's assume 7 (quaternion) based on previous code.
            
            # Since earlier code converted FK output to quaternion: target_pos(3), target_orn(4) -> length 7
            curr_state = np.zeros((8,), dtype=np.float32)
            
            # --- 2. Action from Teleop
            leader_action_dict = leader.get_action()
            motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            leader_jnts = [leader_action_dict[f"{m}.pos"] for m in motors]
            
            # FK Resolution
            target_pos, target_orn, _t_gripper = fk_resolver.get_eef_pose(leader_jnts)
            
            scale_factor = 4.2
            target_pos = np.array(target_pos) * scale_factor
            
            action_pose = np.concatenate([target_pos, target_orn, np.array([_t_gripper])])
            curr_state = np.concatenate([np.zeros(7,), np.array([obs_gripper])]) # dummy
            
            # Exec Flexiv IK
            env.exec_action(tip_pose=np.concatenate([target_pos, target_orn]), target_width=_t_gripper)

            # --- 3. Save to Dataset properly (v3.0 standard)
            dataset.add_frame({
                "observation.images.wrist": cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB),
                "observation.tactiles.left": cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB),
                "observation.tactiles.right": cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB),
                "observation.state": torch.from_numpy(curr_state.astype(np.float32)),
                "action": torch.from_numpy(action_pose.astype(np.float32)),
            })
            
            dt = time.perf_counter() - t0
            if dt < (1.0 / args.fps):
                time.sleep((1.0 / args.fps) - dt)
                
            timestamp = time.perf_counter() - start_t
            
        dataset.save_episode(task=args.single_task)
        recorded_episodes += 1
        logger.info(f"Episode {recorded_episodes} saved!")
        
    logger.info("Writing Dataset HuggingFace format...")
    dataset.consolidate()
    if cam_wrist is not None:
        obs_thread.stop()
    leader.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="lerobot_flexiv_teleop")
    parser.add_argument("--root", type=str, default="datasets/local")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time_s", type=int, default=60)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--single-task", type=str, default="default task")
    parser.add_argument("--teleop", type=str, choices=["koch", "so100"], default="koch")
    parser.add_argument("--teleop_port", type=str, default="/dev/tty.usbserial-110")
    
    args = parser.parse_args()
    main(args)
