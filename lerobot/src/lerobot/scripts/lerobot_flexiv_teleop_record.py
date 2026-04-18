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
import pybullet_data

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
from lerobot.scripts.umi_realworld.flexiv_controller import FlexivInterface
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

    def matrix2quaternion(self, matrix):
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
    def __init__(self, robot_type="koch"):
        self.robot_type = robot_type
        # Pybullet GUI overhead avoidance
        pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        URDF_DICT = {
            "koch": "urdf/assets/low_cost_robot_description/urdf/low_cost_robot.urdf",
            "so100": "urdf/assets/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf",
        }
        urdf_path = URDF_DICT.get(self.robot_type, None)
        if urdf_path is None:
            raise ValueError(f"Unknown URDF for {self.robot_type}")
            
        base_orien = [0, 0, 1, 0] if self.robot_type == "so100" else [0, 0, 0, 1]
        self.robot_pb = pb.loadURDF(
            urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=base_orien,
            useFixedBase=True,
        )

        flexiv_urdf_path = "urdf/assets/leap_hand/flexiv_leap.urdf"
        self.follow_arm = pb.loadURDF(
            flexiv_urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0.7071068, 0.7071068],
            useFixedBase=True,
        )

    def get_target_joints(self, dynamixel_joints, current_flexiv_joints=None):
        """Calculates IK to return 7 joint angles for Rizon 4."""
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
            gripper = -data[-1]

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
            gripper = data[5]

        # Calculate IK using follow_arm
        eef_pos = eef_pos * 4.2
        
        # VERY IMPORTANT: Synchronize PyBullet's simulated follower arm with the REAL physical arm's position
        # BEFORE running Inverse Kinematics! This prevents IK branch jumping/snapping!
        if current_flexiv_joints is not None:
            for i, joint in enumerate(current_flexiv_joints):
                pb.resetJointState(self.follow_arm, i, float(joint))
                
        joints_tuple = pb.calculateInverseKinematics(
            self.follow_arm, 8, eef_pos, new_eef_orn
        )
        
        joints = list(joints_tuple[:7])
        for i, joint in enumerate(joints):
            pb.resetJointState(self.follow_arm, i, joint)
            
        return joints, gripper


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
            
            left_tactile_img = None
            right_tactile_img = None
            if self.cam_tactile_left is not None:
                left_tactile_img, _ = self.cam_tactile_left.get_data()
            if self.cam_tactile_right is not None:
                right_tactile_img, _ = self.cam_tactile_right.get_data()
            
            eepose = self.env.robot.get_ee_pose()
            gripper_width = self.env.robot.get_gripper_width()
            
            with self.lock:
                self.queue.append({
                    'wrist_img': wrist_img.copy() if wrist_img is not None else wrist_img,
                    'left_tactile_img': left_tactile_img.copy() if left_tactile_img is not None else None,
                    'right_tactile_img': right_tactile_img.copy() if right_tactile_img is not None else None,
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
# NewFlexivEnv (RDK 0.9 via FlexivInterface - matching inference script)
# ---------------------------------------------------------------------
class NewFlexivEnv:
    def __init__(self, init_qpos=None, robot_ip="192.168.2.100", local_ip="192.168.2.102"):
        self.init_qpos = init_qpos
        
        # RDK 0.9 Wrapper via FlexivInterface
        self.robot = FlexivInterface(
            robot_ip=robot_ip,
            local_ip=local_ip,
            move_home=False,
            init_offset=None,
            init_qpos=None,
            use_gripper_width_mapping=False
        )
        self.robot.send_gripper_state(0.12, 0.1, 10)
        time.sleep(1)
        
        if init_qpos is not None:
            self.robot.send_joint_position(init_qpos)
            time.sleep(3)

    def reset(self):
        if self.init_qpos is not None:
            self.robot.send_joint_position(self.init_qpos)
            self.robot.send_gripper_state(0.12, 0.1, 10)
            time.sleep(5)

    def exec_action(self, tip_pose, target_width):
        flange_pose = FlexivInterface.tip_to_flange_pose(tip_pose)
        self.robot.send_flange_pose(flange_pose)
        self.robot.send_gripper_state(max(target_width - 0.01, 0.005), 0.1, 10)
        
    def exec_action_joints(self, target_joints, target_width):
        self.robot.send_joint_position(target_joints)
        self.robot.send_gripper_state(max(target_width - 0.01, 0.005), 0.1, 10)


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
    # 1. Init Cameras (MVS only or with tactile depending on flag)
    logger.info(f"Initializing cameras (use_tactile={args.use_tactile})...")
    cam_cfg_name = "handcap_camera.json" if args.use_tactile else "handcap_camera_no_tactile.json"
    possible_cam_paths = [
        Path(__file__).resolve().parent.parent.parent / f"perception/configs/camera/{cam_cfg_name}",
        Path(__file__).resolve().parent.parent.parent.parent.parent / f"Mini-Tele/perception/configs/camera/{cam_cfg_name}",
    ]
    handcap_camera_cfg = None
    for p in possible_cam_paths:
        if p.exists():
            handcap_camera_cfg = p
            break
    
    cam_wrist, cam_tactile_left, cam_tactile_right = None, None, None
    if handcap_camera_cfg is not None:
        logger.info(f"Using camera config: {handcap_camera_cfg}")
        cam_dict = BaseCamera.create_cameras_from_config(config_path=str(handcap_camera_cfg))
        cam_wrist = cam_dict["wrist"]
        cam_tactile_left = cam_dict.get("left_tactile")
        cam_tactile_right = cam_dict.get("right_tactile")
    else:
        logger.warning(f"Camera config not found at any of: {possible_cam_paths}, falling back to None!")
    
    # 2. Init Follower (Flexiv via RDK 0.9)
    robot_ip = os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100")
    local_ip = os.environ.get("FLEXIV_LOCAL_IP", "192.168.2.102")
    init_qpos = eval(os.environ.get("FLEXIV_INIT_POSE", "[-0.0, -0.698, -0.0, 1.571, -0.0, 0.698, -0.0]"))
    env = NewFlexivEnv(init_qpos, robot_ip=robot_ip, local_ip=local_ip)
    
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
            "shape": (1536, 1536, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,), # 7 pos/quat + 1 gripper width
        },
        "action": {
            "dtype": "float32",
            "shape": (8,), # 7 joints + 1 gripper
        }
    }
    if args.use_tactile:
        features["observation.tactiles.left"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }
        features["observation.tactiles.right"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }
    
    dataset_path = Path(args.root) / args.repo_id
    if dataset_path.exists():
        logger.warning(f"\n"
                       f"=========================================\n"
                       f"⚠️ Dataset already exists at: {dataset_path}\n"
                       f"=========================================\n")
        while True:
            choice = input("Do you want to [D]elete, [A]ppend, or [Q]uit? (D/A/Q): ").strip().lower()
            if choice == 'd':
                import shutil
                logger.info(f"Deleting dataset at {dataset_path}...")
                shutil.rmtree(dataset_path, ignore_errors=True)
                break
            elif choice == 'a':
                logger.info("Proceeding to append to the existing dataset...")
                break
            elif choice == 'q':
                logger.info("Exiting pipeline at user request.")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter D, A, or Q.")
                
    if dataset_path.exists() and (dataset_path / "meta" / "info.json").exists():
        dataset = LeRobotDataset(repo_id=args.repo_id, root=str(dataset_path))
    else:
        # Fallback delete just in case it exists but has no meta/info.json (corrupted)
        import shutil
        shutil.rmtree(dataset_path, ignore_errors=True)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=str(dataset_path),
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
                eepose = frame_data['eepose']
                obs_gripper = frame_data['gripper_width']
            else:
                wrist_img = np.zeros((1536, 1536, 3), dtype=np.uint8)
                eepose = env.robot.get_ee_pose()
                obs_gripper = env.robot.get_gripper_width()

            curr_state = np.zeros((8,), dtype=np.float32)
            arm_state = env.robot.get_joint_positions()
            curr_state = np.concatenate([arm_state, np.array([obs_gripper])])
            
            # --- 2. Action from Teleop
            leader_action_dict = leader.get_action()
            motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            leader_jnts = [leader_action_dict[f"{m}.pos"] for m in motors]
            
            # FK Resolution -> PyBullet internal IK -> Joint Actions
            # We seed it with `arm_state` so PyBullet doesn't jump to strange branches
            target_joints, _t_gripper = fk_resolver.get_target_joints(leader_jnts, current_flexiv_joints=arm_state)
            action_state = np.concatenate([target_joints, np.array([_t_gripper])])
            
            # Exec Flexiv Joint Position
            env.exec_action_joints(target_joints=target_joints, target_width=_t_gripper)

            # --- 3. Save to Dataset properly (v3.0 standard)
            frame_dict = {
                "observation.images.wrist": cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB),
                "observation.state": torch.from_numpy(curr_state.astype(np.float32)),
                "action": torch.from_numpy(action_state.astype(np.float32)),
                "task": args.single_task,
            }
            if args.use_tactile:
                left_img = frame_data.get('left_tactile_img') if cam_wrist is not None else None
                right_img = frame_data.get('right_tactile_img') if cam_wrist is not None else None
                frame_dict["observation.tactiles.left"] = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) if left_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                frame_dict["observation.tactiles.right"] = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB) if right_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            dataset.add_frame(frame_dict)
            
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
    parser.add_argument("--use_tactile", action="store_true", help="Enable tactile cameras")
    
    args = parser.parse_args()
    main(args)
