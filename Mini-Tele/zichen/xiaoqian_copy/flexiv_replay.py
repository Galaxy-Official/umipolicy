from .players import TeleopPlayer
import numpy as np
import json
import gym
import time


import torch.nn.functional as F

# import pytorch3d.transforms as p3dt
from scipy.spatial.transform import Rotation
import numpy as np
from datetime import datetime
import copy
import cv2
import h5py
import pickle as pkl
import sys, os, glob

sys.path.append(os.path.join(os.path.dirname(__file__), "../lib_py"))

from ik import IKController
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from scipy.spatial.transform import Rotation as R

import torch
import threading
import flexivrdk

from typing import List
import yaml

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import get_retargeting_config, RetargetingConfig

sys.path.append("/home/rhos_pub/LeapTele/real-training")
from dataloader import *

from algos.bc import BC

# from algos.bc_gmm import BCGMM
# from algos.bc_rnn import BCRNN

# init Leap Motion Detector


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# =============== Hyper params BEGIN ======================
scaling_factor = 2  # scale leap hand motion -> ee pos motion
sigma_scaling_factor = 2.5
leap_lower_bound = np.array([-0.2, 0, -0.1])
leap_upper_bound = np.array([0.2, 0.4, 0.35])
vr_lower_bound = np.array([-1, -1, 0])
vr_upper_bound = np.array([1, 1, 2])
isaacgym_lower_bound = torch.tensor([-2, -2, -2])
isaacgym_upper_bound = torch.tensor([2, 2, 2])

tcp_lower_bound = torch.tensor([0.45, -0.24, 0.03])  # used for real world robot
tcp_upper_bound = torch.tensor([0.87, 0.32, 0.60])

home_joint = torch.tensor([0, -40, 0, 90, 0, 40, 0])  # degree
home_joint = home_joint / 180.0 * 3.1415  # degree -> rad

frequency = 60  # Hz
robot_ip = "192.168.2.100"
local_ip = "192.168.2.102"
log = flexivrdk.Log()
mode = flexivrdk.Mode
test_on_real_robot = True
test_on_real_hand = True
save_realsense = True
robot = None
gripper = None
leap_hand = None
recording_data: bool = True
cam_agent_view = []
cam_wrist_view = []
cam_bird_view = []
cam_left_view = []
cam_right_view = []

keep_hand_pos = True

filepath = "/home/rhos_pub/Desktop/LeapPush/Flexiv_20240804_171436/003/data.pkl"
from .calibration.camera import CameraTactile

thumb_tactile = CameraTactile("/dev/video16")
index_tactile = CameraTactile("/dev/video19")
middle_tactile = CameraTactile("/dev/video12")
little_tactile = CameraTactile("/dev/video14")
# =============== Hyper params END ======================


def set_leap_hand_pos(leaphand, action, nail_handles_):
    # nail_handles_=np.asarray(nail_handles_).reshape(-1)
    set_pose = np.zeros(16)
    for i, key in enumerate([12, 22, 27]):
        tip = key
        tip = tip - 1
        finger_tip = tip - 1
        dip = tip - 2
        pip = tip - 3

        set_pose[i * 4] = action[dip]
        set_pose[i * 4 + 1] = action[pip]
        set_pose[i * 4 + 2] = action[finger_tip]
        set_pose[i * 4 + 3] = action[tip]

    tip = 16
    finger_tip = tip - 1
    dip = tip - 2
    pip = tip - 3
    set_pose[12] = action[pip]
    set_pose[13] = action[dip]
    set_pose[14] = action[finger_tip] * -1
    set_pose[15] = action[tip] * -1

    # print(set_pose)
    # print("tttttttttttteeeeeeeeessssssssssstttttttt")
    leaphand.set_allegro(set_pose)
    time.sleep(0.3)


def handle_viewer_events(task, viewer, agent=None):
    env = task.env
    gym = env.gym
    for evt in gym.query_viewer_action_events(viewer):
        # Reset the DOF states and actor root state to their initial states
        global recording_data

        if (evt.action == "reset") and evt.value > 0:
            print("Reset Env: Home()")
            reset_home(task)
            if agent is not None:
                agent.reset()
            recording_data = False
            task.initialized = False
            task.init_ee_state = None
            task.prev_motion_joint = None
            task.obs_dict = []

        elif (evt.action == "flexiv_data_collection") and evt.value > 0:
            if recording_data:
                recording_data = False
                print("End recording data")
            else:
                recording_data = True
                reset_home(task)
                task.initialized = False
                task.prev_motion_joint = None
                task.obs_dict = []
                cam_agent_view.clear()
                cam_wrist_view.clear()
                print("Start recording data")

        elif (evt.action == "space_shoot") and evt.value > 0:
            if save_realsense:
                dirname = f"/home/rhos_pub/Desktop/Flexiv_{task.timestamp}"
                os.makedirs(dirname, exist_ok=True)

                cnt = len(glob.glob(os.path.join(dirname, "*/")))
                dirname = os.path.join(dirname, f"{cnt:03}")
                os.makedirs(dirname, exist_ok=True)
                for f in os.listdir(dirname):
                    os.remove(os.path.join(dirname, f))

                filename = os.path.join(dirname, "data.pkl")
                with open(filename, "wb") as f:
                    pkl.dump(task.obs_dict, f)

                # save agent view
                if len(cam_agent_view) > 0:
                    with h5py.File(os.path.join(dirname, "agent_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_agent_view])
                        depths = np.array([view["depth"] for view in cam_agent_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                # save wrist view
                if len(cam_wrist_view) > 0:
                    with h5py.File(os.path.join(dirname, "wrist_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_wrist_view])
                        depths = np.array([view["depth"] for view in cam_wrist_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                # save bird view
                if len(cam_bird_view) > 0:
                    with h5py.File(os.path.join(dirname, "bird_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_bird_view])
                        depths = np.array([view["depth"] for view in cam_bird_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                # save left view
                if len(cam_left_view) > 0:
                    with h5py.File(os.path.join(dirname, "left_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_left_view])
                        depths = np.array([view["depth"] for view in cam_left_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                # save right view
                if len(cam_right_view) > 0:
                    with h5py.File(os.path.join(dirname, "right_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_right_view])
                        depths = np.array([view["depth"] for view in cam_right_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                print("Realsense Camera hdf5 Saved!")

            # filename = "obs_0.pkl"
            # motionfile = "faliure_motion_0.pkl"
            # counter = 1
            # os.makedirs(f"../../data/FlexivLeap_{task.timestamp}", exist_ok=True)
            # # os.makedirs(f"../../data/Motion_{task.timestamp}", exist_ok=True)
            # while os.path.exists(f"../../data/FlexivLeap_{task.timestamp}/{filename}"):
            #     filename = f"obs_{counter}.pkl"
            #     motionfile = f"faliure_motion_{counter}.pkl"
            #     counter += 1
            # with open(f"../../data/FlexivLeap_{task.timestamp}/{filename}", 'wb') as f:
            #     pkl.dump(task.obs_dict, f)
            # # with open(f"../../data/Motion_{task.timestamp}/{motionfile}", 'wb') as f:
            # #     pkl.dump(task.motion_dict, f)
            # print("Saved!")


def reset_home(task):
    env = task.env
    gym = env.gym
    env.reset_idx(torch.tensor([0]), torch.tensor([0]))
    env.cur_targets = torch.zeros_like(env.cur_targets)

    if test_on_real_robot:
        move_home(robot)
        reset_joints = get_robot_joints(robot)
        reset_joints = to_torch(reset_joints)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper
        # gripper.move(0.09, 0.1, 10)
    else:
        reset_joints = home_joint

    if test_on_real_hand:
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)

    env.cur_targets[:, : reset_joints.shape[-1]] = reset_joints
    # env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+reset_joints.shape[-1]] = reset_joints

    gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))
    start_time = time.time()
    while True:
        task.step()
        if time.time() - start_time > 2.5:
            break


def reset_joints(task: TeleopPlayer, joints: torch.tensor):
    """reset joints to the given position

    Args:
        task (TeleopPlayer): _description_
        joints (torch.tensor): _description_
    """
    env = task.env
    gym = env.gym
    env.reset_idx(torch.tensor([0]), torch.tensor([0]))
    env.cur_targets = torch.zeros_like(env.cur_targets)
    env.cur_targets[:, : joints.shape[-1]] = joints
    # env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+joints.shape[-1]] = joints

    gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))
    task.step()
    task.initialized = False
    task.prev_motion_joint = None
    task.obs_dict = []


def load_ckpt(agent, expname, ckpt_idx: int = -1):
    basedir = "/home/rhos_pub/LeapTele/real-traning"
    ckpts = [
        os.path.join(basedir, expname, f)
        for f in sorted(os.listdir(os.path.join(basedir, expname)))
        if "tar" in f
    ]

    if len(ckpts) > 0:
        ckpt_path = ckpts[ckpt_idx]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cuda:0")
        _ = ckpt["global_step"]

    agent.load_state_dict(ckpt["agent_state"])
    return agent


def print_robot_states(robot, log):
    """
    Print robot states data
    """
    # Data struct storing robot states
    robot_states = flexivrdk.RobotStates()

    # Get the latest robot states
    robot.getRobotStates(robot_states)

    # Print all gripper states, round all float values to 2 decimals
    log.info("Current robot states:")
    # fmt: off
    print("{")
    print("q: ",  ['%.2f' % i for i in robot_states.q])
    print("theta: ", ['%.2f' % i for i in robot_states.theta])
    print("dq: ", ['%.2f' % i for i in robot_states.dq])
    print("dtheta: ", ['%.2f' % i for i in robot_states.dtheta])
    print("tau: ", ['%.2f' % i for i in robot_states.tau])
    print("tau_des: ", ['%.2f' % i for i in robot_states.tauDes])
    print("tau_dot: ", ['%.2f' % i for i in robot_states.tauDot])
    print("tau_ext: ", ['%.2f' % i for i in robot_states.tauExt])
    print("tcp_pose: ", ['%.2f' % i for i in robot_states.tcpPose])
    print("tcp_pose_d: ", ['%.2f' % i for i in robot_states.tcpPoseDes])
    print("tcp_velocity: ", ['%.2f' % i for i in robot_states.tcpVel])
    print("camera_pose: ", ['%.2f' % i for i in robot_states.camPose])
    print("flange_pose: ", ['%.2f' % i for i in robot_states.flangePose])
    print("FT_sensor_raw_reading: ", ['%.2f' % i for i in robot_states.ftSensorRaw])
    print("F_ext_tcp_frame: ", ['%.2f' % i for i in robot_states.extWrenchInTcp])
    print("F_ext_base_frame: ", ['%.2f' % i for i in robot_states.extWrenchInBase])
    print("}")


def print_gripper_states(gripper, log):
    """
    Print gripper states data @ 1Hz.

    """
    # Data struct storing gripper states
    gripper_states = flexivrdk.GripperStates()

    # Get the latest gripper states
    gripper.getGripperStates(gripper_states)

    # Print all gripper states, round all float values to 2 decimals
    log.info("Current gripper states:")
    print("width: ", round(gripper_states.width, 2))
    print("force: ", round(gripper_states.force, 2))
    print("max_width: ", round(gripper_states.maxWidth, 2))
    print("is_moving: ", gripper_states.isMoving)


def get_robot_joints(robot) -> List[float]:
    """get robot current joints in rad

    Args:
        robot (_type_): _description_

    Returns:
        List[float]: joint positions
    """

    robot_states = flexivrdk.RobotStates()
    robot.getRobotStates(robot_states)
    return robot_states.q


def get_hand_joints(hand):
    return hand.read_pos()


def get_gripper_width(gripper) -> float:
    gripper_states = flexivrdk.GripperStates()
    gripper.getGripperStates(gripper_states)
    return gripper_states.width


def get_flange_pose(robot) -> List[float]:
    robot_states = flexivrdk.RobotStates()
    robot.getRobotStates(robot_states)
    return robot_states.flangePose


def safe_move(robot, dt: float):
    stale_time = time.time() + dt
    robot_states = flexivrdk.RobotStates()
    while True:
        """force update"""
        robot.getRobotStates(robot_states)
        force = np.array(robot_states.extWrenchInTcp[:3])
        force_norm = np.linalg.norm(force)
        if robot_states.tcpPose[2] < 0.04:
            print("tcp is too low")
            raise RuntimeError
        if force_norm > 20:
            print("larger than force threshold")
            raise RuntimeError
        if time.time() > stale_time:
            break


def get_obs(
    task,
    hand = None,
    cam_agent=None,
    cam_wrist=None,
    cam_bird=None,
    cam_left=None,
    cam_right=None,
):
    obs_dict = {}
    obs_dict["obs"] = torch.clamp(
        task.env.obs_buf, -task.env.clip_obs, task.env.clip_obs
    ).to(task.env.rl_device)
    obs_dict["target"] = task.env.cur_targets
    global recording_data
    if not recording_data:
        return

    if test_on_real_robot:
        global robot, gripper
        robot_states = flexivrdk.RobotStates()
        robot.getRobotStates(robot_states)

        obs_dict["joints"] = robot_states.q
        # obs_dict["tcpPose"] = robot_states.tcpPose
        obs_dict["flangePose"] = robot_states.flangePose
    # obs_dict["gripperWidth"] = get_gripper_width(gripper)

    if hand:
        hand_joints = hand.read_pos() - 3.14
        obs_dict["leap_hand"] = hand_joints

        set_pose = np.zeros(28)
        for i, key in enumerate([12, 22, 27]):
            tip = key
            tip = tip - 1
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3

            set_pose[dip] = hand_joints[i * 4]
            set_pose[pip] = hand_joints[i * 4 + 1]
            set_pose[finger_tip] = hand_joints[i * 4 + 2]
            set_pose[tip] = hand_joints[i * 4 + 3]

        tip = 16
        finger_tip = tip - 1
        dip = tip - 2
        pip = tip - 3
        set_pose[pip] = hand_joints[12]
        set_pose[dip] = hand_joints[13]
        set_pose[finger_tip] = hand_joints[14] * -1
        set_pose[tip] = hand_joints[15] * -1

        obs_dict["hand_joints"] = set_pose

    task.obs_dict.append(obs_dict)

    if cam_agent is not None:
        img, depth = cam_agent.get_data()
        img = img.copy()
        depth = depth.copy()
        cam_agent_view.append(
            {
                "image": img,
                "depth": depth,
            }
        )

    if cam_wrist is not None:
        img, depth = cam_wrist.get_data()
        img = img.copy()
        depth = depth.copy()
        cam_wrist_view.append(
            {
                "image": img,
                "depth": depth,
            }
        )

    if cam_bird is not None:
        img, depth = cam_bird.get_data()
        img = img.copy()
        depth = depth.copy()
        cam_bird_view.append(
            {
                "image": img,
                "depth": depth,
            }
        )

    if cam_left is not None:
        img, depth = cam_left.get_data()
        img = img.copy()
        depth = depth.copy()
        cam_left_view.append(
            {
                "image": img,
                "depth": depth,
            }
        )

    if cam_right is not None:
        img, depth = cam_right.get_data()
        img = img.copy()
        depth = depth.copy()
        cam_right_view.append(
            {
                "image": img,
                "depth": depth,
            }
        )

def is_gripper_moving(gripper) -> bool:
    gripper_states = flexivrdk.GripperStates()
    gripper.getGripperStates(gripper_states)
    return gripper_states.isMoving


def self_exam(log):
    global robot
    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        # Wait for the robot to become operational
        while not robot.isOperational():
            time.sleep(1)

        log.info("Robot is now operational")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()
        while not robot.isOperational():
            time.sleep(1)
        log.info("Robot is now operational")

    except Exception as e:
        # Print exception error message
        log.error(str(e))

    # def self_exam_leaphand(log):
    #     global leap_hand
    #     try:
    #         leap_hand = LeapNode()
    #         while True:
    #             leap_hand.set_allegro(np.zeros(16))
    #             time.sleep(0.03)

    except Exception as e:
        # Print exception error message
        log.error(str(e))


@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[0:3] * torch.sign(q_r[3]).unsqueeze(-1)


def leap_axis_correction(vec):
    """zxy coordinate -> xyz coordinate"""
    vec = vec[..., [2, 0, 1]]
    vec[..., :2] = -vec[..., :2]
    return vec


def sigma_axis_correction(vec):
    vec[..., :2] = -vec[..., :2]
    return vec


def is_validpos(tcp_pos):
    """check if tcp_pos is within the valid range"""
    return torch.all(tcp_pos > tcp_lower_bound) and torch.all(tcp_pos < tcp_upper_bound)

def parse_pt_states(pt_states, parse_target):
    """
    Parse the value of a specified primitive state from the pt_states string list.

    Parameters
    ----------
    pt_states : str list
        Primitive states string list returned from Robot::getPrimitiveStates().
    parse_target : str
        Name of the primitive state to parse for.

    Returns
    ----------
    str
        Value of the specified primitive state in string format. Empty string is
        returned if parse_target does not exist.
    """
    for state in pt_states:
        # Split the state sentence into words
        words = state.split()

        if words[0] == parse_target:
            return words[-1]

    return ""


def trans_quat(q):
    """quaternion xyzw -> wxyz"""
    return torch.cat([q[3:], q[:3]])
def move_home_pour(robot):
    log.info("Moving to higher home pose")
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    robot.executePrimitive(
        "MoveL(target=0.0 0 0.0 " + " 0 0 0 " + "TRAJ START, maxVel=0.1)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
    log.info("Moving to home pose")
    robot.executePrimitive("Home()")
    # Wait for the primitive to finish
    while robot.isBusy():
        time.sleep(1)
    robot.executePrimitive(
        "MoveL(target=0.0 0.0 0 " + " -90 0 0 " + "TRAJ START, maxVel=0.1)"
    )
    while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        time.sleep(1)
    robot.executePrimitive("ZeroFTSensor()")


def move_home(robot):
    log.info("Moving to home pose")
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    robot.executePrimitive("Home()")
    # Wait for the primitive to finish
    while robot.isBusy():
        time.sleep(1)
    robot.executePrimitive("ZeroFTSensor()")

def draw_bounding_box(task, tcp_upper_bound, tcp_lower_bound):
    task.draw_line(
        tcp_lower_bound,
        torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
    )
    task.draw_line(
        tcp_lower_bound,
        torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
    )
    task.draw_line(
        tcp_lower_bound,
        torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
    )

    task.draw_line(
        tcp_upper_bound,
        torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
    )
    task.draw_line(
        tcp_upper_bound,
        torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
    )
    task.draw_line(
        tcp_upper_bound,
        torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
    )

    task.draw_line(
        torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
    )
    task.draw_line(
        torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
    )
    task.draw_line(
        torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
    )

    task.draw_line(
        torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
    )
    task.draw_line(
        torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
    )
    task.draw_line(
        torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
    )


def main(task):
    print("main_flexivre")
    # replay_traj(task)
    # diffusion_teleop(task)
    # realworld_exec(task)
    # sigma_teleop(task)
    # leap_teleop(task)
    # flexiv_leap_teleop(task)
    # flexiv_leaphand_diffusion_replay(task)
    # flexiv_leap_diffusion_replay(task)
    # flexiv_leap_teleop_dexretarget(task)
    flexiv_leap_act_replay(task)
    # flexiv_leap_umi_diffusion_replay(task)
    # flexiv_leap_RISE_replay(task)
    
    #flexiv_leap_motion_planning_replay(task)
    # flexiv_leap_motion_planning_replay_pour(task)
    # flexiv_leap_motion_planning_replay_grasp(task)
    # vr_gripper_teleop(task)
    # vr_teleop(task)
    # vr_teleop_dexretarget(task)
    # leap_diffusion_teleop(task)
    # flexiv_leap_dexcap_replay(task)
    # flexiv_leap_dexcap_diffusion_replay(task)


# def realworld_exec(task: TeleopPlayer):
#     assert test_on_real_robot

#     from .calibration.camera import CameraD400

#     cam_agent = CameraD400(0)
#     cam_wrist = CameraD400(1)

#     env = task.env
#     gym = task.env.gym
#     sim = task.env.sim
#     viewer = task.env.viewer
#     task.subscribe_keyboard_events(gym, viewer)

#     # sync sim & real robot joints
#     DOF = -1

#     if test_on_real_robot:
#         self_exam(log)
#         move_home(robot)
#         robot.setMode(mode.NRT_JOINT_POSITION)
#         global gripper
#         gripper = flexivrdk.Gripper(robot)
#         gripper.move(0.09, 0.1, 10)

#         init_joints = get_robot_joints(robot)
#         DOF = len(init_joints)
#         target_vel = [0.0] * DOF
#         target_acc = [0.0] * DOF
#         MAX_VEL = [1.0] * DOF
#         MAX_ACC = [0.5] * DOF

#         init_joints = to_torch(init_joints)

#         reset_joints(task, init_joints)

#     dt = 1.0 / frequency
#     print(
#         "Sending command to robot at",
#         frequency,
#         "Hz, or",
#         dt,
#         "seconds interval",
#     )

#     assert DOF > 0, "DOF must be greater than 0"

#     # ================== Diffusion Actor ====================
#     from torch.utils.data import DataLoader
#     from .diffusha_flexiv.diffusion.train import Trajectory
#     from .diffusha_flexiv.config.default_args import Args
#     from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
#     from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor

#     print(env.flexiv_dof_lower_limits.shape)
#     device = env.device
#     lower_limits = env.flexiv_dof_lower_limits[:DOF]
#     upper_limits = env.flexiv_dof_upper_limits[:DOF]
#     lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
#     upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])

#     # dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=4)
#     # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     # model_pt = "/home/rhos/Desktop/flexiv/data/ddpm/diffusha-flexiv/m5yq5o60/step_00004000.pt"
#     model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/x084y7qb/step_00002500.pt"  # beta_max=1e-2
#     # model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/ybbhnl43/step_00002500.pt" # beta_max=5e-3
#     model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/usx3fnbh/step_00008000.pt"  # beta_max=1e-1, step=50
#     model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006s000.pt"  # hidden_dim=256, step=50, fwd_ratio=0.5
#     model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/zqlkkewt/step_00007000.pt"  # hidden_dim=512, step=100, beta_max=1e-2, horizon=6
#     model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/4wrbxope/step_00007000.pt"  # horizon=12, hidden_dim=512, step=100, beta_max=1e-2
#     model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/00v2rgcl/step_00006000.pt"  # horizon=8, diff+omega(20+20), beta_max=1e-2
#     # model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/pfo1i9r4/step_00006000.pt"  # horizon=8, diff+hum(20+20), beta_max=1e-2

#     diffusion = DiffusionModel(
#         diffusion_core=DiffusionCore(),
#         num_diffusion_steps=Args.num_diffusion_steps,
#         input_size=(Args.copilot_obs_size + Args.act_size),
#         beta_schedule=Args.beta_schedule,
#         beta_min=Args.beta_min,
#         beta_max=Args.beta_max,
#         cond_dim=Args.copilot_obs_size,
#     )

#     diffusion.load_ckpt(model_pt)
#     diffusion_actor = DiffusionAssistedActor(
#         -1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio
#     )
#     # ======================= End of Diffusion Actor ==================

#     env.actions = torch.zeros((1, env.num_dofs))

#     def draw_bounding_box():
#         task.draw_line(
#             tcp_lower_bound,
#             torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
#         )
#         task.draw_line(
#             tcp_lower_bound,
#             torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
#         )
#         task.draw_line(
#             tcp_lower_bound,
#             torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
#         )

#         task.draw_line(
#             tcp_upper_bound,
#             torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
#         )
#         task.draw_line(
#             tcp_upper_bound,
#             torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
#         )
#         task.draw_line(
#             tcp_upper_bound,
#             torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
#         )

#         task.draw_line(
#             torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
#             torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
#         )
#         task.draw_line(
#             torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
#             torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
#         )
#         task.draw_line(
#             torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
#             torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
#         )

#         task.draw_line(
#             torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
#             torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
#         )
#         task.draw_line(
#             torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
#             torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
#         )
#         task.draw_line(
#             torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
#             torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
#         )

#     start_sample_time = time.time()
#     while not env.gym.query_viewer_has_closed(env.viewer):

#         task.step()

#         handle_viewer_events(task, viewer)
#         draw_bounding_box()

#         if not recording_data:
#             continue

#         if test_on_real_robot and robot.isFault():
#             raise Exception("Fault occurred on robot server, exiting ...")

#         # =============== Diffusion Robot ============
#         tq = torch.from_numpy(np.array(get_flange_pose(robot))).to(torch.float32)
#         tq[3:] /= torch.norm(tq[3:])
#         griv = get_gripper_width(gripper)

#         agent_state = torch.cat([tq, torch.tensor([griv]).to(device)]).unsqueeze(0)

#         agent_img, _ = cam_agent.get_data()
#         agent_img = cv2.resize(agent_img, (320, 240))
#         agent_img = agent_img.astype(np.float32).transpose((2, 0, 1)) / 255
#         agent_img = torch.from_numpy(agent_img).unsqueeze(0).to(device)

#         wrist_img, _ = cam_wrist.get_data()
#         wrist_img = cv2.resize(wrist_img, (320, 240))
#         wrist_img = wrist_img.astype(np.float32).transpose((2, 0, 1)) / 255
#         wrist_img = torch.from_numpy(wrist_img).unsqueeze(0).to(device)

#         batch = (agent_state, wrist_img, agent_img)

#         agent_action = torch.zeros((1, (DOF + 1) * Args.horizon)).to(device)

#         agent_action = (
#             diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
#         )
#         agent_action = tensor_clamp(
#             agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action)
#         )
#         agent_action = agent_action.view(-1, 8)
#         agent_action = scale(agent_action, lower_limits, upper_limits)

#         # _ = input()

#         # agent_action = agent_action.mean(dim=0).unsqueeze(0) # TODO:
#         for agent_act in agent_action:  # number of Args.horizon action

#             # ==================== End Diffusion Robot ========================
#             env.cur_targets[:, :DOF] = (
#                 0.7 * env.cur_targets[:, :DOF] + 0.3 * agent_act[:DOF]
#             )
#             gym.set_dof_position_target_tensor(
#                 sim, gymtorch.unwrap_tensor(env.cur_targets)
#             )

#             griv = get_gripper_width(gripper)
#             print(griv, agent_act[-1])
#             griv = agent_act[-1]  # TODO:

#             try:
#                 # action = env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF]
#                 action = env.cur_targets[:, :DOF]
#                 action = action[0][:DOF].tolist()

#                 robot.sendJointPosition(
#                     action, target_vel, target_acc, MAX_VEL, MAX_ACC
#                 )

#                 flag = False
#                 # open gripper
#                 if (
#                     griv > 0.08
#                     and get_gripper_width(gripper) < 0.03
#                     and not is_gripper_moving(gripper)
#                 ):
#                     gripper.move(0.09, 0.1, 20)
#                     flage = True
#                 # close gripper
#                 if (
#                     griv < 0.055
#                     and get_gripper_width(gripper) > 0.08
#                     and not is_gripper_moving(gripper)
#                 ):
#                     gripper.move(0, 0.5, 5)
#                     flag = True

#                 safe_move(robot, dt)

#                 if flag:
#                     break

#             except Exception as e:
#                 log.error(str(e))
#                 exit(0)
#         # print("="*10)
#         # _ = input()

#     exit(0)


def replay_traj(task: TeleopPlayer):
    # sys.path.append('/home/rhos/Desktop/flexiv')
    # from dataloader import *
    from torch.utils.data import DataLoader
    from .diffusha_flexiv.diffusion.train import Trajectory
    from .diffusha_flexiv.config.default_args import Args
    from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
    from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor

    env = task.env
    gym = task.env.gym
    sim = task.env.sim

    griv = 0.0
    DOF = 7

    print(env.flexiv_dof_lower_limits.shape)
    device = env.device
    lower_limits = env.flexiv_dof_lower_limits[:DOF]
    upper_limits = env.flexiv_dof_upper_limits[:DOF]
    lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
    upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])

    dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=6)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # model_pt = "/home/rhos/Desktop/flexiv/data/ddpm/diffusha-flexiv/m5yq5o60/step_00009999.pt"
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/x084y7qb/step_00002500.pt"  # beta_max=1e-2, step=100
    # model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/ybbhnl43/step_00002500.pt" # beta_max=5e-3, step=100
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/usx3fnbh/step_00005000.pt"  # beta_max=1e-1, step=50
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006000.pt"  # hidden_dim=256, step=50
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/zqlkkewt/step_00006000.pt"  # hidden_dim=512, step=100, beta_max=1e-2, horizon=6
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/00v2rgcl/step_00006000.pt"  # horizon=8, diff+omega(20+20), beta_max=1e-2
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/i01k9rov/step_00000500.pt"  # horizon=8, hum+omega(20+20), beta_max=1e-2, seed=233

    diffusion = DiffusionModel(
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=Args.num_diffusion_steps,
        input_size=(Args.copilot_obs_size + Args.act_size),
        beta_schedule=Args.beta_schedule,
        beta_min=Args.beta_min,
        beta_max=Args.beta_max,
        cond_dim=Args.copilot_obs_size,
    )

    diffusion.load_ckpt(model_pt)
    diffusion_actor = DiffusionAssistedActor(
        -1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio
    )

    env.actions = torch.zeros((1, env.num_dofs))

    for _ in range(1):
        reset_joints(task, home_joint)
        for _ in range(100):
            task.step()

        len_data = len(dataloader)
        for _, batch in enumerate(dataloader):

            # for idx in range(len_data):
            # batch = dataloader[idx]

            batch = [b.to(device) for b in batch]

            task.step()

            act = batch[-1]
            act = scale(act.view(-1, 8), lower_limits, upper_limits).reshape(
                -1, Args.act_size
            )

            act_dof = act[..., :DOF]
            print(act_dof, lower_limits)

            act_dof = (
                0.95 * env.cur_targets[:, : env.num_flexiv_dofs] + 0.05 * next_dof_pos
            )
            act_dof = tensor_clamp(
                act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
            )

            print(act_dof, lower_limits, upper_limits)

            # =============== Diffusion Robot ============

            # cur_state = torch.squeeze(env.rigid_body_states[:, env.num_flexiv_bodies:2*env.num_flexiv_bodies])
            cur_state = torch.squeeze(env.rigid_body_states[:, : env.num_flexiv_bodies])
            cur_rb_idx = 8  # flange
            tq = cur_state[cur_rb_idx][:7]

            tq[3:] /= torch.norm(tq[3:])
            tq[3:] = trans_quat(tq[3:])
            # print(tq, batch[0][..., :DOF])
            batch[0][..., :DOF] = tq

            agent_action = (
                diffusion_actor.assisted_act(torch.zeros_like(act), batch)
                .to(device)
                .squeeze()
            )
            agent_action = tensor_clamp(
                agent_action,
                -torch.ones_like(agent_action),
                torch.ones_like(agent_action),
            )

            agent_action = agent_action.view(-1, 8)

            # print("@@@: ", batch[0])
            for agent_act in agent_action:
                agent_act = scale(agent_act, lower_limits, upper_limits)
                # print(batch[0], agent_action)
                # print(act, agent_action)
                # ============================================
                # print(agent_act)
                env.cur_targets[:, :DOF] = (
                    0.7 * env.cur_targets[:, :DOF] + 0.3 * agent_act[:DOF]
                )
                # env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF] = 0.7 * env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF] + 0.3 * agent_action[:DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )

                task.step()
            # print("="*10)
            # idx += Args.horizon - 1

            # _ = input()
            # agent_action = diffusion_actor.assisted_act(act, batch)

    exit(0)


def diffusion_teleop(task: TeleopPlayer):
    sys.path.append("/home/rhos/sigma_sdk")
    import sigma7
    from .calibration.camera import CameraD400

    ik = IKController(damping=0.05)
    task.init_ee_state = None
    cam_agent = CameraD400(0)
    cam_wrist = CameraD400(1)

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = -1

    if test_on_real_robot:
        self_exam(log)
        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        global gripper
        gripper = flexivrdk.Gripper(robot)
        gripper.move(0.09, 0.1, 10)

        init_joints = get_robot_joints(robot)
        DOF = len(init_joints)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 7
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    assert DOF > 0, "DOF must be greater than 0"

    # ================== Diffusion Actor ====================
    from torch.utils.data import DataLoader
    from .diffusha_flexiv.diffusion.train import Trajectory
    from .diffusha_flexiv.config.default_args import Args
    from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
    from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor

    print(env.flexiv_dof_lower_limits.shape)
    device = env.device
    lower_limits = env.flexiv_dof_lower_limits[:DOF]
    upper_limits = env.flexiv_dof_upper_limits[:DOF]
    lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
    upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])

    # dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=4)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # model_pt = "/home/rhos/Desktop/flexiv/data/ddpm/diffusha-flexiv/m5yq5o60/step_00004000.pt"
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/x084y7qb/step_00002500.pt"  # beta_max=1e-2
    # model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/ybbhnl43/step_00003500.pt" # beta_max=5e-3
    model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006000.pt"  # hidden_dim=256, step=50

    diffusion = DiffusionModel(
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=Args.num_diffusion_steps,
        input_size=(Args.copilot_obs_size + Args.act_size),
        beta_schedule=Args.beta_schedule,
        beta_min=Args.beta_min,
        beta_max=Args.beta_max,
        cond_dim=Args.copilot_obs_size,
    )

    diffusion.load_ckpt(model_pt)
    diffusion_actor = DiffusionAssistedActor(
        -1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio
    )
    # ======================= End of Diffusion Actor ==================

    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    sigma7.drdOpen()
    sigma7.drdAutoInit()
    sigma7.drdStart()
    sigma7.drdMoveTo(np.array([0, 0, 0, 0, 0, 0, 0, 0]), block=True)
    sigma7.drdRegulatePos(on=False)
    sigma7.drdRegulateRot(on=False)
    sigma7.drdRegulateGrip(on=False)
    sigma7.drdSetForceAndWristJointTorquesAndGripperForce(0, 0, 0.3, 0.1, 0, 0, 0)

    def display_images():
        while True:
            img_agent, _ = cam_agent.get_data()
            img_agent = img_agent.copy()
            img_wrist, _ = cam_wrist.get_data()
            img_wrist = img_wrist.copy()
            img = np.concatenate((img_agent, img_wrist), axis=1)
            cv2.imshow("Image", img)
            cv2.waitKey(50)

    # thread = threading.Thread(target=display_images)
    # thread.start()

    start_sample_time = time.time()
    set_real_robot_time = time.time()

    while not env.gym.query_viewer_has_closed(env.viewer):
        # for _ in range(10):
        #     reset_home(task)
        #     task.initialized             = False
        #     task.prev_motion_joint       = None
        #     task.obs_dict                = []
        #     for _, batch in enumerate(dataloader):
        #         batch = [b.to(device) for b in batch]

        task.step()
        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if not recording_data:  # TODO: turn on when testing on real robot
            continue

        if time.time() - start_sample_time > dt:
            get_obs(task, cam_agent, cam_wrist)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])

        if task.prev_motion_joint is None:
            print("\033[91mPlease move sigma.7 to a comfortable pose...\033[0m")
            cur_time = time.time()
            while True:
                task.step()
                if time.time() - cur_time > 2.0:
                    break
            print("\033[92mSampled\033[0m")
            task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
            task.init_ee_state = cur_state[8].clone().cpu()

        # === X, Y, Z ===
        _, px, py, pz, oa, ob, og, pg, mat = sigma7.drdGetPositionAndOrientation()
        delta_pos = (
            np.array([px, py, pz]) - task.prev_motion_joint[1:4]
        ) * sigma_scaling_factor
        delta_pos[1] *= 1.5
        delta_pos = sigma_axis_correction(delta_pos)

        dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
        mask = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)

        cur_rb_idx = 8
        cur_pos = cur_state[cur_rb_idx][:3]
        cur_rot_q = cur_state[cur_rb_idx][3:7]
        cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
        z_axis = cur_rot_mat[:, 2]

        target_pos = task.init_ee_state[:3] + delta_pos

        dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
        mask[cur_rb_idx, :3] = True

        delta_euler = np.array([oa, ob, og]) - task.prev_motion_joint[4:7]
        init_rot_e = R.from_quat(task.init_ee_state[3:7].cpu().numpy()).as_euler("xyz")
        delta_euler[1:] *= -1
        target_rot_q = R.from_euler("xyz", init_rot_e) * R.from_euler(
            "xyz", delta_euler
        )
        target_rot_q = torch.from_numpy(target_rot_q.as_quat()).type_as(cur_rot_q)
        # print(delta_euler, target_rot_q)

        target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0)
        )
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0)
        )
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1)
        )

        dpose[cur_rb_idx, 3:] = orientation_error(target_rot_q, cur_rot_q)
        mask[cur_rb_idx, 3:] = True

        tcp_pos = target_pos + z_axis * 0.15

        task.draw_sphere(target_pos)
        task.draw_sphere(tcp_pos, color=(0, 0, 0))

        # if not is_validpos(tcp_pos):
        #     continue

        fake_slice = torch.ones((1, 6, env.num_flexiv_dofs), device=env.device)
        fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

        action = ik.control_ik(dpose, fake_jacobian, mask)
        next_dof_pos = env.flexiv_dof_pos + action

        act_dof = 0.90 * env.cur_targets[:, : env.num_flexiv_dofs] + 0.10 * next_dof_pos
        act_dof = tensor_clamp(
            act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
        )

        # =============== Diffusion Robot ============
        delta_action = act_dof - env.flexiv_dof_pos
        agent_action = env.flexiv_dof_pos + delta_action

        griv = 0.09 if pg < -0.015 else 0.01  # TODO:
        # griv = get_gripper_width(gripper)

        cur_state = torch.squeeze(env.rigid_body_states[:, : env.num_flexiv_bodies])
        cur_rb_idx = 8  # flange
        tq = cur_state[cur_rb_idx][:7]
        tq[3:] /= torch.norm(tq[3:])
        tq[3:] = trans_quat(tq[3:])

        if test_on_real_robot:
            tq = torch.from_numpy(np.array(get_flange_pose(robot))).to(torch.float32)
            tq[3:] /= torch.norm(tq[3:])

        # batch[0][..., :DOF] = tq   # change it into current xyz + quat
        # batch[0][..., DOF:] = griv # change it into current grivWidth

        agent_state = torch.cat([tq, torch.tensor([griv]).to(device)]).unsqueeze(0)

        agent_img, _ = cam_agent.get_data()
        agent_img = cv2.resize(agent_img, (320, 240))
        agent_img = agent_img.astype(np.float32).transpose((2, 0, 1)) / 255
        agent_img = torch.from_numpy(agent_img).unsqueeze(0).to(device)

        wrist_img, _ = cam_wrist.get_data()
        wrist_img = cv2.resize(wrist_img, (320, 240))
        wrist_img = wrist_img.astype(np.float32).transpose((2, 0, 1)) / 255
        wrist_img = torch.from_numpy(wrist_img).unsqueeze(0).to(device)

        batch = (agent_state, wrist_img, agent_img)

        agent_action = torch.cat(
            [agent_action[..., :DOF].squeeze(), torch.tensor([griv]).to(device)]
        )  # add griv dimension
        agent_action = unscale(agent_action.unsqueeze(0), lower_limits, upper_limits)
        agent_action = agent_action.repeat(1, Args.horizon).view(
            -1, 8 * Args.horizon
        )  # repeat horizon times
        # agent_action = torch.zeros_like(agent_action) # TODO: autonomous agent's input

        agent_action = (
            diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
        )
        agent_action = tensor_clamp(
            agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action)
        )
        agent_action = scale(agent_action.view(-1, 8), lower_limits, upper_limits)

        # ==================== End Diffusion Robot ========================
        agent_action = agent_action[:4].mean(dim=0).unsqueeze(0)

        for act_next in agent_action:
            # env.cur_targets[:, :DOF] = 0.7 * env.cur_targets[:, :DOF] + 0.3 * act_next[:DOF]
            env.cur_targets[:, :DOF] = (
                0.3 * env.cur_targets[:, :DOF] + 0.7 * act_next[:DOF]
            )

            env.cur_targets[:, :DOF] = (
                0.7 * env.cur_targets[:, :DOF] + 0.3 * act_dof[..., :DOF]
            )  # TODO: blend with original human action

            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

            task.step(clear_lines=False)
            handle_viewer_events(task, viewer)
            draw_bounding_box()

            if not recording_data:  # TODO: turn on when testing on real robot
                break

            griv = act_next[-1]  # TODO:
            print("grivWidth: ", griv)

            if test_on_real_robot:
                try:
                    # action = env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF]
                    action = env.cur_targets[:, :DOF]
                    action = action[0][:DOF].tolist()

                    robot.sendJointPosition(
                        action, target_vel, target_acc, MAX_VEL, MAX_ACC
                    )
                    # open gripper
                    if (
                        griv > 0.06
                        and get_gripper_width(gripper) < 0.03
                        and not is_gripper_moving(gripper)
                    ):
                        gripper.move(0.09, 0.1, 20)
                    # close gripper
                    if (
                        griv < 0.075
                        and get_gripper_width(gripper) > 0.08
                        and not is_gripper_moving(gripper)
                    ):
                        gripper.grasp(5)

                    safe_move(robot, dt)

                except Exception as e:
                    log.error(str(e))
                    exit(0)

            # ===== sync joint state =====
            if test_on_real_robot:
                joints = get_robot_joints(robot)
                env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )

    exit(0)


def calibration(task: TeleopPlayer):
    sys.path.append("/home/rhos/sigma_sdk")
    import sigma7, cv2

    ik = IKController(damping=0.05)
    task.init_ee_state = None
    from .calibration.camera import CameraD400

    cam = CameraD400(1)
    chessboard_size = (8, 11)
    saved_img = 0

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = -1

    if test_on_real_robot:
        self_exam(log)
        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        gripper = flexivrdk.Gripper(robot)

        init_joints = get_robot_joints(robot)
        DOF = len(init_joints)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

        try:
            gripper.move(0.01, 0.1, 20)
        except Exception as e:
            log.error(str(e))
    else:
        DOF = 7
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    sigma7.drdOpen()
    sigma7.drdAutoInit()
    sigma7.drdStart()
    sigma7.drdMoveTo(np.array([0, 0, 0, 0, 0, 0, 0, 0]), block=True)
    sigma7.drdRegulatePos(on=False)
    sigma7.drdRegulateRot(on=False)
    sigma7.drdRegulateGrip(on=False)
    sigma7.drdSetForceAndWristJointTorquesAndGripperForce(0, 0, 0.3, 0.1, 0, 0, 0)

    calib_time = time.time()
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()
        get_obs(task)
        handle_viewer_events(task, viewer)
        draw_bounding_box()

        # joints = task.detector.detect_right_joints()
        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])

        if task.prev_motion_joint is None:
            print("\033[91mPlease move sigma.7 to a comfortable pose...\033[0m")
            cur_time = time.time()
            while True:
                task.step()
                if time.time() - cur_time > 3.0:
                    break
            print("\033[92mSampled\033[0m")
            task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
            task.init_ee_state = cur_state[8].clone().cpu()

        # === X, Y, Z ===
        _, px, py, pz, oa, ob, og, pg, mat = sigma7.drdGetPositionAndOrientation()
        delta_pos = (
            np.array([px, py, pz]) - task.prev_motion_joint[1:4]
        ) * sigma_scaling_factor
        delta_pos[1] *= 1.5
        delta_pos = sigma_axis_correction(delta_pos)

        dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
        mask = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)

        cur_rb_idx = 8
        cur_pos = cur_state[cur_rb_idx][:3]
        cur_rot_q = cur_state[cur_rb_idx][3:7]
        cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
        z_axis = cur_rot_mat[:, 2]

        target_pos = task.init_ee_state[:3] + delta_pos

        dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
        mask[cur_rb_idx, :3] = True

        delta_euler = np.array([oa, ob, og]) - task.prev_motion_joint[4:7]
        init_rot_e = R.from_quat(task.init_ee_state[3:7].cpu().numpy()).as_euler("xyz")
        delta_euler[1:] *= -1
        target_rot_q = R.from_euler("xyz", init_rot_e) * R.from_euler(
            "xyz", delta_euler
        )
        target_rot_q = torch.from_numpy(target_rot_q.as_quat()).type_as(cur_rot_q)
        # print(delta_euler, target_rot_q)

        target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0)
        )
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0)
        )
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1)
        )

        dpose[cur_rb_idx, 3:] = orientation_error(target_rot_q, cur_rot_q)
        mask[cur_rb_idx, 3:] = True

        tcp_pos = target_pos + z_axis * 0.15

        task.draw_sphere(target_pos)
        task.draw_sphere(tcp_pos, color=(0, 0, 0))

        if not is_validpos(tcp_pos):
            continue

        fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
        fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

        action = ik.control_ik(dpose, fake_jacobian, mask)
        next_dof_pos = env.flexiv_dof_pos + action

        act_dof = 0.95 * env.cur_targets[:, : env.num_flexiv_dofs] + 0.05 * next_dof_pos
        act_dof = tensor_clamp(
            act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
        )

        env.cur_targets[:, : env.num_flexiv_dofs] = act_dof
        env.actions = act_dof
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

        if test_on_real_robot:
            try:
                action = act_dof[0][:DOF].tolist()
                robot.sendJointPosition(
                    action, target_vel, target_acc, MAX_VEL, MAX_ACC
                )

            except Exception as e:
                log.error(str(e))

            img, _ = cam.get_data()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

            if time.time() - calib_time < 1.5:
                continue

            calib_time = time.time()
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret == True:
                saved_img += 1
                filename = f"/home/rhos/Desktop/cali/{saved_img:03}.png"
                print("chessboard scanned!")
                cv2.imwrite(filename, img)
                flange_pose = get_flange_pose(robot)
                np.savetxt(f"/home/rhos/Desktop/cali/{saved_img:03}.txt", flange_pose)
                np.savetxt(f"/home/rhos/Desktop/cali/pose/{saved_img:03}.txt", action)

                cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
                cv2.imshow("Image", img)
                cv2.waitKey(500)


def sigma_teleop(task: TeleopPlayer):
    sys.path.append("/home/rhos/sigma_sdk")
    import sigma7
    from .calibration.camera import CameraD400

    ik = IKController(damping=0.05)
    task.init_ee_state = None
    cam_agent = CameraD400(0)
    cam_wrist = CameraD400(1)

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = -1

    if test_on_real_robot:
        self_exam(log)
        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        global gripper
        gripper = flexivrdk.Gripper(robot)

        init_joints = get_robot_joints(robot)
        DOF = len(init_joints)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

        try:
            gripper.move(0.01, 0.1, 20)
        except Exception as e:
            log.error(str(e))
    else:
        DOF = 7
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    sigma7.drdOpen()
    sigma7.drdAutoInit()
    sigma7.drdStart()
    sigma7.drdMoveTo(np.array([0, 0, 0, 0, 0, 0, 0, 0]), block=True)
    sigma7.drdRegulatePos(on=False)
    sigma7.drdRegulateRot(on=False)
    sigma7.drdRegulateGrip(on=False)
    sigma7.drdSetForceAndWristJointTorquesAndGripperForce(0, 0, 0.3, 0.1, 0, 0, 0)

    def display_images():
        while True:
            img_agent, _ = cam_agent.get_data()
            img_agent = img_agent.copy()
            img_wrist, _ = cam_wrist.get_data()
            img_wrist = img_wrist.copy()
            img = np.concatenate((img_agent, img_wrist), axis=1)
            cv2.imshow("Image", img)
            cv2.waitKey(50)

    # thread = threading.Thread(target=display_images)
    # thread.start()

    start_sample_time = time.time()
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if not recording_data:
            continue

        if time.time() - start_sample_time > dt:
            get_obs(task, cam_agent, cam_wrist)
            start_sample_time = time.time()
        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])

        if task.prev_motion_joint is None:
            print("\033[91mPlease move sigma.7 to a comfortable pose...\033[0m")
            cur_time = time.time()
            while True:
                task.step()
                if time.time() - cur_time > 2.0:
                    break
            print("\033[92mSampled\033[0m")
            task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
            task.init_ee_state = cur_state[8].clone().cpu()

        # === X, Y, Z ===
        _, px, py, pz, oa, ob, og, pg, mat = sigma7.drdGetPositionAndOrientation()
        delta_pos = (
            np.array([px, py, pz]) - task.prev_motion_joint[1:4]
        ) * sigma_scaling_factor
        delta_pos[1] *= 1.5
        delta_pos = sigma_axis_correction(delta_pos)

        dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
        mask = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)

        cur_rb_idx = 8
        cur_pos = cur_state[cur_rb_idx][:3]
        cur_rot_q = cur_state[cur_rb_idx][3:7]
        cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
        z_axis = cur_rot_mat[:, 2]

        target_pos = task.init_ee_state[:3] + delta_pos

        dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
        mask[cur_rb_idx, :3] = True

        delta_euler = np.array([oa, ob, og]) - task.prev_motion_joint[4:7]
        init_rot_e = R.from_quat(task.init_ee_state[3:7].cpu().numpy()).as_euler("xyz")
        delta_euler[1:] *= -1
        target_rot_q = R.from_euler("xyz", init_rot_e) * R.from_euler(
            "xyz", delta_euler
        )
        target_rot_q = torch.from_numpy(target_rot_q.as_quat()).type_as(cur_rot_q)
        # print(delta_euler, target_rot_q)

        target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0)
        )
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0)
        )
        task.draw_line(
            target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1)
        )

        dpose[cur_rb_idx, 3:] = orientation_error(target_rot_q, cur_rot_q)
        mask[cur_rb_idx, 3:] = True

        tcp_pos = target_pos + z_axis * 0.15

        task.draw_sphere(target_pos)
        task.draw_sphere(tcp_pos, color=(0, 0, 0))

        if not is_validpos(tcp_pos):
            # task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
            # task.init_ee_state = cur_state[8].clone().cpu()
            continue

        fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
        fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

        action = ik.control_ik(dpose, fake_jacobian, mask)
        next_dof_pos = env.flexiv_dof_pos + action

        act_dof = 0.95 * env.cur_targets[:, : env.num_flexiv_dofs] + 0.05 * next_dof_pos
        act_dof = tensor_clamp(
            act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
        )

        env.cur_targets[:, : env.num_flexiv_dofs] = act_dof
        env.actions = act_dof
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

        if test_on_real_robot:
            try:
                action = act_dof[0][:DOF].tolist()
                robot.sendJointPosition(
                    action, target_vel, target_acc, MAX_VEL, MAX_ACC
                )

                # open gripper
                if (
                    pg < -0.015
                    and get_gripper_width(gripper) < 0.05
                    and not is_gripper_moving(gripper)
                ):
                    gripper.move(0.09, 0.1, 20)
                # close gripper
                if (
                    pg > -0.007
                    and get_gripper_width(gripper) > 0.06
                    and not is_gripper_moving(gripper)
                ):
                    # gripper.move(0.015, 0.1, 20)
                    gripper.grasp(5)

            except Exception as e:
                log.error(str(e))


def flexiv_leap_diffusion_replay(task: TeleopPlayer):

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    cam_bird = CameraD400(1)
    # cam_left = CameraD400(2)
    cam_right = CameraD400(0)

    task.init_ee_state = None

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints

    if test_on_real_robot:
        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    if test_on_real_hand:
        leap_hand = LeapNode()
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)

    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    # ================== Diffusion Actor ====================
    from torch.utils.data import DataLoader
    from .diffusha_flexiv.diffusion.train import Trajectory
    from .diffusha_flexiv.config.default_args import Args
    from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
    from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor
    from .diffusion_unet import set_seed, Diffusion

    print(env.flexiv_dof_lower_limits.shape)
    device = env.device
    upper_limits = torch.tensor(
        [
            2.7925,
            2.2689,
            2.9671,
            2.6878,
            2.9671,
            4.5379,
            2.9671,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            2.0940,
            2.4430,
            1.2000,
            1.3400,
        ],
        dtype=torch.float32,
    )
    lower_limits = torch.tensor(
        [
            -2.7925,
            -2.2689,
            -2.9671,
            -1.8675,
            -2.9671,
            -1.3963,
            -2.9671,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -0.3490,
            -0.4700,
            -1.9000,
            -1.8800,
        ],
        dtype=torch.float32,
    )

    # lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
    # upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])

    # dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=4)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006000.pt"   # hidden_dim=256, step=50
    # model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/4wrbxope/step_00006000.pt"   #horizon=12, hidden_dim=512, step=100, beta_max=1e-2
    model_pt = "/home/rhos_pub/Desktop/flexiv_training/logs/diffusion_unet/policy_epoch_1600_seed_233.ckpt"

    # diffusion = DiffusionModel(
    #     diffusion_core=DiffusionCore(),
    #     num_diffusion_steps=Args.num_diffusion_steps,
    #     input_size=(Args.copilot_obs_size + Args.act_size),
    #     beta_schedule=Args.beta_schedule,
    #     beta_min=Args.beta_min,
    #     beta_max=Args.beta_max,
    #     cond_dim=Args.copilot_obs_size
    # )

    diffusion = Diffusion()
    set_seed(233)

    # diffusion.load_ckpt(model_pt)
    state_dict = torch.load(model_pt)
    diffusion.load_state_dict(state_dict)
    # diffusion_actor = DiffusionAssistedActor(-1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio)
    # ======================= End of Diffusion Actor ==================

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")
    i = 0

    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_ee_state is None:
            task.init_ee_state = cur_state[8].clone().cpu()

        state = np.zeros(28)

        hand_joints = leap_hand.read_pos() - 3.14

        robot_states = flexivrdk.RobotStates()
        robot.getRobotStates(robot_states)
        state = np.concatenate([robot_states.q, hand_joints])

        state = state.astype(np.float32)

        # =============== Diffusion Robot ============

        # print(griv, finger_dis, griv_width)

        # griv = get_gripper_width(gripper)

        cur_flange_pose = get_flange_pose(robot)

        if (
            cur_flange_pose[1] < 1000
        ):  # TODO: this is hard-coded, you can directly remove the if statement
            cur_state = torch.squeeze(env.rigid_body_states[:, : env.num_flexiv_bodies])

            # batch[0][..., :DOF] = tq   # change it into current xyz + quat
            # batch[0][..., DOF:] = griv # change it into current grivWidth

            bird_img, _ = cam_bird.get_data()
            bird_img = cv2.resize(bird_img, (320, 240))
            bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
            bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)

            # left_img, _ = cam_left.get_data()
            # left_img = cv2.resize(left_img, (320, 240))
            # left_img = left_img.astype(np.float32).transpose((2, 0, 1)) / 255
            # left_img = torch.from_numpy(left_img).unsqueeze(0).to(device)

            right_img, _ = cam_right.get_data()
            right_img = cv2.resize(right_img, (320, 240))
            right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
            right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)

            batch = (torch.tensor(state).view(1, -1), bird_img, right_img)
            # agent_action = torch.tensor(state).to(device)
            # agent_action = torch.randn(state.shape).to(device)
            # agent_action = unscale(agent_action.unsqueeze(0), lower_limits, upper_limits)
            # agent_action = agent_action.repeat(1, Args.horizon).view(-1, 23 * Args.horizon)

            agent_action = diffusion(batch).cpu().squeeze()
            print(agent_action.shape)
            # agent_action = diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
            # agent_action = diffusion_actor.teleop_diffusion_cond_sample(batch,agent_action,True).to(device).squeeze()
            # agent_action = tensor_clamp(agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action))
            # agent_action = scale(agent_action.view(-1, 23), lower_limits, upper_limits)
            # agent_action = scale(agent_action.view(-1, 8), lower_limits, upper_limits)
            # print("execute agent_action", agent_action)
            # print("act_dof: ", act_dof)
            # print("")
            # _ = input()
            start_sample_time = time.time()  # TODO: neglect network inference time
            # ==================== End Diffusion Robot ========================

            act_dof = np.zeros((agent_action.shape[0], 28))
            act_dof[:, :7] = agent_action[:, :7]

            for i, key in enumerate([12, 22, 27]):
                tip = key
                tip = tip - 1
                finger_tip = tip - 1
                dip = tip - 2
                pip = tip - 3

                act_dof[:, dip] = agent_action[:, i * 4 + 7]
                act_dof[:, pip] = agent_action[:, i * 4 + 1 + 7]
                act_dof[:, finger_tip] = agent_action[:, i * 4 + 2 + 7]
                act_dof[:, tip] = agent_action[:, i * 4 + 3 + 7]

            tip = 16
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3
            act_dof[:, pip] = agent_action[:, 12 + 7]
            act_dof[:, dip] = agent_action[:, 13 + 7]
            act_dof[:, finger_tip] = agent_action[:, 14 + 7] * -1
            act_dof[:, tip] = agent_action[:, 15 + 7] * -1

        # agent_action = agent_action[:4].mean(dim=0).unsqueeze(0)
        for act_next in act_dof:
            env.cur_targets[:, :DOF] = (
                0.9 * env.cur_targets[:, :DOF] + 0.1 * act_next[:DOF]
            )
            # env.cur_targets[:, :DOF] = 0.9 * env.cur_targets[:, :DOF] + 0.1 * act_dof[..., :DOF] # TODO: blend with original human action

            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

            task.step()
            handle_viewer_events(task, viewer)
            draw_bounding_box()

            if not recording_data:  # TODO: turn on when testing on real robot
                break

            if time.time() - start_sample_time > dt:
                get_obs(task)
                start_sample_time = time.time()

            if test_on_real_robot:
                try:
                    flexiv_targets = act_next[:Flexiv_DOF]
                    action = flexiv_targets[:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                    # print('action',np.asarray(act_dof).shape)
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_next.tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))

            # ===== sync joint state =====
            if test_on_real_robot:
                joints = get_robot_joints(robot)
                # env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
            time.sleep(0.5)


def flexiv_leaphand_diffusion_replay(task: TeleopPlayer):

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    cam_bird = CameraD400(1)
    # cam_left = CameraD400(2)
    cam_right = CameraD400(0)

    task.init_ee_state = None

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints

    if test_on_real_robot:
        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    if test_on_real_hand:
        leap_hand = LeapNode()
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)

    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    # ================== Diffusion Actor ====================
    from torch.utils.data import DataLoader
    from .diffusha_flexiv.diffusion.train import Trajectory
    from .diffusha_flexiv.config.default_args import Args
    from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
    from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor
    from .diffusion_unet import set_seed, Diffusion

    device = env.device
    upper_limits = torch.tensor(
        [
            2.7925,
            2.2689,
            2.9671,
            2.6878,
            2.9671,
            4.5379,
            2.9671,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            2.0940,
            2.4430,
            1.2000,
            1.3400,
        ],
        dtype=torch.float32,
    )
    lower_limits = torch.tensor(
        [
            -2.7925,
            -2.2689,
            -2.9671,
            -1.8675,
            -2.9671,
            -1.3963,
            -2.9671,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -0.3490,
            -0.4700,
            -1.9000,
            -1.8800,
        ],
        dtype=torch.float32,
    )

    model_pt = "/home/rhos_pub/Desktop/flexiv_training/logs/diffusion_unet/policy_epoch_1600_seed_233.ckpt"

    diffusion = Diffusion()
    set_seed(233)

    state_dict = torch.load(model_pt)
    diffusion.load_state_dict(state_dict)
    # diffusion_actor = DiffusionAssistedActor(-1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio)
    # ======================= End of Diffusion Actor ==================

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")
    i = 0

    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        state = np.zeros(28)

        hand_joints = leap_hand.read_pos() - 3.14

        robot_states = flexivrdk.RobotStates()
        robot.getRobotStates(robot_states)
        state = np.concatenate([robot_states.q, hand_joints])

        state = state.astype(np.float32)
        state = torch.tensor(state).to(device)

        # =============== Diffusion Robot ============

        bird_img, _ = cam_bird.get_data()
        bird_img = cv2.resize(bird_img, (320, 240))
        bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
        print(bird_img)
        bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)

        right_img, _ = cam_right.get_data()
        right_img = cv2.resize(right_img, (320, 240))
        right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
        right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)

        batch = (torch.tensor(state).view(1, -1), bird_img, right_img)

        agent_action = diffusion(batch).cpu().squeeze()
        start_sample_time = time.time()  # TODO: neglect network inference time
        # ==================== End Diffusion Robot ========================

        act_dof = np.zeros((agent_action.shape[0], 28))
        act_dof[:, :7] = agent_action[:, :7]

        for i, key in enumerate([12, 22, 27]):
            tip = key
            tip = tip - 1
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3

            act_dof[:, dip] = agent_action[:, i * 4 + 7]
            act_dof[:, pip] = agent_action[:, i * 4 + 1 + 7]
            act_dof[:, finger_tip] = agent_action[:, i * 4 + 2 + 7]
            act_dof[:, tip] = agent_action[:, i * 4 + 3 + 7]

        tip = 16
        finger_tip = tip - 1
        dip = tip - 2
        pip = tip - 3
        act_dof[:, pip] = agent_action[:, 12 + 7]
        act_dof[:, dip] = agent_action[:, 13 + 7]
        act_dof[:, finger_tip] = agent_action[:, 14 + 7] * -1
        act_dof[:, tip] = agent_action[:, 15 + 7] * -1

        # agent_action = agent_action[:4].mean(dim=0).unsqueeze(0)
        for act_next in act_dof:
            env.cur_targets[:, :DOF] = torch.from_numpy(act_next[:DOF]).to("cpu")
            # env.cur_targets[:, :DOF] = 0.9 * env.cur_targets[:, :DOF] + 0.1 * act_dof[..., :DOF] # TODO: blend with original human action

            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

            if not recording_data:  # TODO: turn on when testing on real robot
                break

            # if time.time() - start_sample_time > dt:
            #     get_obs(task)
            #     start_sample_time = time.time()

            if test_on_real_robot:
                try:
                    flexiv_targets = act_next[:Flexiv_DOF]
                    action = flexiv_targets[:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                    # print('action',np.asarray(act_dof).shape)
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_next.tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))
            task.step()
            handle_viewer_events(task, viewer)
            draw_bounding_box()

        # ===== sync joint state =====
        if test_on_real_robot:
            joints = get_robot_joints(robot)
            # env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
        time.sleep(0.5)


def sim_move_gripper(env, width: float):

    env.cur_targets[:, 7] = width
    """
    8: left_inner_knuckle_joint
    9: left_outer_knuckle_joint
    10: left_inner_finger_joint
    11: right_inner_knuckle_joint
    12: right_outer_knuckle_joint
    13: right_inner_finger_joint
    """

    env.cur_targets[:, 9] = 9.404 * width - 0.155
    env.cur_targets[:, 8] = env.cur_targets[:, 9]
    env.cur_targets[:, 10] = -env.cur_targets[:, 9]

    env.cur_targets[:, 12] = env.cur_targets[:, 9]
    env.cur_targets[:, 11] = env.cur_targets[:, 12]
    env.cur_targets[:, 13] = -env.cur_targets[:, 12]


def leap_teleop(task: TeleopPlayer):
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from leap.LeapDualJointDetector import Detector
    from .calibration.camera import CameraD400

    task.detector = Detector(up_axis="y", scale=0.001)
    ik = IKController(damping=0.05)
    cam_agent = CameraD400(0)
    # cam_wrist = CameraD400(1)

    task.init_ee_state = None
    last_delta_pos = np.zeros(6)
    last_wrist_pos = np.zeros(3)
    last_detect_time = time.time()

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = -1

    if test_on_real_robot:
        self_exam(log)
        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        global gripper
        gripper = flexivrdk.Gripper(robot)
        try:
            gripper.move(0.09, 0.1, 20)
        except Exception as e:
            log.error(str(e))

        init_joints = get_robot_joints(robot)
        DOF = len(init_joints)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 7
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()
    sim_gripper_width = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        joints = task.detector.detect_right_joints()
        hand_detected = (joints is not None) and (len(joints) > 0)

        # if (not recording_data) or (not hand_detected):
        #     continue

        if time.time() - start_sample_time > dt:
            # get_obs(task, cam_agent, cam_wrist)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if task.initialized:
            cur_state = torch.squeeze(
                env.rigid_body_states[..., : env.num_flexiv_bodies]
            )
            if task.init_ee_state is None:
                task.init_ee_state = cur_state[8].clone().cpu()

            # === X, Y, Z ===
            if hand_detected:
                # use leap_lower_bound and leap_upper_bound to cast joints['wrist'] into [-1, 1] real number
                delta_pos = (
                    joints["right_wrist"] - task.prev_motion_joint["right_wrist"]
                ) * scaling_factor
                delta_pos = leap_axis_correction(delta_pos)

                # ==== avoid jitter ====
                # delta_pos_history.append(delta_pos)
                # delta_pos_history = delta_pos_history[-20:]
                # dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
                # if dis <= 0.02:
                #     delta_pos_history = [delta_pos_history[0]] * 20
                #     delta_pos = delta_pos_history[-1]
                # ======================

                dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
                mask = torch.zeros(
                    (env.num_flexiv_bodies, 6), dtype=bool, device=env.device
                )

                cur_rb_idx = 8
                cur_pos = cur_state[cur_rb_idx][:3]
                cur_rot_q = cur_state[cur_rb_idx][3:7]
                cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
                z_axis = cur_rot_mat[:, 2]

                target_pos = task.init_ee_state[:3] + delta_pos

                dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(
                    env.device
                )
                mask[cur_rb_idx, :3] = True

                palm_norm = leap_axis_correction(joints["right_palm_normal"])
                palm_norm /= np.linalg.norm(palm_norm)  # z-axis

                arm_dir = leap_axis_correction(
                    -joints["middle_pos"][0] + joints["right_wrist"]
                )
                arm_dir /= np.linalg.norm(arm_dir)  # x-axis

                target_rot_mat = np.stack(
                    [arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1
                )
                target_rot_q = R.from_matrix(target_rot_mat).as_quat()
                target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)
                # print(np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3]))

                finger_dis = np.linalg.norm(
                    joints["thumb_pos"][3] - joints["index_pos"][3]
                )

                target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
                task.draw_line(
                    target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0)
                )
                task.draw_line(
                    target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0)
                )
                task.draw_line(
                    target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1)
                )

                orn = orientation_error(target_rot_q, cur_rot_q)
                dpose[cur_rb_idx, 3:] = orn
                mask[cur_rb_idx, 3:] = True

                tcp_pos = target_pos + z_axis * 0.15
                nrm = np.linalg.norm(
                    dpose[cur_rb_idx] - last_delta_pos
                )  # filter human hand noise

                # if time.time() - last_detect_time > 1:
                #     task.prev_motion_joint = joints
                #     task.init_ee_state = cur_state[8].clone().cpu()
                #     last_detect_time = time.time()

                # if not is_validpos(tcp_pos):
                #     task.prev_motion_joint = joints
                #     task.init_ee_state = cur_state[8].clone().cpu()
                #     continue

                if not is_validpos(tcp_pos):
                    tcp_pos = torch.clamp(tcp_pos, tcp_lower_bound, tcp_upper_bound)
                    target_pos = tcp_pos - z_axis * 0.15
                    dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(
                        env.device
                    )

                task.draw_sphere(target_pos)
                task.draw_sphere(tcp_pos, color=(0, 0, 0))

                if nrm < 0.003:
                    dpose[cur_rb_idx] = last_delta_pos

                last_delta_pos = dpose[cur_rb_idx]

                fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
                fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

                action = ik.control_ik(dpose, fake_jacobian, mask)
                next_dof_pos = env.flexiv_dof_pos + action

                act_dof = (
                    0.95 * env.cur_targets[:, : env.num_flexiv_dofs]
                    + 0.05 * next_dof_pos
                )
                act_dof = tensor_clamp(
                    act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
                )

                env.cur_targets[:, :DOF] = act_dof[..., :DOF]

                if finger_dis > 0.07 and env.cur_targets[:, DOF] < 0.03:
                    sim_gripper_width = 0.09
                elif finger_dis < 0.04 and env.cur_targets[:, DOF] > 0.08:
                    sim_gripper_width = 0

                print("@@@ ", env.cur_targets[:, :14])
                sim_move_gripper(env, sim_gripper_width)
                print(sim_gripper_width, env.cur_targets[:, :14])
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )

                if test_on_real_robot:
                    try:
                        action = act_dof[0][:DOF].tolist()
                        robot.sendJointPosition(
                            action, target_vel, target_acc, MAX_VEL, MAX_ACC
                        )

                        # open gripper
                        if (
                            finger_dis > 0.07
                            and get_gripper_width(gripper) < 0.05
                            and not is_gripper_moving(gripper)
                        ):
                            gripper.move(0.09, 0.1, 20)
                        # close gripper
                        if (
                            finger_dis < 0.03
                            and get_gripper_width(gripper) > 0.06
                            and not is_gripper_moving(gripper)
                        ):
                            gripper.grasp(5)

                    except Exception as e:
                        log.error(str(e))

                # ===== sync joint state =====
                if test_on_real_robot:
                    joints = get_robot_joints(robot)
                    env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(
                        device
                    )
                    gym.set_dof_position_target_tensor(
                        sim, gymtorch.unwrap_tensor(env.cur_targets)
                    )

        if not task.initialized:
            if hand_detected:
                task.initialized = True
                print("\033[91mSampling human pose, please try to stay still...\033[0m")
                time.sleep(1)
                joints = task.detector.detect_right_joints()
                print("\033[92mSampled\033[0m")
                task.prev_motion_joint = joints


def flexiv_leap_teleop(task: TeleopPlayer, obs=True):
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from leap.LeapDualJointDetector import Detector
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    task.detector = Detector(up_axis="y", scale=0.001)
    ik = IKController(damping=0.05)
    # cam_agent = CameraD400(0)
    # cam_wrist = CameraD400(1)

    if not obs:
        cam_bird = CameraD400(1)
        cam_left = CameraD400(2)
        cam_right = CameraD400(0)

    task.init_ee_state = None

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = 28

    if test_on_real_robot:
        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)
    if test_on_real_hand:
        leap_hand = LeapNode()
    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()
    sim_gripper_width = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        joints = task.detector.detect_right_joints()
        hand_detected = (joints is not None) and (len(joints) > 0)

        # if (not recording_data) or (not hand_detected):
        #     continue

        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if task.initialized:
            cur_state = torch.squeeze(
                env.rigid_body_states[..., : env.num_flexiv_bodies, :3]
            )
            if task.init_hand_wrist_joint is None:
                state = cur_state
                tip_state = state[env.nail_handles]
                wrist_state = state[8]
                task.init_tip_state = tip_state.clone().cpu()
                task.init_hand_wrist_joint = wrist_state.clone().cpu()
            task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)

            # # === X, Y, Z ===
            if hand_detected:
                rate = (
                    joints["right_wrist"] - task.prev_motion_joint["right_wrist"]
                ) / (leap_upper_bound - leap_lower_bound)
                delta_pos = scale(
                    torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound
                )
                delta_pos = leap_axis_correction(delta_pos)
                target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]

                dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
                mask = torch.zeros(
                    (env.num_flexiv_bodies, 6), dtype=bool, device=env.device
                )
                # print(target_wrist_pos.shape,wrist_state.shape)

                dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(
                    env.device
                )
                mask[8, :3] = True
                task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))

                for i, key in enumerate(["index", "thumb", "middle", "ring"]):
                    # for i, key in enumerate(['thumb']):
                    k = key + "_pos"
                    cur_rb_tip_idx = env.nail_handles[i]  # nail_handles:[10,14,18,22]
                    cur_pos = cur_state[cur_rb_tip_idx]
                    real_finger_len = (
                        np.linalg.norm(joints[k][0] - joints[k][1])
                        + np.linalg.norm(joints[k][2] - joints[k][1])
                        + np.linalg.norm(joints[k][3] - joints[k][2])
                        + np.linalg.norm(joints[k][0] - joints["right_wrist"])
                    )
                    # leap_hand_finger_len = 0.298 + 0.295
                    leap_hand_finger_len = 0.23
                    if key == "ring":
                        leap_hand_finger_len *= 1.05
                    if key == "index":
                        leap_hand_finger_len *= 1.1

                    if key == "thumb":
                        leap_hand_finger_len *= 0.8
                    # leap_hand_finger_len /= 2.0
                    target_pos = (
                        target_wrist_pos
                        + leap_axis_correction(joints[k][-1] - joints["right_wrist"])
                        * leap_hand_finger_len
                        / real_finger_len
                    )
                    # target_pos = target_wrist_pos + (joints[k][-1] - joints['right_wrist'])
                    dpose[cur_rb_tip_idx, :3] = (
                        target_pos.to(env.device)
                    ) - cur_pos.to(env.device)
                    mask[cur_rb_tip_idx, :3] = True

                    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
                    task.draw_sphere(target_pos, color=colors[i])

                fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
                fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

                action = ik.control_ik(dpose, fake_jacobian, mask)
                next_dof_pos = env.flexiv_dof_pos + action

                act_dof = (
                    0.95 * env.cur_targets[:, : env.num_flexiv_dofs]
                    + 0.05 * next_dof_pos
                )
                act_dof = tensor_clamp(
                    act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
                )

                # env.cur_targets[:, :DOF] = act_dof[..., :DOF]
                env.cur_targets[:, :DOF] = act_dof[..., :DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )

                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[0][:Flexiv_DOF].tolist()
                        robot.sendJointPosition(
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))
                if test_on_real_hand:
                    try:
                        action = act_dof[0].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))
                # ===== sync joint state =====
                # if test_on_real_robot:
                # joints = get_robot_joints(robot)
                # env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
                # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

        if not task.initialized:
            if hand_detected:
                task.initialized = True
                print("\033[91mSampling human pose\033[0m")
                time.sleep(1)
                joints = task.detector.detect_right_joints()
                print("\033[92mSampled\033[0m")
                task.prev_motion_joint = joints


def flexiv_leap_teleop_dexretarget(task: TeleopPlayer, obs=False):

    obs_buf = []

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    print(device)
    if obs:
        import pickle

        with open(filepath, "rb") as f:
            datalist = pickle.load(f)
        for data in datalist:
            # obs_data = data['obs']
            obs_data = data["hand_joints"]
            obs_data[:7] = data["joints"]
            obs_data = torch.tensor(obs_data.reshape((1, 28)))
            obs_buf.append(obs_data)
        i = 0
    else:
        state_dim = action_dim = 23
        set_seed(42)
        agent = BC(
            state_dim=state_dim,
            action_dim=action_dim * 16,
            img_dim=64,
            actor_lr=1e-3,
            hidden_dim=256,
            device=device,
        )
        load_ckpt(agent, expname="logs/push lr 1e-3", ckpt_idx=-1)
        # lower_limits=task.env.flexiv_dof_lower_limits
        # upper_limits=task.env.flexiv_dof_upper_limits
        upper_limits = torch.tensor(
            [
                2.7925,
                2.2689,
                2.9671,
                2.6878,
                2.9671,
                4.5379,
                2.9671,
                1.0470,
                2.2300,
                1.8850,
                2.0420,
                1.0470,
                2.2300,
                1.8850,
                2.0420,
                1.0470,
                2.2300,
                1.8850,
                2.0420,
                2.0940,
                2.4430,
                1.2000,
                1.3400,
            ],
            dtype=torch.float32,
        )
        lower_limits = torch.tensor(
            [
                -2.7925,
                -2.2689,
                -2.9671,
                -1.8675,
                -2.9671,
                -1.3963,
                -2.9671,
                -1.0470,
                -0.3140,
                -0.5060,
                -0.3660,
                -1.0470,
                -0.3140,
                -0.5060,
                -0.3660,
                -1.0470,
                -0.3140,
                -0.5060,
                -0.3660,
                -0.3490,
                -0.4700,
                -1.9000,
                -1.8800,
            ],
            dtype=torch.float32,
        )

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    if not obs:
        cam_bird = CameraD400(1)
        # cam_left = CameraD400(2)
        cam_right = CameraD400(0)

    task.init_ee_state = None

    # sync sim & real robot joints
    DOF = 28

    if test_on_real_robot:
        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    if test_on_real_hand:
        leap_hand = LeapNode()
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")

    i = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        begin_time = time.time()
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()
        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        # if (not recording_data) or (not hand_detected):
        #     continue
        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        if obs:
            act_dof = obs_buf[i]
            i += 1
        else:
            # state=np.zeros(28)

            # hand_joints = leap_hand.read_pos()-3.14

            # set_pose=np.zeros(28)
            # for i, key in enumerate([12,22,27]):
            #     tip = key
            #     tip = tip-1
            #     finger_tip = tip-1
            #     dip = tip - 2
            #     pip = tip - 3

            #     state[dip] = hand_joints[i*4]
            #     state[pip] = hand_joints[i*4+1]
            #     state[finger_tip] = hand_joints[i*4+2]
            #     state[tip] = hand_joints[i*4+3]

            # tip = 16
            # finger_tip = tip - 1
            # dip = tip - 2
            # pip = tip - 3
            # state[pip] = hand_joints[12]
            # state[dip] = hand_joints[13]
            # state[finger_tip] = hand_joints[14]*-1
            # state[tip] = hand_joints[15]*-1

            # robot_states = flexivrdk.RobotStates()
            # robot.getRobotStates(robot_states)
            # state[:7] = robot_states.q

            # state = state.astype(np.float32)

            hand_joints = leap_hand.read_pos() - 3.14

            robot_states = flexivrdk.RobotStates()
            robot.getRobotStates(robot_states)
            state = np.concatenate([robot_states.q, hand_joints])

            state = state.astype(np.float32)

            # state = np.array(task.env.obs_buf[:,:28]).astype(np.float32).reshape((28))
            print(state)
            state = torch.tensor(state).to(device)

            bird_img, _ = cam_bird.get_data()
            bird_img = cv2.resize(bird_img, (320, 240))
            bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
            bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)

            # left_img, _ = cam_left.get_data()
            # left_img = cv2.resize(left_img, (320, 240))
            # left_img = left_img.astype(np.float32).transpose((2, 0, 1)) / 255
            # left_img = torch.from_numpy(left_img).unsqueeze(0).to(device)

            right_img, _ = cam_right.get_data()
            right_img = cv2.resize(right_img, (320, 240))
            right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
            right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)

            act_time = time.time()
            act_dof = agent.get_action(state.unsqueeze(0), bird_img, right_img)
            act_dof = torch.from_numpy(act_dof).to("cpu")

            act_dof = torch.clamp(
                act_dof, -torch.ones_like(act_dof), torch.ones_like(act_dof)
            ).reshape(-1, action_dim)
            act_dof = scale(act_dof, lower=lower_limits, upper=upper_limits)

            actions = torch.zeros((act_dof.shape[0], 28))
            actions[:, :7] = act_dof[:, :7]

            for i, key in enumerate([12, 22, 27]):
                tip = key
                tip = tip - 1
                finger_tip = tip - 1
                dip = tip - 2
                pip = tip - 3

                actions[:, dip] = act_dof[:, i * 4 + 7]
                actions[:, pip] = act_dof[:, i * 4 + 1 + 7]
                actions[:, finger_tip] = act_dof[:, i * 4 + 2 + 7]
                actions[:, tip] = act_dof[:, i * 4 + 3 + 7]

            tip = 16
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3
            actions[:, pip] = act_dof[:, 12 + 7]
            actions[:, dip] = act_dof[:, 13 + 7]
            actions[:, finger_tip] = act_dof[:, 14 + 7] * -1
            actions[:, tip] = act_dof[:, 15 + 7] * -1

            act_dof = actions
            print("act", time.time() - act_time)

        if obs:
            env.cur_targets[:, :DOF] = act_dof[:, :DOF]
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )
            if test_on_real_robot:
                try:
                    flexiv_targets = act_dof[..., :Flexiv_DOF]
                    action = flexiv_targets[0][:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                    # print('action',np.asarray(act_dof).shape)
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_dof[0].tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))
            time.sleep(0.5)

        else:
            for idx, act in enumerate(act_dof):
                env.cur_targets[:, :DOF] = act_dof[idx, :DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )
                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[idx][:Flexiv_DOF].tolist()
                        robot.sendJointPosition(
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))

                if test_on_real_hand:
                    try:
                        action = act_dof[idx].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))
                task.step()
                time.sleep(0.5)

        # ===== sync joint state =====
        if test_on_real_robot:
            # joints = get_robot_joints(robot)
            # hand_joints = get_hand_joints(leap_hand)
            # print('cur:',env.cur_targets[:,isaac_joints],hand_joints/180*np.pi)
            # env.cur_targets[:, :7] = torch.from_numpy(np.array(joints)).to(device)
            # env.cur_targets[:,isaac_joints] = torch.from_numpy(np.array(hand_joints)[retargeting_joints]).to(device)

            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

        print("loop:", time.time() - begin_time)


def flexiv_leap_act_replay(task: TeleopPlayer, obs=False):
    from .act.imitate_episodes import ACTInference

    obs_buf = []

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    if obs:
        import pickle

        with open(filepath, "rb") as f:
            datalist = pickle.load(f)
        for data in datalist:
            # obs_data = data['obs']
            obs_data = data["hand_joints"]
            obs_data[:7] = data["joints"]
            obs_data = torch.tensor(obs_data.reshape((1, 28)))
            obs_buf.append(obs_data)
        i = 0
    else:
        state_dim = action_dim = 23
        # lower_limits=task.env.flexiv_dof_lower_limits
        # upper_limits=task.env.flexiv_dof_upper_limits

        act = ACTInference()

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    cam_bird = CameraD400(135122074278)
    cam_right = CameraD400(233522075695)

    task.init_ee_state = None

    # sync sim & real robot joints
    DOF = 28

    if test_on_real_robot:
        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    if test_on_real_hand:
        global leap_hand
        leap_hand = LeapNode()
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)

    i = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer, act)
        draw_bounding_box()
        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        # if (not recording_data) or (not hand_detected):
        #     continue
        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        if obs:
            act_dof = obs_buf[i]
            i += 1
        else:

            hand_joints = leap_hand.read_pos() - 3.14

            robot_states = flexivrdk.RobotStates()
            robot.getRobotStates(robot_states)
            state = np.concatenate([robot_states.q, hand_joints])

            state = state.astype(np.float32)

            # state = np.array(task.env.obs_buf[:,:28]).astype(np.float32).reshape((28))
            # print(state)
            state = torch.tensor(state).to(device)

            bird_img, _ = cam_bird.get_data()
            bird_img = cv2.resize(bird_img, (320, 240))
            bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
            bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)

            # left_img, _ = cam_left.get_data()
            # left_img = cv2.resize(left_img, (320, 240))
            # left_img = left_img.astype(np.float32).transpose((2, 0, 1)) / 255
            # left_img = torch.from_numpy(left_img).unsqueeze(0).to(device)

            right_img, _ = cam_right.get_data()
            right_img = cv2.resize(right_img, (320, 240))
            right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
            right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)

            '''cameras'''
            # img = torch.stack([bird_img, right_img], dim=1)
            '''side camera'''
            img = torch.stack([right_img], dim=1)
            '''top camera'''
            # img = torch.stack([bird_img], dim=1)

            act_dof = act.inference(torch.tensor(state).cpu(), img)
            act_dof = torch.from_numpy(act_dof).to("cpu")

            actions = torch.zeros((act_dof.shape[0], 28))
            actions[:, :7] = act_dof[:, :7]

            for i, key in enumerate([12, 22, 27]):
                tip = key
                tip = tip - 1
                finger_tip = tip - 1
                dip = tip - 2
                pip = tip - 3

                actions[:, dip] = act_dof[:, i * 4 + 7]
                actions[:, pip] = act_dof[:, i * 4 + 1 + 7]
                actions[:, finger_tip] = act_dof[:, i * 4 + 2 + 7]
                actions[:, tip] = act_dof[:, i * 4 + 3 + 7]

            tip = 16
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3
            actions[:, pip] = act_dof[:, 12 + 7]
            actions[:, dip] = act_dof[:, 13 + 7]
            actions[:, finger_tip] = act_dof[:, 14 + 7] * -1
            actions[:, tip] = act_dof[:, 15 + 7] * -1

            act_dof = actions

        if obs:
            env.cur_targets[:, :DOF] = act_dof[:, :DOF]
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )
            if test_on_real_robot:
                try:
                    flexiv_targets = act_dof[..., :Flexiv_DOF]
                    action = flexiv_targets[0][:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                    # print('action',np.asarray(act_dof).shape)
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_dof[0].tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))
            time.sleep(0.5)

        else:
            # for idx, _ in enumerate(act_dof):
            for idx in range(1):
                env.cur_targets[:, :DOF] = act_dof[idx, :DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )
                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[idx][:Flexiv_DOF].tolist()
                        robot.sendJointPosition(
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))

                if test_on_real_hand:
                    try:
                        action = act_dof[idx].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))
                task.step()

        # ===== sync joint state =====
        if test_on_real_robot:
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )



def flexiv_leap_RISE_replay(task: TeleopPlayer, obs=False):
    from .RISE.agent import RISE_agent

    obs_buf = []

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    print(device)
    if obs:
        import pickle

        with open(filepath, "rb") as f:
            datalist = pickle.load(f)
        for data in datalist:
            # obs_data = data['obs']
            obs_data = data["hand_joints"]
            obs_data[:7] = data["joints"]
            obs_data = torch.tensor(obs_data.reshape((1, 28)))
            obs_buf.append(obs_data)
        i = 0
    else:
        state_dim = action_dim = 23
        # lower_limits=task.env.flexiv_dof_lower_limits
        # upper_limits=task.env.flexiv_dof_upper_limits

        policy = RISE_agent()

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    if not obs:
        cam_bird = CameraD400(1)
        # cam_left = CameraD400(2)
        cam_right = CameraD400(0)

    task.init_ee_state = None

    # sync sim & real robot joints
    DOF = 28

    if test_on_real_robot:
        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    if test_on_real_hand:
        leap_hand = LeapNode()
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")

    i = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()
        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        # if (not recording_data) or (not hand_detected):
        #     continue
        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        if obs:
            act_dof = obs_buf[i]
            i += 1
        else:

            hand_joints = leap_hand.read_pos() - 3.14

            robot_states = flexivrdk.RobotStates()
            robot.getRobotStates(robot_states)
            state = np.concatenate([robot_states.q, hand_joints])

            state = state.astype(np.float32)

            # state = np.array(task.env.obs_buf[:,:28]).astype(np.float32).reshape((28))
            print(state)
            state = torch.tensor(state).to(device)

            bird_img, bird_depth = cam_bird.get_data()

            # left_img, _ = cam_left.get_data()
            # left_img = cv2.resize(left_img, (320, 240))
            # left_img = left_img.astype(np.float32).transpose((2, 0, 1)) / 255
            # left_img = torch.from_numpy(left_img).unsqueeze(0).to(device)

            right_img, right_depth = cam_right.get_data()

            act_time = time.time()
            # act_dof = policy.predict(
            #     [bird_img, right_img],
            #     [bird_depth, right_depth],
            #     [cam_bird.getIntrinsics(), cam_right.getIntrinsics()],
            # )
            act_dof = policy.predict(
                [bird_img], [bird_depth], [cam_bird.getIntrinsics()]
            )
            act_dof = torch.from_numpy(act_dof).to("cpu")
            print("act", time.time() - act_time)

            # act_dof = torch.clamp(act_dof, -torch.ones_like(act_dof), torch.ones_like(act_dof)).reshape(-1, action_dim)
            # act_dof = scale(act_dof, lower=lower_limits, upper=upper_limits)

            actions = torch.zeros((act_dof.shape[0], 28))
            actions[:, :7] = act_dof[:, :7]

            for i, key in enumerate([12, 22, 27]):
                tip = key
                tip = tip - 1
                finger_tip = tip - 1
                dip = tip - 2
                pip = tip - 3

                actions[:, dip] = act_dof[:, i * 4 + 7]
                actions[:, pip] = act_dof[:, i * 4 + 1 + 7]
                actions[:, finger_tip] = act_dof[:, i * 4 + 2 + 7]
                actions[:, tip] = act_dof[:, i * 4 + 3 + 7]

            tip = 16
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3
            actions[:, pip] = act_dof[:, 12 + 7]
            actions[:, dip] = act_dof[:, 13 + 7]
            actions[:, finger_tip] = act_dof[:, 14 + 7] * -1
            actions[:, tip] = act_dof[:, 15 + 7] * -1

            act_dof = actions

        if obs:
            env.cur_targets[:, :DOF] = act_dof[:, :DOF]
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )
            if test_on_real_robot:
                try:
                    flexiv_targets = act_dof[..., :Flexiv_DOF]
                    action = flexiv_targets[0][:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                    # print('action',np.asarray(act_dof).shape)
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_dof[0].tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))
            time.sleep(0.5)

        else:
            for idx, _ in enumerate(act_dof):
                # for idx in range(1):
                env.cur_targets[:, :DOF] = act_dof[idx, :DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )
                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[idx][:Flexiv_DOF].tolist()
                        robot.sendJointPosition(
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))

                if test_on_real_hand:
                    try:
                        action = act_dof[idx].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))
                task.step()
                time.sleep(0.5)

        # ===== sync joint state =====
        if test_on_real_robot:
            # joints = get_robot_joints(robot)
            # hand_joints = get_hand_joints(leap_hand)
            # print('cur:',env.cur_targets[:,isaac_joints],hand_joints/180*np.pi)
            # env.cur_targets[:, :7] = torch.from_numpy(np.array(joints)).to(device)
            # env.cur_targets[:,isaac_joints] = torch.from_numpy(np.array(hand_joints)[retargeting_joints]).to(device)

            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

        print("loop:", time.time() - begin_time)


def vr_axis_correction(vec):
    """-yzx coordinate -> xyz coordinate"""
    vec = vec[..., [2, 0, 1]]
    vec[..., 1] = -vec[..., 1]
    return vec


def vr_gripper_teleop(task: TeleopPlayer):
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .vr_detector import AllegroHandDetector
    from .calibration.camera import CameraD400
    from .oculus_streamer import OculusStreamer

    task.detector = AllegroHandDetector()
    streamer = OculusStreamer()
    ik = IKController(damping=0.05)
    # cam_agent = CameraD400(0)
    # cam_wrist = CameraD400(1)

    task.init_ee_state = None
    last_delta_pos = np.zeros(6)
    last_wrist_pos = np.zeros(3)
    last_detect_time = time.time()

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = -1

    if test_on_real_robot:
        self_exam(log)
        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        global gripper
        gripper = flexivrdk.Gripper(robot)
        try:
            gripper.move(0.09, 0.1, 20)
        except Exception as e:
            log.error(str(e))

        init_joints = get_robot_joints(robot)
        DOF = len(init_joints)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 7
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
        camera.py,
    )

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()
    sim_gripper_width = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        color_image = None  # TODO: replace with task ...
        while color_image is None:
            color_image = env.gym.get_camera_image_gpu_tensor(
                sim, env.envs[0], env.camera_handle, gymapi.IMAGE_COLOR
            )
            color_image = gymtorch.wrap_tensor(color_image)
            color_image = color_image.cpu().numpy()
            color_image = color_image[:, :, [2, 1, 0]]

        streamer.publish(color_image)

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        joints = task.detector.detect_right_joints()
        hand_detected = (joints is not None) and (len(joints) > 0)

        # if (not recording_data) or (not hand_detected):
        #     continue

        if time.time() - start_sample_time > dt:
            # get_obs(task, cam_agent, cam_wrist)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if task.initialized:
            cur_state = torch.squeeze(
                env.rigid_body_states[..., : env.num_flexiv_bodies]
            )
            if task.init_ee_state is None:
                task.init_ee_state = cur_state[8].clone().cpu()
                task.init_ee_state[:3] += torch.tensor([0.5, 0, -1])

            # === X, Y, Z ===
            if hand_detected:
                # use leap_lower_bound and leap_upper_bound to cast joints['wrist'] into [-1, 1] real number
                delta_pos = (
                    joints["thumb"][0] - task.prev_motion_joint["thumb"][0]
                ) * scaling_factor
                delta_pos = vr_axis_correction(delta_pos)

                # ==== avoid jitter ====
                # delta_pos_history.append(delta_pos)
                # delta_pos_history = delta_pos_history[-20:]
                # dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
                # if dis <= 0.02:
                #     delta_pos_history = [delta_pos_history[0]] * 20
                #     delta_pos = delta_pos_history[-1]
                # ======================

                dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
                mask = torch.zeros(
                    (env.num_flexiv_bodies, 6), dtype=bool, device=env.device
                )

                cur_rb_idx = 8
                cur_pos = cur_state[cur_rb_idx][:3]
                cur_rot_q = cur_state[cur_rb_idx][3:7]
                cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
                z_axis = cur_rot_mat[:, 2]

                target_pos = task.init_ee_state[:3] + delta_pos

                dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(
                    env.device
                )
                mask[cur_rb_idx, :3] = True

                palm_norm = np.cross(
                    vr_axis_correction(joints["index"][1] - joints["thumb"][0]),
                    vr_axis_correction(joints["ring"][1] - joints["thumb"][0]),
                )
                palm_norm /= np.linalg.norm(palm_norm)  # z-axis

                arm_dir = vr_axis_correction(-joints["middle"][1] + joints["thumb"][0])
                arm_dir /= np.linalg.norm(arm_dir)  # x-axis

                target_rot_mat = np.stack(
                    [arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1
                )
                target_rot_q = R.from_matrix(target_rot_mat).as_quat()
                target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)
                # print(np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3]))

                finger_dis = np.linalg.norm(joints["thumb"][-1] - joints["index"][-1])

                target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
                task.draw_line(
                    target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0)
                )
                task.draw_line(
                    target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0)
                )
                task.draw_line(
                    target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1)
                )

                orn = orientation_error(target_rot_q, cur_rot_q)
                dpose[cur_rb_idx, 3:] = orn
                mask[cur_rb_idx, 3:] = True

                tcp_pos = target_pos + z_axis * 0.15
                nrm = np.linalg.norm(
                    dpose[cur_rb_idx] - last_delta_pos
                )  # filter human hand noise

                # if time.time() - last_detect_time > 1:
                #     task.prev_motion_joint = joints
                #     task.init_ee_state = cur_state[8].clone().cpu()
                #     last_detect_time = time.time()

                # if not is_validpos(tcp_pos):
                #     task.prev_motion_joint = joints
                #     task.init_ee_state = cur_state[8].clone().cpu()
                #     continue

                if not is_validpos(tcp_pos):
                    tcp_pos = torch.clamp(tcp_pos, tcp_lower_bound, tcp_upper_bound)
                    target_pos = tcp_pos - z_axis * 0.15
                    dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(
                        env.device
                    )

                task.draw_sphere(target_pos)
                task.draw_sphere(tcp_pos, color=(0, 0, 0))

                if nrm < 0.003:
                    dpose[cur_rb_idx] = last_delta_pos

                last_delta_pos = dpose[cur_rb_idx]

                fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
                fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

                action = ik.control_ik(dpose, fake_jacobian, mask)
                next_dof_pos = env.flexiv_dof_pos + action

                act_dof = (
                    0.95 * env.cur_targets[:, : env.num_flexiv_dofs]
                    + 0.05 * next_dof_pos
                )
                act_dof = tensor_clamp(
                    act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
                )

                env.cur_targets[:, :DOF] = act_dof[..., :DOF]

                if finger_dis > 0.07 and env.cur_targets[:, DOF] < 0.03:
                    sim_gripper_width = 0.09
                elif finger_dis < 0.04 and env.cur_targets[:, DOF] > 0.08:
                    sim_gripper_width = 0

                print("@@@ ", env.cur_targets[:, :14])
                sim_move_gripper(env, sim_gripper_width)
                print(sim_gripper_width, env.cur_targets[:, :14])
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )

                if test_on_real_robot:
                    try:
                        action = act_dof[0][:DOF].tolist()
                        robot.sendJointPosition(
                            action, target_vel, target_acc, MAX_VEL, MAX_ACC
                        )

                        # open gripper
                        if (
                            finger_dis > 0.07
                            and get_gripper_width(gripper) < 0.05
                            and not is_gripper_moving(gripper)
                        ):
                            gripper.move(0.09, 0.1, 20)
                        # close gripper
                        if (
                            finger_dis < 0.03
                            and get_gripper_width(gripper) > 0.06
                            and not is_gripper_moving(gripper)
                        ):
                            gripper.grasp(5)

                    except Exception as e:
                        log.error(str(e))

                # ===== sync joint state =====
                if test_on_real_robot:
                    joints = get_robot_joints(robot)
                    env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(
                        device
                    )
                    gym.set_dof_position_target_tensor(
                        sim, gymtorch.unwrap_tensor(env.cur_targets)
                    )

        if not task.initialized:
            if hand_detected:
                task.initialized = True
                print("\033[91mSampling human pose, please try to stay still...\033[0m")
                time.sleep(1)
                joints = task.detector.detect_right_joints()
                print("\033[92mSampled\033[0m")
                task.prev_motion_joint = joints


def vr_teleop(task: TeleopPlayer):
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .vr_detector import AllegroHandDetector
    from .calibration.camera import CameraD400
    from .oculus_streamer import OculusStreamer

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    task.detector = AllegroHandDetector()
    streamer = OculusStreamer()
    ik = IKController(damping=0.05)
    # cam_agent = CameraD400(0)
    # cam_wrist = CameraD400(1)

    task.init_ee_state = None

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = -1

    if test_on_real_robot:
        self_exam(log)
        move_home(robot)
        # reset_home(task)

        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper
        # gripper = flexivrdk.Gripper(robot)

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    if test_on_real_hand:
        leap_hand = LeapNode()
    dt = 1.0 / frequency
    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    start_sample_time = time.time()

    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()
        color_image = None  # TODO: replace with task ...
        while color_image is None:
            color_image = env.gym.get_camera_image_gpu_tensor(
                sim, env.envs[0], env.camera_handle, gymapi.IMAGE_COLOR
            )
            color_image = gymtorch.wrap_tensor(color_image)
            color_image = color_image.cpu().numpy()
            color_image = color_image[:, :, [2, 1, 0]]

        streamer.publish(color_image)

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        joints_hand = task.detector.detect_right_joints()
        transformed_keypoints = task.detector.detect_transformed_joints()
        print(joints_hand)
        joints_arm = joints_hand["thumb"][0]
        hand_detected = (joints_hand is not None) and (len(joints_hand) > 0)

        # if (not recording_data) or (not hand_detected):
        #     continue

        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if task.initialized:
            cur_state = torch.squeeze(
                env.rigid_body_states[..., : env.num_flexiv_bodies]
            )
            if task.init_hand_wrist_joint is None:
                state = cur_state[..., :3]
                tip_state = state[env.nail_handles]
                wrist_state = state[8]
                task.init_tip_joint = tip_state.clone().cpu()
                task.init_hand_wrist_joint = wrist_state.clone().cpu() + torch.tensor(
                    [0.5, 0, -1]
                )
            task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)
            print("wrist_state:", wrist_state)

            # === X, Y, Z ===
            if hand_detected:
                rate = (joints_arm[:3] - task.prev_motion_joint[:3]) / (
                    vr_upper_bound - vr_lower_bound
                )
                print("init:", task.init_hand_wrist_joint)
                delta_pos = scale(
                    torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound
                )
                delta_pos = vr_axis_correction(delta_pos)
                target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]
                target_wrist_pos = np.clip(
                    target_wrist_pos, tcp_lower_bound, tcp_upper_bound
                )

                # ==== avoid jitter ====
                # delta_pos_history.append(delta_pos)
                # delta_pos_history = delta_pos_history[-20:]
                # dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
                # if dis <= 0.02:
                #     delta_pos_history = [delta_pos_history[0]] * 20
                #     delta_pos = delta_pos_history[-1]
                # ======================

                dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
                mask = torch.zeros(
                    (env.num_flexiv_bodies, 6), dtype=bool, device=env.device
                )

                dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(
                    env.device
                )
                mask[8, :3] = True
                task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))

                for i, key in enumerate(["index", "thumb", "middle", "ring"]):
                    cur_rb_tip_idx = env.nail_handles[i]
                    cur_pos = cur_state[cur_rb_tip_idx][:3]
                    real_finger_len = (
                        np.linalg.norm(joints_hand[key][0] - joints_hand[key][1])
                        + np.linalg.norm(joints_hand[key][2] - joints_hand[key][1])
                        + np.linalg.norm(joints_hand[key][3] - joints_hand[key][2])
                        + np.linalg.norm(joints_hand[key][4] - joints_hand[key][3])
                    )
                    if key == "thumb":
                        real_finger_len += np.linalg.norm(
                            joints_hand[key][5] - joints_hand[key][4]
                        )

                    leap_hand_finger_len = 0.23
                    if key == "ring":
                        leap_hand_finger_len *= 1.05
                    if key == "index":
                        leap_hand_finger_len *= 1.1
                    if key == "thumb":
                        leap_hand_finger_len *= 0.8

                    target_finger_pos = (
                        target_wrist_pos
                        + vr_axis_correction(joints_hand[key][-1] - joints_arm)
                        * leap_hand_finger_len
                        / real_finger_len
                    )

                    dpose[cur_rb_tip_idx, :3] = target_finger_pos.to(
                        env.device
                    ) - cur_pos.to(env.device)
                    mask[cur_rb_tip_idx, :3] = True

                    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
                    task.draw_sphere(target_finger_pos, color=colors[i])

                fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
                fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

                action = ik.control_ik(dpose, fake_jacobian, mask)
                next_dof_pos = env.flexiv_dof_pos + action

                act_dof = (
                    0.95 * env.cur_targets[:, : env.num_flexiv_dofs]
                    + 0.05 * next_dof_pos
                )
                act_dof = tensor_clamp(
                    act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
                )

                env.cur_targets[:, :DOF] = act_dof[..., :DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )

                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[0][:Flexiv_DOF].tolist()
                        robot.sendJointPosition(
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))
                if test_on_real_hand:
                    try:
                        action = act_dof[0].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))

                # ===== sync joint state =====
                # if test_on_real_robot:
                #     joints_arm = get_robot_joints(robot)
                #     env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints_arm)).to(device)
                #     gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

        if not task.initialized:
            if hand_detected:
                task.initialized = True
                print("\033[91mSampling human pose, please try to stay still...\033[0m")
                time.sleep(1)
                joints_hand = task.detector.detect_right_joints()
                joints_arm = joints_hand["thumb"][0]
                print("\033[92mSampled\033[0m")
                task.prev_motion_joint = joints_arm


def vr_teleop_dexretarget(task: TeleopPlayer):
    OPERATOR2MANO_RIGHT = np.array(
        [
            [0, 1, 0.0],
            [0, 0, 1],
            [-1, 0, 0],
        ]
    )
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .vr_detector import AllegroHandDetector
    from .calibration.camera import CameraD400
    from .oculus_streamer import OculusStreamer

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    task.detector = AllegroHandDetector()
    streamer = OculusStreamer()
    ik = IKController(damping=0.05)
    # cam_agent = CameraD400(0)
    # cam_wrist = CameraD400(1)

    task.init_ee_state = None

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints
    DOF = -1

    if test_on_real_robot:
        self_exam(log)
        move_home(robot)
        # reset_home(task)

        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper
        # gripper = flexivrdk.Gripper(robot)

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    if test_on_real_hand:
        leap_hand = LeapNode()
    dt = 1.0 / frequency
    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    start_sample_time = time.time()
    cfg = None
    with open(
        "/home/rhos_pub/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/cfg/retarget/leap_hand_right_.yaml"
    ) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    retargeter = RetargetingConfig.from_dict(cfg).build()

    retargeting_joint_names = retargeter.joint_names
    isaac_joint_names = gym.get_actor_dof_names(env.envs[0], task.env.flexivs[0])
    isaac_joint_indices = {}
    for name in retargeting_joint_names:
        if name in isaac_joint_names:
            isaac_joint_indices[retargeting_joint_names.index(name)] = (
                isaac_joint_names.index(name)
            )
    print(retargeting_joint_names, isaac_joint_names)
    isaac_joints = np.array(list(isaac_joint_indices.values()))
    retargeting_joints = np.array(list(isaac_joint_indices.keys()))

    print(retargeter.optimizer.target_link_human_indices)

    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()
        color_image = None  # TODO: replace with task ...
        while color_image is None:
            color_image = env.gym.get_camera_image_gpu_tensor(
                sim, env.envs[0], env.camera_handle, gymapi.IMAGE_COLOR
            )
            color_image = gymtorch.wrap_tensor(color_image)
            color_image = color_image.cpu().numpy()
            color_image = color_image[:, :, [2, 1, 0]]

        streamer.publish(color_image)

        handle_viewer_events(task, viewer)
        draw_bounding_box()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        joints_hand = task.detector.detect_right_joints()
        transformed_keypoints = task.detector.detect_transformed_joints()
        print(joints_hand)
        joints_arm = joints_hand["thumb"][0]
        hand_detected = (joints_hand is not None) and (len(joints_hand) > 0)

        # if (not recording_data) or (not hand_detected):
        #     continue

        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if task.initialized:
            cur_state = torch.squeeze(
                env.rigid_body_states[..., : env.num_flexiv_bodies]
            )
            if task.init_hand_wrist_joint is None:
                state = cur_state[..., :3]
                tip_state = state[env.nail_handles]
                wrist_state = state[8]
                task.init_tip_joint = tip_state.clone().cpu()
                task.init_hand_wrist_joint = wrist_state.clone().cpu() + torch.tensor(
                    [0.5, 0, -1]
                )
            task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)
            print("wrist_state:", wrist_state)

            # === X, Y, Z ===
            if hand_detected:
                rate = (joints_arm[:3] - task.prev_motion_joint[:3]) / (
                    vr_upper_bound - vr_lower_bound
                )
                print("init:", task.init_hand_wrist_joint)
                delta_pos = scale(
                    torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound
                )
                delta_pos = vr_axis_correction(delta_pos)
                target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]
                target_wrist_pos = np.clip(
                    target_wrist_pos, tcp_lower_bound, tcp_upper_bound
                )

                # ==== avoid jitter ====
                # delta_pos_history.append(delta_pos)
                # delta_pos_history = delta_pos_history[-20:]
                # dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
                # if dis <= 0.02:
                #     delta_pos_history = [delta_pos_history[0]] * 20
                #     delta_pos = delta_pos_history[-1]
                # ======================

                dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
                mask = torch.zeros(
                    (env.num_flexiv_bodies, 6), dtype=bool, device=env.device
                )

                dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(
                    env.device
                )
                mask[8, :3] = True
                task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))

                cur_rot_q = cur_state[8][3:7]
                cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
                z_axis = cur_rot_mat[:, 2]

                palm_norm = -np.cross(
                    vr_axis_correction(
                        joints_hand["index"][1] - joints_hand["thumb"][0]
                    ),
                    vr_axis_correction(
                        joints_hand["ring"][1] - joints_hand["thumb"][0]
                    ),
                )
                palm_norm /= np.linalg.norm(palm_norm)  # z-axis

                arm_dir = vr_axis_correction(
                    joints_hand["middle"][1] - joints_hand["thumb"][0]
                )
                arm_dir /= np.linalg.norm(arm_dir)  # x-axis

                target_rot_mat = np.stack(
                    [arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1
                )
                target_rot_q = R.from_matrix(target_rot_mat).as_quat()
                target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)

                target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
                task.draw_line(
                    target_wrist_pos,
                    target_wrist_pos + target_rot_mat[:, 0] * 0.1,
                    color=(1, 0, 0),
                )
                task.draw_line(
                    target_wrist_pos,
                    target_wrist_pos + target_rot_mat[:, 1] * 0.1,
                    color=(0, 1, 0),
                )
                task.draw_line(
                    target_wrist_pos,
                    target_wrist_pos + target_rot_mat[:, 2] * 0.1,
                    color=(0, 0, 1),
                )

                orn = orientation_error(target_rot_q, cur_rot_q)
                dpose[8, 3:] = orn
                mask[8, 3:] = True

                indices = (retargeter.optimizer.target_link_human_indices / 4).astype(
                    int
                )
                fingertip_pos = np.stack(
                    [
                        np.array(transformed_keypoints["thumb"][0])
                        @ OPERATOR2MANO_RIGHT,
                        np.array(transformed_keypoints["thumb"][-1])
                        @ OPERATOR2MANO_RIGHT,
                        np.array(transformed_keypoints["index"][-1])
                        @ OPERATOR2MANO_RIGHT,
                        np.array(transformed_keypoints["middle"][-1])
                        @ OPERATOR2MANO_RIGHT,
                        np.array(transformed_keypoints["ring"][-1])
                        @ OPERATOR2MANO_RIGHT,
                    ]
                )
                ref_pos = []
                ref_pos = fingertip_pos[indices[1, :]] - fingertip_pos[indices[0, :]]

                # for i, key in enumerate(['thumb','index','middle','ring']):
                # ref_pos.append(np.array(transformed_keypoints[key][-1])@OPERATOR2MANO_RIGHT)
                print("transformed:", transformed_keypoints)
                ref_pos = np.stack(ref_pos)
                qpos = retargeter.retarget(ref_pos)
                print("qpos:", qpos)
                # for i, key in enumerate(['index','thumb','middle','ring']):
                #        cur_rb_tip_idx           = env.nail_handles[i]
                #        cur_pos                  = cur_state[cur_rb_tip_idx][:3]
                #        real_finger_len = np.linalg.norm(joints_hand[key][0] - joints_hand[key][1]) + np.linalg.norm(joints_hand[key][2] - joints_hand[key][1]) \
                #                                + np.linalg.norm(joints_hand[key][3] - joints_hand[key][2]) + np.linalg.norm(joints_hand[key][4] - joints_hand[key][3])
                #        if key=='thumb':
                #            real_finger_len += np.linalg.norm(joints_hand[key][5] - joints_hand[key][4])
                #
                #        leap_hand_finger_len = 0.23
                #        if key == 'ring' :
                #            leap_hand_finger_len *= 1.05
                #        if key == 'index':
                #          leap_hand_finger_len *= 1.1
                #        if key == 'thumb':
                #         leap_hand_finger_len *= 0.8

                #        target_finger_pos = target_wrist_pos + vr_axis_correction(joints_hand[key][-1] - joints_arm) * leap_hand_finger_len / real_finger_len

                #        dpose[cur_rb_tip_idx, :3] = target_finger_pos.to(env.device) - cur_pos.to(env.device)
                #        mask[cur_rb_tip_idx, :3]  = True
                #
                #        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
                #        task.draw_sphere(target_finger_pos, color=colors[i])

                fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
                fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

                action = ik.control_ik(dpose, fake_jacobian, mask)

                next_dof_pos = env.flexiv_dof_pos + action

                next_dof_pos[:, isaac_joints] = torch.tensor(
                    qpos[retargeting_joints], dtype=torch.float
                )

                act_dof = (
                    0.95 * env.cur_targets[:, : env.num_flexiv_dofs]
                    + 0.05 * next_dof_pos
                )
                act_dof = tensor_clamp(
                    act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits
                )

                env.cur_targets[:, :DOF] = act_dof[..., :DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )

                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[0][:Flexiv_DOF].tolist()
                        robot.sendJointPosition(
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))
                if test_on_real_hand:
                    try:
                        action = act_dof[0].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))

                # ===== sync joint state =====
                # if test_on_real_robot:
                #     joints_arm = get_robot_joints(robot)
                #     env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints_arm)).to(device)
                #     gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

        if not task.initialized:
            if hand_detected:
                task.initialized = True
                print("\033[91mSampling human pose, please try to stay still...\033[0m")
                time.sleep(1)
                joints_hand = task.detector.detect_right_joints()
                joints_arm = joints_hand["thumb"][0]
                print("\033[92mSampled\033[0m")
                task.prev_motion_joint = joints_arm


def flexiv_leap_umi_diffusion_replay(task: TeleopPlayer):

    # CKPT_PATH = "/home/rhos_pub/tactile_umi_grav/umi/data/outputs/TeleOP+HumanCap(50)1227/17.07.21_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt"
    # CKPT_PATH = "/home/rhos_pub/tactile_umi_grav/umi/data/outputs/2025.01.02/18.29.21_train_diffusion_unet_timm_umi/checkpoints/epoch=0600-train_loss=0.019.ckpt"
    CKPT_PATH = '/home/rhos_pub/shengcheng/ckpts/MotionPlan100/DP/checkpoints/latest.ckpt'
    # CKPT_PATH = '/home/rhos_pub/tactile_umi_grav/umi/data/outputs/2025.01.09/01.02.15_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt'
    import dill
    import hydra

    sys.path.append(
        "/home/rhos_pub/LeapTele/DexCopilot/rl_games/rl_games/algos_torch/tactile_umi_grav"
    )
    from diffusion_policy.common.cv2_util import get_image_transform
    from diffusion_policy.common.pytorch_util import dict_apply
    from umi.real_world.real_inference_util import (
        get_real_umi_obs_dict,
        get_real_umi_action,
    )

    class Diffusion:
        def __init__(
            self,
            model_pt,
            device="cuda",
        ):

            payload = torch.load(
                open(model_pt, "rb"), map_location="cpu", pickle_module=dill
            )
            cfg = payload["cfg"]
            print("model_name:", cfg.policy.obs_encoder.model_name)
            print("dataset_path:", cfg.task.dataset.dataset_path)

            # creating model
            # have to be done after fork to prevent
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16  # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print("obs_pose_rep", obs_pose_rep)
            print("action_pose_repr", action_pose_repr)

            policy.eval().to(device)


            policy.reset()

            self.policy = policy
            self.device = device
            self.cfg = cfg
            self.obs_pose_rep = obs_pose_rep

            self.tf = get_image_transform(
                input_res=(640, 480),  # TODO
                output_res=(224, 224),
                # obs output rgb
                bgr_to_rgb=False,
            )

        def transform(self, img, obs_float32=True):
            img = np.ascontiguousarray(self.tf(img))
            if obs_float32:
                img = img.astype(np.float32) / 255
            img = np.moveaxis(img, -1, 0)  # from get_real_umi_obs_dict
            return img

        def forward_eval(self, states, bird_imgs, right_imgs, obs_horizon=2):
            bird_imgs = [self.transform(im) for im in bird_imgs]
            right_imgs = [self.transform(im) for im in right_imgs]

            obs = {
                "robot0_joints": states[:, :7].repeat(obs_horizon, 1),
                "robot0_leap_hand": states[:, 7:].repeat(obs_horizon, 1),
                "camera0_rgb": torch.from_numpy(np.stack(bird_imgs)).repeat(
                    obs_horizon, 1, 1, 1
                ),
                "camera1_rgb": torch.from_numpy(np.stack(right_imgs)).repeat(
                    obs_horizon, 1, 1, 1
                ),
            }

            # should be THWC
            # obs = get_real_umi_obs_dict(
            #     env_obs=obs, shape_meta=self.cfg.task.shape_meta,
            #     obs_pose_repr=self.obs_pose_rep)
            obs_dict = dict_apply(obs, lambda x: x.unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)

            # for key, val in obs_dict.items():
            #     print(key, val.shape)
            # print(obs_dict["camera0_rgb"].view(-1, 3).mean(0))
            # print(obs_dict["camera1_rgb"].view(-1, 3).mean(0))

            raw_action = result["action_pred"][0].detach().to("cpu")
            return raw_action

    # end class Diffusion

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    cam_bird = CameraD400(135122074278)
    cam_right = CameraD400(233522075695)

    task.init_ee_state = None

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    device = env.device
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)

    # sync sim & real robot joints

    if test_on_real_robot:
        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    dt = 1.0 / frequency
    # if test_on_real_hand:
    leap_hand = LeapNode()
    set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)

    print(
        "Sending command to robot at",
        frequency,
        "Hz, or",
        dt,
        "seconds interval",
    )

    # ================== Diffusion Actor ====================

    print(env.flexiv_dof_lower_limits.shape)
    device = env.device
    upper_limits = torch.tensor(
        [
            2.7925,
            2.2689,
            2.9671,
            2.6878,
            2.9671,
            4.5379,
            2.9671,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            1.0470,
            2.2300,
            1.8850,
            2.0420,
            2.0940,
            2.4430,
            1.2000,
            1.3400,
        ],
        dtype=torch.float32,
    )
    lower_limits = torch.tensor(
        [
            -2.7925,
            -2.2689,
            -2.9671,
            -1.8675,
            -2.9671,
            -1.3963,
            -2.9671,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -1.0470,
            -0.3140,
            -0.5060,
            -0.3660,
            -0.3490,
            -0.4700,
            -1.9000,
            -1.8800,
        ],
        dtype=torch.float32,
    )

    # lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
    # upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])

    diffusion = Diffusion(CKPT_PATH)
    set_seed(233)
    # ======================= End of Diffusion Actor ==================

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")
    i = 0

    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box(task, tcp_upper_bound, tcp_lower_bound)

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_ee_state is None:
            task.init_ee_state = cur_state[8].clone().cpu()

        state = np.zeros(28)

        hand_joints = leap_hand.read_pos() - 3.14

        robot_states = flexivrdk.RobotStates()
        robot.getRobotStates(robot_states)
        state = np.concatenate([robot_states.q, hand_joints])

        state = state.astype(np.float32)

        # =============== Diffusion Robot ============

        # print(griv, finger_dis, griv_width)

        # griv = get_gripper_width(gripper)

        cur_flange_pose = get_flange_pose(robot)

        if (
            cur_flange_pose[1] < 1000
        ):  # TODO: this is hard-coded, you can directly remove the if statement
            cur_state = torch.squeeze(env.rigid_body_states[:, : env.num_flexiv_bodies])

            # batch[0][..., :DOF] = tq   # change it into current xyz + quat
            # batch[0][..., DOF:] = griv # change it into current grivWidth

            bird_img, _ = cam_bird.get_data()
            right_img, _ = cam_right.get_data()
            bird_img_list = [bird_img]
            right_img_list = [right_img]

            # vis_img = np.concatenate([bird_img, right_img], axis=1)
            # print("vis_img", vis_img.shape)
            # cv2.imshow("default", vis_img[..., ::-1])
            # cv2.imwrite(f'/home/rhos_pub/LeapTele/DexCopilot/rl_games/rl_games/algos_torch/tmp.png', vis_img)

            agent_action = (
                diffusion.forward_eval(
                    torch.tensor(state).view(1, -1), bird_img_list, right_img_list
                )
                .cpu()
                .squeeze()
            )
            # agent_action = agent_action[:1]
            # print(agent_action.shape)
            # agent_action = diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
            # agent_action = diffusion_actor.teleop_diffusion_cond_sample(batch,agent_action,True).to(device).squeeze()
            # agent_action = tensor_clamp(agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action))
            # agent_action = scale(agent_action.view(-1, 23), lower_limits, upper_limits)
            # agent_action = scale(agent_action.view(-1, 8), lower_limits, upper_limits)
            # print("execute agent_action", agent_action)
            # print("act_dof: ", act_dof)
            # print("")
            # _ = input()
            start_sample_time = time.time()  # TODO: neglect network inference time
            # ==================== End Diffusion Robot ========================

            act_dof = np.zeros((agent_action.shape[0], 28))
            act_dof[:, :7] = agent_action[:, :7]

            for i, key in enumerate([12, 22, 27]):
                tip = key
                tip = tip - 1
                finger_tip = tip - 1
                dip = tip - 2
                pip = tip - 3

                act_dof[:, dip] = agent_action[:, i * 4 + 7]
                act_dof[:, pip] = agent_action[:, i * 4 + 1 + 7]
                act_dof[:, finger_tip] = agent_action[:, i * 4 + 2 + 7]
                act_dof[:, tip] = agent_action[:, i * 4 + 3 + 7]

            tip = 16
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3
            act_dof[:, pip] = agent_action[:, 12 + 7]
            act_dof[:, dip] = agent_action[:, 13 + 7]
            act_dof[:, finger_tip] = agent_action[:, 14 + 7] * -1
            act_dof[:, tip] = agent_action[:, 15 + 7] * -1

        # agent_action = agent_action[:4].mean(dim=0).unsqueeze(0)
        print(len(act_dof))
        for act_next in act_dof[:6]:
            env.cur_targets[:, :DOF] = (
                0.8 * env.cur_targets[:, :DOF] + 0.2 * act_next[:DOF]
            )
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

            task.step()
            handle_viewer_events(task, viewer)

            if time.time() - start_sample_time > dt:
                get_obs(task)
                start_sample_time = time.time()

            if test_on_real_robot:
                try:
                    flexiv_targets = act_next[:Flexiv_DOF]
                    action = flexiv_targets[:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_next.tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))

            time.sleep(0.5)


def flexiv_leap_dexcap_diffusion_replay(task: TeleopPlayer, obs=False):
    sys.path.append("/home/rhos_pub/LeapTele/DexCopilot/rl_games/rl_games/algos_torch/")
    from DexCap.STEP3_inference.test_policy import RobotEnv, DPInference
    from DexCap.STEP3_inference.realsense_module import DepthCameraModule, crop_pcd

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    obs_buf = []

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    print(device)

    task.init_ee_state = None

    ckpt_path = "/home/rhos_pub/LeapTele/DexCopilot/rl_games/rl_games/algos_torch/DexCap/STEP2_train_policy/robomimic/trained_models/model_epoch_2400.pth"  # TODO

    # sync sim & real robot joints
    DOF = 28

    if test_on_real_robot:

        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        global leap_hand
        leap_hand = LeapNode()
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    robot_env = RobotEnv(robot, leap_hand)

    if obs:
        import pickle

        with open(filepath, "rb") as f:
            datalist = pickle.load(f)
        for data in datalist:
            # obs_data = data['obs']
            obs_data = data["hand_joints"]
            obs_data[:7] = data["joints"]
            obs_data = torch.tensor(obs_data.reshape((1, 28)))
            obs_buf.append(obs_data)
        i = 0
    else:
        state_dim = action_dim = 23
        # lower_limits=task.env.flexiv_dof_lower_limits
        # upper_limits=task.env.flexiv_dof_upper_limits

        DP = DPInference(robot_env, ckpt_path)

    # if not obs:
    #     cam_bird = CameraD400(1)
    #     # cam_left = CameraD400(2)
    #     cam_right = CameraD400(0)

    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")

    i = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        begin_time = time.time()
        task.step()

        handle_viewer_events(task, viewer, DP)
        draw_bounding_box()
        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        # if (not recording_data) or (not hand_detected):
        #     continue
        if time.time() - start_sample_time > dt:
            # get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        if obs:
            act_dof = obs_buf[i]
            i += 1
        else:

            input_state = robot_env.get_state()

            act_time = time.time()
            act_dof = DP.inference(input_state)
            act_dof = torch.from_numpy(act_dof).to("cpu")
            # print("act", time.time() - act_time)
            # print("Inference Result: ", act_dof)
            # exit()

            # act_dof = torch.clamp(act_dof, -torch.ones_like(act_dof), torch.ones_like(act_dof)).reshape(-1, action_dim)
            # act_dof = scale(act_dof, lower=lower_limits, upper=upper_limits)

            actions = torch.zeros((1, 28))
            actions[:, :7] = act_dof[:7]

            for i, key in enumerate([12, 22, 27]):
                tip = key
                tip = tip - 1
                finger_tip = tip - 1
                dip = tip - 2  # rotation joints
                pip = tip - 3

                # actions[:, dip] = act_dof[i * 4 + 7]
                # actions[:, pip] = act_dof[i * 4 + 1 + 7]
                # actions[:, finger_tip] = act_dof[i * 4 + 2 + 7]
                # actions[:, tip] = act_dof[i * 4 + 3 + 7]

                actions[:, pip] = act_dof[i * 4 + 7]
                actions[:, dip] = act_dof[i * 4 + 1 + 7]
                actions[:, finger_tip] = act_dof[i * 4 + 2 + 7]
                actions[:, tip] = act_dof[i * 4 + 3 + 7]

            tip = 16
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3
            actions[:, pip] = act_dof[12 + 7]
            actions[:, dip] = act_dof[13 + 7]
            actions[:, finger_tip] = act_dof[14 + 7]
            actions[:, tip] = act_dof[15 + 7]

            act_dof = actions

        if obs:
            env.cur_targets[:, :DOF] = act_dof[:, :DOF]
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )
            if test_on_real_robot:
                try:
                    flexiv_targets = act_dof[..., :Flexiv_DOF]
                    action = flexiv_targets[0][:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                    # print('action',np.asarray(act_dof).shape)
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_dof[0].tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))
            time.sleep(0.5)

        else:
            # for idx, _ in enumerate(act_dof):
            for idx in range(1):
                env.cur_targets[:, :DOF] = act_dof[idx, :DOF]
                gym.set_dof_position_target_tensor( #update simulation
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )
                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[idx][:Flexiv_DOF].tolist()
                        robot.sendJointPosition( #flexiv api func
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))

                if test_on_real_hand:
                    try:
                        action = act_dof[idx].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))
                task.step()
                time.sleep(0.2)

        # ===== sync joint state =====
        if test_on_real_robot:
            # joints = get_robot_joints(robot)
            # hand_joints = get_hand_joints(leap_hand)
            # print('cur:',env.cur_targets[:,isaac_joints],hand_joints/180*np.pi)
            # env.cur_targets[:, :7] = torch.from_numpy(np.array(joints)).to(device)
            # env.cur_targets[:,isaac_joints] = torch.from_numpy(np.array(hand_joints)[retargeting_joints]).to(device)

            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

        print("loop:", time.time() - begin_time)


def flexiv_leap_dexcap_replay(task: TeleopPlayer, obs=False):
    sys.path.append("/home/rhos_pub/LeapTele/DexCopilot/rl_games/rl_games/algos_torch/")
    from DexCap.STEP3_inference.test_policy import RobotEnv, DPInference
    from DexCap.STEP3_inference.realsense_module import DepthCameraModule, crop_pcd

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    obs_buf = []

    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    print(device)

    task.init_ee_state = None

    # Load trajectory
    trajectory_path = "arcap_pp_0"
    start_idx = 0
    stride = 3
    traj = np.load(
        f"/home/rhos_pub/shengcheng/dex-data/HumanArcap/pick_1230/arcap_pick_1230_6.npz"
    )
    arm_q_traj = traj["arm_qs"][start_idx::stride]  # 3 is harded coded stride.
    wrist_poses = traj["wrist_poses"][start_idx::stride]
    wrist_orns = traj["wrist_orns"][start_idx::stride]
    hand_q_traj = traj["hand_qs"][start_idx::stride]

    # sync sim & real robot joints
    DOF = 28

    if test_on_real_robot:
        global leap_hand
        leap_hand = LeapNode()
        set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)

        self_exam(log)
        # self_exam_leaphand(log)

        move_home(robot)
        robot.setMode(mode.NRT_JOINT_POSITION)
        # global gripper

        init_joints = get_robot_joints(robot)
        DOF = 28
        Flexiv_DOF = len(init_joints)

        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [0.5] * DOF

        init_joints = to_torch(init_joints)

        reset_joints(task, init_joints)

    else:
        DOF = 27
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [0.10] * DOF
        MAX_ACC = [0.20] * DOF
        reset_joints(task, home_joint)

    # robot_env = RobotEnv(robot, leap_hand)

    if obs:
        import pickle

        with open(filepath, "rb") as f:
            datalist = pickle.load(f)
        for data in datalist:
            # obs_data = data['obs']
            obs_data = data["hand_joints"]
            obs_data[:7] = data["joints"]
            obs_data = torch.tensor(obs_data.reshape((1, 28)))
            obs_buf.append(obs_data)
        i = 0
    else:
        state_dim = action_dim = 23
        # lower_limits=task.env.flexiv_dof_lower_limits
        # upper_limits=task.env.flexiv_dof_upper_limits

    # if not obs:
    #     cam_bird = CameraD400(1)
    #     # cam_left = CameraD400(2)
    #     cam_right = CameraD400(0)

    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")

    assert DOF > 0, "DOF must be greater than 0"
    env.actions = torch.zeros((1, env.num_dofs))

    def draw_bounding_box():
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            tcp_lower_bound,
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            tcp_upper_bound,
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
        )

        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]),
        )
        task.draw_line(
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]),
            torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]),
        )

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")

    i = 0
    step = 0
    while not env.gym.query_viewer_has_closed(env.viewer):
        if step == len(arm_q_traj):
            break
        begin_time = time.time()
        task.step()

        handle_viewer_events(task, viewer)
        draw_bounding_box()
        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        # if (not recording_data) or (not hand_detected):
        #     continue
        if time.time() - start_sample_time > dt:
            get_obs(task)
            start_sample_time = time.time()

        if test_on_real_robot and robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        if obs:
            act_dof = obs_buf[i]
            i += 1
        else:
            act_dof = np.hstack([arm_q_traj[step], hand_q_traj[step]])
            act_dof = torch.from_numpy(act_dof).to("cpu")
            # act_dof[6] += 2 * np.pi / 3

            # act_dof = torch.clamp(act_dof, -torch.ones_like(act_dof), torch.ones_like(act_dof)).reshape(-1, action_dim)
            # act_dof = scale(act_dof, lower=lower_limits, upper=upper_limits)

            actions = torch.zeros((1, 28))
            actions[:, :7] = act_dof[:7]
            # actions[:, :7] = torch.tensor([0, -0.6989, 0, 1.57, 0, 0.6973, 0])
            # actions[:, :7] = torch.tensor([-0.12, -0.91, -0.090, 1.16, 0.08, 0.51, -0.22])

            for i, key in enumerate([12, 22, 27]):
                tip = key
                tip = tip - 1
                finger_tip = tip - 1
                dip = tip - 2  # rotation joints
                pip = tip - 3

                # actions[:, dip] = act_dof[i * 4 + 7]
                # actions[:, pip] = act_dof[i * 4 + 1 + 7]
                # actions[:, finger_tip] = act_dof[i * 4 + 2 + 7]
                # actions[:, tip] = act_dof[i * 4 + 3 + 7]

                actions[:, pip] = act_dof[i * 4 + 7]
                actions[:, dip] = act_dof[i * 4 + 1 + 7]
                actions[:, finger_tip] = act_dof[i * 4 + 2 + 7]
                actions[:, tip] = act_dof[i * 4 + 3 + 7]

            tip = 16
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3
            actions[:, pip] = act_dof[12 + 7]
            actions[:, dip] = act_dof[13 + 7]
            actions[:, finger_tip] = act_dof[14 + 7]
            actions[:, tip] = act_dof[15 + 7]

            act_dof = actions

        if obs:
            env.cur_targets[:, :DOF] = act_dof[:, :DOF]
            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )
            if test_on_real_robot:
                try:
                    flexiv_targets = act_dof[..., :Flexiv_DOF]
                    action = flexiv_targets[0][:Flexiv_DOF].tolist()
                    robot.sendJointPosition(
                        action,
                        target_vel[:Flexiv_DOF],
                        target_acc[:Flexiv_DOF],
                        MAX_VEL[:Flexiv_DOF],
                        MAX_ACC[:Flexiv_DOF],
                    )
                    # print('action',np.asarray(act_dof).shape)
                except Exception as e:
                    log.error(str(e))

            if test_on_real_hand:
                try:
                    action = act_dof[0].tolist()
                    set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
                except Exception as e:
                    log.error(str(e))
            time.sleep(0.5)

        else:
            # for idx, _ in enumerate(act_dof):
            for idx in range(1):
                env.cur_targets[:, :DOF] = act_dof[idx, :DOF]
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(env.cur_targets)
                )
                if test_on_real_robot:
                    try:
                        flexiv_targets = act_dof[..., :Flexiv_DOF]
                        action = flexiv_targets[idx][:Flexiv_DOF].tolist()
                        robot.sendJointPosition(
                            action,
                            target_vel[:Flexiv_DOF],
                            target_acc[:Flexiv_DOF],
                            MAX_VEL[:Flexiv_DOF],
                            MAX_ACC[:Flexiv_DOF],
                        )
                        # print('action',np.asarray(act_dof).shape)
                    except Exception as e:
                        log.error(str(e))

                if test_on_real_hand:
                    try:
                        action = act_dof[idx].tolist()
                        set_leap_hand_pos(
                            leap_hand, np.asarray(action), env.nail_handles
                        )
                    except Exception as e:
                        log.error(str(e))
                task.step()
                time.sleep(0.5)

        # ===== sync joint state =====
        if test_on_real_robot:
            # joints = get_robot_joints(robot)
            # hand_joints = get_hand_joints(leap_hand)
            # print('cur:',env.cur_targets[:,isaac_joints],hand_joints/180*np.pi)
            # env.cur_targets[:, :7] = torch.from_numpy(np.array(joints)).to(device)
            # env.cur_targets[:,isaac_joints] = torch.from_numpy(np.array(hand_joints)[retargeting_joints]).to(device)

            gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(env.cur_targets)
            )

        print("loop:", time.time() - begin_time)

        step += 1
        time.sleep(0.1)


def flexiv_leap_motion_planning_replay_pour(task: TeleopPlayer, obs=False):
    #print("#####################################)))))))))))))))))))))))))))))))))))))")
    obs_buf = []
    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    # Initialize cameras
    # cam_bird = CameraD400(0)
    # cam_right = CameraD400(1)
    cam_bird = CameraD400(135122074278)
    cam_right = CameraD400(233522075695)
    # Perform self-examination and move robot to home position
    self_exam(log)
    move_home(robot)
    robot.setMode(mode.NRT_JOINT_POSITION)

    # Initialize robot joints and degrees of freedom (DOF)
    init_joints = get_robot_joints(robot)
    DOF = 28
    Flexiv_DOF = len(init_joints)

    # Set target velocities and accelerations
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF
    # MAX_VEL = [1.0] * DOF
    # MAX_ACC = [0.5] * DOF

    MAX_VEL = [0.5] * DOF
    MAX_ACC = [0.5] * DOF

    # Convert initial joints to torch tensor and reset joints
    init_joints = to_torch(init_joints)
    reset_joints(task, init_joints)

    # Initialize LeapNode and set hand position
    global leap_hand
    leap_hand = LeapNode()#28 DOFs
    set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)

    # Set time step and initialize actions
    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")
    env.actions = torch.zeros((1, env.num_dofs))

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")
# {
# q:  ['0.00', '-0.70', '0.00', '1.57', '-0.00', '0.70', '-0.00']
# theta:  ['-0.00', '-0.70', '-0.00', '1.57', '0.00', '0.69', '0.01']
# dq:  ['0.00', '0.00', '0.00', '-0.00', '0.00', '0.00', '-0.00']
# dtheta:  ['0.00', '0.00', '0.00', '-0.00', '-0.00', '0.00', '-0.00']
# tau:  ['0.02', '57.17', '1.40', '-25.42', '-2.34', '5.35', '0.16']
# tau_des:  ['0.03', '4.12', '0.43', '-2.14', '0.46', '1.87', '0.08']
# tau_dot:  ['-85.16', '7.38', '-71.11', '-0.12', '26.43', '-29.65', '13.76']
# tau_ext:  ['-1.57', '-1.78', '-1.28', '1.55', '0.49', '-0.36', '0.10']
# tcp_pose:  ['0.68', '-0.11', '0.13', '-0.00', '-0.00', '1.00', '0.00']
# tcp_pose_d:  ['0.68', '-0.11', '0.13', '-0.00', '-0.00', '1.00', '0.00']
# tcp_velocity:  ['0.00', '0.00', '0.00', '0.00', '-0.00', '-0.00']
# camera_pose:  ['0.76', '-0.11', '0.33', '0.00', '0.71', '-0.71', '-0.00']
# flange_pose:  ['0.68', '-0.11', '0.28', '-0.00', '-0.00', '1.00', '0.00']
# FT_sensor_raw_reading:  ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00']
# F_ext_tcp_frame:  ['1.65', '-1.93', '2.66', '-0.36', '-0.41', '0.16']
# F_ext_base_frame:  ['-1.66', '-1.92', '-2.65', '0.36', '-0.41', '-0.16']
# }
    count = 0
    count1 = 0
    cnt2 = 0
    cnt3 = 0
    poition_home = [0.68, -0.11, 0.28, -0.00, -0.00, 1.00, 0.00]
    
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()

        # handle_viewer_events(task, viewer)
        # if time.time() - start_sample_time > dt:
        get_obs(task, leap_hand,cam_bird=cam_bird, cam_right=cam_right)
        # start_sample_time = time.time()

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        hand_joints = leap_hand.read_pos() - 3.14
        robot_states = flexivrdk.RobotStates()#
        robot.getRobotStates(robot_states)
        state = np.concatenate([robot_states.q, hand_joints]) #arm's joint 7 angles and hand's 21 dofs?
        state = state.astype(np.float32)
        state = torch.tensor(state).to(device)

        bird_img, _ = cam_bird.get_data()
        bird_img = cv2.resize(bird_img, (320, 240))
        bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
        bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)# channel w h 1

        right_img, _ = cam_right.get_data()
        right_img = cv2.resize(right_img, (320, 240))
        right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
        right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)

        img = torch.stack([bird_img, right_img], dim=1)

        actions = torch.zeros((28))
        json_path = "/home/rhos_pub/grounded-sam/posedata/start/start.json"
        with open(json_path, "r") as file:
            data = json.load(file)

        q_start = data["posedata/start"]["q_start"]
        position_start = data["posedata/start"]["tcp_start"]

        q_start_floats = [float(value) for value in q_start]
        position_start= [float(value_ps) for value_ps in position_start]

        json_path1 = "/home/rhos_pub/grounded-sam/posedata/end/end.json"
        with open(json_path1, "r") as file:
            data = json.load(file)

        q_end = data["posedata/end"]["q_end"]
        position_end = data["posedata/end"]["tcp_end"]


        q_end_floats = [float(value) for value in q_end]
        position_end= [float(value_pe) for value_pe in position_end]

        step_home2start = F.pairwise_distance(torch.tensor(position_start[:3]).to(device),torch.tensor(poition_home[:3]).to(device),p=2)/0.02 + 35
        #print(step_home2start)
        
        step_start2end = F.pairwise_distance(torch.tensor(position_start[:3]).to(device),torch.tensor(position_end[:3]).to(device),p=2)/0.02 + 35
        #linear interpolation
        actions[:7] = (1/step_home2start) * (step_home2start - count1) * state[:7].to(device) + (1/step_home2start) * count1 * torch.tensor(q_start_floats).to(device)
        if count1 < step_home2start:
            count1 += 1

        first_flag = False

        # if (
        #     torch.dist(
        #         state[:7].to(device), actions[:7].to(device), p=2
        #     )
        #     < 0.0015
        # ):s
        if (torch.dist(state[:7].to(device), torch.tensor(q_start_floats).to(device), p=2)< 0.0020) and not first_flag:
            first_flag = True

        #print(first_flag)
        # print("distance",torch.dist(
        #         state[:7].to(device), torch.tensor(q_start_floats).to(device), p=2
        #     ))
        for i, key in enumerate([12, 22, 27]):
            tip = key
            tip = tip - 1
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3

            actions[dip] = state[i * 4 + 7]
            actions[pip] = state[i * 4 + 1 + 7]
            actions[finger_tip] = state[i * 4 + 2 + 7]
            actions[tip] = state[i * 4 + 3 + 7]
        tip = 16
        finger_tip = tip - 1
        dip = tip - 2
        pip = tip - 3
        actions[pip] = state[12 + 7]
        actions[dip] = state[13 + 7]
        actions[finger_tip] = state[14 + 7] * -1
        actions[tip] = state[15 + 7] * -1

        if first_flag or (count>15):
            actions[8:] = 0.0625 * (16 - count) * actions[8:] + 0.0625 * count * torch.tensor(
                [
                    1.5059,
                    -0.1503,
                    0.2358,
                    0.0477,
                    0.0000,
                    1.8743,
                    0.6292,
                    0.2751,
                    0.12000,
                    0.0000,
                    1.6625,
                    -0.1181,
                    0.2113,
                    -0.0017,
                    0.0000,
                    1.5842,
                    -0.0322,
                    0.3707,
                    0.3441,
                    0.0000,
                ]
            )
            if count < 16:
                count += 1
            # print(count, actions[8:])
        act_dof = actions#target movement
        if count > 15:
            actions[:7] = (1/step_start2end) * (step_start2end - cnt2) * state[:7].to(device) + (1/step_start2end) * cnt2 * torch.tensor(q_end_floats).to(device)
            if cnt2 < step_start2end:
                cnt2 += 1
            if (torch.dist(state[:7].to(device), torch.tensor(q_end_floats).to(device), p=2)< 0.0020):
                actions[8:] = 0.1 * (10 - cnt3) * actions[8:] + 0.1 * cnt3 * torch.tensor([0.0])
                if cnt3 < 10:
                    cnt3 += 1
                # set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
                # dirname = f"/home/rhos_pub/Desktop/Flexiv_{task.timestamp}"
                # os.makedirs(dirname, exist_ok=True)

                # cnt = len(glob.glob(os.path.join(dirname, "*/")))
                # dirname = os.path.join(dirname, f"{cnt:03}")#3 digits
                # os.makedirs(dirname, exist_ok=True)
                # for f in os.listdir(dirname):
                #     os.remove(os.path.join(dirname, f))

                # filename = os.path.join(dirname, "data.pkl")
                # with open(filename, "wb") as f:
                #     pkl.dump(task.obs_dict, f)

                # if len(cam_bird_view) > 0:
                #     with h5py.File(os.path.join(dirname, "bird_img.hdf5"), "w") as hf:
                #         images = np.array([view["image"] for view in cam_bird_view])
                #         depths = np.array([view["depth"] for view in cam_bird_view])
                #         hf.create_dataset("images", data=images)
                #         hf.create_dataset("depths", data=depths)

                # if len(cam_right_view) > 0:
                #     with h5py.File(os.path.join(dirname, "right_img.hdf5"), "w") as hf:
                #         images = np.array([view["image"] for view in cam_right_view])
                #         depths = np.array([view["depth"] for view in cam_right_view])
                #         hf.create_dataset("images", data=images)
                #         hf.create_dataset("depths", data=depths)

                # print("Realsense Camera hdf5 Saved!")
                if cnt3 == 9:
                    # set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
                    dirname = f"/home/rhos_pub/Desktop/Flexiv_{task.timestamp}"
                    os.makedirs(dirname, exist_ok=True)

                    cnt = len(glob.glob(os.path.join(dirname, "*/")))
                    dirname = os.path.join(dirname, f"{cnt:03}")#3 digits
                    os.makedirs(dirname, exist_ok=True)
                    for f in os.listdir(dirname):
                        os.remove(os.path.join(dirname, f))

                    filename = os.path.join(dirname, "data.pkl")
                    with open(filename, "wb") as f:
                        pkl.dump(task.obs_dict, f)

                    if len(cam_bird_view) > 0:
                        with h5py.File(os.path.join(dirname, "bird_img.hdf5"), "w") as hf:
                            images = np.array([view["image"] for view in cam_bird_view])
                            depths = np.array([view["depth"] for view in cam_bird_view])
                            hf.create_dataset("images", data=images)
                            hf.create_dataset("depths", data=depths)

                    if len(cam_right_view) > 0:
                        with h5py.File(os.path.join(dirname, "right_img.hdf5"), "w") as hf:
                            images = np.array([view["image"] for view in cam_right_view])
                            depths = np.array([view["depth"] for view in cam_right_view])
                            hf.create_dataset("images", data=images)
                            hf.create_dataset("depths", data=depths)

                    print("Realsense Camera hdf5 Saved!")
                    break

        env.cur_targets[:, :DOF] = act_dof[:DOF]
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
        if test_on_real_robot:
            try:
                flexiv_targets = act_dof[:Flexiv_DOF]
                action = flexiv_targets[:Flexiv_DOF].tolist()
                robot.sendJointPosition(
                    action,
                    target_vel[:Flexiv_DOF],
                    target_acc[:Flexiv_DOF],
                    MAX_VEL[:Flexiv_DOF],
                    MAX_ACC[:Flexiv_DOF],
                )
            except Exception as e:
                log.error(str(e))

        action = act_dof.tolist()
        
        set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
        task.step()
        time.sleep(0.2)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))




def flexiv_leap_motion_planning_replay(task: TeleopPlayer, obs=False):
    obs_buf = []
    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    # Initialize cameras
    cam_bird = CameraD400(0)
    cam_right = CameraD400(1)

    # Perform self-examination and move robot to home position
    self_exam(log)
    move_home_pour(robot)
    robot.setMode(mode.NRT_JOINT_POSITION)

    # Initialize robot joints and degrees of freedom (DOF)
    init_joints = get_robot_joints(robot)
    DOF = 28
    Flexiv_DOF = len(init_joints)

    # Set target velocities and accelerations
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF
    # MAX_VEL = [1.0] * DOF
    # MAX_ACC = [0.5] * DOF

    MAX_VEL = [0.5] * DOF
    MAX_ACC = [0.5] * DOF

    # Convert initial joints to torch tensor and reset joints
    init_joints = to_torch(init_joints)
    reset_joints(task, init_joints)

    # Initialize LeapNode and set hand position
    global leap_hand
    leap_hand = LeapNode()#28 DOFs
    init_actions = np.zeros(28)
    #init_actions[13] = 1.3
    set_leap_hand_pos(leap_hand, init_actions, env.nail_handles)

    # Set time step and initialize actions
    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")
    env.actions = torch.zeros((1, env.num_dofs))

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")
# {
# q:  ['0.00', '-0.70', '0.00', '1.57', '-0.00', '0.70', '-0.00']
# theta:  ['-0.00', '-0.70', '-0.00', '1.57', '0.00', '0.69', '0.01']
# dq:  ['0.00', '0.00', '0.00', '-0.00', '0.00', '0.00', '-0.00']
# dtheta:  ['0.00', '0.00', '0.00', '-0.00', '-0.00', '0.00', '-0.00']
# tau:  ['0.02', '57.17', '1.40', '-25.42', '-2.34', '5.35', '0.16']
# tau_des:  ['0.03', '4.12', '0.43', '-2.14', '0.46', '1.87', '0.08']
# tau_dot:  ['-85.16', '7.38', '-71.11', '-0.12', '26.43', '-29.65', '13.76']
# tau_ext:  ['-1.57', '-1.78', '-1.28', '1.55', '0.49', '-0.36', '0.10']
# tcp_pose:  ['0.68', '-0.11', '0.13', '-0.00', '-0.00', '1.00', '0.00']
# tcp_pose_d:  ['0.68', '-0.11', '0.13', '-0.00', '-0.00', '1.00', '0.00']
# tcp_velocity:  ['0.00', '0.00', '0.00', '0.00', '-0.00', '-0.00']
# camera_pose:  ['0.76', '-0.11', '0.33', '0.00', '0.71', '-0.71', '-0.00']
# flange_pose:  ['0.68', '-0.11', '0.28', '-0.00', '-0.00', '1.00', '0.00']
# FT_sensor_raw_reading:  ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00']
# F_ext_tcp_frame:  ['1.65', '-1.93', '2.66', '-0.36', '-0.41', '0.16']
# F_ext_base_frame:  ['-1.66', '-1.92', '-2.65', '0.36', '-0.41', '-0.16']
# }


# [2024-11-28 19:29:36.258] [info] Current robot states:
# {
# q:  ['-0.49', '-1.37', '-0.16', '0.63', '1.47', '-0.54', '-0.50']
# theta:  ['-0.49', '-1.37', '-0.16', '0.63', '1.47', '-0.54', '-0.49']
# dq:  ['0.00', '-0.00', '-0.00', '-0.00', '0.00', '-0.00', '-0.00']
# dtheta:  ['0.00', '0.00', '0.00', '-0.00', '0.00', '0.00', '-0.00']
# tau:  ['0.06', '68.55', '5.54', '-23.64', '2.93', '0.40', '-0.11']
# tau_des:  ['0.03', '1.99', '0.46', '-1.26', '0.34', '1.64', '-0.04']
# tau_dot:  ['-59.69', '-40.86', '-18.60', '68.36', '12.80', '15.95', '-37.70']
# tau_ext:  ['-0.04', '-2.16', '-0.35', '1.12', '-0.36', '-1.63', '0.05']
# tcp_pose:  ['0.68', '-0.11', '0.13', '-0.00', '0.00', '0.71', '0.71']
# tcp_pose_d:  ['0.68', '-0.11', '0.13', '-0.00', '0.00', '0.71', '0.71']
# tcp_velocity:  ['-0.00', '-0.00', '0.00', '0.00', '-0.00', '-0.00']
# camera_pose:  ['0.76', '-0.31', '0.13', '0.50', '-0.50', '0.50', '0.50']
# flange_pose:  ['0.68', '-0.26', '0.13', '-0.00', '0.00', '0.71', '0.71']
# FT_sensor_raw_reading:  ['0.00', '0.00', '0.00', '0.00', '0.00', '0.00']
# F_ext_tcp_frame:  ['0.83', '-2.41', '-2.18', '-1.32', '1.47', '0.06']
# F_ext_base_frame:  ['-0.84', '-2.18', '-2.41', '1.33', '0.06', '1.46']
# }
    count = 0
    count1 = 0
    cnt2 = 0
    cnt3 = 0
    poition_home = [0.68, -0.26, 0.13, -0.00, 0.00, 0.71, 0.71]
    # print("*****************************************************88")
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        # handle_viewer_events(task, viewer)
        # if time.time() - start_sample_time > dt:
        get_obs(task, leap_hand,cam_bird=cam_bird, cam_right=cam_right)
        # start_sample_time = time.time()

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        hand_joints = leap_hand.read_pos() - 3.14
        robot_states = flexivrdk.RobotStates()#
        robot.getRobotStates(robot_states)
        state = np.concatenate([robot_states.q, hand_joints]) #arm's joint 7 angles and hand's 21 dofs?
        state = state.astype(np.float32)
        state = torch.tensor(state).to(device)

        bird_img, _ = cam_bird.get_data()
        bird_img = cv2.resize(bird_img, (320, 240))
        bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
        bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)# channel w h 1

        right_img, _ = cam_right.get_data()
        right_img = cv2.resize(right_img, (320, 240))
        right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
        right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)

        img = torch.stack([bird_img, right_img], dim=1)
        print()
        actions = torch.zeros((28))
        json_path = "/home/rhos_pub/grounded-sam/posedata_pour/start/start.json"
        with open(json_path, "r") as file:
            data = json.load(file)

        q_start = data["posedata/start"]["q_start"]
        position_start = data["posedata/start"]["tcp_start"]

        q_start_floats = [float(value) for value in q_start]
        position_start= [float(value_ps) for value_ps in position_start]

        json_path2 = "/home/rhos_pub/grounded-sam/posedata_pour/mid/mid.json"
        with open(json_path2, "r") as file:
            data = json.load(file)

        q_mid = data["posedata/mid"]["q_mid"]
        position_mid = data["posedata/mid"]["tcp_mid"]
        q_mid_floats = [float(value) for value in q_mid]
        position_mid = [float(value_pe) for value_pe in position_mid]

        json_path1 = "/home/rhos_pub/grounded-sam/posedata_pour/end/end.json"
        with open(json_path1, "r") as file:
            data = json.load(file)

        q_end = data["posedata/end"]["q_end"]
        position_end = data["posedata/end"]["tcp_end"]


        q_end_floats = [float(value) for value in q_end]
        position_end= [float(value_pe) for value_pe in position_end]

        step_home2start = F.pairwise_distance(torch.tensor(position_start[:3]).to(device),torch.tensor(poition_home[:3]).to(device),p=2)/0.02
        print(step_home2start)
        
        step_start2mid = F.pairwise_distance(torch.tensor(position_mid[:3]).to(device),torch.tensor(position_start[:3]).to(device),p=2)/0.02

        step_mid2end = F.pairwise_distance(torch.tensor(position_mid[4:]).to(device),torch.tensor(position_end[4:]).to(device),p=2)/0.02
        #linear interpolation
        actions[:7] = (1/step_home2start) * (step_home2start - count1) * state[:7].to(device) + (1/step_home2start) * count1 * torch.tensor(q_start_floats).to(device)
        if count1 < step_home2start:
            count1 += 1

        first_flag = False

        # if (
        #     torch.dist(
        #         state[:7].to(device), actions[:7].to(device), p=2
        #     )
        #     < 0.0015
        # ):s
        if (
            torch.dist(
                state[:7].to(device), torch.tensor(q_start_floats).to(device), p=2
            )
            < 0.0020
        ):
            first_flag = True

        print(first_flag)
        print("distance",torch.dist(
                state[:7].to(device), torch.tensor(q_start_floats).to(device), p=2
            ))
        for i, key in enumerate([12, 22, 27]):
            tip = key
            tip = tip - 1
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3

            actions[dip] = state[i * 4 + 7]
            actions[pip] = state[i * 4 + 1 + 7]
            actions[finger_tip] = state[i * 4 + 2 + 7]
            actions[tip] = state[i * 4 + 3 + 7]
        tip = 16
        finger_tip = tip - 1
        dip = tip - 2
        pip = tip - 3
        actions[pip] = state[12 + 7]
        actions[dip] = state[13 + 7]
        actions[finger_tip] = state[14 + 7] * -1
        actions[tip] = state[15 + 7] * -1

        if first_flag:
            if count == 0:
                actions[13] = 1.2
            actions[8:] = 0.1 * (10 - count) * actions[
                8:
            ] + 0.1 * count * torch.tensor(
                [
                    1.5059,
                    -0.1503,
                    1.2358,
                    1.1477,
                    0.0000,
                    1.5743,
                    0.0107,
                    1.2658,
                    0.8503,
                    0.0000,
                    1.5059,
                    0.,
                    1.2658,
                    0.8503,
                    0.0000,
                    1.5842,
                    0.,
                    1.0819,
                    1.4307,
                    0.0000,
                ]
            )
            if count < 10:
                count += 1

        act_dof = actions#target movement
        if count > 9:
            actions[:7] = (1/step_start2mid) * (step_start2mid - cnt2) * state[:7].to(device) + (1/step_start2mid) * cnt2 * torch.tensor(q_mid_floats).to(device)
            if cnt2 < step_start2mid:
                cnt2 += 1
            print("!!!!!!&&&$$$##",torch.dist(
                    state[:7].to(device), torch.tensor(q_end_floats).to(device), p=2
                ))
            # if (
            #     torch.dist(
            #         state[:7].to(device), torch.tensor(q_end_floats).to(device), p=2
            #     )
            #     < 2.07
            # ):
            if (
                cnt2 >= step_start2mid - 1
            ):
                actions[:7] = 0.1 * (10 - cnt3) * state[:7].to(device) + 0.1 * cnt3 * torch.tensor(q_end_floats).to(device)
                if cnt3 < 10:
                    cnt3 += 1
                # set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
                dirname = f"/home/rhos_pub/Desktop/Flexiv_{task.timestamp}"
                os.makedirs(dirname, exist_ok=True)

                cnt = len(glob.glob(os.path.join(dirname, "*/")))
                dirname = os.path.join(dirname, f"{cnt:03}")#3 digits
                os.makedirs(dirname, exist_ok=True)
                for f in os.listdir(dirname):
                    os.remove(os.path.join(dirname, f))

                filename = os.path.join(dirname, "data.pkl")
                with open(filename, "wb") as f:
                    pkl.dump(task.obs_dict, f)

                if len(cam_bird_view) > 0:
                    with h5py.File(os.path.join(dirname, "bird_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_bird_view])
                        depths = np.array([view["depth"] for view in cam_bird_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                if len(cam_right_view) > 0:
                    with h5py.File(os.path.join(dirname, "right_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_right_view])
                        depths = np.array([view["depth"] for view in cam_right_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                print("Realsense Camera hdf5 Saved!")
                if cnt3 == 20:
                    break

        env.cur_targets[:, :DOF] = act_dof[:DOF]
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
        if test_on_real_robot:
            try:
                flexiv_targets = act_dof[:Flexiv_DOF]
                action = flexiv_targets[:Flexiv_DOF].tolist()
                robot.sendJointPosition(
                    action,
                    target_vel[:Flexiv_DOF],
                    target_acc[:Flexiv_DOF],
                    MAX_VEL[:Flexiv_DOF],
                    MAX_ACC[:Flexiv_DOF],
                )
            except Exception as e:
                log.error(str(e))

        action = act_dof.tolist()
        set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
        task.step()
        time.sleep(0.2)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))




def flexiv_gripper_motion_planning(task: TeleopPlayer, obs=False):
    print("gripper grasping")
    obs_buf = []
    env = task.env
    gym = task.env.gym
    sim = task.env.sim
    viewer = task.env.viewer
    task.subscribe_keyboard_events(gym, viewer)
    device = torch.device("cuda")
    
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../utils/"))
    from .calibration.camera import CameraD400

    sys.path.append("/home/rhos_pub/LeapTele/python/")
    from leaphand import LeapNode

    # Initialize cameras
    cam_bird = CameraD400(0)
    cam_right = CameraD400(1)

    # Perform self-examination and move robot to home position
    self_exam(log)
    move_home(robot)
    robot.setMode(mode.NRT_JOINT_POSITION)

    # Initialize robot joints and degrees of freedom (DOF)
    init_joints = get_robot_joints(robot)
    DOF = 28
    Flexiv_DOF = len(init_joints)

    # Set target velocities and accelerations
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF
    MAX_VEL = [0.5] * DOF
    MAX_ACC = [0.5] * DOF

    # Convert initial joints to torch tensor and reset joints
    init_joints = to_torch(init_joints)
    reset_joints(task, init_joints)

    global leap_hand
    leap_hand = LeapNode()#28 DOFs
    set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
    dt = 1.0 / frequency
    print("Sending command to robot at ", frequency, " Hz")
    env.actions = torch.zeros((1, env.num_dofs))

    delta_pos_history = []
    start_sample_time = time.time()

    for i in range(100):
        task.step()
        time.sleep(0.02)
    print("start")


    count = 0
    count1 = 0
    cnt2 = 0
    cnt3 = 0
    poition_home = [0.68, -0.11, 0.28, -0.00, -0.00, 1.00, 0.00]
    
    while not env.gym.query_viewer_has_closed(env.viewer):
        task.step()
        get_obs(task, leap_hand,cam_bird=cam_bird, cam_right=cam_right)

        cur_state = torch.squeeze(env.rigid_body_states[..., : env.num_flexiv_bodies])
        if task.init_hand_wrist_joint is None:
            state = cur_state[..., :3]
            tip_state = state[env.nail_handles]
            wrist_state = state[8]
            task.init_tip_state = tip_state.clone().cpu()
            task.init_hand_wrist_joint = wrist_state.clone().cpu()

        hand_joints = leap_hand.read_pos() - 3.14
        robot_states = flexivrdk.RobotStates()#
        robot.getRobotStates(robot_states)
        state = np.concatenate([robot_states.q, hand_joints]) #arm's joint 7 angles and hand's 21 dofs?
        state = state.astype(np.float32)
        state = torch.tensor(state).to(device)

        bird_img, _ = cam_bird.get_data()
        bird_img = cv2.resize(bird_img, (320, 240))
        bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
        bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)# channel w h 1

        right_img, _ = cam_right.get_data()
        right_img = cv2.resize(right_img, (320, 240))
        right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
        right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)

        img = torch.stack([bird_img, right_img], dim=1)

        actions = torch.zeros((28))
        json_path = "/home/rhos_pub/grounded-sam/posedata/start/start.json"
        with open(json_path, "r") as file:
            data = json.load(file)

        q_start = data["posedata/start"]["q_start"]
        position_start = data["posedata/start"]["tcp_start"]

        q_start_floats = [float(value) for value in q_start]
        position_start= [float(value_ps) for value_ps in position_start]

        json_path1 = "/home/rhos_pub/grounded-sam/posedata/end/end.json"
        with open(json_path1, "r") as file:
            data = json.load(file)

        q_end = data["posedata/end"]["q_end"]
        position_end = data["posedata/end"]["tcp_end"]


        q_end_floats = [float(value) for value in q_end]
        position_end= [float(value_pe) for value_pe in position_end]

        step_home2start = F.pairwise_distance(torch.tensor(position_start[:3]).to(device),torch.tensor(poition_home[:3]).to(device),p=2)/0.02
        print(step_home2start)
        
        step_start2end = F.pairwise_distance(torch.tensor(position_start[:3]).to(device),torch.tensor(position_end[:3]).to(device),p=2)/0.02
        #linear interpolation
        actions[:7] = (1/step_home2start) * (step_home2start - count1) * state[:7].to(device) + (1/step_home2start) * count1 * torch.tensor(q_start_floats).to(device)
        if count1 < step_home2start:
            count1 += 1

        first_flag = False

        # if (
        #     torch.dist(
        #         state[:7].to(device), actions[:7].to(device), p=2
        #     )
        #     < 0.0015
        # ):s
        if (
            torch.dist(
                state[:7].to(device), torch.tensor(q_start_floats).to(device), p=2
            )
            < 0.0020
        ):
            first_flag = True

        print(first_flag)
        print("distance",torch.dist(
                state[:7].to(device), torch.tensor(q_start_floats).to(device), p=2
            ))
        for i, key in enumerate([12, 22, 27]):
            tip = key
            tip = tip - 1
            finger_tip = tip - 1
            dip = tip - 2
            pip = tip - 3

            actions[dip] = state[i * 4 + 7]
            actions[pip] = state[i * 4 + 1 + 7]
            actions[finger_tip] = state[i * 4 + 2 + 7]
            actions[tip] = state[i * 4 + 3 + 7]
        tip = 16
        finger_tip = tip - 1
        dip = tip - 2
        pip = tip - 3
        actions[pip] = state[12 + 7]
        actions[dip] = state[13 + 7]
        actions[finger_tip] = state[14 + 7] * -1
        actions[tip] = state[15 + 7] * -1

        if first_flag:
            actions[8:] = 0.1 * (10 - count) * actions[
                8:
            ] + 0.1 * count * torch.tensor(
                [
                    1.5059,
                    -0.1503,
                    0.2358,
                    0.0477,
                    0.0000,
                    1.8743,
                    0.6292,
                    0.2751,
                    0.12000,
                    0.0000,
                    1.6625,
                    -0.1181,
                    0.2113,
                    -0.0017,
                    0.0000,
                    1.5842,
                    -0.0322,
                    0.3707,
                    0.3441,
                    0.0000,
                ]
            )
            if count < 10:
                count += 1

        act_dof = actions#target movement
        if count > 9:
            actions[:7] = (1/step_start2end) * (step_start2end - cnt2) * state[:7].to(device) + (1/step_start2end) * cnt2 * torch.tensor(q_end_floats).to(device)
            if cnt2 < step_start2end:
                cnt2 += 1
            if (
                torch.dist(
                    state[:7].to(device), torch.tensor(q_end_floats).to(device), p=2
                )
                < 0.0020
            ):
                actions[8:] = 0.1 * (10 - cnt3) * actions[8:] + 0.1 * cnt3 * torch.tensor([0.0])
                if cnt3 < 10:
                    print("aaaaaccccction",actions.shape)
                    cnt3 += 1
                # set_leap_hand_pos(leap_hand, np.zeros(28), env.nail_handles)
                dirname = f"/home/rhos_pub/Desktop/Flexiv_{task.timestamp}"
                os.makedirs(dirname, exist_ok=True)

                cnt = len(glob.glob(os.path.join(dirname, "*/")))
                dirname = os.path.join(dirname, f"{cnt:03}")#3 digits
                os.makedirs(dirname, exist_ok=True)
                for f in os.listdir(dirname):
                    os.remove(os.path.join(dirname, f))

                filename = os.path.join(dirname, "data.pkl")
                with open(filename, "wb") as f:
                    pkl.dump(task.obs_dict, f)

                if len(cam_bird_view) > 0:
                    with h5py.File(os.path.join(dirname, "right_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_bird_view])
                        depths = np.array([view["depth"] for view in cam_bird_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                if len(cam_right_view) > 0:
                    with h5py.File(os.path.join(dirname, "bird_img.hdf5"), "w") as hf:
                        images = np.array([view["image"] for view in cam_right_view])
                        depths = np.array([view["depth"] for view in cam_right_view])
                        hf.create_dataset("images", data=images)
                        hf.create_dataset("depths", data=depths)

                print("Realsense Camera hdf5 Saved!")
                if cnt3 == 7:
                    break

        env.cur_targets[:, :DOF] = act_dof[:DOF]
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
        if test_on_real_robot:
            try:
                flexiv_targets = act_dof[:Flexiv_DOF]
                action = flexiv_targets[:Flexiv_DOF].tolist()
                robot.sendJointPosition(
                    action,
                    target_vel[:Flexiv_DOF],
                    target_acc[:Flexiv_DOF],
                    MAX_VEL[:Flexiv_DOF],
                    MAX_ACC[:Flexiv_DOF],
                )
            except Exception as e:
                log.error(str(e))

        action = act_dof.tolist()
        
        set_leap_hand_pos(leap_hand, np.asarray(action), env.nail_handles)
        task.step()
        time.sleep(0.2)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
