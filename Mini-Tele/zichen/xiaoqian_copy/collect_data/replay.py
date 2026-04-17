import os
import time
import tqdm
import configargparse
import numpy as np
import transformations as tf
from camera import CameraD400
from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize
import fpsample
import numpy as np
import threading
import cv2
import pickle
import glob
from natsort import natsorted
import pandas as pd


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data")
    parser.add_argument('--global_key', type=str)
    parser.add_argument('--episode', type=int, default=0)

    args = parser.parse_args()
    return args

def pose_to_matrix(pose):
    """
    输入: 
        x, y, z  -- 平移分量
        qx, qy, qz, qw -- 四元数 (假设已归一化)
    输出:
        4x4 齐次变换矩阵
    """
    x, y, z, qw, qx, qy, qz = pose # pppqqqq wxyz
    # 1. 根据四元数得到旋转矩阵 R
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)

    # 2. 组装到 4x4 矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


if __name__ == '__main__':

    args = config_parse()
    joint_position_paths = glob.glob(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/joint_position/*.npy")
    joint_position_paths = natsorted(joint_position_paths)
    gripper_width_paths = glob.glob(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/gripper_width/*.npy")
    gripper_width_paths = natsorted(gripper_width_paths)
    tcp_paths = glob.glob(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/tcp/*.npy")
    tcp_paths = natsorted(tcp_paths)
    print(len(joint_position_paths))

    try:
        robot = Flexiv()
        robot_loaded = True
        
        ## joint replay for miniteleo
        for joint_position_path, gripper_width_path in zip(joint_position_paths, gripper_width_paths):
            
            joint_position = np.load(joint_position_path)
            gripper_width = np.load(gripper_width_path)
            print(joint_position)
            robot.send_joint_position(joint_position)
            robot.set_gripper_width(gripper_width)
            
        
            time.sleep(0.2)

        
    
    except Exception as e:
        print(e)
        if robot_loaded:
            del robot