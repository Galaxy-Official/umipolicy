"""
使用realsense相机录制视频
"""

# !/usr/bin/env python
# coding=utf-8
import time
import pickle
import h5py  # 深度图格式所在库
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import csv
import json
import sys, os, glob
sys.path.append("/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy")
from robot import Flexiv
sys.path.append("/home/yushun/Workspace/Mini-Tele")
import lib_py.flexivrdk as flexivrdk
import time

robot_ip = "192.168.2.100"
local_ip = "192.168.2.103"
log = flexivrdk.Log()
mode = flexivrdk.Mode
robot = None

class Camera(object):
    """
    realsense相机处理类
    """

    def __init__(
        self, width=1280, height=720, fps=30, serial_number=None
    ):  # 图片格式可根据程序需要进行更改

        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        if serial_number is not None:
            print(serial_number)
            self.config.enable_device(serial_number)
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, fps
        )
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, fps
        )

        self.pipeline_profile = self.pipeline.start(self.config)  # 获取图像视频流
        self.device = self.pipeline_profile.get_device()
        self.mtx = self.getIntrinsics()
        print(self.mtx)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()  # 获得frame (包括彩色，深度图)
        colorizer = rs.colorizer()  # 创建伪彩色图对象
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        # 创建对齐对象
        align_to = rs.stream.color  # rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(align_to)  # “align_to”是我们计划对齐深度帧的流类型。
        aligned_frames = align.process(frames)
        # 获取对齐的帧
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame是对齐的深度图
        color_frame = aligned_frames.get_color_frame()

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = (
            aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        )
        # print("color_intrinsics:", color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)
        # print("depth_intrinsics:", depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)

        # left_frame  = frames.get_infrared_frame(1)
        # right_frame = frames.get_infrared_frame(2)
        color_image = np.asanyarray(color_frame.get_data())
        colorizer_depth = np.asanyarray(
            colorizer.colorize(aligned_depth_frame).get_data()
        )
        depthx_image = np.asanyarray(aligned_depth_frame.get_data())  # 原始深度图
        # left_frame   = np.asanyarray(left_frame.get_data())
        # right_frame  = np.asanyarray(right_frame.get_data())

        frame_timestamp = frames.get_timestamp()  # 获取时间戳

        return color_image, depthx_image, colorizer_depth, frame_timestamp
        # left_frame, right_frame

    def getIntrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        intrinsics = (
            color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        )
        print("intr", intrinsics)
        mtx = [
            intrinsics.width,
            intrinsics.height,
            intrinsics.ppx,
            intrinsics.ppy,
            intrinsics.fx,
            intrinsics.fy,
        ]
        camIntrinsics = np.array(
            [[mtx[4], 0, mtx[2]], [0, mtx[5], mtx[3]], [0, 0, 1.0]]
        )
        return camIntrinsics

## ins:
# 233722071807 wrist: [[910.12329102,   0.,         649.51123047],
#  [  0.,         908.83721924, 369.29821777],
#  [  0. ,          0.,           1.        ]]

# 001622071104 top: [[914.21118164,   0. ,        641.01922607],
#  [  0.,         913.02062988, 364.07424927],
#  [  0.,           0.,           1.        ]]


def random_transform(base_pose, rot_bound_deg, trans_bound):
    """
    对base_pose进行随机旋转和平移
    rot_bound_deg: 旋转角度上限（度），各轴在[-rot_bound_deg, rot_bound_deg]范围内随机
    trans_bound: [tx, ty, tz] 平移上限（米），各轴在[-trans_bound[i], trans_bound[i]]范围内随机
    """
    # 随机旋转角度（弧度）
    rot_bound_rad = np.deg2rad(rot_bound_deg)
    rx = np.random.uniform(-rot_bound_rad, rot_bound_rad)
    ry = np.random.uniform(-rot_bound_rad, rot_bound_rad)
    rz = np.random.uniform(-rot_bound_rad, rot_bound_rad)

    # 构建旋转矩阵（ZYX欧拉角顺序）
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_delta = Rz @ Ry @ Rx

    # 随机平移
    tx = np.random.uniform(-trans_bound[0], trans_bound[0])
    ty = np.random.uniform(-trans_bound[1], trans_bound[1])
    tz = np.random.uniform(-trans_bound[2], trans_bound[2])

    # 应用变换：旋转叠加到基准姿态，平移直接叠加
    target_pose = base_pose.copy()
    target_pose[:3, :3] = base_pose[:3, :3] @ R_delta  # 旋转叠加
    target_pose[:3, 3] = base_pose[:3, 3] + np.array([tx, ty, tz])  # 平移叠加

    return target_pose

        
if __name__ == "__main__":
    # -------------------------- 核心配置参数 --------------------------
    # sn = "233522075695"  # left side
    sn = "135122074278"  # top
    # sn = "332322072673"  # right side

    # root_dir = "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-side"
    root_dir = "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-top2"
    # root_dir = "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-aside5"

    cycle_count = 20  # 姿态循环次数
    record_duration = 1  # 单次录制时长（秒）
    cycle_interval = 10.0  # 单次循环总间隔（秒，每10秒发一次移动指令）
    fps = 30  # 相机帧率
    target_frames = fps * record_duration  # 单次录制帧数（60帧）
    gripper_close_width = 0.001  # 夹爪闭合宽度
    arrive_stable_time = 1.0  # 机器人到位后稳定等待时间（秒）\
    width = 640
    height = 480

    # -------------------------- 随机变换边界参数 --------------------------
    rotation_bound_deg = 5.0  # 旋转角度上限（度），各轴在[-5°, 5°]范围内随机
    translation_bound = np.array([0.1, 0.05, 0.05])  # xyz平移上限（米）
    
    # --- 初始化适应相机位置 --- #
    x_offset = 0.0          # left side: -0.3   top: -0.1
    y_offset = 0.1
    z_offset = 0.1          # left side: 0.1   top: 0   


    # -------------------------- 初始化全局资源 --------------------------
    print("===== 初始化机器人/相机资源 =====")
    robot = Flexiv()  # 初始化机器人（含夹爪）
    cam = Camera(width=width, height=height, fps=fps, serial_number=sn)  # 初始化相机
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)  # 创建预览窗口

    # -------------------------- 第一阶段：原地不动 + 闭合夹爪 --------------------------
    print("\n===== 第一阶段：原地闭合夹爪 =====")
    robot.move_to_home()
    current_pose = robot.readPose()

    # 移动到相机可见位置
    current_pose[0, 3] += x_offset
    current_pose[1, 3] += y_offset
    current_pose[2, 3] += z_offset
    robot.movePosePrimitive(current_pose)
    current_pose = robot.readPose()
    print(f"当前机器人位姿：\n{current_pose}")

    print(f"闭合夹爪（目标宽度：{gripper_close_width}m）...")
    robot.set_gripper_width(gripper_close_width)
    time.sleep(1.0)  # 等待夹爪完全闭合
    print("\n===== 第一阶段完成 =====")

    # -------------------------- 终端询问是否进入第二阶段 --------------------------
    while True:
        user_input = input("\n是否进入第二阶段（8个姿态循环录制）？输入 y/n 并回车：").strip().lower()
        if user_input in ["y", "n"]:
            break
        else:
            print(f"无效输入：{user_input}，请输入 y（是）或 n（否）！")

    if user_input == "n":
        print("\n用户选择不进入第二阶段，释放资源并结束程序...")
        cv2.destroyAllWindows()
        cam.pipeline.stop()
        print("资源已释放，程序结束")
        exit(0)
    else:
        print("\n用户选择进入第二阶段，开始8个姿态循环录制...")

    # -------------------------- 第二阶段：8个姿态循环录制（核心优化） --------------------------
    print("\n===== 第二阶段：8个姿态循环录制 =====")
    for cycle_idx in range(cycle_count):
        print(f"\n----- 开始第 {cycle_idx+1}/{cycle_count} 次循环 -----")
        start_cycle_time = time.time()
        
        # 1. 创建保存目录
        global_key = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        root_path = f"{root_dir}/{global_key}_{sn}"
        os.makedirs(root_path, exist_ok=True)
        video_path = f"{root_path}/rgb.mp4"
        video_depth16_path = f"{root_path}/depth.h5"

        # 2. 对current_pose做随机旋转和平移，生成目标位姿
        target_pose = random_transform(current_pose, rotation_bound_deg, translation_bound)
        print(f"下发第 {cycle_idx+1} 个目标位姿移动指令...")
        print(f"随机生成的目标位姿：\n{target_pose}")
        robot.movePosePrimitive(target_pose)  # 核心：到位后才返回
        
        # 3. 到位后稳定等待4秒（避免机器人微动）
        print(f"机器人已到位，等待 {arrive_stable_time} 秒稳定...")
        time.sleep(arrive_stable_time)

        # 4. 初始化视频/深度写入器
        mp4 = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(video_path, mp4, fps, (width, height), isColor=True)
        wr_depth = h5py.File(video_depth16_path, "w")
        robot_poses = {}

        # 5. 录制2秒视频（精准录制，无抖动）
        print(f"开始录制视频（时长{record_duration}秒）...")
        frame_id = 0
        while frame_id < target_frames:
            color_image, depthxy_image, colorizer_depth, frame_timestamp = cam.get_frame()
            wr.write(color_image)
            depth_map_name = str(frame_id).zfill(5) + "_depth.png"
            wr_depth[depth_map_name] = depthxy_image
            robot_poses[frame_id] = robot.readPose()
            cv2.imshow("RealSense", color_image)
            cv2.waitKey(1)
            frame_id += 1
        print(f"第 {cycle_idx+1} 次录制完成，共录制 {frame_id} 帧")

        # 6. 释放录制资源
        wr.release()
        wr_depth.close()

        # 7. 保存关键数据（首帧RGB/深度/位姿）
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{root_path}/rgb.jpg", frame)
        cap.release()
        
        depth_0 = h5py.File(video_depth16_path, "r")[str(0).zfill(5) + "_depth.png"]
        pickle.dump(np.array(depth_0), open(f"{root_path}/depth.pkl", "wb"))
        
        json.dump(robot_poses[0].tolist(), open(f"{root_path}/robot_pose.json", "w"))

        # 8. 清理临时文件
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(video_depth16_path):
            os.remove(video_depth16_path)

        # 9. 等待剩余时间（保证每10秒下发一次移动指令）
        cycle_elapsed_time = time.time() - start_cycle_time
        remaining_time = cycle_interval - cycle_elapsed_time
        if remaining_time > 0:
            print(f"等待 {remaining_time:.2f} 秒进入下一轮（保证10秒指令间隔）...")
            time.sleep(remaining_time)
        else:
            print(f"警告：本轮耗时 {cycle_elapsed_time:.2f} 秒，超过设定10秒间隔")

    # -------------------------- 结束流程 --------------------------
    print("\n===== 所有流程执行完成 =====")
    cv2.destroyAllWindows()
    cam.pipeline.stop()
    # robot.set_gripper_width(0.08)  # 可选：最后张开夹爪
    print("资源已释放，程序结束")