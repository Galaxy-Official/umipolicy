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
from robot import Flexiv

sys.path.append("flexiv_api/lib_py")

import flexivrdk
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
        self.pipeline = None
        self.config = rs.config()
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        try:
            # 尝试停止任何已存在的pipeline
            self.pipeline = rs.pipeline()
            self.pipeline.stop()
        except:
            pass

        # 创建新的pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_number is not None:
            print(f"使用相机序列号: {serial_number}")
            self.config.enable_device(serial_number)

        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, fps
        )
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, fps
        )

        try:
            self.pipeline_profile = self.pipeline.start(self.config)  # 获取图像视频流
            self.device = self.pipeline_profile.get_device()
            self.mtx = self.getIntrinsics()
            print(f"相机内参: {self.mtx}")
        except Exception as e:
            print(f"相机初始化失败: {e}")
            raise

    def __del__(self):
        """析构函数，确保正确释放相机资源"""
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("相机资源已释放")
            except Exception as e:
                print(f"释放相机资源时出错: {e}")

    def release(self):
        """手动释放相机资源"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            print("相机资源已手动释放")

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


if __name__ == "__main__":

    while True:

        root_dir = "collect_data/data/kp"
        global_key = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))  
        root_path = f"{root_dir}/{global_key}/init_capture"
        assert not os.path.exists(root_path)
        os.makedirs(root_path, exist_ok=True)

        video_path = f"{root_path}/rgb.mp4"
        video_depth16_path = f"{root_path}/depth.h5"

        # 初始化参数
        fps, w, h = 30, 1280, 720
        mp4 = cv2.VideoWriter_fourcc(*"mp4v")  # 视频格式
        sn = "001622071104" # todo

        # 完成相机初始化
        try:
            cam = Camera(w, h, fps, serial_number=sn)
        except Exception as e:
            print(f"相机初始化失败: {e}")
            print("请检查：1) 相机是否被其他程序占用 2) 序列号是否正确 3) 相机是否连接")

            # 尝试不使用序列号连接
            print("尝试自动检测相机...")
            try:
                cam = Camera(w, h, fps, serial_number=None)
                print("使用自动检测成功连接相机")
            except Exception as e2:
                print(f"自动检测也失败: {e2}")
                continue
        flag_V = 0
        idx = 0
        id = 0
        print("录制视频请按: s, 保存视频或退出请按： q")

        ## Robot初始化
        robot = Flexiv()
        # robot.move_to_home()
        HOME_POSE = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0.51175809],#0.51175809
                             [-0.05031023 , 0.99846963 , 0.02296254 ,-0.01212505],#-0.01212505
                             [-0.11582299,  0.0170036 , -0.99312432  ,0.45597902],#0.45597902
                             [ 0.      ,    0.      ,    0.  ,        1.        ]])
        robot.movePosePrimitive(HOME_POSE)
        cam2robot = robot.readCamPose()
        # np.savez(f"{root_path}/cam2robot.npz", cam2robot)
        robot_poses = {}

        while True:
            # 读取图像帧，包括RGB图和深度图
            color_image, depthxy_image, colorizer_depth, frame_timestamp = (
                cam.get_frame()
            )

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", color_image)
            # tcp_pose, tcp_pose_d, camera_pose = get_robot_states(robot, log)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("s"):
                flag_V = 1
                # 创建视频文件
                wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)
                wr_depth = h5py.File(video_depth16_path, "w")
    
                print("...录制视频中...")
            if flag_V == 1:
                # 保存图像帧
                wr.write(color_image)  # 保存RGB图像帧
                depth_map_name = str(id).zfill(5) + "_depth.png"
                wr_depth[depth_map_name] = (
                    depthxy_image 
                )
                # 保存时间戳
                # csvwriter.writerow([id, system_timestamp, frame_timestamp])
                robot_poses[id] = robot.readPose()
                id = id + 1
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                print("...录制结束/直接退出...")
                break
        
        depth_0 = wr_depth[str(0).zfill(5) + "_depth.png"]
        print(depth_0)
        pickle.dump(np.array(depth_0), open(f"{root_path}/depth.pkl", "wb"))


        # 录制完毕，释放对象
        wr.release()
        wr_depth.close()
        cam.release()  # 释放相机资源
        print(f"若保存视频，则视频保存在：{video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        output_image_path = f"{root_path}/rgb.jpg"
        cv2.imwrite(output_image_path, frame)
        cap.release()

        json.dump(robot_poses[0].tolist(), open(f"{root_path}/robot_pose.json", "w"))

        os.system(f"rm -r {video_path}")
        os.system(f"rm -r {video_depth16_path}")


        # Ask the user if they want to record another video or exit
        user_input = input("是否要录制另一个视频? (y/n): ").strip().lower()
        if user_input != "y":
            print("...退出程序...")
            break
