#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json, yaml, cv2
import numpy as np
from pathlib import Path
import open3d as o3d

# 1. 相机内参 -----------------------------------------------------------------
## top 001622071104
# K = np.array([[914.21118164,   0. ,        641.01922607],
#             [  0.,         913.02062988, 364.07424927],
#             [  0.,           0.,           1.        ]])
## side 233722071807 
K = np.array([[910.12329102,   0.,         649.51123047],
 [  0.,         908.83721924, 369.29821777],
 [  0. ,          0.,           1.        ]])

D = np.array([0., 0., 0., 0.])

def cal_ext():
    # ========= 必填常量 ==========================================================
    DATASET_ROOT   = 'data/calib_eye_to_hand/2025-06-14_14'   # 数据目录

    DICT_NAME  = 'DICT_6X6_250'      # 6×6_250 词典
    TARGET_ID  = 0                # 若已知 marker ID，填整数；未知留 None
    MARKER_LEN = 0.15               # marker 物理边长 (m)
    # ============================================================================

    # 2. ArUco 字典 --------------------------------------------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))

    # 3. 采集 hand-eye 配对姿态 ---------------------------------------------------
    Rs_tgt2cam, ts_tgt2cam = [], []
    Rs_g2b,     ts_g2b     = [], []

    ts_dirs = sorted(d for d in glob.glob(os.path.join(DATASET_ROOT, '*'))
                    if os.path.isdir(d))

    for ts in ts_dirs:
        pose_path = os.path.join(ts, 'robot_pose.json')
        rgb_list  = sorted(glob.glob(os.path.join(ts, 'rgb.jpg')))
        if not rgb_list or not os.path.isfile(pose_path):
            continue

        # 3-1 读取机器人 EE→Base ---------------------------------------------------
        T44 = np.array(json.load(open(pose_path)), dtype=np.float64)
        T44 = np.linalg.inv(T44)
        Rs_g2b.append(T44[:3, :3])
        ts_g2b.append(T44[:3, 3:4])

        # 3-2 Marker 位姿 Target→Cam ---------------------------------------------
        img  = cv2.imread(rgb_list[0], cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if ids is None:
            print(f'⚠️  {ts}: no marker detected')
            Rs_g2b.pop(); ts_g2b.pop()
            continue
        # else: ## visualization
        #     cv2.aruco.drawDetectedMarkers(img, corners, ids)

        #     # --- 为每个 marker 画坐标轴 ---
        #     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        #         corners, MARKER_LEN, K, D)
        #     for rvec, tvec in zip(rvecs, tvecs):
        #         cv2.drawFrameAxes(img, K, D, rvec, tvec, MARKER_LEN * 0.5)
        #     VIZ_DIR = Path("./")
        #     out_path = VIZ_DIR / f"{Path(ts).name}.png"
        #     cv2.imwrite(str(out_path), img)

        # 选定目标 marker ----------------------------------------------------------
        if TARGET_ID is None:
            idx = 0                        # 用第一张
        else:
            hits = np.where(ids.flatten() == TARGET_ID)[0]
            if len(hits) == 0:
                print(f'⚠️  {ts}: target ID {TARGET_ID} not found')
                Rs_g2b.pop(); ts_g2b.pop()
                continue
            idx = hits[0]

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[idx], MARKER_LEN, K, D)
        R_, _ = cv2.Rodrigues(rvec[0])
        Rs_tgt2cam.append(R_)
        ts_tgt2cam.append(tvec[0].reshape(3, 1))

    N = len(Rs_tgt2cam)
    print(f'🧮 有效 hand-eye 帧数 = {N}')
    assert N >= 8, '姿态太少；建议采 ≥ 8 组不同方位'

    # 4. hand-eye 求解 ------------------------------------------------------------
    R_c2b, t_c2b = cv2.calibrateHandEye(
        Rs_g2b, ts_g2b,
        Rs_tgt2cam, ts_tgt2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS)
    T = np.eye(4)
    T[:3, :3] = R_c2b
    T[:3, 3] = t_c2b.reshape(1, -1)
    # print(T)
    print(np.linalg.inv(T))

def verify():
    ## top
    # camera2base = np.array([[ 0.01765401,  0.99974247, -0.01425958,  0.02838739],
    #                     [ 0.9907336,  -0.01941227, -0.13442504, -0.77229811],
    #                     [-0.13466724, -0.01175431, -0.99082116,  0.94939382],
    #                     [ 0.,          0.,          0.,          1.        ]])
    # side
    camera2base = np.array([[ 0.99739231,  0.02812688, -0.06646399, -0.38234544],
                    [-0.04064356, -0.54210141, -0.83932959,  0.10983865],
                    [-0.05963795,  0.83984221, -0.5395446,   1.12547454],
                    [ 0.      ,    0.  ,        0.   ,       1.        ]])
    
    from camera import CameraD400
    # camera_id = "233722071807"  # top-view camera
    camera_id = "233722071467" # side-view camera
    camera = CameraD400(camera_id=camera_id)
    colors, depths = camera.get_data(hole_filling=False)

    # Construct Open3D camera intrinsic
    H, W = colors.shape[:2]
    # Example intrinsics, replace with actual if needed
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # Convert to Open3D images
    color_o3d = o3d.geometry.Image(colors)
    depth_o3d = o3d.geometry.Image(depths.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Visualize and pick a point
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    print("请在点云上点击一个点，按 q 退出。")
    vis.run()
    vis.destroy_window()
    picked = vis.get_picked_points()
    if picked:
        idx = picked[0]
        point = np.asarray(pcd.points)[idx]
        print(f"选中点索引: {idx}, 坐标: {point}")

        # Transform point from camera to base
        point_h = np.append(point, 1.0)  # homogeneous
        point_base = np.linalg.inv(camera2base)@point_h
        print(f"点在 base 坐标系下的坐标: {point_base[:3]}")
    else:
        print("未选中任何点。")

    # from robot import Flexiv
    # robot = Flexiv()
    # pose = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0.54],#0.51175809
    #                 [-0.05031023 , 0.99846963 , 0.02296254 ,-0.08],#-0.01212505
    #                 [-0.11582299,  0.0170036 , -0.99312432  ,0.06],#0.45597902
    #                 [ 0.      ,    0.      ,    0.  ,        1.        ]])
    # pose[:3, 3] = point_base[:3]
    # print("Moving robot to: ", pose[:3, 3])
    # robot.movePosePrimitive(pose)


if __name__ == "__main__":
    
    verify()