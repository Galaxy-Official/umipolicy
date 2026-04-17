#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, json, yaml, cv2
import numpy as np
from pathlib import Path
import open3d as o3d
import sys
sys.path.append("/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy")
from robot import Flexiv
from camera import CameraD400
# camera_id = "233522075695"  # left side camera
# camera_id = "135122074278"  # top camera
# camera_id = "332322072673"  # right side camera
# camera_id = "233622078525"  # wrist camera

# ===== 预设集合：camera_id、DATASET_ROOT、K 一并管理 =====
PRESETS = {
    "left_side": {
        "camera_id": "233522075695",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-side",
        # K 来源：可选 "camera"（通过设备读取）或 "file"（从npz加载）
        "K_source": "camera",
        "K_file": f"chess_capture/233522075695/intrinsics_opencv.npz",
    },
    "top": {
        "camera_id": "135122074278",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-top2",
        "K_source": "camera",
        "K_file": f"chess_capture/135122074278/intrinsics_opencv.npz",
    },
    "right_side": {
        "camera_id": "332322072673",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-aside5",
        "K_source": "camera",
        "K_file": f"chess_capture/332322072673/intrinsics_opencv.npz",
    },
    "wrist": {
        "camera_id": "233622078525",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/wrist",
        "K_source": "camera",
        "K_file": f"chess_capture/233622078525/intrinsics_opencv.npz",
    },
}

# 选择预设（可以改为 "left_side"、"top"、"right_side"、"wrist"、"all"）
PRESET_NAME = "all"

# wrist 模式：跳过 PRESETS，直接在主流程中初始化
if PRESET_NAME == "wrist":
    camera_id = "233622078525"  # wrist camera SN
    camera = CameraD400(camera_id=camera_id, width=640, height=480)
    K = camera.getIntrinsics()
    D = np.array([0., 0., 0., 0.])
    DATASET_ROOT = "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-wrist"
elif PRESET_NAME == "all":
    # 同屏显示四预设时，不进行单相机初始化，后续在 visualize_all_presets_in_base() 中处理
    camera = None
    K = None
    D = np.array([0., 0., 0., 0.])
    DATASET_ROOT = None
else:
    # 其余模式：从 PRESETS 读取一套参数
    preset = PRESETS[PRESET_NAME]
    camera_id = preset["camera_id"]
    camera = CameraD400(camera_id=camera_id, width=640, height=480)
    if preset["K_source"] == "camera":
        K = camera.getIntrinsics()
    elif preset["K_source"] == "file":
        K = np.load(preset["K_file"])["K"]
    else:
        raise ValueError(f"Unknown K_source: {preset['K_source']}")
    D = np.array([0., 0., 0., 0.])
    DATASET_ROOT = preset["dataset_root"]

# ========= 常量 ==========================================================
DICT_NAME  = 'DICT_6X6_1000'      # 6×6_250 词典
TARGET_ID  = 0                     # 若已知 marker ID，填整数；未知留 None
MARKER_LEN = 0.15                  # marker 物理边长 (m)
# ========================================================================

def cal_ext():
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
    camera_pose = np.linalg.inv(T)

    # 保存 camera_pose 到 json 文件
    camera_pose_path = os.path.join(DATASET_ROOT, 'camera_pose.json')
    with open(camera_pose_path, 'w') as f:
        json.dump(camera_pose.tolist(), f, indent=2)
    print(f'📁 camera_pose 已保存到 {camera_pose_path}')

    return camera_pose, DATASET_ROOT

def verify(camera_pose, dataset_root):
    colors, depths = camera.get_data(hole_filling=False)

    # Construct Open3D camera intrinsic
    H, W = colors.shape[:2]
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

    # 保存相机坐标系下的点云
    camera_ply_path = os.path.join(dataset_root, 'camera.ply')
    o3d.io.write_point_cloud(camera_ply_path, pcd)
    print(f'📁 相机坐标系点云已保存到 {camera_ply_path}')

    # 通过 camera_pose 将点云转换到 base 坐标系
    camera_to_base = np.linalg.inv(camera_pose)
    pcd_base = pcd.transform(camera_to_base)

    # 保存 base 坐标系下的点云
    base_ply_path = os.path.join(dataset_root, 'base.ply')
    o3d.io.write_point_cloud(base_ply_path, pcd_base)
    print(f'📁 Base 坐标系点云已保存到 {base_ply_path}')

    # 可视化 pcd_base 并显示坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd_base, coord_frame],
                                       window_name="Base 坐标系点云",
                                       width=1280, height=720)

# ===== 工具函数 =====
def build_intrinsic_from_K(K, H, W):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

def capture_pcd(camera, K):
    colors, depths = camera.get_data(hole_filling=False)
    H, W = colors.shape[:2]
    intrinsic = build_intrinsic_from_K(K, H, W)
    color_o3d = o3d.geometry.Image(colors)
    depth_o3d = o3d.geometry.Image(depths.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

def get_preset_runtime(preset_name):
    preset = PRESETS[preset_name]
    cam = CameraD400(camera_id=preset["camera_id"], width=640, height=480)
    if preset["K_source"] == "camera":
        K_ = cam.getIntrinsics()
    elif preset["K_source"] == "file":
        K_ = np.load(preset["K_file"])["K"]
    else:
        raise ValueError(f"Unknown K_source: {preset['K_source']}")
    return cam, K_, preset["dataset_root"]

# ===== 同屏显示四预设，共享同一 Base 坐标系（wrist 为基准，不标定）=====
def visualize_all_presets_in_base():
    """
    Build four point clouds aligned to the same base frame (wrist pose from robot, others via calibration),
    then open an interactive window. Use keys 1-5 to control visibility:
      1: toggle left_side
      2: toggle top
      3: toggle right_side
      4: toggle wrist
      5: toggle all on/off
    """
    robot = Flexiv()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])

    # Load runtime for presets
    left_cam, left_K, left_root = get_preset_runtime("left_side")
    top_cam, top_K, top_root = get_preset_runtime("top")
    right_cam, right_K, right_root = get_preset_runtime("right_side")
    wrist_cam, wrist_K, wrist_root = get_preset_runtime("wrist")

    # Wrist as base (base->cam), skip calibration
    T_wrist_to_base = robot.readCamPose()

    # Calibrate three presets (keep cal_ext unchanged; switch global DATASET_ROOT/camera/K to reuse it)
    global DATASET_ROOT, camera, K
    DATASET_ROOT = left_root;  camera = left_cam;  K = left_K
    T_base_to_left, _ = cal_ext()
    DATASET_ROOT = top_root;   camera = top_cam;   K = top_K
    T_base_to_top, _ = cal_ext()
    DATASET_ROOT = right_root; camera = right_cam; K = right_K
    T_base_to_right, _ = cal_ext()

    # Build real-color point clouds transformed to base
    pcd_wrist = capture_pcd(wrist_cam, wrist_K).transform(T_wrist_to_base)
    pcd_left  = capture_pcd(left_cam,  left_K ).transform(np.linalg.inv(T_base_to_left))
    pcd_top   = capture_pcd(top_cam,   top_K  ).transform(np.linalg.inv(T_base_to_top))
    pcd_right = capture_pcd(right_cam, right_K).transform(np.linalg.inv(T_base_to_right))

    # Interactive viewer with 1-5 key toggles
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="All presets in Base frame", width=1280, height=720)

    geoms = {
        "coord": coord_frame,
        "left_side": pcd_left,
        "top": pcd_top,
        "right_side": pcd_right,
        "wrist": pcd_wrist,
    }
    visible = {
        "coord": True,
        "left_side": True,
        "top": True,
        "right_side": True,
        "wrist": True,
    }

    # Add initial geometries
    vis.add_geometry(geoms["coord"])
    vis.add_geometry(geoms["left_side"])
    vis.add_geometry(geoms["top"])
    vis.add_geometry(geoms["right_side"])
    vis.add_geometry(geoms["wrist"])

    def toggle(name: str):
        # Preserve current view parameters to avoid resetting the view
        vc = vis.get_view_control()
        params = vc.convert_to_pinhole_camera_parameters()

        if visible[name]:
            vis.remove_geometry(geoms[name], reset_bounding_box=False)
            visible[name] = False
        else:
            vis.add_geometry(geoms[name])
            visible[name] = True

        # Restore view and update renderer
        vc.convert_from_pinhole_camera_parameters(params)
        vis.update_renderer()

    def toggle_all(v):
        # Preserve current view parameters to avoid resetting the view
        vc = vis.get_view_control()
        params = vc.convert_to_pinhole_camera_parameters()

        any_on = any(visible[k] for k in ["left_side", "top", "right_side", "wrist"])
        for k in ["left_side", "top", "right_side", "wrist"]:
            if any_on and visible[k]:
                vis.remove_geometry(geoms[k], reset_bounding_box=False)
                visible[k] = False
            elif not any_on and not visible[k]:
                vis.add_geometry(geoms[k])
                visible[k] = True

        # Restore view and update renderer
        vc.convert_from_pinhole_camera_parameters(params)
        vis.update_renderer()

    # Key bindings: 1-4 toggle individual presets, 5 toggles all
    vis.register_key_callback(ord('1'), lambda v: toggle("left_side"))
    vis.register_key_callback(ord('2'), lambda v: toggle("top"))
    vis.register_key_callback(ord('3'), lambda v: toggle("right_side"))
    vis.register_key_callback(ord('4'), lambda v: toggle("wrist"))
    vis.register_key_callback(ord('5'), toggle_all)

    # Quit with Q/q without resetting the view
    def quit_cb(v):
        v.close()
        return False
    vis.register_key_callback(ord('Q'), quit_cb)
    vis.register_key_callback(ord('q'), quit_cb)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if PRESET_NAME == "wrist":
        robot = Flexiv()
        camera_pose = robot.readCamPose()
        camera_pose = np.linalg.inv(camera_pose)
        dataset_root = "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-wrist"
        verify(camera_pose, dataset_root)
    elif PRESET_NAME == "all":
        visualize_all_presets_in_base()
    else:
        camera_pose, dataset_root = cal_ext()
        verify(camera_pose, dataset_root)