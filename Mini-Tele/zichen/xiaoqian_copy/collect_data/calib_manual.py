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
# camera_id = "332322073416"  # right side camera
# camera_id = "233622078525"  # wrist camera

# ===== 预设集合：camera_id、DATASET_ROOT、K 一并管理 =====
PRESETS = {
    "left_side": {
        "camera_id": "233522075695",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-side2",
        # K 来源：可选 "camera"（通过设备读取）或 "file"（从npz加载）
        "K_source": "camera",
        "K_file": f"/home/yushun/Workspace/Mini-Tele/zichen/chess_capture/233522075695/intrinsics_opencv.npz",
    },
    "top": {
        "camera_id": "135122074278",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-top2",
        "K_source": "camera",
        "K_file": f"/home/yushun/Workspace/Mini-Tele/zichen/chess_capture/135122074278/intrinsics_opencv.npz",
    },
    # "right_side": {
    #     "camera_id": "332322073416",
    #     "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-aside5",
    #     "K_source": "camera",
    #     "K_file": f"/home/yushun/Workspace/Mini-Tele/zichen/chess_capture/332322073416/intrinsics_opencv.npz",
    # },
    "wrist": {
        "camera_id": "233622078525",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/wrist",
        "K_source": "camera",
        "K_file": f"/home/yushun/Workspace/Mini-Tele/zichen/chess_capture/233622078525/intrinsics_opencv.npz",
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
        K = np.load(preset["K_file"])['K']
    else:
        raise ValueError(f"Unknown K_source: {preset['K_source']}")
    D = np.array([0., 0., 0., 0.])
    DATASET_ROOT = preset["dataset_root"]

# ========= 常量 ==========================================================
DICT_NAME  = 'DICT_6X6_1000'      # 6×6_250 词典
TARGET_ID  = 0                     # 若已知 marker ID，填整数；未知留 None
MARKER_LEN = 0.15                  # marker 物理边长 (m)
# ========================================================================

# --- 新增：统一读取外参（优先 finetuned，其次已有结果，再次计算） ---
def load_extrinsics_for_root(dataset_root: str, compute_if_missing: bool = True):
    # Try to load base->cam extrinsics for the given dataset root.
    # Priority:
    #   1) camera_pose_finetuned.json
    #   2) camera_pose.json
    #   3) compute via calibration if compute_if_missing is True
    # Initialize required local variables used below
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
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
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
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

# New: Load extrinsics with priority (finetuned > camera_pose.json > cal_ext)
def load_extrinsics_priority(dataset_root, cam_obj, K_mat):
    # Return T_base_to_cam (4x4) for a preset, preferring finetuned file.
    # - If <dataset_root>/camera_pose_finetuned.json exists and contains key 'T_base_to_cam', load and return it.
    # - Else if <dataset_root>/camera_pose.json exists, load and return it (expects a 4x4 matrix or object convertible to np.array).
    # - Else fall back to running cal_ext() using the provided camera & K in globals.
    finetuned_path = os.path.join(dataset_root, "camera_pose_finetuned.json")
    if os.path.isfile(finetuned_path):
        try:
            with open(finetuned_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "T_base_to_cam" in data:
                T = np.array(data["T_base_to_cam"], dtype=np.float64)
            else:
                T = np.array(data, dtype=np.float64)
            if T.shape == (4, 4):
                print(f"[extrinsics] Loaded finetuned T_base_to_cam from {finetuned_path}")
                return T
            else:
                print(f"[extrinsics] Invalid shape in {finetuned_path}: {T.shape}")
        except Exception as e:
            print(f"[extrinsics] Failed to read finetuned file {finetuned_path}: {e}")

    pose_path = os.path.join(dataset_root, "camera_pose.json")
    if os.path.isfile(pose_path):
        try:
            with open(pose_path, "r") as f:
                data = json.load(f)
            T = np.array(data, dtype=np.float64)
            if T.shape == (4, 4):
                print(f"[extrinsics] Loaded camera_pose from {pose_path}")
                return T
            else:
                print(f"[extrinsics] Invalid shape in {pose_path}: {T.shape}")
        except Exception as e:
            print(f"[extrinsics] Failed to read {pose_path}: {e}")

    print(f"[extrinsics] No saved extrinsics found for {dataset_root}, running calibration...")
    global DATASET_ROOT, camera, K
    DATASET_ROOT = dataset_root
    camera = cam_obj
    K = K_mat
    T_base_to_cam, _ = cal_ext()
    return T_base_to_cam

# Provide a small wrapper so other parts can call cal_ext() as before
# Uses the current global DATASET_ROOT/camera/K

def cal_ext():
    global DATASET_ROOT
    camera_pose, dataset_root = load_extrinsics_for_root(DATASET_ROOT, compute_if_missing=True)
    return camera_pose, dataset_root

# ===== 同屏显示四预设，共享同一 Base 坐标系（wrist 为基准，不标定）=====
def visualize_all_presets_in_base():
    # Build four point clouds aligned to the same base frame (wrist pose from robot, others via calibration),
    # then open an interactive window.
    #
    # Keyboard controls:
    #   Visibility
    #     1: toggle left_side
    #     2: toggle top
    #     3: toggle right_side
    #     4: toggle wrist
    #     Q/q: quit viewer
    #
    #   Selection (which preset to manipulate)
    #     5: select left_side
    #     6: select top
    #     7: select right_side
    #     8: select wrist
    #
    #   Axis selection (which axis to manipulate)
    #     X/x: select X axis
    #     Y/y: select Y axis
    #     Z/z: select Z axis
    #
    #   Manipulation (applies to the currently selected preset point cloud ONLY)
    #     W/w: rotate +rot_step around selected axis
    #     S/s: rotate -rot_step around selected axis
    #     A/a: translate -trans_step along selected axis
    #     D/d: translate +trans_step along selected axis
    #
    #   Coordinate frame scaling:
    #     O/o: enlarge axes
    #     P/p: shrink axes
    #
    #   Color mode:
    #     U/u: toggle between real image colors and pure uniform colors per preset
    #
    #   Point size:
    #     K/k: increase point size
    #     L/l: decrease point size
    #
    #   Extrinsics:
    #     E/e: save current extrinsics (base->cam) for selected preset to dataset root
    #
    #   Help:
    #     H/h: show help overlay window
    #
    #   Arrow Up/Down:
    #     Up:    increase manipulation step (translation & rotation)
    #     Down:  decrease manipulation step (translation & rotation)
    robot = Flexiv()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])

    # Load runtime for presets
    left_cam, left_K, left_root = get_preset_runtime("left_side")
    top_cam, top_K, top_root = get_preset_runtime("top")
    # right_cam, right_K, right_root = get_preset_runtime("right_side")  # right side camera disabled
    wrist_cam, wrist_K, wrist_root = get_preset_runtime("wrist")

    # Wrist as base reference (do not calibrate wrist)
    T_wrist_to_base = robot.readCamPose()

    # Load extrinsics with priority for the other three presets
    T_base_to_left  = load_extrinsics_priority(left_root,  left_cam,  left_K)
    T_base_to_top   = load_extrinsics_priority(top_root,   top_cam,   top_K)
    # T_base_to_right = load_extrinsics_priority(right_root, right_cam, right_K)  # disabled

    # Keep a mutable copy of current extrinsics (Base->Cam) per preset, will be updated by user interactions
    current_T_base_to_cam = {
        "left_side": T_base_to_left.copy(),
        "top": T_base_to_top.copy(),
        # "right_side": T_base_to_right.copy(),  # right side disabled
        "wrist": T_wrist_to_base.copy(),  # wrist from robot; we keep it for consistency
    }

    # Build real-color point clouds transformed to base
    pcd_wrist = capture_pcd(wrist_cam, wrist_K).transform(T_wrist_to_base)
    pcd_left  = capture_pcd(left_cam,  left_K ).transform(np.linalg.inv(T_base_to_left))
    pcd_top   = capture_pcd(top_cam,   top_K  ).transform(np.linalg.inv(T_base_to_top))
    # pcd_right = capture_pcd(right_cam, right_K).transform(np.linalg.inv(T_base_to_right))  # disabled

    # pcd_wrist = pcd_wrist.random_down_sample(0.0625)
    # pcd_left  = pcd_left.random_down_sample(0.0625)
    # pcd_top   = pcd_top.random_down_sample(0.0625)
    # pcd_right = pcd_right.random_down_sample(0.0625)

    # Interactive viewer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="All presets in Base frame", width=1280, height=720)

    # --- Point size control (affects rendering of all point clouds) ---
    render_opt = vis.get_render_option()
    point_size_step = 0.1  # was 1.0
    min_point_size = 0.2    # was 1.0
    max_point_size = 20.0

    # Initialize a reasonable default point size
    render_opt.point_size = max(render_opt.point_size, 3.0)

    def set_point_size(new_size: float):
        # Clamp and apply without resetting view
        clamped = float(np.clip(new_size, min_point_size, max_point_size))
        render_opt.point_size = clamped
        vis.update_renderer()
        print(f"[viewer] Point size = {render_opt.point_size}")

    def inc_point_size(_):
        set_point_size(render_opt.point_size + point_size_step)

    def dec_point_size(_):
        set_point_size(render_opt.point_size - point_size_step)

    # Register key callbacks for point size
    vis.register_key_callback(ord('K'), inc_point_size)
    vis.register_key_callback(ord('k'), inc_point_size)
    vis.register_key_callback(ord('L'), dec_point_size)
    vis.register_key_callback(ord('l'), dec_point_size)

    geoms = {
        "coord": coord_frame,
        "left_side": pcd_left,
        "top": pcd_top,
        # "right_side": pcd_right,  # disabled
        "wrist": pcd_wrist,
    }
    visible = {
        "coord": True,
        "left_side": True,
        "top": True,
        # "right_side": False,  # disabled
        "wrist": True,
    }

    # Cache original per-point colors to restore real image colors on demand
    original_colors = {
        "left_side": o3d.utility.Vector3dVector(np.asarray(pcd_left.colors)),
        "top":       o3d.utility.Vector3dVector(np.asarray(pcd_top.colors)),
        # "right_side":o3d.utility.Vector3dVector(np.asarray(pcd_right.colors)),  # disabled
        "wrist":     o3d.utility.Vector3dVector(np.asarray(pcd_wrist.colors)),
    }
    # Uniform color palette (RGBA not supported in point colors; use RGB)
    uniform_palette = {
        "left_side": [1.0, 0.0, 0.0],  # red
        "top":       [0.0, 1.0, 0.0],  # green
        # "right_side":[0.0, 0.0, 1.0],  # blue (disabled)
        "wrist":     [1.0, 1.0, 0.0],  # yellow
    }
    # Current color mode: "real" or "uniform"
    color_mode = "real"

    # Add initial geometries
    vis.add_geometry(geoms["coord"])
    vis.add_geometry(geoms["left_side"])
    vis.add_geometry(geoms["top"])
    # vis.add_geometry(geoms["right_side"])  # disabled
    vis.add_geometry(geoms["wrist"])

    # Selection and manipulation state
    selected_name = "wrist"
    selected_axis = "x"
    trans_step = 0.01
    rot_step_deg = 2.0

    # --- Adjust manipulation steps with Arrow Up / Arrow Down ---
    # Arrow key codes: Up=265, Down=264 in Open3D GLFW backend
    def increase_steps(_):
        nonlocal trans_step, rot_step_deg
        trans_step *= 1.25
        rot_step_deg *= 1.25
        print(f"[viewer] Steps increased: trans_step={trans_step:.5f}, rot_step_deg={rot_step_deg:.3f}")

    def decrease_steps(_):
        nonlocal trans_step, rot_step_deg
        trans_step = max(1e-4, trans_step / 1.25)
        rot_step_deg = max(0.01, rot_step_deg / 1.25)
        print(f"[viewer] Steps decreased: trans_step={trans_step:.5f}, rot_step_deg={rot_step_deg:.3f}")

    # Register arrow keys
    vis.register_key_callback(265, increase_steps)  # Up arrow
    vis.register_key_callback(264, decrease_steps)  # Down arrow

    # --- Save current intrinsics K on SPACE for the selected preset ---
    # Map preset name -> (K, root) gathered above in this function
    preset_intrinsics = {
        "left_side": (left_K, left_root),
        "top": (top_K, top_root),
        # "right_side": (right_K, right_root),  # disabled
        "wrist": (wrist_K, wrist_root),
    }

    # --- Save current extrinsics (base->cam) on E for the selected preset ---
    # Map preset name -> dataset root
    preset_roots = {
        "left_side": left_root,
        "top": top_root,
        # "right_side": right_root,  # disabled
        "wrist": wrist_root,
    }
    # vis.register_key_callback(ord('B'), save_current_K)
    # vis.register_key_callback(ord('b'), save_current_K)

    def save_current_extrinsics(_):
        # Persist current extrinsics (4x4) of selected preset
        if selected_name not in current_T_base_to_cam:
            return
        if selected_name == "wrist":
            # Optional: skip saving wrist extrinsics (robot provides it), remove this guard if needed
            print("[viewer] Wrist extrinsics come from robot; skipping save.")
            return
        T_sel = current_T_base_to_cam[selected_name]
        root_sel = preset_roots[selected_name]
        try:
            os.makedirs(root_sel, exist_ok=True)
            out_path = os.path.join(root_sel, "demo_camera_pose_finetuned.json")
            with open(out_path, "w") as f:
                json.dump({"T_base_to_cam": np.asarray(T_sel).tolist()}, f, indent=2)
            print(f"[viewer] Saved NEW extrinsics for '{selected_name}' to {out_path}")
        except Exception as e:
            print(f"[viewer] Failed to save extrinsics for '{selected_name}': {e}")
    # Re-register callbacks to ensure save uses new logic
    vis.register_key_callback(ord('E'), save_current_extrinsics)
    vis.register_key_callback(ord('e'), save_current_extrinsics)


    # Helpers to preserve view
    def preserve_view():
        vc = vis.get_view_control()
        params = vc.convert_to_pinhole_camera_parameters()
        return vc, params

    def restore_view(vc, params):
        vc.convert_from_pinhole_camera_parameters(params)
        vis.update_renderer()

    # Visibility toggle
    def toggle(name: str):
        vc, params = preserve_view()
        if visible[name]:
            vis.remove_geometry(geoms[name], reset_bounding_box=False)
            visible[name] = False
        else:
            vis.add_geometry(geoms[name])
            visible[name] = True
        restore_view(vc, params)

    # Color mode toggle: U/u
    def toggle_color_mode(v):
        nonlocal color_mode
        vc, params = preserve_view()
        color_mode = "uniform" if color_mode == "real" else "real"
        # Apply colors to each preset point cloud without changing geometry
        for k in ["left_side", "top", "wrist"]:
            if color_mode == "uniform":
                geoms[k].paint_uniform_color(uniform_palette[k])
            else:
                # restore original per-point colors
                geoms[k].colors = o3d.utility.Vector3dVector(np.asarray(original_colors[k]))
            vis.update_geometry(geoms[k])
        restore_view(vc, params)

    # Selection
    def select_preset(name: str):
        nonlocal selected_name
        selected_name = name
        print(f"[viewer] Selected preset: {selected_name}")

    def select_axis(ax: str):
        nonlocal selected_axis
        selected_axis = ax
        print(f"[viewer] Selected axis: {selected_axis.upper()}")

    # Transforms
    def axis_index(ax: str) -> int:
        return {"x": 0, "y": 1, "z": 2}[ax]

    def apply_translation(sign: int):
        if selected_name not in geoms:
            return
        vc, params = preserve_view()
        t = np.zeros(3)
        t[axis_index(selected_axis)] = sign * trans_step
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = t
        # Update geometry in Base frame
        geoms[selected_name].transform(T)
        vis.update_geometry(geoms[selected_name])
        # Correctly accumulate into Base->Cam: new = old @ inv(delta_in_base)
        if selected_name in current_T_base_to_cam:
            try:
                delta_inv = np.linalg.inv(T)
                current_T_base_to_cam[selected_name] = current_T_base_to_cam[selected_name] @ delta_inv
            except np.linalg.LinAlgError:
                print("[viewer] Warning: translation delta not invertible (should not happen)")
        restore_view(vc, params)

    def apply_rotation(sign: int):
        if selected_name not in geoms:
            return
        vc, params = preserve_view()
        theta = np.deg2rad(sign * rot_step_deg)
        c, s = np.cos(theta), np.sin(theta)
        ax = selected_axis
        R = np.eye(3, dtype=np.float64)
        if ax == "x":
            R = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]], dtype=np.float64)
        elif ax == "y":
            R = np.array([[ c, 0, s],
                          [ 0, 1, 0],
                          [-s, 0, c]], dtype=np.float64)
        elif ax == "z":
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        # Update geometry in Base frame
        geoms[selected_name].transform(T)
        vis.update_geometry(geoms[selected_name])
        # Correctly accumulate into Base->Cam: new = old @ inv(delta_in_base)
        if selected_name in current_T_base_to_cam:
            delta_inv = np.linalg.inv(T)
            current_T_base_to_cam[selected_name] = current_T_base_to_cam[selected_name] @ delta_inv
        restore_view(vc, params)

    # On-screen help overlay using an OpenCV window (non-blocking)
    help_text_lines = [
        "Viewer Hotkeys:",
        "1/2/3/4: Toggle left/top/right/wrist",
        "5/6/7/8: Select preset to manipulate",
        "X/Y/Z : Select axis",
        "W/S   : Rotate +/- around selected axis",
        "A/D   : Move -/+ along selected axis",
        "O/P   : Enlarge/Shrink coordinate axes",
        "U     : Toggle color mode (real/uniform)",
        "K/L   : Increase/Decrease point size",
        "E/e   : Save current extrinsics (base->cam) for selected preset",
        "Q     : Quit, H : Toggle this help"
    ]
    help_visible = True

    def draw_help_overlay():
        img = np.zeros((360, 520, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (519, 359), (32, 32, 32), thickness=-1)
        y = 30
        for i, line in enumerate(help_text_lines):
            color = (255, 255, 255) if i == 0 else (220, 220, 220)
            cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y += 28
        cv2.imshow("Viewer Help", img)
        cv2.waitKey(1)

    def toggle_help(_):
        nonlocal help_visible
        help_visible = not help_visible
        if help_visible:
            draw_help_overlay()
        else:
            try:
                cv2.destroyWindow("Viewer Help")
            except Exception:
                pass

    # Show help at start
    # draw_help_overlay()

    # Register H/h to toggle help overlay
    vis.register_key_callback(ord('H'), toggle_help)
    vis.register_key_callback(ord('h'), toggle_help)

    # Print a help banner to terminal and allow re-print with H
    help_text = (
        "\n=== Viewer Hotkeys ===\n"
        "1 2 3 4 : Toggle visibility (left/top/right/wrist)\n"
        "5 6 7 8 : Select preset to manipulate (left/top/right/wrist)\n"
        "X Y Z   : Select axis (x/y/z)\n"
        "W/S     : Rotate +/− around selected axis\n"
        "A/D     : Move −/+ along selected axis\n"
        "O/P     : Enlarge/Shrink coordinate axes\n"
        "U       : Toggle color mode (real ↔ uniform per preset)\n"
        "K/L     : Increase/Decrease point size\n"
        "E/e     : Save current extrinsics (base->cam) for selected preset\n"
        "Q       : Quit\n"
        "H       : Show this help again\n"
    )
    print(help_text)

    # Key bindings
    vis.register_key_callback(ord('1'), lambda v: toggle("left_side"))
    vis.register_key_callback(ord('2'), lambda v: toggle("top"))
    # vis.register_key_callback(ord('3'), lambda v: toggle("right_side"))  # disabled
    vis.register_key_callback(ord('4'), lambda v: toggle("wrist"))

    vis.register_key_callback(ord('5'), lambda v: select_preset("left_side"))
    vis.register_key_callback(ord('6'), lambda v: select_preset("top"))
    # vis.register_key_callback(ord('7'), lambda v: select_preset("right_side"))  # disabled
    vis.register_key_callback(ord('8'), lambda v: select_preset("wrist"))

    vis.register_key_callback(ord('X'), lambda v: select_axis("x"))
    vis.register_key_callback(ord('x'), lambda v: select_axis("x"))
    vis.register_key_callback(ord('Y'), lambda v: select_axis("y"))
    vis.register_key_callback(ord('y'), lambda v: select_axis("y"))
    vis.register_key_callback(ord('Z'), lambda v: select_axis("z"))
    vis.register_key_callback(ord('z'), lambda v: select_axis("z"))

    vis.register_key_callback(ord('W'), lambda v: apply_rotation(+1))
    vis.register_key_callback(ord('w'), lambda v: apply_rotation(+1))
    vis.register_key_callback(ord('S'), lambda v: apply_rotation(-1))
    vis.register_key_callback(ord('s'), lambda v: apply_rotation(-1))
    vis.register_key_callback(ord('A'), lambda v: apply_translation(-1))
    vis.register_key_callback(ord('a'), lambda v: apply_translation(-1))
    vis.register_key_callback(ord('D'), lambda v: apply_translation(+1))
    vis.register_key_callback(ord('d'), lambda v: apply_translation(+1))

    # Coordinate frame scaling
    coord_scale_step = 1.25
    def scale_coord(factor: float):
        vc, params = preserve_view()
        coord_frame.scale(factor, center=[0.0, 0.0, 0.0])
        vis.update_geometry(coord_frame)
        restore_view(vc, params)
    vis.register_key_callback(ord('O'), lambda v: scale_coord(coord_scale_step))
    vis.register_key_callback(ord('o'), lambda v: scale_coord(coord_scale_step))
    vis.register_key_callback(ord('P'), lambda v: scale_coord(1.0 / coord_scale_step))
    vis.register_key_callback(ord('p'), lambda v: scale_coord(1.0 / coord_scale_step))

    # Color mode toggle I/i
    vis.register_key_callback(ord('I'), toggle_color_mode)
    vis.register_key_callback(ord('i'), toggle_color_mode)

    # Quit
    def quit_cb(v):
        v.close()
        return False
    vis.register_key_callback(ord('Q'), quit_cb)
    vis.register_key_callback(ord('q'), quit_cb)

    vis.run()
    vis.destroy_window()
    cv2.destroyAllWindows()
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