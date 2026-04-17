#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, cv2
import numpy as np
from pathlib import Path
import open3d as o3d
import sys
sys.path.append("/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy")
from camera import CameraD400
# 仅保留：从RGB+Depth生成点云，并与GLB模型同屏对比
camera = CameraD400(camera_id="233622078525", width=640, height=480)
K = camera.getIntrinsics()

def build_intrinsic_from_K(K, H, W):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

def generate_pcd_from_files(
    camera_sn: str,
    rgb_path: str,
    depth_path: str,
    K_input: np.ndarray,
    depth_scale: float = 1000.0,
    extrinsic_path: str | None = None,
    to_base: bool = True,
):
    """
    仅生成点云（不保存、不交互），并可选用外参将其变换到Base坐标系。
    直接使用传入的K矩阵（ndarray），不读取任何K路径。
    """
    rgb_path = str(Path(rgb_path).expanduser())
    depth_path = str(Path(depth_path).expanduser())

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB 文件不存在: {rgb_path}")
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"Depth 文件不存在: {depth_path}")

    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise RuntimeError(f"无法读取 RGB: {rgb_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise RuntimeError(f"无法读取 Depth: {depth_path}")

    Hc, Wc = color_rgb.shape[:2]
    Hd, Wd = depth_img.shape[:2]
    if (Hc, Wc) != (Hd, Wd):
        raise ValueError(f"RGB 与 Depth 尺寸不一致: RGB={Hc}x{Wc}, Depth={Hd}x{Wd}")

    if not isinstance(K_input, np.ndarray):
        raise TypeError("K_input 必须是 numpy.ndarray")
    K_use = K_input.astype(np.float64)

    intrinsic = build_intrinsic_from_K(K_use, Hc, Wc)

    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=depth_scale, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # 外参（可选）
    T_cam_in_base = None
    if extrinsic_path is not None:
        extrinsic_path = str(Path(extrinsic_path).expanduser())
        if not os.path.isfile(extrinsic_path):
            raise FileNotFoundError(f"未找到外参文件: {extrinsic_path}")
        T = np.array(json.load(open(extrinsic_path)), dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"外参矩阵形状错误，应为4x4，实际: {T.shape}")
        T_cam_in_base = T
    if to_base and T_cam_in_base is not None:
        pcd = pcd.transform(T_cam_in_base)

    return pcd

def preview_with_glb(pcd, glb_path, T_glb_to_base=None, init_scale=1.0, window_name="PCD + GLB"):
    """
    仅显示点云与GLB模型，支持显示/隐藏、缩放、平移与保存。
    热键：
      G/g  显示/隐藏GLB
      ]/[  放大/缩小GLB
      X/x  沿 +X / -X 平移GLB
      Y/y  沿 +Y / -Y 平移GLB
      Z/z  沿 +Z / -Z 平移GLB
      Up   增大平移步长
      Down 减小平移步长
      S/s  保存当前GLB到同目录 *_edited.glb
      Q/q  退出
    """
    glb_path = str(Path(glb_path).expanduser())
    mesh = o3d.io.read_triangle_mesh(glb_path)
    if mesh.is_empty():
        raise RuntimeError(f"加载GLB失败: {glb_path}")
    mesh.compute_vertex_normals()

    if T_glb_to_base is not None:
        mesh.transform(T_glb_to_base)
    center = mesh.get_center()
    if init_scale != 1.0:
        mesh.scale(init_scale, center=center)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=1280, height=720)

    # 添加点云与GLB
    vis.add_geometry(pcd)
    vis.add_geometry(mesh)
    visible = {"glb": True}
    scale_step = 1.1
    translate_step = {"value": 0.01}  # 平移步长（单位取决于坐标尺度）

    def preserve_view():
        vc = vis.get_view_control()
        params = vc.convert_to_pinhole_camera_parameters()
        return vc, params

    def restore_view(vc, params):
        vc.convert_from_pinhole_camera_parameters(params)
        vis.update_renderer()

    def toggle_glb(_):
        vc, params = preserve_view()
        if visible["glb"]:
            vis.remove_geometry(mesh, reset_bounding_box=False)
        else:
            vis.add_geometry(mesh)
        visible["glb"] = not visible["glb"]
        restore_view(vc, params)

    def scale_glb(factor):
        vc, params = preserve_view()
        mesh.scale(factor, center=mesh.get_center())
        vis.update_geometry(mesh)
        restore_view(vc, params)

    def translate_glb(axis, sign):
        vc, params = preserve_view()
        delta = translate_step["value"] * sign
        if axis == 'x':
            t = np.array([delta, 0.0, 0.0])
        elif axis == 'y':
            t = np.array([0.0, delta, 0.0])
        else:
            t = np.array([0.0, 0.0, delta])
        mesh.translate(t, relative=True)
        vis.update_geometry(mesh)
        restore_view(vc, params)

    def adjust_step(increase=True):
        if increase:
            translate_step["value"] = min(translate_step["value"] * 2.0, 1.0)
        else:
            translate_step["value"] = max(translate_step["value"] / 2.0, 1e-5)
        print(f"当前平移步长: {translate_step['value']}")

    def save_glb(_):
        src = Path(glb_path)
        out = src.with_name(src.stem + "_edited" + src.suffix)
        ok = o3d.io.write_triangle_mesh(str(out), mesh, write_triangle_uvs=True)
        print(f"已保存: {out}" if ok else "保存失败")

    # 注册按键
    vis.register_key_callback(ord('G'), toggle_glb)
    vis.register_key_callback(ord('g'), toggle_glb)
    vis.register_key_callback(ord(']'), lambda v: scale_glb(scale_step))
    vis.register_key_callback(ord('['), lambda v: scale_glb(1.0 / scale_step))

    # 平移：X/Y/Z 正负方向
    vis.register_key_callback(ord('X'), lambda v: translate_glb('x', +1))
    vis.register_key_callback(ord('x'), lambda v: translate_glb('x', -1))
    vis.register_key_callback(ord('Y'), lambda v: translate_glb('y', +1))
    vis.register_key_callback(ord('y'), lambda v: translate_glb('y', -1))
    vis.register_key_callback(ord('Z'), lambda v: translate_glb('z', +1))
    vis.register_key_callback(ord('z'), lambda v: translate_glb('z', -1))

    # 调整步长：GLFW Up(265) / Down(264)
    vis.register_key_callback(265, lambda v: (adjust_step(True), None)[1])
    vis.register_key_callback(264, lambda v: (adjust_step(False), None)[1])

    # 保存
    vis.register_key_callback(ord('S'), save_glb)
    vis.register_key_callback(ord('s'), save_glb)

    def quit_cb(v):
        v.close()
        return False
    vis.register_key_callback(ord('Q'), quit_cb)
    vis.register_key_callback(ord('q'), quit_cb)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 用法：
    # python obj_manual.py <SN> <RGB_PATH> <DEPTH_PATH> [EXTRINSIC_JSON] [DEPTH_SCALE] [GLB_PATH] [GLB_INIT_SCALE]
    # 仅生成点云并与GLB模型同屏对比，不进行任何保存/平移/旋转操作。

    args = sys.argv[1:]
    DEFAULT_SN = "233622078525"
    DEFAULT_RGB = "/home/yushun/Workspace/Mini-Tele/zichen/obj_total/frame_000021_rgb.png"
    DEFAULT_DEPTH = "/home/yushun/Workspace/Mini-Tele/zichen/obj_total/frame_000021_depth.png"
    DEFAULT_EXTRINSIC = None
    DEFAULT_DEPTH_SCALE = 1000.0
    DEFAULT_GLB = "/home/yushun/Workspace/Mini-Tele/zichen/obj_total/result(1).glb"
    DEFAULT_GLB_INIT_SCALE = 1.0

    if len(args) < 3:
        sn = DEFAULT_SN
        rgb_path = DEFAULT_RGB
        depth_path = DEFAULT_DEPTH
        K_input = K  # 直接使用内存中的K
        extrinsic_path = DEFAULT_EXTRINSIC
        depth_scale = DEFAULT_DEPTH_SCALE
        glb_path = DEFAULT_GLB
        glb_init_scale = DEFAULT_GLB_INIT_SCALE
    else:
        sn = args[0]
        rgb_path = args[1]
        depth_path = args[2]
        # 第4个参数改为外参，可选
        extrinsic_path = args[3] if len(args) >= 4 else None
        depth_scale = float(args[4]) if len(args) >= 5 else 1000.0
        glb_path = args[5] if len(args) >= 6 else DEFAULT_GLB
        glb_init_scale = float(args[6]) if len(args) >= 7 else DEFAULT_GLB_INIT_SCALE
        K_input = K  # 始终使用内存中的K

    pcd = generate_pcd_from_files(
        camera_sn=sn,
        rgb_path=rgb_path,
        depth_path=depth_path,
        K_input=K_input,
        depth_scale=depth_scale,
        extrinsic_path=extrinsic_path,
        to_base=True,
    )

    preview_with_glb(
        pcd=pcd,
        glb_path=glb_path,
        T_glb_to_base=np.eye(4),
        init_scale=glb_init_scale,
        window_name="PointCloud vs GLB",
    )