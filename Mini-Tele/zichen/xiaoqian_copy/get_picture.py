#!/usr/bin/env python3
import os
import cv2
import json
import time
import argparse
from datetime import datetime
import numpy as np

# 按你的工程结构导入
from camera import CameraD400
from robot import Flexiv

def save_rgb_depth(cam: CameraD400, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    color, depth = cam.get_data(hole_filling=False)

    # 保存RGB（BGR直接写入PNG）
    rgb_path = os.path.join(out_dir, "rgb.png")
    cv2.imwrite(rgb_path, color)

    # 读取相机depth_scale（深度值 * depth_scale = 米）
    depth_scale = cam.pipeline_profile.get_device().first_depth_sensor().get_depth_scale()

    # 保存Depth原始z16（与camera.py一致，供后处理使用）
    depth_z16_path = os.path.join(out_dir, "depth_z16.png")
    depth_z16 = depth.astype(np.uint16) if depth.dtype != np.uint16 else depth
    cv2.imwrite(depth_z16_path, depth_z16)

    # 同时保存米单位深度为npy，方便加载计算
    depth_m = depth.astype(np.float32) * float(depth_scale)
    depth_m_path = os.path.join(out_dir, "depth_m.npy")
    np.save(depth_m_path, depth_m)

    # 生成易查看的“距离越远越暗”的灰度深度图（8-bit）
    depth_gray_path = os.path.join(out_dir, "depth_gray.png")
    valid = depth_m > 0
    if np.any(valid):
        v = depth_m[valid]
        vmin = float(np.percentile(v, 5.0))
        vmax = float(np.percentile(v, 95.0))
        if vmax <= vmin:
            vmin = float(np.min(v))
            vmax = float(np.max(v) + 1e-6)
        d_norm = np.clip((depth_m - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
        # 反转：远处更暗（值更小）
        d_gray = (1.0 - d_norm) * 255.0
    else:
        d_gray = np.zeros_like(depth_m, dtype=np.float32)
    depth_gray_u8 = d_gray.astype(np.uint8)
    cv2.imwrite(depth_gray_path, depth_gray_u8)

    # 生成伪彩色深度图（可选，便于快速查看）
    depth_vis_path = os.path.join(out_dir, "depth_vis.png")
    depth_vis = cv2.applyColorMap(depth_gray_u8, cv2.COLORMAP_JET)
    cv2.imwrite(depth_vis_path, depth_vis)

    # 记录元信息
    meta = {
        "depth_scale": float(depth_scale),
        "notes": "depth_in_meters = depth_z16 * depth_scale; depth_gray: farther=darker"
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return rgb_path, depth_z16_path, depth_gray_path, depth_vis_path

def main():
    parser = argparse.ArgumentParser("Capture RGB/Depth from 4 RealSense cameras and save robot cam pose")
    parser.add_argument("--sns", nargs="+", required=True, help="4 camera serial numbers")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out-root", type=str, default="/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy")
    args = parser.parse_args()

    if len(args.sns) != 4:
        raise ValueError("Please provide exactly 4 serial numbers to --sns")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(args.out_root, f"multi_cam_capture_{ts}")
    os.makedirs(session_dir, exist_ok=True)

    # 保存机器人腕相机位姿（Base<-Cam 或 Base->Cam 依据你的API；这里沿用 Flexiv.readCamPose()）
    try:
        robot = Flexiv()
        T_wrist_to_base = robot.readCamPose()  # 假定返回4x4
        pose_path = os.path.join(session_dir, "robot_cam_pose.json")
        with open(pose_path, "w") as f:
            json.dump({"T_wrist_to_base": [list(row) for row in T_wrist_to_base]}, f, indent=2)
        print(f"Saved robot cam pose: {pose_path}")
    except Exception as e:
        print(f"Warning: failed to read/save robot cam pose: {e}")

    # 逐个相机抓拍并保存
    for sn in args.sns:
        cam_dir = os.path.join(session_dir, sn)
        print(f"Initializing camera {sn} ...")
        cam = CameraD400(camera_id=sn, width=args.width, height=args.height)
        # 小延时保证第一帧稳定
        time.sleep(0.2)
        rgb_path, depth_png_path, depth_gray_path, depth_vis_path = save_rgb_depth(cam, cam_dir)
        print(f"Saved {sn} -> {rgb_path}, {depth_png_path}, {depth_gray_path}, {depth_vis_path}")
        del cam  # 触发 __del__ 停止pipeline

    print(f"All done. Output dir: {session_dir}")

if __name__ == "__main__":
    main()