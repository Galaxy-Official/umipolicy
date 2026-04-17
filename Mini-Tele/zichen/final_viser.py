import os
import time
import threading
import numpy as np
import pandas as pd
import polars as pl
import trimesh
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF
from pathlib import Path
from scipy.spatial.transform import Rotation
from PIL import Image
from typing import Optional, List, Dict
import tyro

# ===================== 🔧 统一配置区域 =====================
URDF_PATH = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/real2sim/urdf/assets/rizon/robot_with_grav.urdf"
DATA_ROOT = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/openpi/datasets/real/panblock/01-06-19-16-20/lerobot/test"
TRACKING_DATA_PATH = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/real2sim/mvtracker/datasets_panblock/result_data.npy"
GLB_FILE_PATH = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/real2sim/assets/panblock/scaled_mesh.glb"

EPISODE_INDEX = 0
TARGET_CAM_INDICES = [0, 1, 3] 
CAM_ID_TO_NAME = {0: "head", 1: "side", 2: "aside", 3: "wrist"}

# Mesh 初始校准
CALIB_TRANS = np.array([0.0, 0.0, 0.0])
CALIB_ROT_EULER = np.array([0.0, 0.0, 0.0])
CALIB_SCALE = 1.0
# ==========================================================

def compute_rigid_transform(source_points, target_points):
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = centroid_target - R @ centroid_source
    return R, T

def rotation_matrix_to_wxyz(R):
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat() 
    return np.array([quat[3], quat[0], quat[1], quat[2]])

class UnifiedVisualizer:
    def __init__(self):
        self.server = viser.ViserServer(port=8080)
        self.state = {"playing": False, "frame": 0, "manual_mode": False}
        
        # 1. 加载数据
        self._load_robot_data()
        self._load_tracking_data()
        self._load_scene_metadata()
        
        # 统一总帧数 (取最小值防止越界)
        self.num_frames = min(len(self.robot_traj), len(self.obj_traj))
        print(f"✅ 数据加载完成，总显示帧数: {self.num_frames}")

        # 2. 初始化场景节点
        self._setup_scene()
        
        # 3. 构建GUI
        self._setup_gui()

    def _load_robot_data(self):
        print("🤖 Loading URDF & Robot Trajectory...")
        self.urdf = URDF.load(URDF_PATH)
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=self.urdf)
        self.actuated_names = list(self.viser_urdf.get_actuated_joint_limits().keys())
        self.joint_limits = self.viser_urdf.get_actuated_joint_limits()

        # 读取 Parquet
        data_path = Path(DATA_ROOT)
        chunk_dirs = sorted((data_path / "data").glob("chunk-*"))
        self.robot_traj = None
        for chunk_dir in chunk_dirs:
            for pq in sorted(chunk_dir.glob("episode_*.parquet")):
                df = pl.read_parquet(pq)
                if df["episode_index"][0] == EPISODE_INDEX:
                    self.robot_traj = df["observation.state"].to_numpy()
                    break
            if self.robot_traj is not None: break
        
        if self.robot_traj is None: raise ValueError("Robot data not found!")

    def _load_tracking_data(self):
        print("📦 Loading Object Mesh & Tracking...")
        data_dict = np.load(TRACKING_DATA_PATH, allow_pickle=True).item()
        
        # 处理 Query points
        raw_query = data_dict.get("query_points", data_dict.get("query", None))
        if raw_query.ndim == 3: raw_query = raw_query[0]
        self.query_pts = raw_query[:, 1:4] 

        # 处理 Trajectory
        raw_traj = data_dict.get("traj_e", None)
        if raw_traj.ndim == 4: raw_traj = raw_traj[0]
        if raw_traj.shape[0] != len(self.robot_traj) and raw_traj.shape[1] == len(self.robot_traj):
            raw_traj = raw_traj.transpose(1, 0, 2)
        self.obj_traj = raw_traj

        # 加载 Mesh
        mesh = trimesh.load(GLB_FILE_PATH, force='mesh')
        T_calib = np.eye(4)
        T_calib[:3, :3] = Rotation.from_euler('xyz', CALIB_ROT_EULER, degrees=True).as_matrix() * CALIB_SCALE
        T_calib[:3, 3] = CALIB_TRANS
        mesh.apply_transform(T_calib)
        self.obj_mesh = mesh
        
        # 加载新mesh
        self.pan_mesh = trimesh.load("/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/real2sim/assets/panblock/object1/pan_scaled_mesh.glb", force='mesh')

        # 预计算物体位姿
        self.mesh_transforms = []
        for f in range(len(self.obj_traj)):
            R, T = compute_rigid_transform(self.query_pts, self.obj_traj[f])
            Mat = np.eye(4); Mat[:3, :3] = R; Mat[:3, 3] = T
            self.mesh_transforms.append(Mat)

    def _load_scene_metadata(self):
        print("🖼️ Loading Scene Metadata (RGB-D)...")
        search = os.path.join(DATA_ROOT, "data", "chunk-000", "*.parquet")
        pq_path = glob.glob(search)[0]
        self.meta_df = pd.read_parquet(pq_path)
        self.images_root = os.path.join(DATA_ROOT, "images")

    def _setup_scene(self):
        # 机器人已由 ViserUrdf 初始化
        # 物体 Mesh
        self.mesh_handle = self.server.scene.add_mesh_trimesh(
            "/object_mesh", self.obj_mesh, position=(0,0,0)
        )
        self.pan_handle = self.server.scene.add_mesh_trimesh(
            "/pan_mesh", self.pan_mesh, position=(0,0,0)
        )
        self.grid = self.server.scene.add_grid("grid", width=2, height=2)
        self.track_handle = None
        self.cloud_handle = None

    def _setup_gui(self):
        # 1. 播放控制
        with self.server.gui.add_folder("Timeline Control"):
            self.play_btn = self.server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            self.pause_btn = self.server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
            self.speed_slider = self.server.gui.add_slider("Speed", 0.1, 5.0, 0.1, 1.0)
            self.timeline = self.server.gui.add_slider("Frame", 0, self.num_frames - 1, 1, 0)

        @self.play_btn.on_click
        def _(_): self.state["playing"] = True; self.state["manual_mode"] = False
        @self.pause_btn.on_click
        def _(_): self.state["playing"] = False
        @self.timeline.on_update
        def _(_):
            self.state["frame"] = self.timeline.value
            self.update_frame(self.timeline.value)

        # 2. 机器人关节手动控制
        self.joint_sliders = {}
        with self.server.gui.add_folder("Robot Joints"):
            for name in self.actuated_names:
                low, high = self.joint_limits[name]
                s = self.server.gui.add_slider(name, float(low), float(high), 0.001, 0.0)
                self.joint_sliders[name] = s
                
                def make_callback(n=name):
                    @self.joint_sliders[n].on_update
                    def _(event):
                        if self.state["playing"]: return
                        self.state["manual_mode"] = True
                        self.viser_urdf.update_cfg({n: event.target.value})
                make_callback()

        # 3. 视觉控制
        with self.server.gui.add_folder("Visibility"):
            self.show_mesh = self.server.gui.add_checkbox("Show Object", True)
            self.show_robot = self.server.gui.add_checkbox("Show Robot", True)
            self.show_track = self.server.gui.add_checkbox("Show Track Points", False)
            self.show_cloud = self.server.gui.add_checkbox("Show Scene Cloud", True)
            self.cloud_step = self.server.gui.add_slider("Cloud Step", 1, 20, 1, 8)

        self.show_robot.on_update(lambda _: setattr(self.viser_urdf, 'show_visual', self.show_robot.value))
        # 其他属性在 update_frame 中实时检查

    def _get_cloud(self, frame_idx, step):
        all_pts, all_cols = [], []
        row = self.meta_df.iloc[frame_idx]
        raw_idx = row['frame_index'] if 'frame_index' in self.meta_df.columns else frame_idx
        ep_folder = f"episode_{int(row['episode_index']):06d}"
        
        for cam_idx in TARGET_CAM_INDICES:
            name = CAM_ID_TO_NAME[cam_idx]
            rgb_p = os.path.join(self.images_root, f"observation.images.{name}", ep_folder, f"frame_{int(raw_idx):06d}.png")
            if not os.path.exists(rgb_p): continue
            
            rgb = np.array(Image.open(rgb_p).convert("RGB"))
            depth_p = rgb_p.replace(f"observation.images.{name}", f"observation.depth.{name}")
            depth = np.array(Image.open(depth_p)).astype(np.float32) / 1000.0
            
            K = row[f"observation.intrinsics.{name}"].reshape(3,3)
            T_cw = row[f"observation.extrinsics.{name}"].reshape(4,4)
            
            rgb_s, depth_s = rgb[::step, ::step], depth[::step, ::step]
            h, w = depth_s.shape
            ys, xs = np.indices((h, w))
            u, v, z = (xs * step).flatten(), (ys * step).flatten(), depth_s.flatten()
            
            mask = (z > 0.1) & (z < 2.0)
            if not np.any(mask): continue
            
            z, u, v, cols = z[mask], u[mask], v[mask], rgb_s.reshape(-1, 3)[mask]
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            pts_c = np.stack([x, y, z, np.ones_like(x)], axis=1)
            pts_w = (np.linalg.inv(T_cw) @ pts_c.T).T 
            all_pts.append(pts_w[:, :3]); all_cols.append(cols)
            
        return (np.concatenate(all_pts), np.concatenate(all_cols)) if all_pts else (None, None)

    def update_frame(self, frame_idx):
        # --- 1. 更新机器人 ---
        current_action = self.robot_traj[frame_idx].copy()
        # 这里的硬编码逻辑保留自你的原脚本 (根据具体URDF调整)
        current_action[3] *= -1 
        
        cfg = {}
        for i in range(min(7, len(self.actuated_names))):
            name = self.actuated_names[i]
            cfg[name] = current_action[i]
            if name in self.joint_sliders:
                self.joint_sliders[name].value = current_action[i]
        
        # 夹爪
        gripper_val = current_action[7]
        for name in self.actuated_names:
            if any(k in name.lower() for k in ["finger", "gripper", "joint8"]):
                cfg[name] = gripper_val
                if name in self.joint_sliders: self.joint_sliders[name].value = gripper_val
        self.viser_urdf.update_cfg(cfg)

        # --- 2. 更新物体 Mesh ---
        if self.show_mesh.value:
            self.mesh_handle.visible = True
            T = self.mesh_transforms[frame_idx]
            self.mesh_handle.position = T[:3, 3]
            self.mesh_handle.wxyz = rotation_matrix_to_wxyz(T[:3, :3])
        else:
            self.mesh_handle.visible = False

        # --- 3. 更新追踪点 ---
        if self.show_track.value:
            self.track_handle = self.server.scene.add_point_cloud(
                "/track_points", points=self.obj_traj[frame_idx], colors=[0, 255, 0], point_size=0.02
            )
        elif self.track_handle:
            self.track_handle.remove()
            self.track_handle = None

        # --- 4. 更新点云 ---
        if self.show_cloud.value:
            pts, cols = self._get_cloud(frame_idx, self.cloud_step.value)
            if pts is not None:
                self.cloud_handle = self.server.scene.add_point_cloud(
                    "/scene_cloud", points=pts, colors=cols, point_size=0.01
                )
        elif self.cloud_handle:
            self.cloud_handle.remove()
            self.cloud_handle = None

    def playback_loop(self):
        while True:
            if self.state["playing"]:
                self.state["frame"] = (self.state["frame"] + 1) % self.num_frames
                self.timeline.value = self.state["frame"]
            
            fps = 30 * self.speed_slider.value
            time.sleep(1.0 / fps)

    def run(self):
        threading.Thread(target=self.playback_loop, daemon=True).start()
        print(f"🚀 Viser Server running at http://localhost:8080")
        while True:
            time.sleep(1)

import glob
if __name__ == "__main__":
    app = UnifiedVisualizer()
    app.run()