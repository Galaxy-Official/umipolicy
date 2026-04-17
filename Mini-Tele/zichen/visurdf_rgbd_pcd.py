#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viser 可视化工具：结合 URDF 机械臂点云和多相机 RGB-D 融合点云

功能：
1. 从 URDF 文件加载机械臂 mesh，采样为点云显示
2. 从多个 RealSense 相机（side、wrist、head）获取 RGB-D 数据，融合为点云
3. 使用 viser 进行实时可视化
4. 支持关节角控制和重置
5. 可调节点云采样率和点大小

依赖：
- viser: 可视化框架
- yourdfpy: URDF 加载
- open3d: 点云处理
- numpy: 数值计算
- pyrealsense2: RealSense 相机（可选）
"""

from __future__ import annotations

import time
import threading
import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Literal, List
import json
import sys

# 可选导入 RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. Camera features will be disabled.")


# ===== 相机预设配置 =====
CAMERA_PRESETS = {
    "side": {
        "camera_id": "233522075695",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-side2",
        "color": [1.0, 0.0, 0.0],  # 红色
        "invert_extrinsics": False,
    },
    "head": {
        "camera_id": "135122074278",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/2025-12-25-top2",
        "color": [0.0, 1.0, 0.0],  # 绿色
        "invert_extrinsics": False,
    },
    "wrist": {
        "camera_id": "233622078525",
        "dataset_root": "/home/yushun/Workspace/Mini-Tele/zichen/calib_eyehand/data/wrist",
        "color": [1.0, 1.0, 0.0],  # 黄色
        "invert_extrinsics": True,  # wrist 相机外参需要取逆
    },
}


@dataclass
class MultiCameraConfig:
    """多相机配置"""
    enabled_cameras: List[str] = field(default_factory=lambda: ["side", "head", "wrist"])
    width: int = 640
    height: int = 480
    fps: int = 30

    # 外参文件路径（覆盖预设中的外参）
    extrinsics_override: Dict[str, str] = field(default_factory=dict)

    # 是否连接真实机器人获取腕部相机外参
    connect_robot_for_wrist: bool = False
    robot_ip: str = "192.168.2.100"
    local_ip: str = "192.168.2.102"


@dataclass
class PointCloudConfig:
    """点云配置"""
    # RGB-D 点云配置
    rgbd_downsample_voxel_size: float = 0.005  # 体素下采样大小
    rgbd_point_size: float = 0.01  # 点云显示大小（viser 单位，约等于屏幕像素）
    rgbd_max_depth: float = 3.0  # 最大深度 (m)

    # 是否使用统一的颜色（而不是图像颜色）
    use_uniform_colors: bool = False

    # 点云更新频率
    update_rate: float = 10.0  # Hz


@dataclass
class RobotConfig:
    """机器人配置"""
    # 连接真实机器人读取状态
    connect_robot: bool = False
    robot_ip: str = "192.168.2.100"
    local_ip: str = "192.168.2.102"

    # 或者从数据集读取状态
    data_root: Optional[str] = None
    episode_index: int = 0

    # 关节角映射偏移（根据 URDF 调整）
    joint_offsets: List[float] = field(default_factory=lambda: [0.0] * 7)
    joint_multipliers: List[float] = field(default_factory=lambda: [1.0] * 7)


class RealSenseCamera:
    """RealSense 相机封装"""

    def __init__(self, camera_id: str, width: int = 640, height: int = 480, fps: int = 30):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not available")

        self.width = width
        self.height = height
        self.camera_id = camera_id

        # 配置管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 根据序列号查找设备
        ctx = rs.context()
        devices = ctx.query_devices()
        device_found = False

        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == camera_id:
                self.config.enable_device(camera_id)
                device_found = True
                break

        if not device_found:
            print(f"Warning: Camera {camera_id} not found, skipping...")
            self.pipeline = None
            return

        # 配置流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 启动管道
        try:
            self.profile = self.pipeline.start(self.config)

            # 获取深度传感器参数
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # 创建对齐对象
            self.align = rs.align(rs.stream.color)

            print(f"RealSense camera {camera_id} initialized: {width}x{height} @ {fps}fps")
        except Exception as e:
            print(f"Failed to start camera {camera_id}: {e}")
            self.pipeline = None

    def is_initialized(self) -> bool:
        return self.pipeline is not None

    def get_intrinsics(self) -> np.ndarray:
        """获取相机内参矩阵 K"""
        if not self.is_initialized():
            return np.eye(3)

        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        K = np.eye(3)
        K[0, 0] = intrinsics.fx
        K[1, 1] = intrinsics.fy
        K[0, 2] = intrinsics.ppx
        K[1, 2] = intrinsics.ppy

        return K

    def get_rgbd_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取 RGB-D 数据
        Returns:
            color: RGB 图像 (H, W, 3)
            depth: 深度图 (H, W)，单位：米
        """
        if not self.is_initialized():
            raise RuntimeError(f"Camera {self.camera_id} not initialized")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get frames")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # 转换深度到米
        depth = depth * self.depth_scale

        # BGR -> RGB
        color = color[:, :, [2, 1, 0]]

        return color, depth

    def stop(self):
        """停止相机"""
        if self.pipeline is not None:
            self.pipeline.stop()


class MultiCameraManager:
    """多相机管理器"""

    def __init__(self, config: MultiCameraConfig):
        self.config = config
        self.cameras: Dict[str, RealSenseCamera] = {}
        self.extrinsics: Dict[str, np.ndarray] = {}  # Base -> Camera 变换矩阵
        self.point_sizes: Dict[str, float] = {}  # 每个相机的独立 point_size
        self.robot = None  # 用于读取 wrist 相机的实时位姿

        # 如果需要从机器人读取 wrist 外参，先连接机器人
        if "wrist" in config.enabled_cameras and config.connect_robot_for_wrist:
            try:
                sys.path.append("/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy")
                from robot import Flexiv
                self.robot = Flexiv(
                    robot_ip=config.robot_ip,
                    local_ip=config.local_ip,
                    move_home=False,
                )
                print("Robot connected for wrist camera extrinsics")
            except Exception as e:
                print(f"Failed to connect robot for wrist camera: {e}")

        # 初始化启用的相机
        for cam_name in self.config.enabled_cameras:
            if cam_name not in CAMERA_PRESETS:
                print(f"Warning: Unknown camera preset '{cam_name}', skipping...")
                continue

            preset = CAMERA_PRESETS[cam_name]

            # 创建相机
            camera = RealSenseCamera(
                camera_id=preset["camera_id"],
                width=config.width,
                height=config.height,
                fps=config.fps,
            )

            if camera.is_initialized():
                self.cameras[cam_name] = camera
                print(f"Camera '{cam_name}' initialized: {preset['camera_id']}")

                # 对于 wrist 相机，如果连接了机器人，使用实时位姿（不需要外参文件）
                if cam_name == "wrist" and self.robot is not None:
                    print(f"  -> Using real-time robot pose for '{cam_name}'")
                    self.extrinsics[cam_name] = None  # 标记为使用实时位姿
                else:
                    # 加载外参
                    if cam_name in self.config.extrinsics_override:
                        # 使用覆盖的外参文件
                        T = self.load_extrinsics(
                            self.config.extrinsics_override[cam_name]
                        )
                    else:
                        # 从预设的数据集目录加载外参
                        T = self.load_extrinsics_from_dataset(
                            preset["dataset_root"], cam_name
                        )

                    # 根据预设决定是否取逆
                    if preset.get("invert_extrinsics", False):
                        T = np.linalg.inv(T)
                        print(f"  -> Extrinsics inverted for '{cam_name}'")

                    self.extrinsics[cam_name] = T

                # 初始化 point_size（使用默认值）
                self.point_sizes[cam_name] = 0.005

        print(f"Initialized {len(self.cameras)} cameras: {list(self.cameras.keys())}")

    def get_extrinsics(self, cam_name: str) -> np.ndarray:
        """获取相机外参（支持 wrist 相机的实时位姿）"""
        T = self.extrinsics.get(cam_name)
        if T is None and cam_name == "wrist" and self.robot is not None:
            # 从机器人实时读取 wrist 相机位姿
            try:
                T = self.robot.readCamPose()
                # 根据预设，wrist 需要取逆
                preset = CAMERA_PRESETS.get(cam_name, {})
                if preset.get("invert_extrinsics", False):
                    T = np.linalg.inv(T)
                    print(f"  -> Real-time wrist pose inverted")
                return T
            except Exception as e:
                print(f"Error reading wrist pose from robot: {e}")
                return np.eye(4)
        elif T is None:
            return np.eye(4)
        return T

    def load_extrinsics(self, filepath: str) -> np.ndarray:
        """加载外参文件"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # 支持 T_base_to_cam 格式
        if isinstance(data, dict) and "T_base_to_cam" in data:
            T = np.array(data["T_base_to_cam"])
        else:
            T = np.array(data)

        return T

    def load_extrinsics_from_dataset(self, dataset_root: str, cam_name: str) -> np.ndarray:
        """从数据集目录加载外参"""
        import os

        # 优先级: finetuned > camera_pose.json > 机器人读取 (仅wrist，如果配置了connect_robot)
        finetuned_path = os.path.join(dataset_root, "camera_pose_finetuned.json")
        pose_path = os.path.join(dataset_root, "camera_pose.json")

        # 尝试加载 finetuned 外参
        if os.path.isfile(finetuned_path):
            try:
                with open(finetuned_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "T_base_to_cam" in data:
                    T = np.array(data["T_base_to_cam"])
                else:
                    T = np.array(data)
                print(f"Loaded finetuned extrinsics for '{cam_name}' from {finetuned_path}")
                return T
            except Exception as e:
                print(f"Failed to load finetuned extrinsics: {e}")

        # 尝试加载普通外参
        if os.path.isfile(pose_path):
            try:
                with open(pose_path, "r") as f:
                    data = json.load(f)
                T = np.array(data)
                print(f"Loaded extrinsics for '{cam_name}' from {pose_path}")
                return T
            except Exception as e:
                print(f"Failed to load extrinsics: {e}")

        # 对于腕部相机，如果配置了 connect_robot，尝试从机器人读取
        if cam_name == "wrist" and self.config.connect_robot_for_wrist:
            try:
                T = self.load_robot_camera_pose()
                print(f"Loaded wrist camera pose from robot")
                return T
            except Exception as e:
                print(f"Failed to load wrist pose from robot: {e}")

        print(f"Warning: No extrinsics found for '{cam_name}', using identity")
        return np.eye(4)

    def load_robot_camera_pose(self) -> np.ndarray:
        """从机器人读取腕部相机位姿"""
        import sys
        sys.path.append("/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy")
        from robot import Flexiv

        robot = Flexiv(
            robot_ip=self.config.robot_ip,
            local_ip=self.config.local_ip,
            move_home=False,
        )
        T_wrist_to_base = robot.readCamPose()
        return T_wrist_to_base

    def get_all_rgbd_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """获取所有相机的 RGBD 数据
        Returns:
            Dict mapping camera name to (color, depth, K)
        """
        result = {}
        for cam_name, camera in self.cameras.items():
            try:
                color, depth = camera.get_rgbd_data()
                K = camera.get_intrinsics()
                result[cam_name] = (color, depth, K)
            except Exception as e:
                print(f"Error getting data from camera '{cam_name}': {e}")
        return result

    def stop_all(self):
        """停止所有相机"""
        for camera in self.cameras.values():
            camera.stop()


class PointCloudVisualizer:
    """点云可视化器"""

    def __init__(
        self,
        urdf_path: str,
        camera_config: MultiCameraConfig,
        pcd_config: PointCloudConfig,
        robot_config: RobotConfig,
    ):
        self.server = viser.ViserServer()
        self.camera_config = camera_config
        self.pcd_config = pcd_config
        self.robot_config = robot_config

        # 加载 URDF
        print(f"Loading URDF from: {urdf_path}")
        self.urdf = URDF.load(urdf_path)
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=self.urdf)

        # 获取关节信息
        self.actuated_names = list(self.viser_urdf.get_actuated_joint_limits().keys())
        self.joint_limits = self.viser_urdf.get_actuated_joint_limits()

        print(f"Detected joints: {self.actuated_names}")
        for name, (lower, upper) in self.joint_limits.items():
            print(f"  {name}: [{lower:.3f}, {upper:.3f}]")

        # 初始化多相机
        self.camera_manager = None
        if REALSENSE_AVAILABLE and camera_config.enabled_cameras:
            try:
                self.camera_manager = MultiCameraManager(camera_config)
            except Exception as e:
                print(f"Failed to initialize camera manager: {e}")

        # 初始化机器人连接
        self.robot = None
        if robot_config.connect_robot:
            try:
                sys.path.append("/home/yushun/Workspace/Mini-Tele/zichen/xiaoqian_copy")
                from robot import Flexiv
                self.robot = Flexiv(
                    robot_ip=robot_config.robot_ip,
                    local_ip=robot_config.local_ip,
                    move_home=False,
                )
                print("Robot connected successfully")
            except Exception as e:
                print(f"Failed to connect robot: {e}")
                self.robot = None

        # 初始关节角
        self.initial_joint_positions = self.get_initial_joint_positions()
        self.current_joint_positions = self.initial_joint_positions.copy()

        # 点云节点
        self.rgbd_pcd_node = None
        self.individual_pcd_nodes: Dict[str, viser.GlbHandle] = {}

        # 运行状态
        self.running = True
        self.update_thread = None

        # 设置 GUI
        self.setup_gui()

        # 添加坐标轴
        self.grid = self.server.scene.add_grid("grid", width=2, height=2)
        # self.server.scene.add_axes("axes", length=0.5, radius=0.01)

    def get_initial_joint_positions(self) -> Dict[str, float]:
        """获取初始关节角"""
        # 使用固定的初始关节角
        initial_joints = [
            1.5159254871832673e-05,
            -0.6991473436355591,
            -0.00017347678658552468,
            1.5714093446731567,
            1.901722134789452e-05,
            0.697081983089447,
            -0.00010770193330245093,
            0.10000000149011612
        ]

        # 参照 urdf_move.py，对关节角进行调整
        adjusted_joints = initial_joints.copy()
        # joint4 取负
        # adjusted_joints[3] *= -1

        # 映射到 URDF 关节，并 clip 到 bounds 内
        cfg = {}
        # 前7个是机械臂关节角
        for i in range(min(7, len(self.actuated_names))):
            joint_name = self.actuated_names[i]
            value = float(adjusted_joints[i])
            # Clip 到关节限制范围内
            lower, upper = self.joint_limits[joint_name]
            value = max(lower, min(upper, value))
            cfg[joint_name] = value
        # 第8个是gripper_width，查找夹爪关节
        gripper_val = adjusted_joints[7]
        for name in self.actuated_names:
            if "finger" in name.lower() or "gripper" in name.lower() or "joint8" in name.lower():
                value = float(gripper_val)
                # Clip 到关节限制范围内
                lower, upper = self.joint_limits[name]
                value = max(lower, min(upper, value))
                cfg[name] = value

        return cfg

    def load_joint_positions_from_dataset(self, data_root: str, episode_index: int) -> Dict[str, float]:
        """从数据集加载关节角"""
        import polars as pl

        data_path = Path(data_root)
        chunk_dirs = sorted((data_path / "data").glob("chunk-*"))

        if not chunk_dirs:
            print(f"Warning: No chunk directories found in {data_path}, using zero positions")
            return {name: 0.0 for name in self.actuated_names}

        for chunk_dir in chunk_dirs:
            parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
            for parquet_file in parquet_files:
                df = pl.read_parquet(parquet_file)
                if df["episode_index"][0] == episode_index:
                    state = df["observation.state"][0]
                    print(f"Loaded episode {episode_index} from {parquet_file}")
                    # 前7个是关节角
                    return {self.actuated_names[i]: float(state[i]) for i in range(min(7, len(self.actuated_names)))}

        print(f"Warning: Episode {episode_index} not found, using zero positions")
        return {name: 0.0 for name in self.actuated_names}

    def setup_gui(self):
        """设置 GUI 控件"""
        # 关节控制
        with self.server.gui.add_folder("Joint Control"):
            self.joint_sliders = {}

            for name in self.actuated_names:
                lower, upper = self.joint_limits[name]
                slider = self.server.gui.add_slider(
                    name,
                    min=float(lower),
                    max=float(upper),
                    step=0.001,
                    initial_value=self.initial_joint_positions.get(name, 0.0),
                )

                @slider.on_update
                def _(event, joint_name=name):
                    self.current_joint_positions[joint_name] = event.target.value
                    self.update_robot_pose()

                self.joint_sliders[name] = slider

            # 重置按钮
            reset_button = self.server.gui.add_button("Reset to Initial", icon=viser.Icon.REFRESH)

            @reset_button.on_click
            def _(_):
                self.reset_to_initial()

        # 点云配置
        with self.server.gui.add_folder("Point Cloud Settings"):
            # RGB-D 点云设置
            voxel_slider = self.server.gui.add_slider(
                "RGBD Voxel Size",
                min=0.001,
                max=0.02,
                step=0.001,
                initial_value=self.pcd_config.rgbd_downsample_voxel_size,
            )

            @voxel_slider.on_update
            def _(event):
                self.pcd_config.rgbd_downsample_voxel_size = event.target.value

            # 更新点云按钮
            update_pcd_button = self.server.gui.add_button("Update Point Cloud", icon=viser.Icon.REFRESH)

            @update_pcd_button.on_click
            def _(_):
                self.update_rgbd_point_cloud()

            # 每个相机的独立控制（开关 + point_size）
            self.camera_point_sliders = {}
            self.camera_enabled_checks = {}
            for cam_name in self.camera_config.enabled_cameras:
                if cam_name not in CAMERA_PRESETS:
                    continue

                # 为每个相机创建一个子文件夹
                with self.server.gui.add_folder(f"{cam_name.capitalize()} Camera"):
                    # 开关控制
                    enabled_checkbox = self.server.gui.add_checkbox(
                        f"Show {cam_name.capitalize()}",
                        True,
                    )
                    self.camera_enabled_checks[cam_name] = enabled_checkbox

                    @enabled_checkbox.on_update
                    def _(event, cn=cam_name):
                        if cn in self.individual_pcd_nodes:
                            self.individual_pcd_nodes[cn].visible = event.target.value

                    # Point Size 滑条
                    slider = self.server.gui.add_slider(
                        "Point Size",
                        min=0.001,
                        max=0.01,
                        step=0.001,
                        initial_value=0.005,
                    )
                    self.camera_point_sliders[cam_name] = slider

                    @slider.on_update
                    def _(event, cn=cam_name):
                        new_point_size = event.target.value
                        if self.camera_manager is not None and cn in self.camera_manager.point_sizes:
                            self.camera_manager.point_sizes[cn] = new_point_size
                        # 更新对应的独立点云节点 - 移除并重新创建
                        if cn in self.individual_pcd_nodes:
                            node = self.individual_pcd_nodes[cn]
                            points = node.points
                            colors = node.colors
                            visible = node.visible
                            node.remove()
                            self.individual_pcd_nodes[cn] = self.server.scene.add_point_cloud(
                                name=f"/rgbd_pcd_{cn}",
                                points=points,
                                colors=colors,
                                point_size=new_point_size,
                            )
                            self.individual_pcd_nodes[cn].visible = visible

            max_depth_slider = self.server.gui.add_slider(
                "Max Depth (m)",
                min=0.5,
                max=5.0,
                step=0.1,
                initial_value=self.pcd_config.rgbd_max_depth,
            )

            @max_depth_slider.on_update
            def _(event):
                self.pcd_config.rgbd_max_depth = event.target.value

            # 颜色模式
            uniform_color_checkbox = self.server.gui.add_checkbox(
                "Use Uniform Camera Colors",
                self.pcd_config.use_uniform_colors
            )

            @uniform_color_checkbox.on_update
            def _(event):
                self.pcd_config.use_uniform_colors = event.target.value

        # 显示控制
        with self.server.gui.add_folder("Visibility"):
            # Grid 开关
            show_grid = self.server.gui.add_checkbox("Show Grid", True)

            @show_grid.on_update
            def _(event):
                self.grid.visible = event.target.value

            # 背景颜色选择器
            bg_color_picker = self.server.gui.add_rgb(
                "Background Color",
                initial_value=(40, 40, 40),  # 默认深灰色 (RGB 0-255)
            )

            @bg_color_picker.on_update
            def _(event):
                # 创建纯色背景图 (numpy array 格式)
                r_int = int(event.target.value[0])
                g_int = int(event.target.value[1])
                b_int = int(event.target.value[2])

                # 创建 2x2 的纯色图像数组 (H, W, 3) uint8
                img_array = np.full((2, 2, 3), [r_int, g_int, b_int], dtype=np.uint8)

                # 使用 set_background_image 设置背景
                self.server.scene.set_background_image(img_array)

            show_urdf_mesh = self.server.gui.add_checkbox("Show URDF Mesh", True)

            @show_urdf_mesh.on_update
            def _(event):
                self.viser_urdf.show_visual = event.target.value

        # 初始化点云
        self.update_robot_pose()

        # 初始更新点云
        self.update_rgbd_point_cloud()

    def update_robot_from_hardware(self):
        """从真实机器人更新关节角"""
        if self.robot is None:
            return

        try:
            joints = self.robot.get_joint_positions()
            # 应用偏移和乘数
            offsets = self.robot_config.joint_offsets
            multipliers = self.robot_config.joint_multipliers
            adjusted_joints = []
            for i, j in enumerate(joints):
                adjusted = j * multipliers[i] + offsets[i]
                adjusted_joints.append(adjusted)

            # 更新当前关节角
            for i in range(min(len(adjusted_joints), len(self.actuated_names))):
                self.current_joint_positions[self.actuated_names[i]] = float(adjusted_joints[i])

            # 更新滑条显示
            for name, value in self.current_joint_positions.items():
                if name in self.joint_sliders:
                    self.joint_sliders[name].value = value

            # 更新 URDF
            self.update_robot_pose()
        except Exception as e:
            print(f"Error updating robot from hardware: {e}")

    def reset_to_initial(self):
        """重置到初始位置"""
        self.current_joint_positions = self.get_initial_joint_positions()
        for name, slider in self.joint_sliders.items():
            slider.value = self.current_joint_positions.get(name, 0.0)
        self.update_robot_pose()

    def update_robot_pose(self):
        """更新机器人姿态"""
        cfg = {name: self.current_joint_positions[name] for name in self.actuated_names}
        self.viser_urdf.update_cfg(cfg)

    def rgbd_to_point_cloud(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_base_to_cam: np.ndarray,
        uniform_color: Optional[np.ndarray] = None,
        downsample_step: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将 RGB-D 数据转换为点云（在 Base 坐标系中）

        Returns:
            points: (N, 3) 点云坐标
            colors: (N, 3) RGB 颜色 (0-255)
        """
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 下采样
        depth_ds = depth[::downsample_step, ::downsample_step]
        color_ds = color[::downsample_step, ::downsample_step]

        h, w = depth_ds.shape
        ys, xs = np.indices((h, w))
        u = (xs * downsample_step).flatten()
        v = (ys * downsample_step).flatten()
        z = depth_ds.flatten()

        # 过滤有效深度
        mask = (z > 0.1) & (z < self.pcd_config.rgbd_max_depth)
        if not np.any(mask):
            return np.empty((0, 3)), np.empty((0, 3))

        z = z[mask]
        u = u[mask]
        v = v[mask]
        colors = color_ds.reshape(-1, 3)[mask]

        # 相机坐标系下的 3D 点
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_cam = np.stack([x, y, z], axis=1)

        # 齐次坐标
        points_cam_h = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)

        # 转换到 Base 坐标系
        points_base = (np.linalg.inv(T_base_to_cam) @ points_cam_h.T).T[:, :3]

        # 设置颜色
        if uniform_color is not None:
            colors = np.tile(uniform_color * 255, (len(colors), 1))

        # 简单的体素下采样
        if self.pcd_config.rgbd_downsample_voxel_size > 0:
            voxel_size = self.pcd_config.rgbd_downsample_voxel_size
            voxel_indices = (points_base / voxel_size).astype(int)
            # 使用字典去重
            unique_keys = {tuple(v): i for i, v in enumerate(voxel_indices)}
            indices = list(unique_keys.values())
            points_base = points_base[indices]
            colors = colors[indices]

        return points_base, colors.astype(np.uint8)

    def update_rgbd_point_cloud(self):
        """更新多相机 RGB-D 融合点云"""
        if self.camera_manager is None:
            return

        # 获取所有相机的数据
        all_data = self.camera_manager.get_all_rgbd_data()

        print(f"update_rgbd_point_cloud: got data from {list(all_data.keys())} cameras")

        if not all_data:
            print("No camera data available!")
            return

        # 只创建独立的相机点云，不创建融合点云
        for cam_name, (color, depth, K) in all_data.items():
            print(f"Processing camera '{cam_name}': color shape={color.shape}, depth shape={depth.shape}")
            # 获取外参（支持 wrist 相机的实时位姿）
            T_base_to_cam = self.camera_manager.get_extrinsics(cam_name)
            print(f"  Extrinsics shape: {T_base_to_cam.shape}")
            # 获取外参（支持 wrist 相机的实时位姿）
            T_base_to_cam = self.camera_manager.get_extrinsics(cam_name)

            # 获取预设颜色（用于统一颜色模式）
            preset = CAMERA_PRESETS.get(cam_name, {})
            uniform_color = np.array(preset.get("color", [1.0, 1.0, 1.0]))

            # 转换为点云
            if self.pcd_config.use_uniform_colors:
                points, colors = self.rgbd_to_point_cloud(color, depth, K, T_base_to_cam, uniform_color)
            else:
                points, colors = self.rgbd_to_point_cloud(color, depth, K, T_base_to_cam, None)

            if len(points) > 0:
                # 过滤无效点
                valid = (points[:, 2] > 0) & (np.abs(points[:, 0]) < 2) & (np.abs(points[:, 1]) < 2) & (points[:, 2] < self.pcd_config.rgbd_max_depth)
                points = points[valid]
                colors = colors[valid]

                if len(points) > 0:
                    # 先移除旧节点
                    if cam_name in self.individual_pcd_nodes:
                        self.individual_pcd_nodes[cam_name].remove()

                    # 获取该相机的 point_size
                    cam_point_size = self.camera_manager.point_sizes.get(cam_name, 0.005)

                    # 创建新的点云节点
                    self.individual_pcd_nodes[cam_name] = self.server.scene.add_point_cloud(
                        name=f"/rgbd_pcd_{cam_name}",
                        points=points,
                        colors=colors,
                        point_size=cam_point_size,
                    )

                    # 根据开关设置可见性
                    if cam_name in self.camera_enabled_checks:
                        self.individual_pcd_nodes[cam_name].visible = self.camera_enabled_checks[cam_name].value
                    else:
                        self.individual_pcd_nodes[cam_name].visible = True

        # 删除融合点云节点（如果存在）
        if self.rgbd_pcd_node is not None:
            self.rgbd_pcd_node.remove()
            self.rgbd_pcd_node = None

    def run(self):
        """运行可视化"""
        print("Visualization running. Press Ctrl+C to exit.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            if self.camera_manager is not None:
                self.camera_manager.stop_all()


def main(
    # URDF 路径
    urdf_path: str = "/home/yushun/Workspace/Mini-Tele/urdf/assets/rizon/flexiv_rizon4_better.urdf",

    # 相机配置
    enabled_cameras: List[str] = ["side", "head", "wrist"],
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    connect_robot_for_wrist: bool = True,  # 默认启用，获取 wrist 相机外参
    robot_ip: str = "192.168.2.100",
    local_ip: str = "192.168.2.102",

    # 点云配置
    rgbd_voxel_size: float = 0.005,
    rgbd_point_size: float = 0.01,
    max_depth: float = 3.0,
    use_uniform_colors: bool = False,
    update_rate: float = 10.0,

    # 机器人/数据集配置
    connect_robot: bool = False,
    data_root: Optional[str] = None,
    episode_index: int = 0,
    joint_offsets: List[float] = None,
    joint_multipliers: List[float] = None,
) -> None:
    """运行 URDF + 多相机 RGBD 融合点云可视化

    Args:
        urdf_path: URDF 文件路径
        enabled_cameras: 启用的相机列表 (side, head, wrist)
        width: 相机图像宽度
        height: 相机图像高度
        fps: 相机帧率
        connect_robot_for_wrist: 是否连接机器人获取腕部相机外参
        robot_ip: 机器人 IP 地址
        local_ip: 本地 IP 地址
        rgbd_voxel_size: RGBD 点云体素下采样大小
        rgbd_point_size: RGBD 点云显示大小
        max_depth: 最大深度距离 (米)
        use_uniform_colors: 是否使用统一的相机颜色
        update_rate: 点云更新频率 (Hz)
        connect_robot: 是否连接真实机器人读取关节角
        data_root: 数据集根目录 (用于读取初始关节角)
        episode_index: 要加载的 episode 索引
        joint_offsets: 关节角偏移
        joint_multipliers: 关节角乘数
    """

    if joint_offsets is None:
        joint_offsets = [0.0] * 7
    if joint_multipliers is None:
        joint_multipliers = [1.0] * 7

    camera_config = MultiCameraConfig(
        enabled_cameras=enabled_cameras,
        width=width,
        height=height,
        fps=fps,
        connect_robot_for_wrist=connect_robot_for_wrist,
        robot_ip=robot_ip,
        local_ip=local_ip,
    )

    pcd_config = PointCloudConfig(
        rgbd_downsample_voxel_size=rgbd_voxel_size,
        rgbd_point_size=rgbd_point_size,
        rgbd_max_depth=max_depth,
        use_uniform_colors=use_uniform_colors,
        update_rate=update_rate,
    )

    robot_config = RobotConfig(
        connect_robot=connect_robot,
        robot_ip=robot_ip,
        local_ip=local_ip,
        data_root=data_root,
        episode_index=episode_index,
        joint_offsets=joint_offsets,
        joint_multipliers=joint_multipliers,
    )

    visualizer = PointCloudVisualizer(
        urdf_path=urdf_path,
        camera_config=camera_config,
        pcd_config=pcd_config,
        robot_config=robot_config,
    )

    visualizer.run()


if __name__ == "__main__":
    tyro.cli(main)
