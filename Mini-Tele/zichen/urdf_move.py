from __future__ import annotations

import time
import threading
import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF
from pathlib import Path
import polars as pl

def main(
    # URDF路径
    urdf_path: str = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/real2sim/urdf/assets/rizon/robot_with_grav.urdf",
    # Lerobot dataset路径
    data_root: str = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/openpi/datasets/real/panblock/01-06-19-16-20/lerobot/test",
    # 选择episode索引
    episode_index: int = 0,
) -> None:
    server = viser.ViserServer()

    # 1. 加载 URDF
    urdf = URDF.load(urdf_path)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
    )

    # 获取所有可驱动关节的名称和限制，以便我们按顺序对齐数据
    actuated_names = list(viser_urdf.get_actuated_joint_limits().keys())
    joint_limits = viser_urdf.get_actuated_joint_limits()
    print(f"检测到关节: {actuated_names}")
    for name, (lower, upper) in joint_limits.items():
        print(f"  {name}: [{lower:.3f}, {upper:.3f}]")

    # 2. 从lerobot dataset读取轨迹数据
    data_path = Path(data_root)
    # 查找parquet文件
    chunk_dirs = sorted((data_path / "data").glob("chunk-*"))
    if not chunk_dirs:
        raise ValueError(f"在 {data_path / 'data'} 中未找到chunk-*目录")

    # 读取指定episode的action数据
    traj_data = None
    for chunk_dir in chunk_dirs:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for parquet_file in parquet_files:
            df = pl.read_parquet(parquet_file)
            if df["episode_index"][0] == episode_index:
                traj_data = df["observation.state"].to_numpy()
                print(f"从 {parquet_file} 读取到episode {episode_index}，共 {len(traj_data)} 帧")
                break
        if traj_data is not None:
            break

    if traj_data is None:
        raise ValueError(f"未找到episode_index={episode_index}的数据")

    num_frames = len(traj_data)
    print(f"轨迹数据形状: {traj_data.shape}")
    print(f"action前7维为关节角，第8维为gripper_width")

    # 3. GUI 控制组件
    with server.gui.add_folder("Timeline Control"):
        play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
        speed_slider = server.gui.add_slider("Playback Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        timeline = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )

    # 播放状态变量
    state = {"playing": False, "frame": 0, "manual_mode": False}

    @play_button.on_click
    def _(_): state["playing"] = True; state["manual_mode"] = False

    @pause_button.on_click
    def _(_): state["playing"] = False

    # 5. 关节手动控制
    joint_sliders = {}

    with server.gui.add_folder("Joint Control"):
        for name in actuated_names:
            lower, upper = joint_limits[name]
            # 为每个关节创建滑条，初始值为0
            slider = server.gui.add_slider(
                name,
                min=float(lower),
                max=float(upper),
                step=0.001,
                initial_value=0.0,
            )
            joint_sliders[name] = slider

            @slider.on_update
            def _(event, joint_name=name):
                # 如果正在播放轨迹，忽略滑条更新回调
                if state.get("playing", False):
                    return
                # 手动调整时，停止自动播放
                state["manual_mode"] = True
                state["playing"] = False
                # 更新该关节的角度
                cfg = {joint_name: event.target.value}
                viser_urdf.update_cfg(cfg)

    # 4. 更新函数：将action数据应用到 URDF，并同步更新滑条
    def update_robot_to_frame(frame_idx: int):
        frame_idx = int(frame_idx)
        if frame_idx >= num_frames: return

        current_action = traj_data[frame_idx].copy()
        
        # joint4 -pi
        # current_action[1] += 40 * np.pi / 180  # init 40 degree
        current_action[3] *= -1  # init 180 degree
        # current_action[5] -= 40 * np.pi / 180  # init 90 degree
        # current_action[-1] *= -1
        

        # 核心逻辑：将action的 8 维向量映射到 URDF 的关节配置字典中
        # 前7个是机械臂关节角，第8个是gripper_width
        cfg = {}

        # 自动映射前7个关节
        for i in range(min(7, len(actuated_names))):
            cfg[actuated_names[i]] = current_action[i]
            # 同步更新滑条显示（使用临时禁用回调避免递归）
            if actuated_names[i] in joint_sliders:
                joint_sliders[actuated_names[i]].value = current_action[i]

        # 夹爪处理 (寻找名字里带 finger 或 gripper 的关节)
        gripper_val = current_action[7]
        for name in actuated_names:
            if "finger" in name.lower() or "gripper" in name.lower() or "joint8" in name.lower():
                cfg[name] = gripper_val
                # 同步更新夹爪滑条
                if name in joint_sliders:
                    joint_sliders[name].value = gripper_val

        viser_urdf.update_cfg(cfg)

    # 当手动拖动滑块时更新
    @timeline.on_update
    def _(_):
        state["frame"] = timeline.value
        update_robot_to_frame(timeline.value)

    # 6. 播放循环 (在后台运行)
    def playback_loop():
        while True:
            if state["playing"]:
                # 计算下一帧
                state["frame"] = (state["frame"] + 1) % num_frames
                # 同步更新滑块（这会触发 timeline.on_update）
                timeline.value = state["frame"]
            
            # 控制播放速度
            fps = 30 * speed_slider.value
            time.sleep(1.0 / fps)

    # 启动后台线程
    threading.Thread(target=playback_loop, daemon=True).start()

    # 添加网格
    grid = server.scene.add_grid("grid", width=2, height=2)
    
    # 7. 辅助功能：显示/隐藏
    with server.gui.add_folder("Visibility"):
        show_mesh = server.gui.add_checkbox("Show Mesh", True)
        @show_mesh.on_update
        def _(_): viser_urdf.show_visual = show_mesh.value
        
        show_grid = server.gui.add_checkbox("Show Grid", True)
        @show_grid.on_update
        def _(_): grid.visible = show_grid.value

    # 保持主进程运行
    while True:
        time.sleep(1)

if __name__ == "__main__":
    tyro.cli(main)