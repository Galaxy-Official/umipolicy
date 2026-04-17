import time
from pathlib import Path
import numpy as np
import viser
import viser.transforms as tf
from yourdfpy import URDF

def main(urdf_path: str):
    # 1. 加载 URDF
    # yourdfpy 会自动处理模型路径
    robot = URDF.load(urdf_path)
    
    # 2. 启动 Viser 服务器
    server = viser.ViserServer()
    
    # 用于存储 viser 中的节点句柄
    viser_nodes = {}

    def update_robot():
        """根据当前的关节角更新 Viser 中所有 Link 的位姿"""
        # 获取正向运动学计算后的所有 Link 的变换矩阵
        # cfg 格式为 {joint_name: value}
        fk = robot.forward_kinematics()
        
        for link_name, transform in fk.items():
            if link_name in viser_nodes:
                # 更新位姿
                viser_nodes[link_name].position = transform[:3, 3]
                viser_nodes[link_name].orientation = tf.SO3.from_matrix(transform[:3, :3])

    # 3. 初始化可视化模型
    with server.add_gui_folder("Robot Joints"):
        # 遍历 URDF 中的视觉元素
        for link_name, link in robot.robot.link_map.items():
            # 创建一个 Viser 节点作为容器
            viser_nodes[link_name] = server.add_frame(f"/robot/{link_name}")
            
            # 添加该 link 下的所有视觉网格
            for i, visual in enumerate(link.visuals):
                # 提取几何体（Mesh）
                # yourdfpy 使用 trimesh 加载数据
                meshes = robot.robot.link_map[link_name].visuals[i].geometry.meshes
                for j, mesh in enumerate(meshes):
                    server.add_mesh_trimesh(
                        name=f"/robot/{link_name}/visual_{i}_{j}",
                        mesh=mesh,
                    )

    # 4. 创建 GUI 滑块来控制关节
    gui_joints = {}
    
    # 遍历所有非固定关节
    for joint_name in robot.actuated_joint_names:
        joint = robot.robot.joint_map[joint_name]
        
        # 获取关节限位，如果没有则设置默认值
        lower = joint.limit.lower if joint.limit else -np.pi
        upper = joint.limit.upper if joint.limit else np.pi
        
        # 创建滑块
        slider = server.add_gui_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=0.01,
            initial_value=0.0,
        )
        
        # 监听滑块变化
        @slider.on_update
        def _(_):
            # 获取所有滑动条的当前值
            current_config = {name: s.value for name, s in gui_joints.items()}
            # 更新 yourdfpy 内部状态
            robot.update_cfg(current_config)
            # 更新 Viser 显示
            update_robot()
            
        gui_joints[joint_name] = slider

    # 初始刷新一次位置
    update_robot()

    print("Viser server 正在运行，请访问浏览器查看。")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    # 将此路径替换为你自己的 URDF 文件路径
    import sys
    urdf_file = "/home/yushun/Workspace/Mini-Tele/urdf/assets/rizon/flexiv_rizon4.urdf" 
    if len(sys.argv) > 1:
        urdf_file = sys.argv[1]
    
    main(urdf_file)