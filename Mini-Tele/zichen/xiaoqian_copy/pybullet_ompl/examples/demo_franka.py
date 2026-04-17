import os.path as osp
import pybullet as p
import math
import sys
import time
import pybullet_data
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

try:
    import pb_ompl
except ImportError:
    # Handle import when running from different directories
    sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
    import pb_ompl

class SimpleEnv:
    """Simple environment wrapper for PbOMPLRobot"""
    def __init__(self, gui=True):
        # 选择连接模式：GUI用于图形化显示，DIRECT用于无头模式
        if gui:
            try:
                self.id = p.connect(p.GUI)
                # 设置相机视角以获得更好的观察角度
                p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-30,
                                           cameraTargetPosition=[0, 0, 0.5], physicsClientId=self.id)
            except:
                print("无法连接到GUI，切换到无头模式")
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)

class BoxDemo():
    def __init__(self, gui=True):
        self.obstacles = []

        # Create simple environment with GUI support
        self.env = SimpleEnv(gui=gui)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        # 启用渲染
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.env.id)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # load robot
        robot_urdf_path = osp.join(osp.dirname(osp.abspath(__file__)), "../models/franka_description/robots/panda_arm.urdf")
        robot_id = p.loadURDF(robot_urdf_path, (0,0,0), useFixedBase = 1)
        robot = pb_ompl.PbOMPLRobot(robot_id, env=self.env)
        # Initialize robot state
        robot.set_state([0,0,0,-1,0,1.5,0])  # Set initial joint configuration
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("RRTConnect")  # Try different planners: RRTConnect, BITstar, RRTstar

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        # add box
        self.add_box([1, 0, 0.7], [0.5, 0.5, 0.05])

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def visualize_path(self, path, color=[1, 0, 0], line_width=3):
        """可视化规划路径"""
        # 清除之前的路径可视化
        p.removeAllUserDebugItems(physicsClientId=self.env.id)

        # 获取末端执行器链接ID（通常是最后一个链接）
        num_joints = p.getNumJoints(self.robot.id, physicsClientId=self.env.id)
        end_effector_link = num_joints - 1  # 假设末端执行器是最后一个链接

        # 绘制路径轨迹
        for i in range(len(path) - 1):
            # 获取末端执行器位置
            self.robot.set_state(path[i])
            result1 = p.getLinkState(self.robot.id, end_effector_link, physicsClientId=self.env.id)
            if result1 is not None:
                pos1 = result1[0]

                self.robot.set_state(path[i + 1])
                result2 = p.getLinkState(self.robot.id, end_effector_link, physicsClientId=self.env.id)
                if result2 is not None:
                    pos2 = result2[0]

                    # 绘制线段
                    p.addUserDebugLine(pos1, pos2, color, line_width, physicsClientId=self.env.id)

        print(f"路径可视化完成，共绘制了 {len(path)-1} 条线段")

    def execute_with_visualization(self, path, time_step=0.05):
        """执行路径并显示动画"""
        print(f"开始执行轨迹，共 {len(path)} 个路点...")

        # 获取末端执行器链接ID
        num_joints = p.getNumJoints(self.robot.id, physicsClientId=self.env.id)
        end_effector_link = num_joints - 1

        # 设置初始状态
        self.robot.set_state(path[0])

        # 检查是否为GUI模式
        is_gui_mode = p.getConnectionInfo(physicsClientId=self.env.id)['connectionMethod'] == p.GUI

        if is_gui_mode:
            # 执行前暂停一下，让用户观察初始状态
            print("准备执行，按回车键开始...")
            input()

        # 逐步执行路径
        for i, q in enumerate(path):
            self.robot.set_state(q)
            p.stepSimulation(physicsClientId=self.env.id)

            # 添加当前位置的标记点（仅在GUI模式下）
            if is_gui_mode:
                result = p.getLinkState(self.robot.id, end_effector_link, physicsClientId=self.env.id)
                if result is not None:
                    pos = result[0]
                    p.addUserDebugPoints([pos], [[0, 1, 0]], pointSize=3, lifeTime=0.1, physicsClientId=self.env.id)

            # 控制执行速度（仅在GUI模式下）
            if is_gui_mode:
                time.sleep(time_step)

            if i % 10 == 0:
                print(f"执行进度: {i+1}/{len(path)} ({(i+1)/len(path)*100:.1f}%)")

        print("轨迹执行完成！")

    def demo(self):
        # More reasonable start and goal configurations
        start = [0, -0.5, 0, -1.5, 0, 1.5, 0.5]
        goal = [0.5, -0.3, 0.2, -1.2, 0.1, 1.3, 0.7]

        print(f"Planning from start: {start}")
        print(f"to goal: {goal}")

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal)

        if res:
            print(f"Planning successful! Found path with {len(path)} waypoints")
            print(f"First few waypoints: {path[:3]}")
            print(f"Last few waypoints: {path[-3:]}")

            # 可视化规划路径
            print("正在可视化规划路径...")
            self.visualize_path(path, color=[1, 0, 0], line_width=3)  # 红色轨迹

            # 执行轨迹并显示动画
            self.execute_with_visualization(path, time_step=0.02)

        else:
            print("Planning failed!")

        return res, path

if __name__ == '__main__':
    import argparse

    # 添加命令行参数
    parser = argparse.ArgumentParser(description='OMPL机械臂轨迹规划演示')
    parser.add_argument('--gui', action='store_true', help='使用图形界面显示')
    parser.add_argument('--no-gui', dest='gui', action='store_false', help='使用无头模式')
    parser.set_defaults(gui=True)

    args = parser.parse_args()

    print(f"运行模式: {'图形界面' if args.gui else '无头模式'}")
    print("=" * 50)

    env = BoxDemo(gui=args.gui)
    env.demo()