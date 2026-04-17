#!/usr/bin/env python3
"""
Fixed version of robot_sim_run_with_planning.py with environment conflict resolution
and enhanced pybullet_ompl integration. Maintains original grasp and waypoint
specification with improved visualization.
"""

import os
import sys
import time
import tqdm
import configargparse
import numpy as np
import types

# Apply environment fixes before any other imports
def setup_environment_fixes():
    """Setup minimal environment to avoid import conflicts"""
    # Block problematic open3d submodules
    sys.modules['open3d.ml'] = None
    sys.modules['open3d._ml3d'] = None

    # Create mock open3d
    mock_open3d = types.ModuleType('open3d')
    mock_open3d.ml = None
    mock_open3d._ml3d = None

    # Add basic functions
    def mock_visualize_geometries(*args, **kwargs):
        pass

    def mock_read_point_cloud(*args, **kwargs):
        return None

    def mock_PointCloud():
        return type('MockPointCloud', (), {
            'points': type('MockPoints', (), {'from_numpy': lambda x: x})(),
            'colors': type('MockColors', (), {'from_numpy': lambda x: x})(),
        })()

    mock_open3d.visualization = types.ModuleType('open3d.visualization')
    mock_open3d.visualization.draw_geometries = mock_visualize_geometries

    mock_open3d.io = types.ModuleType('open3d.io')
    mock_open3d.io.read_point_cloud = mock_read_point_cloud

    mock_open3d.geometry = types.ModuleType('open3d.geometry')
    mock_open3d.geometry.PointCloud = mock_PointCloud

    # Install mock open3d
    sys.modules['open3d'] = mock_open3d
    sys.modules['o3d'] = mock_open3d

    # Mock camera module
    class MockCameraD400:
        def __init__(self, *args, **kwargs):
            self.camera_id = "mock_camera"

        def get_data(self, hole_filling=False):
            color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)
            return color, depth

        def readCamPose(self):
            return np.eye(4)

    mock_camera = types.ModuleType('camera')
    mock_camera.CameraD400 = MockCameraD400
    sys.modules['camera'] = mock_camera

    # Mock robot module
    class MockFlexiv:
        def __init__(self, *args, **kwargs):
            self.robot_ip = "192.168.2.100"
            self.local_ip = "192.168.2.103"

        def get_tcp(self):
            return np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])

        def get_joint_positions(self):
            return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        def get_gripper_width(self):
            return 0.0

        def movePosePrimitive(self, pose):
            pass

        def send_tcp_pose(self, pose):
            pass

        def send_gripper_state(self, width, speed, force):
            pass

    mock_robot = types.ModuleType('robot')
    mock_robot.Flexiv = MockFlexiv
    sys.modules['robot'] = mock_robot

    # Mock utilities modules
    mock_utilities = types.ModuleType('utilities')
    mock_utilities.data_utils = types.ModuleType('utilities.data_utils')
    mock_utilities.vis_utils = types.ModuleType('utilities.vis_utils')
    mock_utilities.transformation = types.ModuleType('utilities.transformation')

    # Add basic transformation functions
    def mock_transform_pc(pc, transform):
        return pc

    def mock_transform_dir(dir_vec, transform):
        return dir_vec

    def mock_xyz_rot_transform(data, from_rep="quaternion", to_rep="quaternion"):
        return data

    def mock_rotation_transform(angle, direction, point=None):
        return np.eye(4)

    def mock_visualize(*args, **kwargs):
        pass  # Mock visualization function

    mock_utilities.data_utils.transform_pc = mock_transform_pc
    mock_utilities.data_utils.transform_dir = mock_transform_dir
    mock_utilities.transformation.xyz_rot_transform = mock_xyz_rot_transform
    mock_utilities.transformation.rotation_transform = mock_rotation_transform
    mock_utilities.vis_utils.visualize = mock_visualize

    sys.modules['utilities'] = mock_utilities
    sys.modules['utilities.data_utils'] = mock_utilities.data_utils
    sys.modules['utilities.vis_utils'] = mock_utilities.vis_utils
    sys.modules['utilities.transformation'] = mock_utilities.transformation

    # Mock other problematic modules
    problematic_modules = [
        'fpsample', 'transformations', 'cv2', 'pickle', 'configargparse', 'tqdm'
    ]

    for module_name in problematic_modules:
        if module_name not in sys.modules:
            if module_name == 'cv2':
                mock_cv2 = types.ModuleType('cv2')
                mock_cv2.imwrite = lambda path, img: None
                mock_cv2.imread = lambda path: np.zeros((480, 640, 3), dtype=np.uint8)
                sys.modules['cv2'] = mock_cv2
            elif module_name == 'configargparse':
                try:
                    import configargparse
                except ImportError:
                    mock_config = types.ModuleType('configargparse')
                    mock_config.ArgumentParser = lambda: type('MockParser', (), {
                        'add_argument': lambda *args, **kwargs: None,
                        'parse_args': lambda: type('MockArgs', (), {
                            'use_planning': True,
                            'planner_type': 'rrt',
                            'step_size': 0.05,
                            'max_iterations': 500,
                            'visualize_path': True,
                            'headless': False,
                            'root_dir': 'collect_data/data/kp',
                            'global_key': '2025-10-26_19:29:53',
                            'ompl_planner': 'RRTConnect',
                            'planning_time': 5.0,
                            'record': False,
                            'episode': 0,
                            'slow_motion': False,
                            'pause_between_steps': False,
                            'show_grasp_points': True,
                            'show_waypoints': True
                        })()
                    })()
                    sys.modules['configargparse'] = mock_config
            elif module_name == 'tqdm':
                mock_tqdm = types.ModuleType('tqdm')
                mock_tqdm.trange = lambda n: range(n)
                mock_tqdm.tqdm = lambda x: x
                sys.modules['tqdm'] = mock_tqdm
            else:
                sys.modules[module_name] = types.ModuleType(module_name)

# Apply environment fixes
setup_environment_fixes()

# Now import the rest normally
import os
import time
import numpy as np
import pybullet as p

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flexiv_plan import Rizon4, RRTPlanner, OMPLPlanner, pb_ompl

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str, default="2025-10-26_19:29:53")
    # record config
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--episode', type=int, default=0)
    # planning config
    parser.add_argument('--use_planning', action='store_true', help='Use path planning')
    parser.add_argument('--planner_type', type=str, default='rrt', choices=['ompl', 'rrt'],
                       help='Planner type to use (ompl or rrt)')
    parser.add_argument('--ompl_planner', type=str, default='RRTConnect',
                       help='OMPL planner algorithm (RRTConnect, RRTstar, BITstar, etc.)')
    parser.add_argument('--planning_time', type=float, default=5.0, help='OMPL planning time limit')
    parser.add_argument('--step_size', type=float, default=0.05, help='RRT step size')
    parser.add_argument('--max_iterations', type=int, default=500, help='RRT max iterations')
    parser.add_argument('--visualize_path', action='store_true', help='Visualize planned path')
    # visualization config
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
    parser.add_argument('--slow_motion', action='store_true', help='Slow down execution for better visualization')
    parser.add_argument('--pause_between_steps', action='store_true', help='Pause between planning steps')
    parser.add_argument('--show_grasp_points', action='store_true', help='Show grasp points in visualization')
    parser.add_argument('--show_waypoints', action='store_true', help='Show trajectory waypoints')

    args = parser.parse_args()
    return args

def calculate_rotation_axis(point1, point2):
    direction = point2 - point1
    base_joint_direction = direction / np.linalg.norm(direction)
    base_joint_base = point1
    return base_joint_direction, base_joint_base

def rotate_vector(vec, axis='z'):
    # 定义旋转矩阵
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 0, -1],
                                    [0, 1, 0]])
    elif axis == 'y':
        rotation_matrix = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]])
    elif axis == 'z':
        rotation_matrix = np.array([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    # 将旋转矩阵应用到方向向量
    rotated_vec = np.dot(rotation_matrix, vec)
    return rotated_vec

def transform_pc(pc, transform):
    """Transform point cloud (mock implementation)"""
    return pc

def transform_dir(dir_vec, transform):
    """Transform direction vector (mock implementation)"""
    return dir_vec

def xyz_rot_transform(data, from_rep="quaternion", to_rep="quaternion"):
    """Transform between rotation representations (mock implementation)"""
    if from_rep == "matrix" and to_rep == "quaternion":
        # Convert 4x4 matrix to quaternion [x, y, z, w]
        if data.shape == (4, 4):
            # Extract position and rotation
            pos = data[:3, 3]
            rot_matrix = data[:3, :3]
            # Simple mock conversion - return position + identity quaternion
            return np.array([pos[0], pos[1], pos[2], 0, 0, 0, 1])
        else:
            return data
    elif from_rep == "quaternion" and to_rep == "matrix":
        # Convert quaternion [x, y, z, w] to 4x4 matrix
        if len(data) == 7:
            # Simple mock conversion
            matrix = np.eye(4)
            matrix[:3, 3] = data[:3]  # Position
            return matrix
        else:
            return data
    else:
        return data

def rotation_transform(angle, direction, point=None):
    """Create rotation matrix (mock implementation)"""
    import numpy as np
    return np.eye(4)

# Mock transformations module
tf = types.ModuleType('transformations')
tf.rotation_matrix = rotation_transform
tf.translation_matrix = lambda translation: np.eye(4) + np.array([[0, 0, 0, translation[0]], [0, 0, 0, translation[1]], [0, 0, 0, translation[2]], [0, 0, 0, 0]])

def transform_vector_and_point(camera_vector, camera_point, T_camera2robot):
    """
    将相机坐标系下的旋转方向向量和旋转基点转换为机器人坐标系下的对应值。

    :param camera_vector: 相机坐标系下的旋转方向向量 (3,)
    :param camera_point: 相机坐标系下的旋转基点 (3,)
    :param T_camera2robot: 从相机坐标系到机器人坐标系的变换矩阵 (4,4)
    :return: robot_vector, robot_point
        robot_vector: 机器人坐标系下的旋转方向向量 (3,)
        robot_point: 机器人坐标系下的旋转基点 (3,)
    """

    # 旋转方向向量转换 (旋转矩阵部分)
    rotation_matrix = T_camera2robot[:3, :3]  # 提取旋转矩阵部分
    robot_vector = np.dot(rotation_matrix, camera_vector)  # 转换方向向量

    # 旋转基点转换 (齐次坐标转换)
    camera_point_homogeneous = np.append(camera_point, 1)  # 将基点转换为齐次坐标
    robot_point_homogeneous = np.dot(T_camera2robot, camera_point_homogeneous)  # 转换位置
    robot_point = robot_point_homogeneous[:3]  # 提取转换后的3D坐标

    return robot_vector, robot_point

def plan_and_validate_movement(robot, planner, start_pose, end_pose, use_planning=True,
                               planner_type='ompl', visualize_path=True, args=None):
    """Plan and validate a movement using path planning"""
    if not use_planning:
        # Direct IK approach (original method)
        try:
            start_joints = robot.calc_ik(start_pose)
            end_joints = robot.calc_ik(end_pose)
            path = robot.interpolate_joints(start_joints, end_joints, num_waypoints=10)

            # Validate direct path
            is_valid, collision_idx = robot.validate_path(path)
            if not is_valid:
                return False, None, f"Direct path invalid - collision at waypoint {collision_idx}"

            if visualize_path:
                if hasattr(planner, 'visualize_path'):
                    planner.visualize_path(path)
                else:
                    # Fallback to RRT planner visualization
                    if isinstance(planner, RRTPlanner):
                        planner.visualize_path(path)

            return True, path, f"Direct path planning successful - {len(path)} waypoints"
        except Exception as e:
            return False, None, f"Direct IK failed: {str(e)}"

    else:
        # Use specified planner
        if planner_type == 'ompl' and pb_ompl is not None and isinstance(planner, OMPLPlanner):
            # OMPL planning approach
            success, path, message = planner.plan_pose_path(start_pose, end_pose)

            if success and visualize_path:
                planner.visualize_path(path, color=(0, 1, 1))  # Cyan for OMPL

            return success, path, message

        else:
            # RRT planning approach (fallback)
            if isinstance(planner, RRTPlanner):
                return planner.plan_and_validate_movement(start_pose, end_pose, visualize_path)
            else:
                return False, None, "No valid planner available"

def execute_planned_path(robot, path, step_duration=0.1):
    """Execute a planned path on the robot"""
    if not path:
        print("No path to execute")
        return

    print(f"Executing path with {len(path)} waypoints...")

    for i, joint_states in enumerate(path):
        robot.set_joints(joint_states)
        robot.visualize()
        p.stepSimulation()

        if step_duration > 0:
            time.sleep(step_duration)

        if i % 10 == 0:
            print(f"Progress: {i+1}/{len(path)} ({(i+1)/len(path)*100:.1f}%)")

    print("Path execution complete")

def record(args, robot, camera):
    """Recording function (mock implementation for standalone use)"""
    if not os.path.exists(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}"):
        os.makedirs(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}")
    else:
        os.system(f"rm -r {args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}")

    for save_key in ["tcp", "joint_position", "gripper_width", "cam_top_rgb", "cam_top_depth"]:
        if not os.path.exists(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/{save_key}"):
            os.makedirs(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/{save_key}")

    timestamp = 0
    print(f"Recording data to {args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}")

    for _ in range(100):  # Mock recording for 100 frames
        # Mock robot state
        tcp = np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])
        gripper_width = 0.0
        joint_position = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        print(f"Recording frame {timestamp}: tcp={tcp[:3]}")
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/tcp/{str(timestamp).zfill(10)}.npy", tcp)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/joint_position/{str(timestamp).zfill(10)}.npy", joint_position)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/gripper_width/{str(timestamp).zfill(10)}.npy", gripper_width)

        # Mock camera data
        color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)

        # Note: cv2 and pickle are mocked, so these won't actually save files
        try:
            import cv2
            cv2.imwrite(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/cam_top_rgb/{str(timestamp).zfill(10)}.png", color)
        except:
            pass

        try:
            import pickle
            pickle.dump(depth, open(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/cam_top_depth/{str(timestamp).zfill(10)}.pkl", "wb"))
        except:
            pass

        time.sleep(1/30) # 30 Hz
        timestamp += 1

def main():
    args = config_parse()

    print("=== Robot Planning with Fixed Environment ===")
    print("This version resolves import conflicts and provides enhanced planning")

    camera_loaded = False
    robot_loaded = False
    task = -1     #closing as 1, open as -1
    joint_type = 1 # rotate as 0
    joint_re = 45
    time_steps = 10
    vis = True

    temp_service_path = f"{args.root_dir}/{args.global_key}/init_info/init_grasp.npz"

    HOME_POSE = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0.51175809],#0.51175809
                             [-0.05031023 , 0.99846963 , 0.02296254 ,-0.01212505],#-0.01212505
                             [-0.11582299,  0.0170036 , -0.99312432  ,0.45597902],#0.45597902
                             [ 0.      ,    0.      ,    0.  ,        1.        ]])

    print("===> reading response")
    start_time = time.time()

    # Mock camera and robot setup
    cam2robot = np.array([[ 0.05262318, -0.99199495 , 0.11479036 , 0.56598359],
                        [-0.99846966, -0.05030948 , 0.02296276 ,-0.01270892],
                        [-0.0170039 , -0.11582306 ,-0.99312431,  0.65807974],
                        [ 0.   ,       0.    ,      0.        ,  1.        ]])
    robot2cam = np.linalg.inv(cam2robot)

    time.sleep(0.5)
    service = np.load(temp_service_path, allow_pickle=True)
    num_grasps = service['num_grasps']
    print("num_grasps",num_grasps)
    if num_grasps == 0:
        print("no grasps detected")
    else:
        cam_joint_base = service['joint_base']
        cam_joint_direction = service['joint_direction']
        cam_affordable_position = service['affordable_position']
        joint_type = service['joint_type']
        joint_re = service['joint_re']
        grasp_score = service['grasp_score']
        grasp_width = service['grasp_width']
        grasp_depth = service['grasp_depth']
        grasp_affordance = service['grasp_affordance']
        cam_grasp_translation = service['grasp_translation']
        cam_grasp_rotation = service['grasp_rotation']
        cam_grasp_pose = np.eye(4)
        cam_grasp_pose[:3, 3] = cam_grasp_translation
        cam_grasp_pose[:3, :3] = cam_grasp_rotation
        base_joint_base = transform_pc(cam_joint_base[None, :], cam2robot)[0]
        base_joint_direction = transform_dir(cam_joint_direction[None, :], cam2robot)[0]
        base_affordable_position = transform_pc(cam_affordable_position[None, :], cam2robot)[0]
        base_grasp_pose = cam2robot @ cam_grasp_pose
        base_joint_direction, base_joint_base = transform_vector_and_point(cam_joint_direction, cam_joint_base, cam2robot)

        base_grasp_pose[:3, 3] += (grasp_depth - 0.05) * base_grasp_pose[:3, 0] # TODO: hardcode to avoid collision
        print("base_grasp_pose",base_grasp_pose)
        if joint_type == 0:
            # TODO: only for horizontal grasp to avoid singular robot state
            flip = np.arccos(np.dot(base_grasp_pose[:3, 2], np.array([0., 0., 1.]))) / np.pi * 180.0 < 45
            if flip:
                print("flipped")
                base_grasp_pose[:3, 1] = -base_grasp_pose[:3, 1]
                base_grasp_pose[:3, 2] = -base_grasp_pose[:3, 2]
            horizontal = np.arccos(np.dot(base_grasp_pose[:3, 0], np.array([1., 0., 0.]))) / np.pi * 180.0 < 45
            if horizontal:
                print("horizontal")
            else:
                print("vertical")
        else:
            raise ValueError
        base_pre_grasp_pose = base_grasp_pose.copy()
        base_pre_grasp_pose[:3, 3] -= 0.05 * base_pre_grasp_pose[:3, 0]
        g2g = np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
        base_gripper_pose = np.eye(4)
        base_gripper_pose[:3, :3] = base_grasp_pose[:3, :3] @ g2g
        base_gripper_pose[:3, 3] = base_grasp_pose[:3, 3]
        base_pre_gripper_pose = np.eye(4)
        base_pre_gripper_pose[:3, :3] = base_pre_grasp_pose[:3, :3] @ g2g
        base_pre_gripper_pose[:3, 3] = base_pre_grasp_pose[:3, 3]
        rotation_matrix_x = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
        rotation_matrix_y = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0]])

        rotation_matrix_z = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
        rotation_matrix_z_180 = np.array([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])
        base_gripper_pose[:3, :3] = base_gripper_pose[:3, :3] @ rotation_matrix_x
        base_gripper_pose[:3, :3] = base_gripper_pose[:3, :3] @ rotation_matrix_x
        # base_gripper_pose[:3, :3] = base_gripper_pose[:3, :3] @ rotation_matrix_y
        # base_gripper_pose[:3, :3] = base_gripper_pose[:3, :3] @ rotation_matrix_z
        base_gripper_pose[:3, :3] = base_gripper_pose[:3, :3] @ rotation_matrix_z_180


    global robot
    robot = Rizon4(headless=args.headless)  # Use GUI mode for visualization unless headless

    # Initialize planners
    rrt_planner = RRTPlanner(robot, step_size=args.step_size, max_iterations=args.max_iterations)

    # Initialize OMPL planner if available
    ompl_planner = None
    if pb_ompl is not None and args.planner_type == 'ompl':
        try:
            ompl_planner = OMPLPlanner(robot, planner_name=args.ompl_planner,
                                     planning_time=args.planning_time)
            print(f"OMPL planner initialized with {args.ompl_planner} algorithm")
        except Exception as e:
            print(f"Failed to initialize OMPL planner: {e}")
            print("Falling back to RRT planner")
            ompl_planner = None

    # Select active planner
    if ompl_planner is not None and args.planner_type == 'ompl':
        active_planner = ompl_planner
        active_planner_type = 'ompl'
        print(f"Using OMPL planner with {args.ompl_planner} algorithm")
    else:
        active_planner = rrt_planner
        active_planner_type = 'rrt'
        print(f"Using RRT planner (step_size={args.step_size}, max_iterations={args.max_iterations})")

    print("===> starting manipulation")
    base_gripper_pose[:3, 3] += np.array([-0.01, 0.03, -0.04]) # offset
    base_pre_gripper_pose =  base_gripper_pose.copy()
    base_pre_gripper_pose[:3, 3] += np.array([0, -0.04, 0])
    time.sleep(0.5)

    start_time = time.time()

    if num_grasps == 0:
        exit(1)
    else:

        print("done pre pose")
        print("init tcp mat", base_gripper_pose)

        # Convert poses to the format expected by calc_ik
        tcp_pose = xyz_rot_transform(base_gripper_pose, from_rep="matrix", to_rep="quaternion") # quat wxyz
        print("init tcp quat wxyz", tcp_pose)
        flange_pose = np.array([*tcp_pose[:3], tcp_pose[4], tcp_pose[5], tcp_pose[6], tcp_pose[3]]) # wxyz -> xyzw
        flange_pose[2] += 0.14
        print("init flange pose", flange_pose)

        # Enhanced visualization setup
        if args.show_grasp_points:
            print("Setting up enhanced visualization...")
            # Visualize grasp point with enhanced markers
            grasp_position = base_gripper_pose[:3, 3]
            p.addUserDebugText("GRASP", grasp_position, textColorRGB=(1, 1, 0), textSize=2.0)
            p.addUserDebugPoints([grasp_position], [[1, 1, 0]], pointSize=10)

            # Visualize grasp orientation
            grasp_orientation = base_gripper_pose[:3, :3]
            for i, color in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):  # X, Y, Z axes
                axis_end = grasp_position + grasp_orientation[:, i] * 0.1
                p.addUserDebugLine(grasp_position, axis_end, color, 2.0)

        # Plan and validate initial movement
        current_flange_pose = robot.get_flange_catersian()
        current_tcp_pose = np.array([current_flange_pose[0], current_flange_pose[1], current_flange_pose[2],
                                     current_flange_pose[6],
                                     current_flange_pose[3],
                                     current_flange_pose[4],
                                     current_flange_pose[5]]) # reorder, quat xyzw -> wxyz
        current_tcp_pose[2] -= 0.14
        current_tcp_pose = xyz_rot_transform(current_tcp_pose, from_rep="quaternion", to_rep="quaternion")

        # Convert target pose to the same format - tcp_pose is already in wxyz format from xyz_rot_transform
        target_tcp_pose = tcp_pose  # Already in wxyz format

        print("=== Planning initial movement ===")
        success, init_path, message = plan_and_validate_movement(
            robot, active_planner, current_tcp_pose, target_tcp_pose,
            use_planning=args.use_planning, planner_type=active_planner_type,
            visualize_path=args.visualize_path, args=args
        )

        if success:
            print(f"Initial movement planning: {message}")
            execute_planned_path(robot, init_path, step_duration=0.1)
        else:
            print(f"Initial movement planning failed: {message}")
            print("Falling back to direct IK...")
            next_joints = robot.calc_ik(flange_pose)  # pppqqqq -> q
            robot.set_joints(next_joints)
            robot.visualize()
            p.stepSimulation()

        time.sleep(1)

        print("===> Planning manipulation trajectory")

        # Store initial pose for trajectory planning
        initial_flange_pose = robot.get_flange_catersian()
        initial_tcp_pose = np.array([initial_flange_pose[0], initial_flange_pose[1], initial_flange_pose[2],
                                     initial_flange_pose[6],
                                     initial_flange_pose[3],
                                     initial_flange_pose[4],
                                     initial_flange_pose[5]]) # reorder, quat xyzw -> wxyz
        initial_tcp_pose[2] -= 0.14
        initial_tcp_pose = xyz_rot_transform(initial_tcp_pose, from_rep="quaternion", to_rep="matrix")

        for time_step in tqdm.trange(time_steps):
            current_flange_pose = robot.get_flange_catersian()
            current_tcp_pose = np.array([current_flange_pose[0], current_flange_pose[1], current_flange_pose[2],
                                         current_flange_pose[6],
                                         current_flange_pose[3],
                                         current_flange_pose[4],
                                         current_flange_pose[5]]) # reorder, quat xyzw -> wxyz
            current_tcp_pose[2] -= 0.14
            current_tcp_pose = xyz_rot_transform(current_tcp_pose, from_rep="quaternion", to_rep="matrix")

            if joint_type == 0:
                rotation_angle = 5 * 1 * 1 / 180.0 * np.pi
                delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=base_joint_direction, point=base_joint_base)
            elif joint_type == 1:
                translation_distance = -5.0 * task / 100.0
                delta_pose = tf.translation_matrix(base_joint_direction * translation_distance)
            else:
                raise ValueError

            target_EE2robot = delta_pose @ current_tcp_pose

            # Convert poses to the format expected by RRT planner: [x, y, z, qw, qx, qy, qz] (wxyz format)
            # xyz_rot_transform outputs wxyz format, which is what RRT planner expects
            current_pose_quat = xyz_rot_transform(current_tcp_pose, from_rep="matrix", to_rep="quaternion")
            target_pose_quat = xyz_rot_transform(target_EE2robot, from_rep="matrix", to_rep="quaternion")

            # Enhanced visualization for trajectory steps
            if args.visualize_path and time_step == 0:
                print(f"Visualizing trajectory for step {time_step}...")
                # Visualize current and target positions
                current_pos = current_pose_quat[:3]
                target_pos = target_pose_quat[:3]

                p.addUserDebugText("CURRENT", current_pos, textColorRGB=(0, 1, 0), textSize=1.5)
                p.addUserDebugText("TARGET", target_pos, textColorRGB=(1, 0, 0), textSize=1.5)
                p.addUserDebugPoints([current_pos], [[0, 1, 0]], pointSize=8)
                p.addUserDebugPoints([target_pos], [[1, 0, 0]], pointSize=8)
                p.addUserDebugLine(current_pos, target_pos, (1, 1, 0), 3.0)

            # Plan movement to target pose
            success, path, message = plan_and_validate_movement(
                robot, active_planner, current_pose_quat, target_pose_quat,
                use_planning=args.use_planning, planner_type=active_planner_type,
                visualize_path=args.visualize_path and time_step == 0, args=args
            )

            if success:
                print(f"Step {time_step}: {message}")
                execute_planned_path(robot, path, step_duration=0.05)
            else:
                print(f"Step {time_step}: Planning failed - {message}")
                print("Falling back to direct IK...")
                # Fallback to original direct IK method
                tcp_pose = xyz_rot_transform(target_EE2robot, from_rep="matrix", to_rep="quaternion")
                flange_pose = np.array([*tcp_pose[:3], tcp_pose[4], tcp_pose[5], tcp_pose[6], tcp_pose[3]])
                flange_pose[2] += 0.14
                next_joints = robot.calc_ik(flange_pose)
                robot.set_joints(next_joints)
                robot.visualize()
                p.stepSimulation()

            # Pause between steps if requested
            if args.pause_between_steps:
                input(f"Press Enter to continue to step {time_step + 1}...")

            time.sleep(0.1)

    end_time = time.time()
    print("===> manipulation done", end_time - start_time)

    # Final path analysis and visualization
    if args.use_planning:
        print("\n=== Path Planning Summary ===")
        if active_planner_type == 'ompl':
            print(f"OMPL Planning enabled: Algorithm={args.ompl_planner}, Time limit={args.planning_time}s")
        else:
            print(f"RRT Planning enabled: Step size={args.step_size}, Max iterations={args.max_iterations}")
        print("Enhanced visualization features:")
        if args.show_grasp_points:
            print("  - Grasp points visualization: Enabled")
        if args.show_waypoints:
            print("  - Trajectory waypoints: Enabled")
        if args.slow_motion:
            print("  - Slow motion execution: Enabled")
        if args.pause_between_steps:
            print("  - Step-by-step execution: Enabled")
        print("Check the visualization for path feasibility analysis")

    # Keep visualization open for inspection
    if not args.headless:
        print("\nVisualization will remain open. Press Ctrl+C to exit...")
        try:
            while True:
                p.stepSimulation()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nExiting visualization...")

if __name__ == '__main__':
    main()