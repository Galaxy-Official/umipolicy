import os
import time
import tqdm
import configargparse
import numpy as np
import transformations as tf
from camera import CameraD400
from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize
import fpsample
import numpy as np
import threading
import cv2
import pickle
import pybullet as p
from utilities.transformation import rotation_transform, xyz_rot_transform
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
    parser.add_argument('--planner_type', type=str, default='ompl', choices=['ompl', 'rrt'],
                       help='Planner type to use (ompl or rrt)')
    parser.add_argument('--ompl_planner', type=str, default='RRTConnect',
                       help='OMPL planner algorithm (RRTConnect, RRTstar, BITstar, etc.)')
    parser.add_argument('--planning_time', type=float, default=5.0, help='OMPL planning time limit')
    parser.add_argument('--step_size', type=float, default=0.05, help='RRT step size')
    parser.add_argument('--max_iterations', type=int, default=500, help='RRT max iterations')
    parser.add_argument('--visualize_path', action='store_true', help='Visualize planned path')

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
    """Plan and validate a movement using path planning

    Args:
        robot: Robot instance
        planner: Planner instance (OMPL or RRT)
        start_pose: Starting pose (position + orientation)
        end_pose: Target pose (position + orientation)
        use_planning: Whether to use path planning (if False, uses direct IK)
        planner_type: Type of planner to use ('ompl' or 'rrt')
        visualize_path: Whether to visualize the planned path
        args: Command line arguments

    Returns:
        Tuple[bool, Optional[List[np.ndarray]], str]: (success, path, message)
    """
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
    """Execute a planned path on the robot

    Args:
        robot: Robot instance
        path: List of joint configurations
        step_duration: Time to wait between steps
    """
    print(f"Executing planned path with {len(path)} waypoints...")

    for i, joint_states in enumerate(path):
        robot.set_joints(joint_states)
        robot.visualize()
        p.stepSimulation()

        if step_duration > 0:
            time.sleep(step_duration)

    print("Path execution complete")


def record(args, robot, camera):

    if not os.path.exists(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}"):
        os.makedirs(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}")
    else:
        #print("NOT RECORDED")
        #return
        os.system(f"rm -r {args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}")

    for save_key in ["tcp", "joint_position", "gripper_width", "cam_top_rgb", "cam_top_depth"]:
        if not os.path.exists(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/{save_key}"):
            os.makedirs(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/{save_key}")

    timestamp = 0
    while True:

        ## robot state
        tcp = robot.get_tcp()
        gripper_width = robot.get_gripper_width()
        joint_position = robot.get_joint_positions()
        print("tcp", tcp, "joint_position", joint_position, "gripper_width", gripper_width)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/tcp/{str(timestamp).zfill(10)}.npy", tcp)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/joint_position/{str(timestamp).zfill(10)}.npy", joint_position)
        np.save(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/gripper_width/{str(timestamp).zfill(10)}.npy", gripper_width)

        ## camera
        color, depth = camera.get_data(hole_filling=False)
        cv2.imwrite(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/cam_top_rgb/{str(timestamp).zfill(10)}.png", color)
        pickle.dump(depth, open(f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}/cam_top_depth/{str(timestamp).zfill(10)}.pkl", "wb"))

        time.sleep(1/30) # 30 Hz
        timestamp += 1


if __name__ == '__main__':

    args = config_parse()

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

    # cam2robot = robot.readCamPose()
    # print("pose",cam2robot)
    cam2robot = np.array([[ 0.05262318, -0.99199495 , 0.11479036 , 0.56598359],
                        [-0.99846966, -0.05030948 , 0.02296276 ,-0.01270892],
                        [-0.0170039 , -0.11582306 ,-0.99312431,  0.65807974],
                        [ 0.   ,       0.    ,      0.        ,  1.        ]])
    # cam2robot = np.load(f"{args.root_dir}/{args.global_key}/init_capture/cam2robot.npz")
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
        # base_grasp_pose = cam2robot @ cam_grasp_pose
        base_joint_direction, base_joint_base = transform_vector_and_point(cam_joint_direction, cam_joint_base, cam2robot)
        # print("########",base_joint_base)
        # print("$$$$$$$$",base_joint_direction)

        base_grasp_pose[:3, 3] += (grasp_depth - 0.05) * base_grasp_pose[:3, 0] # TODO: hardcode to avoid collision
        print("base_grasp_pose",base_grasp_pose)
        if joint_type == 0:
            # TODO: only for horizontal grasp to avoid singular robot state
            flip = np.arccos(np.dot(base_grasp_pose[:3, 2], np.array([0., 0., 1.]))) / np.pi * 180.0 < 45
            if flip:
                print("flipped")
                base_grasp_pose[:3, 1] = -base_grasp_pose[:3, 1]
                base_grasp_pose[:3, 2] = -base_grasp_pose[:3, 2]
            rotate = base_grasp_pose[:3, 0][2] > 0
            if rotate:
                print("rotated")
                target_x_axis = base_grasp_pose[:3, 0].copy()
                target_x_axis[2] = -target_x_axis[2]
                rotation_angle = np.arccos(np.dot(base_grasp_pose[:3, 0], target_x_axis))
                rotation_direction = np.array([base_grasp_pose[:3, 0][0], base_grasp_pose[:3, 0][1]])
                rotation_direction /= np.linalg.norm(rotation_direction)
                rotation_direction = np.array([-rotation_direction[1], rotation_direction[0], 0.])
                rotation_matrix = tf.rotation_matrix(angle=rotation_angle, direction=rotation_direction, point=base_grasp_pose[:3, 3])
                base_grasp_pose = rotation_matrix @ base_grasp_pose
        elif joint_type == 1:
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
    robot = Rizon4(headless=False)

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
    # gripper_pose_mat_list = []
    # gripper_pose_mat_list.append(base_gripper_pose)

    if num_grasps == 0:
        exit(1)
    else:

        print("done pre pose")
        # robot.movePosePrimitive(base_gripper_pose)
        print("init tcp mat", base_gripper_pose)

        # Convert poses to the format expected by calc_ik
        tcp_pose = xyz_rot_transform(base_gripper_pose, from_rep="matrix", to_rep="quaternion") # quat wxyz
        print("init tcp quat wxyz", tcp_pose)
        flange_pose = np.array([*tcp_pose[:3], tcp_pose[4], tcp_pose[5], tcp_pose[6], tcp_pose[3]]) # wxyz -> xyzw
        flange_pose[2] += 0.14
        print("init flange pose", flange_pose)

        # Plan and validate initial movement
        current_flange_pose = robot.get_flange_catersian()
        current_tcp_pose = np.array([current_flange_pose[0], current_flange_pose[1], current_flange_pose[2],
                                     current_flange_pose[6],
                                     current_flange_pose[3],
                                     current_flange_pose[4],
                                     current_flange_pose[5]]) # reorder, quat xyzw -> wxyz
        current_tcp_pose[2] -= 0.14
        # Convert to the format expected by RRT planner: [x, y, z, qw, qx, qy, qz] (wxyz format)
        # xyz_rot_transform outputs wxyz format, which is what RRT planner expects
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

        # print("base_gripper_pose", base_gripper_pose)
        # print("tcp_pose", tcp_pose)
        # robot.send_tcp_pose(tcp_pose)
        # time.sleep(1)
        # robot.send_gripper_state(0.005 ,0.1, 20)
        # time.sleep(1)

        # # move
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
            print("current_flange_pose", current_flange_pose)
            current_tcp_pose = np.array([current_flange_pose[0], current_flange_pose[1], current_flange_pose[2], 
                                         current_flange_pose[6], 
                                         current_flange_pose[3], 
                                         current_flange_pose[4], 
                                         current_flange_pose[5]]) # reorder, quat xyzw -> wxyz
            current_tcp_pose[2] -= 0.14
            # print("current_tcp_pose wxyz", current_tcp_pose)
            current_tcp_pose = xyz_rot_transform(current_tcp_pose, from_rep="quaternion", to_rep="matrix")
            # print("current_tcp_pose", current_tcp_pose)
            if joint_type == 0:
                rotation_angle = 5 * 1 * 1 / 180.0 * np.pi
                delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=base_joint_direction, point=base_joint_base)
            elif joint_type == 1:
                translation_distance = -5.0 * task / 100.0
                delta_pose = tf.translation_matrix(base_joint_direction * translation_distance)
            else:
                raise ValueError
            target_EE2robot = delta_pose @ current_tcp_pose
            # next_joints = send_tcp_mat(target_EE2robot, is_tcp=True)
            # print("next tcp mat", target_EE2robot)
            tcp_pose = xyz_rot_transform(target_EE2robot, from_rep="matrix", to_rep="quaternion") # quat wxyz
            # print("next tcp quat wxyz", tcp_pose)
            flange_pose = np.array([*tcp_pose[:3], tcp_pose[4], tcp_pose[5], tcp_pose[6], tcp_pose[3]]) # wxyz -> xyzw
            flange_pose[2] += 0.14
            # print("next flange pose", flange_pose)
            next_joints = robot.calc_ik(flange_pose)  # pppqqqq -> q
            robot.set_joints(next_joints)
            robot.visualize()
            p.stepSimulation()

            time.sleep(0.1)

    end_time = time.time()
    # robot.send_gripper_state(0.5 ,0.1, 20)
    print("===> manipulation done", end_time - start_time)

    ### planning with OMPL
    final_flange_pose = flange_pose

    # # Final path analysis
    # if args.use_planning:
    #     print("\n=== Path Planning Summary ===")
    #     if active_planner_type == 'ompl':
    #         print(f"OMPL Planning enabled: Algorithm={args.ompl_planner}, Time limit={args.planning_time}s")
    #     else:
    #         print(f"RRT Planning enabled: Step size={args.step_size}, Max iterations={args.max_iterations}")
    #     print("Check the visualization for path feasibility analysis")