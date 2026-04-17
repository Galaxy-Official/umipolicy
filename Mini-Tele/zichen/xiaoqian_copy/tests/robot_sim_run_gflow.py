import os
import time
import tqdm
import configargparse
import numpy as np
# import transformations as tf  # 注释掉，因为已经使用numpy实现
# from camera import CameraD400
# from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize
import fpsample
import numpy as np
import threading
import cv2
import pickle
import pybullet as p
from utilities.transformation import rotation_transform, xyz_rot_transform
from flexiv import *
import open3d as o3d
from geometry_utils import *
import json


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="pred_keypoints/capture_0301/box/view_000_pose_000")
    parser.add_argument('--global_key', type=str, default="gflow")
    # record config
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--selected_grasp_idx', type=int, default=0)

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


if __name__ == '__main__':

    args = config_parse()

    camera_loaded = False
    robot_loaded = False
    task = -1     #closing as 1, open as -1
    joint_type = 1 # rotate as 0
    joint_re = 45
    time_steps = 10
    vis = True

    temp_service_path = f"{args.root_dir}/{args.global_key}/init_grasp/{str(args.selected_grasp_idx).zfill(3)}.npz"
    
    print("#####vis grasp#####")
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ## vis point cloud
    point_cloud = pickle.load(open(f"{args.root_dir}/point_cloud.pkl", "rb"))
    cam_pc = point_cloud["points"]
    pc_rgb = point_cloud["colors"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    vis.add_geometry(pcd)

    ## vis grasp
    gg_grasp = pickle.load(open(f"{args.root_dir}/{args.global_key}/init_grasp_orig/{str(args.selected_grasp_idx).zfill(3)}.pkl", "rb"))
    # grippers_o3d = gg_grasp.to_open3d_geometry_list()
    # frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # for gripper in grippers_o3d:
    #     vis.add_geometry(gripper)

    ## vis axis
    # gt_keypoints = json.load(open(f"{args.root_dir}/gt_keypoints.json", "r"))
    # pred_selected_affordable_position = np.array(gt_keypoints["kp_contact"])
    # point1 = np.array(gt_keypoints["kp_axis1"])
    # point2 = np.array(gt_keypoints["kp_axis2"])

    # points = np.array(pcd.points)
    # radius = points[0].max() - points[0].min()
    # radius = 0.01 * radius

    # mesh_sphere = draw_point(pred_selected_affordable_position, [1, 0, 0], radius=radius)
    # vis.add_geometry(mesh_sphere)
    # mesh_sphere = draw_point(point1, [0, 1, 0], radius=radius)
    # vis.add_geometry(mesh_sphere)
    # mesh_sphere = draw_point(point2, [0, 1, 1], radius=radius)
    # vis.add_geometry(mesh_sphere)

    # Load pred_gflow.pkl and compute rotation axis from trajectories
    pred_results = pickle.load(open(f"{args.root_dir}/pred_gflow.pkl", "rb"))

    # Extract gripper position as part point
    pred_part_points = pred_selected_affordable_position = pred_results["gripper_3d_pos"]

    ## vis transformed grasp
    color_map = [[0.0, 0.0, 1.0], [0.098, 0.0, 1.0], [0.2, 0.0, 1.0], [0.298, 0.0, 1.0], [0.4, 0.0, 1.0], [0.498, 0.0, 1.0], [0.6, 0.0, 1.0], [0.698, 0.0, 1.0], [0.8, 0.0, 1.0], [0.898, 0.0, 1.0], [1.0, 0.0, 1.0]]

    g_grasp = gg_grasp[0]
    cam_grasp_translation = g_grasp.translation
    cam_grasp_rotation = g_grasp.rotation_matrix
    cam_grasp_pose = np.eye(4)
    cam_grasp_pose[:3, 3] = cam_grasp_translation
    cam_grasp_pose[:3, :3] = cam_grasp_rotation
    service = np.load(temp_service_path, allow_pickle=True)
    cam_joint_direction = service["joint_direction"]
    cam_joint_base = service["joint_base"]

    # print(pred_results["motion_plan"])
    transforms = []
    for (motion_R, motion_t, _) in pred_results["motion_plan"]:
        transform = np.eye(4)
        transform[:3, :3] = motion_R
        transform[:3, 3] = motion_t.flatten()
        transforms.append(transform)

    # 保持原始transforms，不进行插值
    print(f"Using original transforms: {len(transforms)} transforms")

    for time_step in tqdm.trange(min(10, len(transforms))):  # 使用插值后的transforms
        delta_pose = transforms[time_step]
        cam_grasp_pose = delta_pose @ cam_grasp_pose
        g_grasp.translation = cam_grasp_pose[:3, 3]
        g_grasp.rotation_matrix = cam_grasp_pose[:3, :3]
        gg_grasp.add(g_grasp)
    # grippers_o3d = gg_grasp.to_open3d_geometry_list()
    # for gripper in grippers_o3d:
    #     vis.add_geometry(gripper)
    # grippers_o3d = g_grasp.to_open3d_geometry()
    # vis.add_geometry(grippers_o3d)
    for idx, g_grasp in enumerate(gg_grasp):
        grippers_o3d = g_grasp.to_open3d_geometry(color=color_map[idx*2+1])
        vis.add_geometry(grippers_o3d)
    frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(frame_o3d)

    # 添加trajectory_plan可视化（参考general_flow/vis_exec.py）
    if 'traj_prediction' in pred_results and pred_results['traj_prediction'] is not None:
        print("Adding trajectory_plan visualization...")
        traj_prediction_vis = pred_results['traj_prediction'][0]  # 取第一个推理结果
        nQ = traj_prediction_vis.shape[0]
        max_traj_plan = min(30, nQ)  # 限制显示的轨迹数量，避免过于拥挤

        # 轨迹可视化 - 使用统一的绿色
        color_traj_plan = [0.0, 1.0, 0.0]  # 绿色

        for idx_t, pred_vec in enumerate(traj_prediction_vis[:max_traj_plan]):
            n_points = pred_vec.shape[0]

            for ii in range(n_points - 1):
                point2, point1 = pred_vec[ii + 1], pred_vec[ii]

                # 创建简单的线条表示轨迹
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([point1, point2])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([color_traj_plan])
                vis.add_geometry(line)

        print(f"Visualized {max_traj_plan} trajectories from trajectory_plan with uniform green color")

    vis.run()
    vis.destroy_window()

    assert 0

    HOME_POSE = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0.51175809],#0.51175809
                             [-0.05031023 , 0.99846963 , 0.02296254 ,-0.01212505],#-0.01212505
                             [-0.11582299,  0.0170036 , -0.99312432  ,0.45597902],#0.45597902
                             [ 0.      ,    0.      ,    0.  ,        1.        ]])
    
    

    print("===> reading response")
    start_time = time.time()
    
    # cam2robot = robot.readCamPose()
    
    robot_pose = json.load(open(f"{args.root_dir}/robot_pose.json", "r"))
    robot_pose = np.array(robot_pose)
    # cam2robot = np.array([[ 0.05262318, -0.99199495 , 0.11479036 , 0.56598359],
    #                     [-0.99846966, -0.05030948 , 0.02296276 ,-0.01270892],
    #                     [-0.0170039 , -0.11582306 ,-0.99312431,  0.65807974],
    #                     [ 0.   ,       0.    ,      0.        ,  1.        ]])
    robot_pose_mapto_cam2robot = json.load(open(f"pred_keypoints/robot_pose_mapto_cam2robot.json", "r"))
    for (robot_pose_cand, cam2robot_cand) in robot_pose_mapto_cam2robot:
        robot_pose_cand = np.array(robot_pose_cand)
        cam2robot_cand = np.array(cam2robot_cand)
        if np.allclose(robot_pose, robot_pose_cand):
            cam2robot = cam2robot_cand
            break
    print("cam2robot",cam2robot)
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
        # cam_affordable_position = service['affordable_position']
        joint_type = service['joint_type']
        joint_re = service['joint_re']
        grasp_score = service['grasp_score']
        grasp_width = service['grasp_width']
        grasp_depth = service['grasp_depth']
        # grasp_affordance = service['grasp_affordance']
        cam_grasp_translation = service['grasp_translation']
        cam_grasp_rotation = service['grasp_rotation']
        cam_grasp_pose = np.eye(4)
        cam_grasp_pose[:3, 3] = cam_grasp_translation
        cam_grasp_pose[:3, :3] = cam_grasp_rotation
        # base_joint_base = transform_pc(cam_joint_base[None, :], cam2robot)[0]
        # base_joint_direction = transform_dir(cam_joint_direction[None, :], cam2robot)[0]
        # base_affordable_position = transform_pc(cam_affordable_position[None, :], cam2robot)[0]
        base_grasp_pose = cam2robot @ cam_grasp_pose
        # base_grasp_pose = cam2robot @ cam_grasp_pose
        # base_joint_direction, base_joint_base = transform_vector_and_point(cam_joint_direction, cam_joint_base, cam2robot)
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
    

    print("===> starting manipulation")
    base_gripper_pose[:3, 3] += np.array([-0.01, 0.03, -0.04]) # offset
    base_pre_gripper_pose =  base_gripper_pose.copy()
    base_pre_gripper_pose[:3, 3] += np.array([0, -0.04, 0])
    # base_gripper_pose = np.array([[-0.99199492 ,-0.0526239  , 0.11479028 , 0],
    #                         [-0.05031023 , 0.99846963 , 0.02296254 , 0],
    #                         [-0.11582299,  0.0170036 , -0.99312432  , 0],
    #                         [ 0.      ,    0.      ,    0.  ,        1.        ]])
    # base_gripper_pose[:3, 3] += base_affordable_position # offset
    # base_gripper_pose[:3, 3] += np.array([0.065, 0.07, -0.02])
    time.sleep(0.5)
    
    
    start_time = time.time()
    # gripper_pose_mat_list = []
    # gripper_pose_mat_list.append(base_gripper_pose)

    if num_grasps == 0:
        exit(1)
    else:

        print("done pre pose")
        # robot.movePosePrimitive(base_gripper_pose)
        # print("init tcp mat", base_gripper_pose)
        # next_joints = send_tcp_mat(base_gripper_pose)
        tcp_pose = xyz_rot_transform(base_gripper_pose, from_rep="matrix", to_rep="quaternion") # quat wxyz
        # print("init tcp quat wxyz", tcp_pose)
        flange_pose = np.array([*tcp_pose[:3], tcp_pose[4], tcp_pose[5], tcp_pose[6], tcp_pose[3]]) # wxyz -> xyzw
        flange_pose[2] += 0.14
        print("init flange pose", flange_pose)
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

        flange_pose_waypoints = [flange_pose]
        # # move
        for time_step in tqdm.trange(10):
            current_flange_pose = robot.get_flange_catersian()
            # print("current_flange_pose", current_flange_pose)
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
                # 使用numpy实现旋转矩阵（替换tf.rotation_matrix）
                from scipy.spatial.transform import Rotation as R
                rot = R.from_rotvec(rotation_angle * np.array(base_joint_direction))
                delta_pose = np.eye(4)
                delta_pose[:3, :3] = rot.as_matrix()
                delta_pose[:3, 3] = np.array(base_joint_base) - rot.apply(np.array(base_joint_base))
            elif joint_type == 1:
                translation_distance = -5.0 * task / 100.0
                # 使用numpy实现平移矩阵（替换tf.translation_matrix）
                delta_pose = np.eye(4)
                delta_pose[:3, 3] = base_joint_direction * translation_distance
            else:
                raise ValueError
            target_EE2robot = delta_pose @ current_tcp_pose
            # next_joints = send_tcp_mat(target_EE2robot, is_tcp=True)
            # print("next tcp mat", target_EE2robot)
            tcp_pose = xyz_rot_transform(target_EE2robot, from_rep="matrix", to_rep="quaternion") # quat wxyz
            # print("next tcp quat wxyz", tcp_pose)
            flange_pose = np.array([*tcp_pose[:3], tcp_pose[4], tcp_pose[5], tcp_pose[6], tcp_pose[3]]) # wxyz -> xyzw
            flange_pose[2] += 0.14
            print("next flange pose", flange_pose)
            flange_pose_waypoints.append(flange_pose)
            next_joints = robot.calc_ik(flange_pose)  # pppqqqq -> q
            robot.set_joints(next_joints)
            robot.visualize()
            p.stepSimulation()
            time.sleep(1)
            # break
    end_time = time.time()
    # robot.send_gripper_state(0.5 ,0.1, 20)
    print("===> manipulation done", end_time - start_time)

    flange_pose_waypoints = np.array(flange_pose_waypoints)
    os.makedirs(f"{args.root_dir}/oracle_oracle/flange_pose_waypoints", exist_ok=True)
    np.save(f"{args.root_dir}/oracle_oracle/flange_pose_waypoints/{str(args.selected_grasp_idx).zfill(3)}.npy", flange_pose_waypoints)
    
