import os
import time
import tqdm
import configargparse
import numpy as np
import transformations as tf
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
    parser.add_argument('--global_key', type=str, default="ours")
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

def project_point_to_plane(point, plane_point1, plane_point2, plane_point3):
    """
    将点投影到由三个点定义的平面上

    输入:
    - point: 要投影的点
    - plane_point1, plane_point2, plane_point3: 定义平面的三个点

    返回:
    - projected_point: 投影后的点
    """
    # 计算平面的法向量
    v1 = plane_point2 - plane_point1
    v2 = plane_point3 - plane_point1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # 计算点到平面的距离
    distance = np.dot(point - plane_point1, normal)

    # 计算投影点
    projected_point = point - distance * normal

    return projected_point

def find_closest_point_on_line_segment(point, line_start, line_end):
    """
    找到线段上距离给定点最近的点

    输入:
    - point: 给定点
    - line_start, line_end: 线段的两个端点

    返回:
    - closest_point: 线段上最近的点
    """
    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)
    line_dir = line_vec / line_length

    # 计算点到线段起点的向量
    point_vec = point - line_start

    # 计算投影长度
    projection_length = np.dot(point_vec, line_dir)

    # 将投影长度限制在线段范围内
    projection_length = max(0, min(projection_length, line_length))

    # 计算最近的点
    closest_point = line_start + projection_length * line_dir

    return closest_point

def clamp_points_to_pc_bounds(point1, point2, cam_pc):
    """
    将point1和point2限制在点云范围内，同时保持它们在point1-point2轴上

    输入:
    - point1: 第一个点 (x1, y1, z1)
    - point2: 第二个点 (x2, y2, z2)
    - cam_pc: 点云数据 (N, 3)

    返回:
    - clamped_point1: 限制后的point1
    - clamped_point2: 限制后的point2
    """
    # 计算点云的边界框
    pc_min = np.min(cam_pc, axis=0)
    pc_max = np.max(cam_pc, axis=0)

    # 计算轴的方向向量
    axis_direction = point2 - point1
    axis_length = np.linalg.norm(axis_direction)
    axis_direction = axis_direction / axis_length

    # 计算轴与点云边界框的交点
    def ray_aabb_intersection(origin, direction, bbox_min, bbox_max):
        """计算射线与轴对齐边界框的交点参数t的范围"""
        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            if abs(direction[i]) < 1e-6:  # 平行于该轴
                if origin[i] < bbox_min[i] or origin[i] > bbox_max[i]:
                    return None, None  # 在边界外
            else:
                t1 = (bbox_min[i] - origin[i]) / direction[i]
                t2 = (bbox_max[i] - origin[i]) / direction[i]

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    return None, None  # 无交点

        return t_min, t_max

    # 计算轴与点云边界框的交点
    t_min, t_max = ray_aabb_intersection(point1, axis_direction, pc_min, pc_max)

    if t_min is None or t_max is None:
        # 如果轴与边界框无交点，返回原始点
        return point1, point2

    # 将t限制在有效范围内
    t_min = max(t_min, -0.1 * axis_length)  # 允许稍微超出边界框
    t_max = min(t_max, 1.1 * axis_length)

    # 计算限制后的点
    clamped_point1 = point1 + t_min * axis_direction
    clamped_point2 = point1 + t_max * axis_direction

    return clamped_point1, clamped_point2
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
    service = np.load(temp_service_path, allow_pickle=True)
    
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
    gt_keypoints = json.load(open(f"{args.root_dir}/gt_keypoints.json", "r"))
    gt_pred_selected_affordable_position = np.array(gt_keypoints["kp_contact"])
    gt_point1 = np.array(gt_keypoints["kp_axis1"])
    gt_point2 = np.array(gt_keypoints["kp_axis2"])
    # pred_keypoints = pickle.load(open(f"{args.root_dir}/pred_keypoints.pkl", "rb"))
    pred_keypoints = pickle.load(open(f"{args.root_dir}/pred_keypoints_new.pkl", "rb"))
    pred_joint_directions = pred_keypoints["pred_joint_directions"]
    point1 = pred_anchor_points = pred_keypoints["pred_anchor_points"]
    pred_part_points = pred_keypoints["pred_part_points"]
    point2 = pred_anchor_points + pred_joint_directions
    cam_joint_base = service['joint_base']
    cam_joint_direction = service['joint_direction']
    # point1 = cam_joint_base
    # point2 = cam_joint_base + cam_joint_direction

    points = np.array(pcd.points)
    radius = points[:, 0].max() - points[:, 0].min()
    radius = 0.03 * radius

    # 将point1和point2限制在点云范围内
    clamped_point1, clamped_point2 = clamp_points_to_pc_bounds(point1, point2, cam_pc)

    # 调整pred_part_points位置：投影到由三个点组成的平面上，并找到距离gt_pred_selected_affordable_position最近的点
    # 首先将gt_pred_selected_affordable_position投影到由clamped_point1, clamped_point2, pred_part_points定义的平面上
    projected_gt_point = project_point_to_plane(gt_pred_selected_affordable_position, clamped_point1, clamped_point2, pred_part_points)

    # 找到投影后的点在clamped_point1-clamped_point2线段上最近的点
    # 这里我们希望在平面上找到一个合适的位置，既保持平面关系，又接近目标点

    # 计算平面上的坐标系
    v1 = clamped_point2 - clamped_point1
    v2 = pred_part_points - clamped_point1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # 将目标点投影到平面上
    target_on_plane = project_point_to_plane(gt_pred_selected_affordable_position, clamped_point1, clamped_point2, pred_part_points)

    # 计算调整后的pred_part_points位置（在平面上，且距离目标点最近）
    adjusted_pred_part_points = target_on_plane

    # 打印调试信息
    print(f"Original point1: {point1}, clamped point1: {clamped_point1}")
    print(f"Original point2: {point2}, clamped point2: {clamped_point2}")
    print(f"Original pred_part_points: {pred_part_points}, adjusted pred_part_points: {adjusted_pred_part_points}")
    print(f"GT affordable position: {gt_pred_selected_affordable_position}")
    print(f"Point cloud bounds - min: {np.min(cam_pc, axis=0)}, max: {np.max(cam_pc, axis=0)}")

    # 绘制点
    mesh_sphere = draw_point(clamped_point1, [1, 0, 0], radius=radius)
    vis.add_geometry(mesh_sphere)
    mesh_sphere = draw_point(clamped_point2, [1, 0, 0], radius=radius)
    vis.add_geometry(mesh_sphere)
    mesh_sphere = draw_point(adjusted_pred_part_points, [0, 1, 0], radius=radius)
    vis.add_geometry(mesh_sphere)

    # # 绘制GT点用于参考
    # mesh_sphere = draw_point(gt_pred_selected_affordable_position, [0, 0, 1], radius=radius)
    # vis.add_geometry(mesh_sphere)

    # 绘制clamped_point1和clamped_point2之间的红色连线
    line_points = [clamped_point1, clamped_point2]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(line_points)
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.paint_uniform_color([1, 0, 0])  # 红色
    vis.add_geometry(line)

    # # 绘制三角形平面（可选，用于可视化平面关系）
    # all_keypoints = np.array([
    #     clamped_point1, clamped_point2, adjusted_pred_part_points
    # ])
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(all_keypoints)
    # mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    # mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色，半透明效果
    # vis.add_geometry(mesh)

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

    for time_step in tqdm.trange(10):
        rotation_angle =  5 * 1 * 1 / 180.0 * np.pi # +5 or -5
    
        delta_pose = tf.rotation_matrix(angle=rotation_angle, 
                                        direction=cam_joint_direction, 
                                        point=cam_joint_base
                                        )
        cam_grasp_pose = delta_pose @ cam_grasp_pose
        g_grasp.translation = cam_grasp_pose[:3, 3]
        g_grasp.rotation_matrix = cam_grasp_pose[:3, :3]
        gg_grasp.add(g_grasp)
    for idx, g_grasp in enumerate(gg_grasp):
        grippers_o3d = g_grasp.to_open3d_geometry(color=color_map[idx])
        vis.add_geometry(grippers_o3d)
    frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(frame_o3d)

    vis.run()
    vis.destroy_window()

    # assert 0

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
        base_joint_base = transform_pc(cam_joint_base[None, :], cam2robot)[0]
        base_joint_direction = transform_dir(cam_joint_direction[None, :], cam2robot)[0]
        # base_affordable_position = transform_pc(cam_affordable_position[None, :], cam2robot)[0]
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
    
