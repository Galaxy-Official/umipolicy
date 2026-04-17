import os
import configargparse
import time
import numpy as np
import pickle
import open3d as o3d
import json

import sys
sys.path.append('/Disk3/xiaoqian')

from utilities.env_utils import setup_seed
from utilities.data_utils import transform_pc
from utilities.constants import max_grasp_width
from geometry_utils import *
from tests.gflow_axis_util import (
    compute_rotation_axis_from_trajectories_corrected,
    visualize_trajectory_plane_normals,
    visualize_trajectory_axis_computation_detailed
)
from utilities.metrics_utils import invaffordance_metrics, invaffordances2affordance


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # data config
    parser.add_argument('--cat', type=str, default='Microwave', help='the category of the object')
    # grasp config
    # parser.add_argument('--graspnet', action='store_true', help='whether call graspnet')
    parser.add_argument('--gsnet_weight_path', type=str, default='anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar', help='the path to graspnet weight')
    parser.add_argument('--max_grasp_width', type=float, default=max_grasp_width, help='the max width of the gripper')
    # task config
    parser.add_argument('--selected_part', type=int, default=0, help='the selected part of the object')
    # others
    parser.add_argument('--seed', type=int, default=42, help='the random seed')
    # path config
    parser.add_argument('--root_dir', type=str, default="pred_keypoints/capture_0301/box/view_000_pose_000")
    parser.add_argument('--global_key', type=str, default="gflow")
    parser.add_argument('--selected_grasp_idx', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = config_parse()
    print("please clear temporary data directory")

    def calculate_rotation_axis(point1, point2):
        """
        计算旋转轴的方向和旋转中心
        输入:
        - point1: 第一个点 (x1, y1, z1)
        - point2: 第二个点 (x2, y2, z2)

        返回:
        - base_joint_direction: 旋转轴的方向 (单位向量)
        - base_joint_base: 旋转中心 (point1)
        """
        # 计算旋转轴的方向
        direction = point2 - point1
        
        # 计算单位化方向向量 (旋转轴方向)
        base_joint_direction = direction / np.linalg.norm(direction)
        
        # 旋转中心就是 point1
        base_joint_base = point1
        
        return base_joint_direction, base_joint_base

    point_cloud = pickle.load(open(f"{args.root_dir}/point_cloud.pkl", "rb"))
    cam_pc = point_cloud["points"]
    pc_rgb = point_cloud["colors"]
    print("cam_pc", cam_pc.shape)
    print("pc_rgb", pc_rgb.shape)
    
    # Load pred_gflow.pkl and compute rotation axis from trajectories
    pred_results = pickle.load(open(f"{args.root_dir}/pred_gflow.pkl", "rb"))

    # Extract gripper position as part point
    pred_part_points = pred_selected_affordable_position = pred_results["gripper_3d_pos"]

    """# Extract trajectories and compute rotation axis from them
    traj_prediction = pred_results["traj_prediction"][0]  # (Q, T, 3) - take first inference

    # 使用新的模块计算旋转轴
    axis_result = compute_rotation_axis_from_trajectories(traj_prediction)

    if axis_result['success']:
        pred_joint_directions = axis_result['pred_joint_directions']
        pred_anchor_points = axis_result['pred_anchor_points']
        action_data = axis_result['action_data']
        all_intersection_data = axis_result['all_intersection_data']
        all_actions_data = axis_result['all_actions_data']
        axis_candidates = axis_result['axis_candidates']
    else:
        # 回退到默认值
        pred_joint_directions = axis_result['pred_joint_directions']
        pred_anchor_points = axis_result['pred_anchor_points']
        action_data = axis_result['action_data']
        all_intersection_data = axis_result['all_intersection_data']
        all_actions_data = axis_result['all_actions_data']
        axis_candidates = axis_result['axis_candidates']

    # Set point1 and point2 for compatibility with existing code
    point1 = pred_anchor_points
    point2 = pred_anchor_points + pred_joint_directions

    pred_selected_joint_direction, pred_selected_joint_base = calculate_rotation_axis(point1, point2)"""


    """gt_keypoints = json.load(open(f"{args.root_dir}/gt_keypoints.json", "r"))
    pred_selected_affordable_position = np.array(gt_keypoints["kp_contact"])
    point1 = np.array(gt_keypoints["kp_axis1"])
    point2 = np.array(gt_keypoints["kp_axis2"])


    def generate_trajectories_from_axis(point1, point2, contact_point, num_trajectories=128, radius_range=0.05):
        '''
        # 根据旋转轴和接触点生成三段式轨迹，每段旋转15°
        # 正确的逻辑：每个点向轴做垂线，然后该垂线围绕轴做screw运动
        '''
        axis_direction = point2 - point1
        axis_direction = axis_direction / np.linalg.norm(axis_direction)

        # 三段式轨迹参数
        num_segments = 3
        points_per_segment = 5  # 每段的点数
        total_timesteps = 1 + num_segments * (points_per_segment - 1)  # 起始点 + 3段×4新点 = 13个点

        trajectories = np.zeros((num_trajectories, total_timesteps, 3))

        for i in range(num_trajectories):
            # 在接触点附近随机采样起点
            random_offset = np.random.randn(3) * radius_range * 0.3
            start_point = contact_point + random_offset

            # 计算起点到旋转轴的垂足（最近点）
            point_to_axis_vec = start_point - point1
            projection_length = np.dot(point_to_axis_vec, axis_direction)
            closest_point_on_axis = point1 + projection_length * axis_direction

            # 垂线向量（从轴上最近点到起始点）
            perpendicular_vec = start_point - closest_point_on_axis
            perpendicular_length = np.linalg.norm(perpendicular_vec)

            if perpendicular_length < 1e-6:
                perpendicular_length = radius_range
                # 创建垂直于轴的随机方向作为垂线
                random_vec = np.random.randn(3)
                perpendicular_vec = random_vec - np.dot(random_vec, axis_direction) * axis_direction
                perpendicular_vec = perpendicular_vec / np.linalg.norm(perpendicular_vec) * perpendicular_length
            else:
                perpendicular_vec = perpendicular_vec / perpendicular_length * radius_range
                perpendicular_length = radius_range

            # 设置起始点（第0个点）
            trajectories[i, 0] = start_point

            # 生成三段式轨迹，每段15°旋转
            current_point = start_point
            point_counter = 1

            for segment in range(num_segments):
                # 每段的总旋转角度为15°
                segment_start_angle = segment * np.pi / 12  # 15° * segment
                segment_end_angle = (segment + 1) * np.pi / 12  # 15° * (segment + 1)

                # 在段内生成多个点
                for point_idx in range(1, points_per_segment):  # 从1开始，因为0是起始点
                    # 在段内的角度插值
                    t_segment = point_idx / (points_per_segment - 1)
                    current_angle = segment_start_angle + t_segment * (segment_end_angle - segment_start_angle)

                    # 使用Rodrigues旋转公式旋转垂线向量
                    # 旋转轴：axis_direction，旋转角度：current_angle
                    # 向量：perpendicular_vec

                    # Rodrigues旋转公式: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
                    # 其中k是旋转轴，v是要旋转的向量，θ是旋转角度

                    cos_angle = np.cos(current_angle)
                    sin_angle = np.sin(current_angle)

                    # 计算叉积 (k×v)
                    cross_product = np.cross(axis_direction, perpendicular_vec)

                    # 计算点积 (k·v)
                    dot_product = np.dot(axis_direction, perpendicular_vec)

                    # 应用Rodrigues旋转公式
                    rotated_perpendicular = (perpendicular_vec * cos_angle +
                                           cross_product * sin_angle +
                                           axis_direction * dot_product * (1 - cos_angle))

                    # 新的轨迹点 = 轴上最近点 + 旋转后的垂线向量
                    new_point = closest_point_on_axis + rotated_perpendicular

                    if point_counter < total_timesteps:
                        trajectories[i, point_counter] = new_point
                        current_point = new_point
                        point_counter += 1

        return trajectories

    # 生成新的轨迹
    traj_prediction_new = generate_trajectories_from_axis(point1, point2, pred_selected_affordable_position)
    print(f"Generated {traj_prediction_new.shape[0]} trajectories")
    print(f"Each trajectory has {traj_prediction_new.shape[1]} points in 3 segments")
    print(f"Trajectory generation: contact point → perpendicular to axis → screw motion around axis")
    print(f"Each segment: perpendicular vector rotates 15° around axis → new trajectory point")
    print(f"Total rotation: 3 segments × 15° = 45°")

    # 可选：显示一些调试信息来验证轨迹生成
    for debug_i in range(min(3, traj_prediction_new.shape[0])):  # 只显示前3个轨迹的详细信息
        sample_traj = traj_prediction_new[debug_i]
        print(f"\nSample trajectory {debug_i} points:")
        for j, point in enumerate(sample_traj):
            print(f"  Point {j}: [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
        print(f"  Trajectory length: {len(sample_traj)} points")
        print(f"  Start to end distance: {np.linalg.norm(sample_traj[-1] - sample_traj[0]):.4f}")


    ## 再按照L74-106注释的方法，从traj_prediction_new计算point1, point2
    # 使用修正的算法：每条轨迹由T=4个点组成一个平面，做平面垂线，所有垂线做平均
    axis_result_new = compute_rotation_axis_from_trajectories_corrected(traj_prediction_new)

    # 可视化垂线计算过程（调试用）
    if axis_result_new['success'] and len(axis_result_new['plane_data']) > 0:
        print("\n=== Visualizing trajectory plane normals (perpendicular vectors) ===")
        print("This shows how the axis direction is computed from trajectory planes")
        visualize_trajectory_plane_normals(
            axis_result_new['plane_data'],
            cam_pc,
            pc_rgb,
            window_name="Trajectory Plane Normals - Axis Direction Computation"
        )

    if axis_result_new['success']:
        pred_joint_directions_new = axis_result_new['pred_joint_directions']
        pred_anchor_points_new = axis_result_new['pred_anchor_points']

        # 详细可视化轴位置计算过程：线段、中垂线、交点
        print("\n=== Visualizing detailed axis position computation ===")
        print("This shows segments, perpendicular lines, and intersection points")
        visualize_trajectory_axis_computation_detailed(
            traj_prediction_new,
            pred_joint_directions_new,
            pred_anchor_points_new,
            cam_pc,
            pc_rgb,
            window_name="Trajectory Axis Position Computation - Segments, Perpendiculars, Intersections"
        )
    else:
        # 回退到原始值
        pred_joint_directions_new = point2 - point1
        pred_joint_directions_new = pred_joint_directions_new / np.linalg.norm(pred_joint_directions_new)
        pred_anchor_points_new = point1

    # 计算新的point1和point2
    point1_new = pred_anchor_points_new
    point2_new = pred_anchor_points_new + pred_joint_directions_new

    pred_selected_joint_direction_new, pred_selected_joint_base_new = calculate_rotation_axis(point1_new, point2_new)

    print(f"Original axis: point1={point1}, point2={point2}")
    print(f"Computed axis: point1={point1_new}, point2={point2_new}")
    print(f"Direction difference: {np.arccos(np.abs(np.dot((point2-point1)/np.linalg.norm(point2-point1), pred_joint_directions_new))) * 180 / np.pi:.2f} degrees")

    # 详细可视化轴位置计算过程：线段、中垂线、交点
    print("\n=== Visualizing detailed axis position computation ===")
    print("This shows segments, perpendicular lines, and intersection points")
    visualize_trajectory_axis_computation_detailed(
        traj_prediction_new,
        pred_joint_directions_new,
        pred_anchor_points_new,
        cam_pc,
        pc_rgb,
        window_name="Trajectory Axis Position Computation - Segments, Perpendiculars, Intersections"
    )

    ## 将point1, point2, pred_selected_affordable_position和物体点云在open3d可视化，这样我可以验证正确性
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Axis Verification", width=1200, height=800)

    # 添加点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    vis.add_geometry(pcd)

    # 添加原始旋转轴（红色）
    original_axis = o3d.geometry.LineSet()
    original_axis.points = o3d.utility.Vector3dVector([point1, point2])
    original_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
    original_axis.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # 红色
    vis.add_geometry(original_axis)

    # 原始轴端点
    orig_start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    orig_start_sphere.translate(point1)
    orig_start_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    vis.add_geometry(orig_start_sphere)

    orig_end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    orig_end_sphere.translate(point2)
    orig_end_sphere.paint_uniform_color([1.0, 0.3, 0.3])  # 浅红色
    vis.add_geometry(orig_end_sphere)

    # 添加计算得到的旋转轴（蓝色）
    computed_axis = o3d.geometry.LineSet()
    computed_axis.points = o3d.utility.Vector3dVector([point1_new, point2_new])
    computed_axis.lines = o3d.utility.Vector2iVector([[0, 1]])
    computed_axis.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]])  # 蓝色
    vis.add_geometry(computed_axis)

    # 计算得到的轴端点
    comp_start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    comp_start_sphere.translate(point1_new)
    comp_start_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
    vis.add_geometry(comp_start_sphere)

    # 添加接触点（紫色）
    contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
    contact_sphere.translate(pred_selected_affordable_position)
    contact_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # 紫色
    vis.add_geometry(contact_sphere)

    # 添加少量轨迹用于可视化（分段显示不同颜色）
    max_trajs = min(5, traj_prediction_new.shape[0])
    for i in range(max_trajs):
        trajectory = traj_prediction_new[i]

        # 分段显示轨迹，每段用不同颜色
        num_segments = 3
        points_per_segment = 5

        colors = [
            [0.0, 1.0, 0.0],  # 绿色 - 第一段
            [0.0, 0.8, 0.2],  # 深绿色 - 第二段
            [0.0, 0.6, 0.4]   # 墨绿色 - 第三段
        ]

        # 新的轨迹结构：起始点 + 3段 × 每段4个新点 = 13个点
        # 索引结构: [0] + [1-4] + [5-8] + [9-12]

        for segment in range(num_segments):
            # 计算正确的起始索引
            segment_start_idx = 1 + segment * (points_per_segment - 1)
            segment_end_idx = segment_start_idx + (points_per_segment - 1)

            # 为每段创建线条
            for t in range(segment_start_idx, min(segment_end_idx, len(trajectory) - 1)):
                if t + 1 < len(trajectory):  # 确保不越界
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector([trajectory[t], trajectory[t + 1]])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.colors = o3d.utility.Vector3dVector([colors[segment]])
                    vis.add_geometry(line)

        # 添加轨迹点标记，显示关键点
        if i == 0:  # 只为第一条轨迹添加点标记
            # 起点标记（白色）
            start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            start_sphere.translate(trajectory[0])
            start_sphere.paint_uniform_color([1.0, 1.0, 1.0])  # 白色
            vis.add_geometry(start_sphere)

            # 每段的起始点标记
            for segment in range(num_segments):
                segment_start_idx = 1 + segment * (points_per_segment - 1)
                if segment_start_idx < len(trajectory):
                    segment_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
                    segment_sphere.translate(trajectory[segment_start_idx])
                    segment_sphere.paint_uniform_color(colors[segment])
                    vis.add_geometry(segment_sphere)

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])
    view_ctl.set_lookat(point1)
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.5)

    print("Running axis verification visualization... (Press Q to close)")
    print("Color coding:")
    print("  - Red: Original rotation axis from GT keypoints")
    print("  - Blue: Computed rotation axis from generated trajectories")
    print("  - Purple: Contact point (pred_selected_affordable_position)")
    print("  - White sphere: Trajectory start point")
    print("  - Green gradient: 3-segment trajectories (each segment 15° rotation)")
    print("    - Light green: Segment 1 (0-15°)")
    print("    - Medium green: Segment 2 (15-30°)")
    print("    - Dark green: Segment 3 (30-45°)")
    print("  - Colored spheres: Segment start points")

    vis.update_renderer()
    vis.poll_events()
    vis.run()
    vis.destroy_window()

    # 更新最终的变量用于后续处理
    pred_selected_joint_direction = pred_selected_joint_direction_new
    pred_selected_joint_base = pred_selected_joint_base_new"""


    setup_seed(args.seed)
    # temp_request_path = 'temp_data/observation.npz'
    # temp_response_path = 'temp_data_server/service.npz'
    # temp_flag_path = 'temp_data_server/flag.npy'
    if args.cat == "Microwave":
        joint_types = [0]
        joint_res = [-1]
    elif args.cat == "Refrigerator":
        joint_types = [0]
        joint_res = [1]
    elif args.cat == "Safe":
        joint_types = [0]
        joint_res = [1]
    elif args.cat == "StorageFurniture":
        joint_types = [1, 0]
        joint_res = [0, -1]
    elif args.cat == "Drawer":
        joint_types = [1, 1, 1]
        joint_res = [0, 0, 0]
    elif args.cat == "WashingMachine":
        joint_types = [0]
        joint_res = [-1]
    else:
        raise ValueError(f"Unknown category {args.cat}")

    # if args.graspnet:
    print("===> loading graspnet")
    start_time = time.time()
    from munch import DefaultMunch
    from gsnet import AnyGrasp
    grasp_detector_cfg = {
        'checkpoint_path': args.gsnet_weight_path, 
        'max_gripper_width': args.max_grasp_width, 
        'gripper_height': 0.03, 
        'top_down_grasp': False, 
        'add_vdistance': True, 
        'debug': True
    }
    grasp_detector_cfg = DefaultMunch.fromDict(grasp_detector_cfg)
    grasp_detector = AnyGrasp(grasp_detector_cfg)
    grasp_detector.load_net()
    
    end_time = time.time()
    print(f"===> loaded graspnet {end_time - start_time}")
    
    # while True:
    # print("===> listening to request")
    start_time = time.time()
    serviced = np.array(False)
    # np.save(temp_flag_path, serviced)
    got_request = False



    # 可视化已移除，如需查看可单独调用可视化模块

    # # 创建点云对象
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cam_pc)
    # pcd.colors = o3d.utility.Vector3dVector(pc_rgb)

    # # 调用可视化窗口显示点云
    # o3d.visualization.draw_geometries([pcd])

    # observation = np.load(temp_request_path, allow_pickle=True)
    # cam_pc = observation['point_cloud']
    # print(cam_pc)
    # pc_rgb = observation['rgb']
    c2c = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    cam_pc_model = transform_pc(cam_pc, c2c)
    time.sleep(0.5)
    #os.remove(temp_request_path)
    end_time = time.time()
    print(f"===> got request {end_time - start_time}")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # pcd_o3d_cam = o3d.geometry.PointCloud()
    # pcd_o3d_cam.points = o3d.utility.Vector3dVector(cam_pc_model)
    # pcd_o3d_cam.colors = o3d.utility.Vector3dVector(pc_rgb)
    # vis.add_geometry(pcd_o3d_cam)

    print("===> detecting grasps")
    # if args.graspnet:
    start_time = time.time()
    try:
        # gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='fast')
        gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, apply_object_mask=False, dense_grasp=True, collision_detection=True)
    except:
        gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='slow')
    # print("gg_grasp", gg_grasp)
    if gg_grasp is None:
        gg_grasp = []
    else:
        if len(gg_grasp) != 2:
            gg_grasp = []
        else:
            gg_grasp, pcd_o3d = gg_grasp
            gg_grasp = gg_grasp.nms().sort_by_score()
            grippers_o3d = gg_grasp.to_open3d_geometry_list()
            frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # o3d.visualization.draw_geometries([*grippers_o3d, pcd_o3d, frame_o3d])
            for gripper in grippers_o3d:
                vis.add_geometry(gripper)
                vis.add_geometry(pcd_o3d)
                vis.add_geometry(frame_o3d)
    
    vis.update_renderer()
    vis.poll_events()
    vis.run()
    vis.destroy_window()

    grasp_scores, grasp_widths, grasp_depths, grasp_translations, grasp_rotations, grasp_invaffordances = [], [], [], [], [], []
    for g_idx, g_grasp in enumerate(gg_grasp):
        grasp_score = g_grasp.score
        grasp_scores.append(grasp_score)
        grasp_width = g_grasp.width
        grasp_widths.append(grasp_width)
        grasp_depth = g_grasp.depth
        grasp_depths.append(grasp_depth)
        grasp_translation = g_grasp.translation
        grasp_translations.append(grasp_translation)
        grasp_rotation = g_grasp.rotation_matrix
        grasp_rotations.append(grasp_rotation)
        grasp_invaffordance = invaffordance_metrics(grasp_translation, grasp_rotation, grasp_score, pred_selected_affordable_position, 
                                                    None, None, joint_type=joint_types[args.selected_part])
        grasp_invaffordances.append(grasp_invaffordance)
    grasp_affordances = invaffordances2affordance(grasp_invaffordances)
    
    ## select the best grasp
    # selected_grasp_idxs = np.argsort(grasp_affordances)[::-1]
    orig_scores = gg_grasp.scores
    orig_scores = (orig_scores - np.mean(orig_scores)) / np.std(orig_scores)
    grasp_affordances = ( grasp_affordances - np.mean(grasp_affordances)) / np.std(grasp_affordances)
    new_scores = 0.25 * orig_scores + grasp_affordances
    selected_grasp_idxs = np.argsort(new_scores)[::-1] 
    
    
    for rank, selected_grasp_idx in enumerate(selected_grasp_idxs):
        selected_grasp_score = grasp_scores[selected_grasp_idx]
        selected_grasp_width = grasp_widths[selected_grasp_idx]
        selected_grasp_width = max(min(selected_grasp_width * 1.5, args.max_grasp_width), 0.0)
        selected_grasp_depth = grasp_depths[selected_grasp_idx]
        selected_grasp_translation = grasp_translations[selected_grasp_idx]
        selected_grasp_rotation = grasp_rotations[selected_grasp_idx]
        # selected_grasp_affordance = grasp_affordances[selected_grasp_idx]

        os.makedirs(f"{args.root_dir}/{args.global_key}/init_grasp", exist_ok=True)
        temp_response_path = f"{args.root_dir}/{args.global_key}/init_grasp/{str(rank).zfill(3)}.npz"
        print("temp_response_path", temp_response_path)
        np.savez(temp_response_path,
                    joint_base=None,
                    joint_direction=None, 
                    # affordable_position=pred_selected_affordable_position, 
                    joint_type=joint_types[args.selected_part], 
                    joint_re=joint_res[args.selected_part], 
                    num_grasps=len(gg_grasp), 
                    grasp_score=selected_grasp_score, 
                    grasp_width=selected_grasp_width, 
                    grasp_depth=selected_grasp_depth, 
                    grasp_translation=selected_grasp_translation, 
                    grasp_rotation=selected_grasp_rotation, 
                    # grasp_affordance=selected_grasp_affordance
                    )
        os.makedirs(f"{args.root_dir}/{args.global_key}/init_grasp_orig", exist_ok=True)
        pickle.dump(gg_grasp[selected_grasp_idx:selected_grasp_idx+1], 
                    open(f"{args.root_dir}/{args.global_key}/init_grasp_orig/{str(rank).zfill(3)}.pkl", "wb"))

    end_time = time.time()
    print(f"===> anygrasp detected {end_time - start_time} {len(gg_grasp)}")
    
    ## final vis
    gg_grasp.scores = new_scores
    gg_grasp = gg_grasp.sort_by_score()
    gg_grasp = gg_grasp[:5]

    ## vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    grippers_o3d = gg_grasp.to_open3d_geometry_list()
    frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    for gripper in grippers_o3d:
        vis.add_geometry(gripper)
    vis.add_geometry(pcd_o3d)
    vis.add_geometry(frame_o3d)

    vis.update_renderer()
    vis.poll_events()
    vis.run()
    vis.destroy_window()

    