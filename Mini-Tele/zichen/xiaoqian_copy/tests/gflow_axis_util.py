#!/usr/bin/env python3
"""
GFlow轨迹计算工具模块
包含从轨迹数据计算旋转轴的所有相关函数
"""

import numpy as np
import open3d as o3d


def extract_action_directions(trajectories):
    """Extract action directions from trajectories with visualization data"""
    N, T, _ = trajectories.shape
    action_directions = []
    action_data = []  # 用于可视化的中间数据

    for i in range(N):
        traj = trajectories[i]  # (T, 3)
        for t in range(T - 1):
            pos = traj[t]
            direction = traj[t+1] - traj[t]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
                action_directions.append(direction)
                # 保存可视化数据
                action_data.append({
                    'pos': pos,
                    'direction': direction,
                    'traj_idx': i,
                    'step': t
                })

    return np.array(action_directions), action_data


def fit_revolute_axis_direction(action_directions):
    """Fit revolute joint axis direction using the common action plane method"""
    # Compute covariance matrix of action directions
    cov = np.dot(action_directions.T, action_directions)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # The axis direction is the eigenvector with smallest eigenvalue
    axis_direction = eigenvectors[:, 0]

    # Ensure consistent orientation
    if axis_direction[2] < 0:
        axis_direction = -axis_direction

    return axis_direction


def compute_closest_point_between_lines(p1, d1, p2, d2):
    """Compute closest point (midpoint) between two lines in 3D"""
    w = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)

    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return (p1 + p2) / 2

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    pt1 = p1 + t * d1
    pt2 = p2 + s * d2

    return (pt1 + pt2) / 2


def fit_axis_position_from_single_trajectory(trajectory, axis_direction):
    """Fit axis position from a single trajectory with visualization data"""
    T = len(trajectory)
    actions = []
    intersection_data = []  # 用于可视化的交点数据

    for t in range(T - 1):
        pos = trajectory[t]
        direction = trajectory[t+1] - trajectory[t]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
            # Construct line perpendicular to action and axis
            line_dir = np.cross(direction, axis_direction)
            line_norm = np.linalg.norm(line_dir)
            if line_norm > 1e-8:
                line_dir = line_dir / line_norm
                actions.append({
                    'pos': pos,
                    'line_dir': line_dir,
                    'original_dir': direction,
                    'step': t
                })

    # Compute all pairwise intersections within this trajectory
    intersections = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            p1 = actions[i]['pos']
            d1 = actions[i]['line_dir']
            p2 = actions[j]['pos']
            d2 = actions[j]['line_dir']

            intersection = compute_closest_point_between_lines(p1, d1, p2, d2)
            intersections.append(intersection)

            # 保存交点可视化数据
            intersection_data.append({
                'intersection_point': intersection,
                'line1_pos': p1,
                'line1_dir': d1,
                'line2_pos': p2,
                'line2_dir': d2,
                'line1_step': actions[i]['step'],
                'line2_step': actions[j]['step']
            })

    # Return mean of all intersections from this trajectory and intersection data
    axis_candidate = np.mean(intersections, axis=0) if intersections else trajectory[0]
    return axis_candidate, intersection_data, actions


def fit_revolute_axis_position(trajectories, axis_direction):
    """Fit revolute joint axis position using per-trajectory voting"""
    N = len(trajectories)

    # Get one axis candidate from each trajectory
    axis_candidates = []
    for i in range(N):
        candidate = fit_axis_position_from_single_trajectory(
            trajectories[i], axis_direction
        )
        axis_candidates.append(candidate)

    axis_candidates = np.array(axis_candidates)
    axis_position = np.mean(axis_candidates, axis=0)

    return axis_position


def compute_trajectory_plane_normals(traj_prediction):
    """
    计算每条轨迹所在平面的法向量（垂线）
    正确的逻辑：每条轨迹由T=4个点组成一个平面，做平面垂线

    Args:
        traj_prediction: (Q, T, 3) 轨迹预测数据

    Returns:
        plane_normals: (Q, 3) 每个轨迹平面的法向量
        plane_data: 可视化数据
    """
    Q, T, _ = traj_prediction.shape
    plane_normals = []
    plane_data = []  # 用于可视化的中间数据

    for i in range(Q):
        trajectory = traj_prediction[i]  # (T, 3)

        if T < 3:
            # 如果点数不够，使用默认法向量
            normal = np.array([0, 0, 1])
            plane_normals.append(normal)
            continue

        # 使用轨迹点拟合平面
        # 平面方程: ax + by + cz + d = 0
        # 平面法向量: [a, b, c]

        # 计算轨迹点的质心
        centroid = np.mean(trajectory, axis=0)

        # 构建协方差矩阵
        centered_points = trajectory - centroid
        cov_matrix = np.dot(centered_points.T, centered_points)

        # 使用SVD找到平面的法向量（最小特征值对应的特征向量）
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 找到最小特征值对应的特征向量（平面法向量）
        min_idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, min_idx]

        # 确保法向量方向一致（Z分量为正）
        if normal[2] < 0:
            normal = -normal

        plane_normals.append(normal)

        # 保存可视化数据
        plane_data.append({
            'trajectory_idx': i,
            'centroid': centroid,
            'normal': normal,
            'trajectory_points': trajectory,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        })

    return np.array(plane_normals), plane_data

def compute_rotation_axis_from_trajectories_corrected(traj_prediction):
    """
    修正的旋转轴计算函数
    正确的逻辑：每条轨迹由T=4个点组成一个平面，做平面垂线，所有垂线做平均

    Args:
        traj_prediction: (Q, T, 3) 轨迹预测数据

    Returns:
        dict: 包含计算结果和中间数据的字典
    """
    # Step 1: 计算每条轨迹所在平面的法向量（垂线）
    plane_normals, plane_data = compute_trajectory_plane_normals(traj_prediction)

    if len(plane_normals) == 0:
        # 如果没有找到平面法向量，返回默认值
        return {
            'success': False,
            'pred_joint_directions': np.array([0, 0, 1]),
            'pred_anchor_points': traj_prediction[0, 0, :] if traj_prediction.size > 0 else np.array([0, 0, 0]),
            'plane_data': [],
            'all_intersection_data': [],
            'all_actions_data': [],
            'axis_candidates': np.array([])
        }

    # Step 2: 所有轨迹平面的垂线做平均，获得轴的方向
    pred_joint_directions = np.mean(plane_normals, axis=0)
    pred_joint_directions = pred_joint_directions / np.linalg.norm(pred_joint_directions)

    # 确保Z分量为正
    if pred_joint_directions[2] < 0:
        pred_joint_directions = -pred_joint_directions

    # Step 3: 拟合轴位置（每轨迹投票）
    all_intersection_data = []
    all_actions_data = []
    axis_candidates = []

    for i in range(len(traj_prediction)):
        candidate, intersection_data, actions_data = fit_axis_position_from_single_trajectory(
            traj_prediction[i], pred_joint_directions
        )
        axis_candidates.append(candidate)
        all_intersection_data.extend(intersection_data)
        all_actions_data.extend(actions_data)

    axis_candidates = np.array(axis_candidates)
    pred_anchor_points = np.mean(axis_candidates, axis=0)

    return {
        'success': True,
        'pred_joint_directions': pred_joint_directions,
        'pred_anchor_points': pred_anchor_points,
        'plane_data': plane_data,
        'all_intersection_data': all_intersection_data,
        'all_actions_data': all_actions_data,
        'axis_candidates': axis_candidates
    }

def compute_rotation_axis_from_trajectories(traj_prediction):
    """
    从轨迹数据计算旋转轴的主函数

    Args:
        traj_prediction: (Q, T, 3) 轨迹预测数据

    Returns:
        dict: 包含计算结果和中间数据的字典
    """
    # Step 1: Extract all actions to find axis direction
    action_directions, action_data = extract_action_directions(traj_prediction)

    if len(action_directions) == 0:
        # 如果没有找到动作方向，返回默认值
        return {
            'success': False,
            'pred_joint_directions': np.array([0, 0, 1]),
            'pred_anchor_points': traj_prediction[0, 0, :] if traj_prediction.size > 0 else np.array([0, 0, 0]),
            'action_data': [],
            'all_intersection_data': [],
            'all_actions_data': [],
            'axis_candidates': np.array([])
        }

    # Step 2: Fit axis direction (global, from all actions)
    pred_joint_directions = fit_revolute_axis_direction(action_directions)

    # Step 3: Fit axis position (per-trajectory voting) with intermediate data
    all_intersection_data = []
    all_actions_data = []
    axis_candidates = []

    for i in range(len(traj_prediction)):
        candidate, intersection_data, actions_data = fit_axis_position_from_single_trajectory(
            traj_prediction[i], pred_joint_directions
        )
        axis_candidates.append(candidate)
        all_intersection_data.extend(intersection_data)
        all_actions_data.extend(actions_data)

    axis_candidates = np.array(axis_candidates)
    pred_anchor_points = np.mean(axis_candidates, axis=0)

    return {
        'success': True,
        'pred_joint_directions': pred_joint_directions,
        'pred_anchor_points': pred_anchor_points,
        'action_data': action_data,
        'all_intersection_data': all_intersection_data,
        'all_actions_data': all_actions_data,
        'axis_candidates': axis_candidates
    }

def generate_trajectories_from_axis(point1, point2, contact_point, num_trajectories=128, radius_range=0.05):
    """
    根据旋转轴和接触点生成三段式轨迹，每段旋转15°
    正确的逻辑：每个点向轴做垂线，然后该垂线围绕轴做screw运动
    """
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
            segment_start_angle = segment * np.pi / 6  # 15° * segment
            segment_end_angle = (segment + 1) * np.pi / 6  # 15° * (segment + 1)

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
                    point_counter += 1

    return trajectories

def visualize_trajectory_plane_normals(plane_data, cam_pc, pc_rgb, window_name="Trajectory Plane Normals"):
    """
    可视化轨迹平面法向量（垂线）

    Args:
        plane_data: 来自compute_trajectory_plane_normals的数据
        cam_pc: 点云数据
        pc_rgb: 点云颜色
        window_name: 窗口名称
    """
    print(f"=== Visualizing {len(plane_data)} trajectory plane normals ===")

    # 创建新的可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1200, height=800)

    # 添加点云作为背景
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(cam_pc)
    pcd_bg.colors = o3d.utility.Vector3dVector(pc_rgb)
    vis.add_geometry(pcd_bg)

    # 1. 显示轨迹点
    max_trajectories = min(10, len(plane_data))  # 限制显示的轨迹数量
    colors = [
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 1.0, 0.0],  # 绿色
        [0.0, 0.0, 1.0],  # 蓝色
        [1.0, 1.0, 0.0],  # 黄色
        [1.0, 0.0, 1.0],  # 紫色
        [0.0, 1.0, 1.0],  # 青色
        [1.0, 0.5, 0.0],  # 橙色
        [0.5, 0.0, 1.0],  # 紫罗兰
        [0.0, 1.0, 0.5],  # 春绿色
        [1.0, 0.0, 0.5]   # 深红色
    ]

    for i, data in enumerate(plane_data[:max_trajectories]):
        trajectory_points = data['trajectory_points']
        centroid = data['centroid']
        normal = data['normal']

        # 显示轨迹点
        for _, point in enumerate(trajectory_points):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            sphere.translate(point)
            sphere.paint_uniform_color(colors[i % len(colors)])
            vis.add_geometry(sphere)

        # 显示轨迹质心（较大球体）
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        centroid_sphere.translate(centroid)
        centroid_sphere.paint_uniform_color(colors[i % len(colors)])
        vis.add_geometry(centroid_sphere)

        # 2. 显示平面法向量（垂线）
        # 法向量箭头（从质心出发）
        normal_length = 0.03  # 法向量长度

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.005, cone_radius=0.002,
            cylinder_height=normal_length - 0.005, cylinder_radius=0.001
        )

        # 计算旋转矩阵使箭头指向法向量方向
        z_axis = np.array([0, 0, 1])
        if not np.allclose(normal, z_axis):
            # 计算旋转轴和角度
            cross_prod = np.cross(z_axis, normal)
            dot_prod = np.dot(z_axis, normal)
            if np.abs(dot_prod + 1.0) < 1e-6:
                rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            elif np.abs(dot_prod - 1.0) < 1e-6:
                rotation_matrix = np.eye(3)
            else:
                cross_prod_mat = np.array([
                    [0, -cross_prod[2], cross_prod[1]],
                    [cross_prod[2], 0, -cross_prod[0]],
                    [-cross_prod[1], cross_prod[0], 0]
                ])
                rotation_matrix = np.eye(3) + cross_prod_mat + np.matmul(cross_prod_mat, cross_prod_mat) / (1 + dot_prod)

            arrow.rotate(rotation_matrix, center=[0, 0, 0])

        arrow.translate(centroid)
        arrow.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色箭头表示法向量
        vis.add_geometry(arrow)

    # 3. 显示平均法向量（最终结果）
    if len(plane_data) > 0:
        avg_normal = np.mean([data['normal'] for data in plane_data], axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        avg_centroid = np.mean([data['centroid'] for data in plane_data], axis=0)

        # 平均法向量箭头（更大，红色）
        avg_normal_length = 0.05
        avg_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.01, cone_radius=0.004,
            cylinder_height=avg_normal_length - 0.01, cylinder_radius=0.002
        )

        # 旋转箭头
        z_axis = np.array([0, 0, 1])
        if not np.allclose(avg_normal, z_axis):
            cross_prod = np.cross(z_axis, avg_normal)
            dot_prod = np.dot(z_axis, avg_normal)
            if np.abs(dot_prod + 1.0) < 1e-6:
                rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            elif np.abs(dot_prod - 1.0) < 1e-6:
                rotation_matrix = np.eye(3)
            else:
                cross_prod_mat = np.array([
                    [0, -cross_prod[2], cross_prod[1]],
                    [cross_prod[2], 0, -cross_prod[0]],
                    [-cross_prod[1], cross_prod[0], 0]
                ])
                rotation_matrix = np.eye(3) + cross_prod_mat + np.matmul(cross_prod_mat, cross_prod_mat) / (1 + dot_prod)

            avg_arrow.rotate(rotation_matrix, center=[0, 0, 0])

        avg_arrow.translate(avg_centroid)
        avg_arrow.paint_uniform_color([1.0, 0.0, 0.0])  # 红色表示平均法向量
        vis.add_geometry(avg_arrow)

        # 平均质心（红色球体）
        avg_centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        avg_centroid_sphere.translate(avg_centroid)
        avg_centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometry(avg_centroid_sphere)

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])
    if len(plane_data) > 0:
        avg_centroid = np.mean([data['centroid'] for data in plane_data], axis=0)
        view_ctl.set_lookat(avg_centroid)
    else:
        view_ctl.set_lookat([0, 0, 0])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.5)

    print("Running trajectory plane normals visualization... (Press Q to close)")
    print("Color coding:")
    print("  - Colored spheres: Trajectory points (each color = one trajectory)")
    print("  - Larger colored spheres: Trajectory centroids")
    print("  - Yellow arrows: Individual trajectory plane normals (perpendicular to plane)")
    print("  - Red arrow: Average normal vector (final axis direction)")
    print("  - Red sphere: Average centroid")

    vis.update_renderer()
    vis.poll_events()
    vis.run()
    vis.destroy_window()
    print("=== Trajectory plane normals visualization complete ===")

def visualize_trajectory_axis_computation_detailed(traj_prediction, axis_direction, axis_position,
                                                   cam_pc, pc_rgb, window_name="Trajectory Axis Computation Detailed"):
    """
    详细可视化轨迹轴计算过程：线段、中垂线、交点

    Args:
        traj_prediction: (Q, T, 3) 轨迹数据
        axis_direction: 轴方向向量
        axis_position: 计算得到的轴位置
        cam_pc: 点云数据
        pc_rgb: 点云颜色
        window_name: 窗口名称
    """
    print(f"=== Visualizing detailed trajectory axis computation ===")
    print(f"Showing segments, perpendicular lines, and intersection points")

    # 创建新的可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)

    # 添加点云作为背景
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(cam_pc)
    pcd_bg.colors = o3d.utility.Vector3dVector(pc_rgb)
    vis.add_geometry(pcd_bg)

    # 颜色方案
    trajectory_colors = [
        [1.0, 0.0, 0.0],  # 红色 - 轨迹1
        [0.0, 1.0, 0.0],  # 绿色 - 轨迹2
        [0.0, 0.0, 1.0],  # 蓝色 - 轨迹3
        [1.0, 1.0, 0.0],  # 黄色 - 轨迹4
        [1.0, 0.0, 1.0],  # 紫色 - 轨迹5
    ]

    all_intersections = []
    all_perpendicular_lines = []

    # 限制显示的轨迹数量
    max_trajectories = min(5, len(traj_prediction))

    for traj_idx in range(max_trajectories):
        trajectory = traj_prediction[traj_idx]
        color = trajectory_colors[traj_idx % len(trajectory_colors)]

        print(f"\nProcessing trajectory {traj_idx} with {len(trajectory)} points")

        # 1. 显示轨迹点
        for i, point in enumerate(trajectory):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sphere.translate(point)
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere)

        # 2. 显示轨迹线段
        for i in range(len(trajectory) - 1):
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([trajectory[i], trajectory[i + 1]])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([color])
            vis.add_geometry(line)

        # 3. 计算中垂线（垂直于线段和轴方向）
        actions = []
        for t in range(len(trajectory) - 1):
            pos = trajectory[t]
            direction = trajectory[t+1] - trajectory[t]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
                # 构造垂直于动作和轴的直线
                line_dir = np.cross(direction, axis_direction)
                line_norm = np.linalg.norm(line_dir)
                if line_norm > 1e-8:
                    line_dir = line_dir / line_norm
                    actions.append({
                        'pos': pos,
                        'line_dir': line_dir,
                        'original_dir': direction,
                        'step': t
                    })

        # 4. 显示中垂线（青色）
        for i, action in enumerate(actions):
            start_pos = action['pos']
            line_dir = action['line_dir']

            # 中垂线（双向，更长以便可视化）
            line_length = 0.04  # 增加长度以便观察
            line_start = start_pos - line_dir * line_length/2
            line_end = start_pos + line_dir * line_length/2

            perp_line = o3d.geometry.LineSet()
            perp_line.points = o3d.utility.Vector3dVector([line_start, line_end])
            perp_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            perp_line.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 1.0]])  # 青色
            vis.add_geometry(perp_line)

            # 保存垂线数据
            all_perpendicular_lines.append({
                'start': line_start,
                'end': line_end,
                'center': start_pos,
                'direction': line_dir,
                'trajectory_idx': traj_idx,
                'segment_idx': i
            })

        # 5. 计算并显示交点
        intersections = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                p1 = actions[i]['pos']
                d1 = actions[i]['line_dir']
                p2 = actions[j]['pos']
                d2 = actions[j]['line_dir']

                intersection = compute_closest_point_between_lines(p1, d1, p2, d2)
                intersections.append(intersection)

                # 显示交点（黄色球体）
                intersection_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
                intersection_sphere.translate(intersection)
                intersection_sphere.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
                vis.add_geometry(intersection_sphere)

                all_intersections.append({
                    'point': intersection,
                    'trajectory_idx': traj_idx,
                    'line1_idx': i,
                    'line2_idx': j
                })

        print(f"  Trajectory {traj_idx}: {len(actions)} perpendicular lines, {len(intersections)} intersections")

    # 6. 显示所有交点的平均值（轴位置候选）
    if all_intersections:
        intersection_points = np.array([data['point'] for data in all_intersections])
        avg_intersection = np.mean(intersection_points, axis=0)

        # 平均交点（大黄色球体）
        avg_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        avg_sphere.translate(avg_intersection)
        avg_sphere.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
        vis.add_geometry(avg_sphere)

        print(f"\nAverage intersection point: {avg_intersection}")

    # 7. 显示最终计算的轴位置（红色大球）
    if axis_position is not None:
        final_axis_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        final_axis_sphere.translate(axis_position)
        final_axis_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        vis.add_geometry(final_axis_sphere)

        print(f"Final computed axis position: {axis_position}")

    # 8. 显示轴方向（红色箭头）
    if axis_direction is not None and axis_position is not None:
        axis_length = 0.08

        axis_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.01, cone_radius=0.004,
            cylinder_height=axis_length - 0.01, cylinder_radius=0.002
        )

        # 旋转箭头指向轴方向
        z_axis = np.array([0, 0, 1])
        if not np.allclose(axis_direction, z_axis):
            cross_prod = np.cross(z_axis, axis_direction)
            dot_prod = np.dot(z_axis, axis_direction)
            if np.abs(dot_prod + 1.0) < 1e-6:
                rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            elif np.abs(dot_prod - 1.0) < 1e-6:
                rotation_matrix = np.eye(3)
            else:
                cross_prod_mat = np.array([
                    [0, -cross_prod[2], cross_prod[1]],
                    [cross_prod[2], 0, -cross_prod[0]],
                    [-cross_prod[1], cross_prod[0], 0]
                ])
                rotation_matrix = np.eye(3) + cross_prod_mat + np.matmul(cross_prod_mat, cross_prod_mat) / (1 + dot_prod)

            axis_arrow.rotate(rotation_matrix, center=[0, 0, 0])

        axis_arrow.translate(axis_position)
        axis_arrow.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        vis.add_geometry(axis_arrow)

    # 9. 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])

    # 设置观察点
    if all_intersections:
        all_points = np.array([data['point'] for data in all_intersections])
        center_point = np.mean(all_points, axis=0)
        view_ctl.set_lookat(center_point)
    elif len(traj_prediction) > 0:
        view_ctl.set_lookat(traj_prediction[0][0])
    else:
        view_ctl.set_lookat([0, 0, 0])

    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.6)

    print(f"\n=== 轴位置计算可视化统计 ===")
    print(f"总轨迹数: {len(traj_prediction)}")
    print(f"显示轨迹数: {max_trajectories}")
    print(f"总垂线数: {len(all_perpendicular_lines)}")
    print(f"总交点数: {len(all_intersections)}")

    print("\nRunning detailed axis computation visualization... (Press Q to close)")
    print("Color coding:")
    print("  - Colored lines: Trajectory segments (each color = one trajectory)")
    print("  - Cyan lines: Perpendicular lines (中垂线)")
    print("  - Yellow spheres: Intersection points (交点)")
    print("  - Large yellow sphere: Average of intersections")
    print("  - Large red sphere: Final computed axis position")
    print("  - Red arrow: Computed axis direction")

    vis.update_renderer()
    vis.poll_events()
    vis.run()
    vis.destroy_window()
    print("=== Detailed axis computation visualization complete ===")


def visualize_trajectory_intermediate_results(action_data, all_intersection_data, all_actions_data,
                                             axis_candidates, pred_anchor_points, pred_joint_directions,
                                             cam_pc, pc_rgb, window_name="Trajectory Intermediate Results"):
    """可视化轨迹计算的中间结果"""
    print("=== Visualizing trajectory intermediate calculation results ===")

    # 创建新的可视化器
    vis_intermediate = o3d.visualization.Visualizer()
    vis_intermediate.create_window(window_name=window_name, width=1024, height=768)

    # 添加点云作为背景
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(cam_pc)
    pcd_bg.colors = o3d.utility.Vector3dVector(pc_rgb)
    vis_intermediate.add_geometry(pcd_bg)

    # 1. 显示动作方向向量（红色箭头）
    if action_data and len(action_data) > 0:
        print(f"Visualizing {len(action_data)} action direction vectors...")
        max_arrows = min(100, len(action_data))  # 限制箭头数量
        step = max(1, len(action_data) // max_arrows)

        for i in range(0, len(action_data), step):
            action = action_data[i]
            start_pos = action['pos']
            direction = action['direction'] * 0.02  # 缩放箭头长度
            end_pos = start_pos + direction

            # 创建箭头
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cone_height=0.005, cone_radius=0.002,
                cylinder_height=0.015, cylinder_radius=0.001
            )

            # 计算旋转矩阵使箭头指向正确方向
            z_axis = np.array([0, 0, 1])
            if not np.allclose(direction, z_axis):
                # 计算旋转轴和角度
                cross_prod = np.cross(z_axis, direction)
                dot_prod = np.dot(z_axis, direction)
                if np.abs(dot_prod + 1.0) < 1e-6:
                    rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                elif np.abs(dot_prod - 1.0) < 1e-6:
                    rotation_matrix = np.eye(3)
                else:
                    cross_prod_mat = np.array([
                        [0, -cross_prod[2], cross_prod[1]],
                        [cross_prod[2], 0, -cross_prod[0]],
                        [-cross_prod[1], cross_prod[0], 0]
                    ])
                    rotation_matrix = np.eye(3) + cross_prod_mat + np.matmul(cross_prod_mat, cross_prod_mat) / (1 + dot_prod)

                arrow.rotate(rotation_matrix, center=[0, 0, 0])

            arrow.translate(start_pos)
            arrow.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            vis_intermediate.add_geometry(arrow)
    else:
        print("No action direction data to visualize")

    # 2. 显示轨迹线段的中垂线（青色线条）
    if all_actions_data and len(all_actions_data) > 0:
        print(f"Visualizing perpendicular lines from {len(all_actions_data)} actions...")
        max_lines = min(50, len(all_actions_data))
        line_step = max(1, len(all_actions_data) // max_lines)

        for i in range(0, len(all_actions_data), line_step):
            action = all_actions_data[i]
            start_pos = action['pos']
            line_dir = action['line_dir'] * 0.03  # 缩放线条长度
            end_pos = start_pos + line_dir

            # 创建中垂线
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([start_pos, end_pos])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 1.0]])  # 青色
            vis_intermediate.add_geometry(line)

            # 添加起点标记
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            sphere.translate(start_pos)
            sphere.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色
            vis_intermediate.add_geometry(sphere)
    else:
        print("No perpendicular line data to visualize")

    # 3. 显示交点（黄色球体）
    if all_intersection_data and len(all_intersection_data) > 0:
        print(f"Visualizing {len(all_intersection_data)} intersection points...")
        max_intersections = min(100, len(all_intersection_data))
        intersection_step = max(1, len(all_intersection_data) // max_intersections)

        for i in range(0, len(all_intersection_data), intersection_step):
            intersection = all_intersection_data[i]
            intersection_point = intersection['intersection_point']

            # 交点球
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            sphere.translate(intersection_point)
            sphere.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
            vis_intermediate.add_geometry(sphere)

            # 显示相交的两条线段（橙色）
            line1 = o3d.geometry.LineSet()
            line1_start = intersection['line1_pos']
            line1_end = line1_start + intersection['line1_dir'] * 0.02
            line1.points = o3d.utility.Vector3dVector([line1_start, line1_end])
            line1.lines = o3d.utility.Vector2iVector([[0, 1]])
            line1.colors = o3d.utility.Vector3dVector([[1.0, 0.5, 0.0]])  # 橙色
            vis_intermediate.add_geometry(line1)

            line2 = o3d.geometry.LineSet()
            line2_start = intersection['line2_pos']
            line2_end = line2_start + intersection['line2_dir'] * 0.02
            line2.points = o3d.utility.Vector3dVector([line2_start, line2_end])
            line2.lines = o3d.utility.Vector2iVector([[0, 1]])
            line2.colors = o3d.utility.Vector3dVector([[1.0, 0.5, 0.0]])  # 橙色
            vis_intermediate.add_geometry(line2)
    else:
        print("No intersection data to visualize")

    # 4. 显示单条轨迹的轴候选点（绿色球体）
    if axis_candidates is not None and len(axis_candidates) > 0:
        print(f"Visualizing {len(axis_candidates)} axis candidate points...")
        for i, candidate in enumerate(axis_candidates):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.translate(candidate)
            sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
            vis_intermediate.add_geometry(sphere)
    else:
        print("No axis candidate data to visualize")

    # 5. 显示最终计算的轴（紫色大球和粗线条）
    if pred_anchor_points is not None and pred_joint_directions is not None:
        print("Visualizing final computed axis...")
        # 轴位置点
        axis_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        axis_sphere.translate(pred_anchor_points)
        axis_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # 紫色
        vis_intermediate.add_geometry(axis_sphere)

        # 轴方向线（更粗）
        axis_end = pred_anchor_points + pred_joint_directions * 0.05
        axis_line = o3d.geometry.LineSet()
        axis_line.points = o3d.utility.Vector3dVector([pred_anchor_points, axis_end])
        axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        axis_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0]])  # 紫色
        vis_intermediate.add_geometry(axis_line)
    else:
        print("No final axis data to visualize")

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis_intermediate.add_geometry(coordinate_frame)

    # 设置视角
    view_ctl = vis_intermediate.get_view_control()
    view_ctl.set_front([0, 0, -1])
    if pred_anchor_points is not None:
        view_ctl.set_lookat(pred_anchor_points)
    else:
        view_ctl.set_lookat([0, 0, 0])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.5)

    # 运行可视化
    print("Running intermediate results visualization... (Press Q to close)")
    print("Color coding:")
    print("  - Red arrows: Action direction vectors")
    print("  - Cyan lines: Perpendicular lines (trajectory segment normals)")
    print("  - Yellow spheres: Intersection points between perpendicular lines")
    print("  - Orange lines: Pairs of intersecting perpendicular lines")
    print("  - Green spheres: Axis candidate points from each trajectory")
    print("  - Purple sphere and line: Final computed rotation axis")

    vis_intermediate.update_renderer()
    vis_intermediate.poll_events()
    vis_intermediate.run()
    vis_intermediate.destroy_window()
    print("=== Intermediate visualization complete ===")