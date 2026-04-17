#!/usr/bin/env python3
"""
验证轨迹算法正确性的脚本
通过对比ground truth和算法计算结果
"""

import pickle
import numpy as np
import open3d as o3d
import sys

sys.path.append('/Disk3/xiaoqian')
from tests.gflow_axis_util import compute_rotation_axis_from_trajectories, visualize_trajectory_intermediate_results

def load_pickle(pickle_file):
    """Load pickle file with error handling"""
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'Unable to load data from {pickle_file}: {e}')
        raise
    return pickle_data

def calculate_perpendicular_vector(point1, point2, query_point):
    """计算点到直线的垂足向量"""
    line_vec = point2 - point1
    line_vec = line_vec / np.linalg.norm(line_vec)

    point_vec = query_point - point1
    projection_length = np.dot(point_vec, line_vec)
    projection = point1 + projection_length * line_vec

    return query_point - projection

def generate_gt_trajectories(gt_point1, gt_point2, start_points, num_timesteps=10, radius_range=0.1):
    """
    根据ground truth旋转轴生成轨迹

    Args:
        gt_point1: 旋转轴起点
        gt_point2: 旋转轴终点
        start_points: 轨迹起点列表 (N, 3)
        num_timesteps: 每条轨迹的时间步数
        radius_range: 轨迹半径范围

    Returns:
        gt_trajectories: (N, T, 3) 轨迹数据
    """
    axis_direction = gt_point2 - gt_point1
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    num_trajectories = len(start_points)
    trajectories = np.zeros((num_trajectories, num_timesteps, 3))

    for i, start_point in enumerate(start_points):
        # 计算起点到旋转轴的距离（半径）
        radius_vec = calculate_perpendicular_vector(gt_point1, gt_point2, start_point)
        radius = np.linalg.norm(radius_vec)

        if radius < 1e-6:  # 如果起点在轴上，使用默认半径
            radius = radius_range
            # 创建一个垂直于轴的随机方向
            random_vec = np.random.randn(3)
            radius_vec = random_vec - np.dot(random_vec, axis_direction) * axis_direction
            radius_vec = radius_vec / np.linalg.norm(radius_vec) * radius

        # 归一化半径向量
        radius_vec = radius_vec / np.linalg.norm(radius_vec)

        # 计算切线方向（垂直于轴和半径向量）
        tangent_vec = np.cross(axis_direction, radius_vec)
        tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)

        # 生成圆弧轨迹
        for t in range(num_timesteps):
            angle = (t / (num_timesteps - 1)) * np.pi / 6  # 30度圆弧

            # 旋转半径向量
            rotated_radius = (np.cos(angle) * radius_vec + np.sin(angle) * tangent_vec) * radius

            # 轨迹点 = 轴上最近点 + 旋转后的半径向量
            closest_point = gt_point1 + np.dot(start_point - gt_point1, axis_direction) * axis_direction
            trajectories[i, t] = closest_point + rotated_radius

    return trajectories

def visualize_comparison(gt_trajectories, algorithm_result, gt_point1, gt_point2, cam_pc, pc_rgb):
    """可视化对比ground truth和算法结果"""

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GT vs Algorithm Comparison", width=1200, height=800)

    # 添加点云背景
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    vis.add_geometry(pcd)

    # 1. 可视化Ground Truth轨迹（绿色）
    print(f"Visualizing {gt_trajectories.shape[0]} GT trajectories...")
    for i in range(gt_trajectories.shape[0]):
        trajectory = gt_trajectories[i]

        # 为每条轨迹创建线条
        for t in range(trajectory.shape[0] - 1):
            point1 = trajectory[t]
            point2 = trajectory[t + 1]

            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([point1, point2])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])  # 绿色
            vis.add_geometry(line)

    # 2. 可视化Ground Truth旋转轴（红色）
    gt_axis_line = o3d.geometry.LineSet()
    gt_axis_line.points = o3d.utility.Vector3dVector([gt_point1, gt_point2])
    gt_axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    gt_axis_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # 红色
    vis.add_geometry(gt_axis_line)

    # GT轴端点
    gt_start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    gt_start_sphere.translate(gt_point1)
    gt_start_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    vis.add_geometry(gt_start_sphere)

    gt_end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    gt_end_sphere.translate(gt_point2)
    gt_end_sphere.paint_uniform_color([1.0, 0.3, 0.3])  # 浅红色
    vis.add_geometry(gt_end_sphere)

    # 3. 可视化算法计算结果（蓝色）
    if algorithm_result['success']:
        alg_point1 = algorithm_result['pred_anchor_points']
        alg_direction = algorithm_result['pred_joint_directions']
        alg_point2 = alg_point1 + alg_direction * 0.05  # 缩放用于可视化

        # 算法计算的轴（蓝色）
        alg_axis_line = o3d.geometry.LineSet()
        alg_axis_line.points = o3d.utility.Vector3dVector([alg_point1, alg_point2])
        alg_axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        alg_axis_line.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]])  # 蓝色
        vis.add_geometry(alg_axis_line)

        # 算法计算的轴端点
        alg_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        alg_sphere.translate(alg_point1)
        alg_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
        vis.add_geometry(alg_sphere)

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])
    if algorithm_result['success']:
        view_ctl.set_lookat(algorithm_result['pred_anchor_points'])
    else:
        view_ctl.set_lookat([0, 0, 0])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.5)

    print("Running comparison visualization... (Press Q to close)")
    print("Color coding:")
    print("  - Green lines: Ground truth trajectories")
    print("  - Red line and spheres: Ground truth rotation axis")
    print("  - Blue line and sphere: Algorithm computed rotation axis")

    vis.update_renderer()
    vis.poll_events()
    vis.run()
    vis.destroy_window()

def verify_algorithm_accuracy(gt_point1, gt_point2, algorithm_result, tolerance=0.01):
    """验证算法准确性"""
    if not algorithm_result['success']:
        print("❌ Algorithm failed to compute rotation axis")
        return False

    alg_point1 = algorithm_result['pred_anchor_points']
    alg_direction = algorithm_result['pred_joint_directions']

    # 计算Ground Truth轴方向
    gt_direction = gt_point2 - gt_point1
    gt_direction = gt_direction / np.linalg.norm(gt_direction)

    # 1. 检查轴方向一致性
    direction_error = np.arccos(np.abs(np.dot(gt_direction, alg_direction))) * 180 / np.pi

    # 2. 检查轴位置一致性（点到直线的距离）
    # 计算gt_point1到算法计算轴的距离
    axis_line_vec = alg_direction
    point_to_alg_axis = calculate_perpendicular_vector(alg_point1, alg_point1 + axis_line_vec, gt_point1)
    position_error = np.linalg.norm(point_to_alg_axis)

    print(f"=== Algorithm Verification Results ===")
    print(f"Direction error: {direction_error:.2f} degrees")
    print(f"Position error: {position_error:.4f} meters")

    # 判断是否在容差范围内
    direction_ok = direction_error < 5.0  # 5度误差
    position_ok = position_error < tolerance

    if direction_ok and position_ok:
        print("✅ Algorithm PASSED - Both direction and position within tolerance")
        return True
    else:
        print("❌ Algorithm FAILED - Exceeds tolerance")
        if not direction_ok:
            print(f"   Direction error {direction_error:.2f}° > 5.0°")
        if not position_ok:
            print(f"   Position error {position_error:.4f}m > {tolerance}m")
        return False

def main():
    """主验证函数"""
    print("=== Ground Truth Trajectory Verification ===")

    try:
        # 加载数据
        print("Loading data...")
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        # 尝试加载ground truth数据
        try:
            gt_data = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints.pkl')
            if 'point1' in gt_data and 'point2' in gt_data:
                gt_point1 = gt_data['point1']
                gt_point2 = gt_data['point2']
                print(f"✓ Loaded GT points from gt_keypoints.pkl")
            else:
                print("⚠️  gt_keypoints.pkl does not contain point1/point2, using default GT")
                # 使用默认的ground truth点（基于常见物体结构）
                gt_point1 = np.array([0.0, 0.0, 0.1])
                gt_point2 = np.array([0.0, 0.0, 0.2])
        except:
            print("⚠️  No gt_keypoints.pkl found, using default GT points")
            # 使用默认的ground truth点
            gt_point1 = np.array([0.0, 0.0, 0.1])
            gt_point2 = np.array([0.0, 0.0, 0.2])

        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]
        traj_prediction = pred_results["traj_prediction"][0]

        print(f"Point cloud shape: {cam_pc.shape}")
        print(f"Trajectory prediction shape: {traj_prediction.shape}")
        print(f"GT Point1: {gt_point1}")
        print(f"GT Point2: {gt_point2}")

        # 使用当前轨迹的起点作为生成GT轨迹的起点
        start_points = traj_prediction[:, 0, :]  # 每条轨迹的第一个点作为起点

        # 生成Ground Truth轨迹
        print("Generating GT trajectories...")
        gt_trajectories = generate_gt_trajectories(gt_point1, gt_point2, start_points)

        # 运行算法计算旋转轴
        print("Running algorithm on GT trajectories...")
        algorithm_result = compute_rotation_axis_from_trajectories(gt_trajectories)

        # 可视化对比
        print("Visualizing comparison...")
        visualize_comparison(gt_trajectories, algorithm_result, gt_point1, gt_point2, cam_pc, pc_rgb)

        # 验证算法准确性
        print("Verifying algorithm accuracy...")
        is_accurate = verify_algorithm_accuracy(gt_point1, gt_point2, algorithm_result)

        # 可选：可视化算法中间过程（用于调试）
        if algorithm_result['success'] and input("Show intermediate results? (y/n): ").lower() == 'y':
            print("Showing intermediate calculation results...")
            visualize_trajectory_intermediate_results(
                algorithm_result['action_data'],
                algorithm_result['all_intersection_data'],
                algorithm_result['all_actions_data'],
                algorithm_result['axis_candidates'],
                algorithm_result['pred_anchor_points'],
                algorithm_result['pred_joint_directions'],
                cam_pc, pc_rgb,
                window_name="GT Trajectory Intermediate Results"
            )

        print("\n=== Verification Complete ===")
        if is_accurate:
            print("✅ The algorithm correctly computes rotation axis from trajectories!")
        else:
            print("❌ The algorithm needs adjustment or the GT data may be incorrect.")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()