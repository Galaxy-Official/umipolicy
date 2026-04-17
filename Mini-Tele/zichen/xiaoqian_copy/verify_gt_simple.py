#!/usr/bin/env python3
"""
简化版的ground truth验证脚本
"""

import pickle
import numpy as np
import sys

sys.path.append('/Disk3/xiaoqian')
from tests.gflow_axis_util import compute_rotation_axis_from_trajectories

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

def generate_gt_trajectories(gt_point1, gt_point2, start_points, num_timesteps=4, radius_range=0.1):
    """
    根据ground truth旋转轴生成轨迹
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

        # 生成圆弧轨迹（3段，每段15度）
        for t in range(num_timesteps):
            angle = (t / (num_timesteps - 1)) * np.pi / 6  # 30度圆弧

            # 旋转半径向量
            rotated_radius = (np.cos(angle) * radius_vec + np.sin(angle) * tangent_vec) * radius

            # 轨迹点 = 轴上最近点 + 旋转后的半径向量
            closest_point = gt_point1 + np.dot(start_point - gt_point1, axis_direction) * axis_direction
            trajectories[i, t] = closest_point + rotated_radius

    return trajectories

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
    print(f"GT axis direction: {gt_direction}")
    print(f"Algorithm axis direction: {alg_direction}")
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
    print("=== Ground Truth Trajectory Verification (Simple) ===")

    try:
        # 加载数据
        print("Loading data...")
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')
        gt_data = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints.pkl')

        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]
        traj_prediction = pred_results["traj_prediction"][0]

        gt_point1 = gt_data['point1']
        gt_point2 = gt_data['point2']

        print(f"Point cloud shape: {cam_pc.shape}")
        print(f"Trajectory prediction shape: {traj_prediction.shape}")
        print(f"GT Point1: {gt_point1}")
        print(f"GT Point2: {gt_point2}")

        # 使用当前轨迹的起点作为生成GT轨迹的起点
        start_points = traj_prediction[:, 0, :]  # 每条轨迹的第一个点作为起点

        # 生成Ground Truth轨迹
        print("Generating GT trajectories...")
        gt_trajectories = generate_gt_trajectories(gt_point1, gt_point2, start_points)
        print(f"Generated GT trajectories shape: {gt_trajectories.shape}")

        # 运行算法计算旋转轴
        print("Running algorithm on GT trajectories...")
        algorithm_result = compute_rotation_axis_from_trajectories(gt_trajectories)

        if algorithm_result['success']:
            print(f"Algorithm computed axis point: {algorithm_result['pred_anchor_points']}")
            print(f"Algorithm computed axis direction: {algorithm_result['pred_joint_directions']}")
        else:
            print("❌ Algorithm failed to compute axis")
            return False

        # 验证算法准确性
        print("Verifying algorithm accuracy...")
        is_accurate = verify_algorithm_accuracy(gt_point1, gt_point2, algorithm_result)

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