#!/usr/bin/env python3
"""
分析轨迹数据的特征
"""

import numpy as np
import pickle
import sys

sys.path.append('/Disk3/xiaoqian')

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

def analyze_trajectory_data():
    """分析轨迹数据"""
    print("=== Analyzing Trajectory Data ===")

    # 加载数据
    pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
    traj_prediction = pred_results["traj_prediction"][0]

    print(f"Trajectory shape: {traj_prediction.shape}")
    print(f"Number of trajectories: {traj_prediction.shape[0]}")
    print(f"Points per trajectory: {traj_prediction.shape[1]}")

    # 分析轨迹特征
    all_points = traj_prediction.reshape(-1, 3)
    center_point = np.mean(all_points, axis=0)
    print(f"\nTrajectory center: {center_point}")

    # 计算轨迹的协方差矩阵
    cov_matrix = np.cov(all_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    print(f"\nCovariance matrix eigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):")
    for i, vec in enumerate(eigenvectors.T):
        print(f"  Vector {i}: {vec} (eigenvalue: {eigenvalues[i]:.6f})")

    # 找到最小特征值对应的特征向量（轨迹平面的法向量）
    min_idx = np.argmin(eigenvalues)
    axis_direction = eigenvectors[:, min_idx]
    print(f"\nEstimated axis direction (min eigenvalue): {axis_direction}")

    # 分析单个轨迹
    print(f"\n=== Single Trajectory Analysis ===")
    for i in range(min(3, traj_prediction.shape[0])):
        traj = traj_prediction[i]
        print(f"\nTrajectory {i}:")
        print(f"  Start point: {traj[0]}")
        print(f"  End point: {traj[-1]}")

        # 计算轨迹的局部方向
        if traj.shape[0] > 1:
            directions = []
            for j in range(traj.shape[0] - 1):
                direction = traj[j+1] - traj[j]
                direction = direction / np.linalg.norm(direction)
                directions.append(direction)

            avg_direction = np.mean(directions, axis=0)
            print(f"  Average direction: {avg_direction}")

            # 计算轨迹的平面法向量
            if len(directions) >= 2:
                # 使用叉积找到垂直于轨迹方向的向量
                normal = np.cross(directions[0], directions[1]) if len(directions) > 1 else directions[0]
                if np.linalg.norm(normal) > 1e-6:
                    normal = normal / np.linalg.norm(normal)
                    print(f"  Local plane normal: {normal}")

def analyze_gt_data():
    """分析GT数据"""
    print("\n=== Analyzing GT Data ===")

    try:
        gt_data = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints.pkl')

        point1 = gt_data['point1']
        point2 = gt_data['point2']

        print(f"GT Point1: {point1}")
        print(f"GT Point2: {point2}")

        gt_direction = point2 - point1
        gt_direction = gt_direction / np.linalg.norm(gt_direction)
        print(f"GT Axis direction: {gt_direction}")

        # 计算GT轴的长度和中心
        axis_length = np.linalg.norm(point2 - point1)
        axis_center = (point1 + point2) / 2
        print(f"GT Axis length: {axis_length:.4f} meters")
        print(f"GT Axis center: {axis_center}")

        return point1, point2, gt_direction

    except Exception as e:
        print(f"No GT data found: {e}")
        return None, None, None

def compare_results():
    """对比算法结果和GT"""
    print("\n=== Comparing Algorithm Results with GT ===")

    from tests.gflow_axis_util import compute_rotation_axis_from_trajectories

    # 加载轨迹数据
    pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
    traj_prediction = pred_results["traj_prediction"][0]

    # 运行算法
    result = compute_rotation_axis_from_trajectories(traj_prediction)

    if result['success']:
        alg_point = result['pred_anchor_points']
        alg_direction = result['pred_joint_directions']

        print(f"Algorithm axis point: {alg_point}")
        print(f"Algorithm axis direction: {alg_direction}")

        # 加载GT数据
        gt_point1, gt_point2, gt_direction = analyze_gt_data()

        if gt_direction is not None:
            # 计算方向误差
            direction_error = np.arccos(np.abs(np.dot(gt_direction, alg_direction))) * 180 / np.pi
            print(f"Direction error: {direction_error:.2f} degrees")

            # 计算位置误差
            # 计算GT point1到算法轴的距离
            point_to_axis = (gt_point1 - alg_point) - np.dot(gt_point1 - alg_point, alg_direction) * alg_direction
            position_error = np.linalg.norm(point_to_axis)
            print(f"Position error (GT point1 to algorithm axis): {position_error:.4f} meters")

            # 计算算法点到GT轴的距离
            point_to_gt_axis = (alg_point - gt_point1) - np.dot(alg_point - gt_point1, gt_direction) * gt_direction
            gt_position_error = np.linalg.norm(point_to_gt_axis)
            print(f"Position error (algorithm point to GT axis): {gt_position_error:.4f} meters")

            # 计算中心点距离
            gt_center = (gt_point1 + gt_point2) / 2
            center_distance = np.linalg.norm(alg_point - gt_center)
            print(f"Center point distance: {center_distance:.4f} meters")

def main():
    """主函数"""
    print("=== Trajectory Data Analysis ===")

    # 分析轨迹数据
    analyze_trajectory_data()

    # 分析GT数据
    analyze_gt_data()

    # 对比结果
    compare_results()

if __name__ == "__main__":
    main()