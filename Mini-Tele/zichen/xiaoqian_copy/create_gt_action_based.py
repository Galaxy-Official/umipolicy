#!/usr/bin/env python3
"""
基于动作方向创建GT keypoints（匹配原始算法）
"""

import pickle
import numpy as np
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

def extract_action_directions(traj_prediction):
    """提取轨迹的动作方向（模仿原始算法）"""
    action_directions = []

    for trajectory in traj_prediction:
        # 对于每条轨迹，提取相邻点之间的方向向量
        for i in range(len(trajectory) - 1):
            direction = trajectory[i+1] - trajectory[i]
            if np.linalg.norm(direction) > 1e-6:  # 避免零向量
                direction = direction / np.linalg.norm(direction)
                action_directions.append(direction)

    return np.array(action_directions)

def create_gt_from_action_directions(traj_prediction):
    """从动作方向估计ground truth旋转轴（匹配原始算法逻辑）"""

    print("Analyzing action directions to estimate GT rotation axis...")

    # 提取动作方向
    action_directions = extract_action_directions(traj_prediction)
    print(f"Extracted {len(action_directions)} action directions")

    if len(action_directions) == 0:
        print("No action directions found, using default")
        return None

    # 计算协方差矩阵（模仿fit_revolute_axis_direction）
    cov = np.dot(action_directions.T, action_directions)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # 排序（升序）
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 最小特征值对应的特征向量就是轴方向
    axis_direction = eigenvectors[:, 0]

    # 确保Z分量为正（模仿原始算法）
    if axis_direction[2] < 0:
        axis_direction = -axis_direction

    # 计算轨迹中心作为轴位置
    all_points = traj_prediction.reshape(-1, 3)
    center_point = np.mean(all_points, axis=0)

    # 创建旋转轴点（以中心点为基准，向两边延伸）
    axis_length = 0.1  # 10cm长的轴用于可视化
    point1 = center_point - axis_direction * axis_length / 2
    point2 = center_point + axis_direction * axis_length / 2

    gt_data = {
        'point1': point1,
        'point2': point2,
        'description': 'Estimated rotation axis from action directions (matching original algorithm)',
        'object_type': 'action_based',
        'center_point': center_point,
        'axis_direction': axis_direction,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'action_directions_count': len(action_directions)
    }

    print(f"Trajectory center: {center_point}")
    print(f"Estimated axis direction: {axis_direction}")
    print(f"GT Point1: {point1}")
    print(f"GT Point2: {point2}")

    return gt_data

def main():
    """主函数"""

    print("=== GT Keypoints Creation Tool (Action-Based) ===")

    try:
        # 加载现有数据
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        cam_pc = point_cloud["points"]
        traj_prediction = pred_results["traj_prediction"][0]

        print(f"Loaded trajectory data: {traj_prediction.shape}")
        print(f"Loaded point cloud: {cam_pc.shape}")

        # 创建GT keypoints（基于动作方向）
        gt_file_path = 'pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints_action_based.pkl'

        # 从动作方向创建GT keypoints
        gt_data = create_gt_from_action_directions(traj_prediction)

        if gt_data is not None:
            # 保存GT keypoints
            with open(gt_file_path, 'wb') as f:
                pickle.dump(gt_data, f)

            print(f"\n✅ Action-based GT keypoints saved to {gt_file_path}")
            print(f"Point1: {gt_data['point1']}")
            print(f"Point2: {gt_data['point2']}")
            print(f"Description: {gt_data['description']}")
        else:
            print("❌ Failed to create GT keypoints")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()