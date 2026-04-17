#!/usr/bin/env python3
"""
修正GT keypoints的方向
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

def fix_gt_keypoints():
    """修正GT keypoints的方向"""
    print("=== Fixing GT Keypoints Direction ===")

    # 加载轨迹数据
    pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
    traj_prediction = pred_results["traj_prediction"][0]

    # 计算正确的轴方向（从轨迹协方差矩阵）
    all_points = traj_prediction.reshape(-1, 3)
    center_point = np.mean(all_points, axis=0)

    cov_matrix = np.cov(all_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 找到最小特征值对应的特征向量（轨迹平面的法向量，即旋转轴方向）
    min_idx = np.argmin(eigenvalues)
    axis_direction = eigenvectors[:, min_idx]

    print(f"Trajectory center: {center_point}")
    print(f"Correct axis direction: {axis_direction}")

    # 创建新的GT keypoints（使用正确的方向）
    axis_length = 0.1  # 10cm长的轴用于可视化
    point1 = center_point - axis_direction * axis_length / 2
    point2 = center_point + axis_direction * axis_length / 2

    gt_data = {
        'point1': point1,
        'point2': point2,
        'description': 'Corrected rotation axis from trajectory analysis',
        'object_type': 'corrected',
        'center_point': center_point,
        'axis_direction': axis_direction,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }

    print(f"Corrected GT Point1: {point1}")
    print(f"Corrected GT Point2: {point2}")

    # 保存修正后的GT keypoints
    gt_file_path = 'pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints_fixed.pkl'
    with open(gt_file_path, 'wb') as f:
        pickle.dump(gt_data, f)

    print(f"✅ Fixed GT keypoints saved to {gt_file_path}")

    # 也覆盖原来的文件
    original_path = 'pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints.pkl'
    with open(original_path, 'wb') as f:
        pickle.dump(gt_data, f)

    print(f"✅ Original GT keypoints updated at {original_path}")

if __name__ == "__main__":
    fix_gt_keypoints()