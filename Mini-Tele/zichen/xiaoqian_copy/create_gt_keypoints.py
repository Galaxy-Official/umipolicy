#!/usr/bin/env python3
"""
创建ground truth keypoints的脚本
用于验证轨迹算法
"""

import pickle
import numpy as np
import os

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

def create_default_gt_keypoints():
    """创建默认的ground truth keypoints"""

    # 默认的旋转轴点（可以根据实际物体调整）
    # 这里假设一个垂直的旋转轴，比如门的铰链
    gt_data = {
        'point1': np.array([0.0, 0.0, 0.05]),   # 旋转轴起点
        'point2': np.array([0.0, 0.0, 0.15]),   # 旋转轴终点
        'description': 'Default ground truth rotation axis for verification',
        'object_type': 'door_hinge'  # 或其他类型
    }

    return gt_data

def create_gt_from_trajectory_analysis(traj_prediction):
    """从轨迹数据估计ground truth旋转轴"""

    print("Analyzing trajectories to estimate GT rotation axis...")

    # 简单估计：计算轨迹的中心和主要方向
    all_points = traj_prediction.reshape(-1, 3)
    center_point = np.mean(all_points, axis=0)

    # 计算轨迹的协方差矩阵
    cov_matrix = np.cov(all_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 找到最小特征值对应的特征向量（轨迹平面的法向量，即旋转轴方向）
    min_idx = np.argmin(eigenvalues)
    axis_direction = eigenvectors[:, min_idx]

    # 确保Z分量为正
    if axis_direction[2] < 0:
        axis_direction = -axis_direction

    # 创建旋转轴点（以中心点为基准，向两边延伸）
    axis_length = 0.1  # 10cm长的轴用于可视化
    point1 = center_point - axis_direction * axis_length / 2
    point2 = center_point + axis_direction * axis_length / 2

    gt_data = {
        'point1': point1,
        'point2': point2,
        'description': 'Estimated rotation axis from trajectory analysis',
        'object_type': 'estimated',
        'center_point': center_point,
        'axis_direction': axis_direction,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }

    print(f"Estimated center point: {center_point}")
    print(f"Estimated axis direction: {axis_direction}")
    print(f"GT Point1: {point1}")
    print(f"GT Point2: {point2}")

    return gt_data

def interactive_create_gt_keypoints(traj_prediction):
    """交互式创建ground truth keypoints"""

    print("=== Interactive GT Keypoints Creation ===")
    print("Trajectory center:")
    traj_center = np.mean(traj_prediction.reshape(-1, 3), axis=0)
    print(f"  {traj_center}")

    print("\nOptions:")
    print("1. Use default GT keypoints (vertical axis)")
    print("2. Estimate from trajectory analysis")
    print("3. Enter custom coordinates")

    choice = input("Select option (1-3): ").strip()

    if choice == '1':
        return create_default_gt_keypoints()

    elif choice == '2':
        return create_gt_from_trajectory_analysis(traj_prediction)

    elif choice == '3':
        print("Enter coordinates for rotation axis points:")
        try:
            x1 = float(input(f"Point1 X coordinate: "))
            y1 = float(input(f"Point1 Y coordinate: "))
            z1 = float(input(f"Point1 Z coordinate: "))

            x2 = float(input(f"Point2 X coordinate: "))
            y2 = float(input(f"Point2 Y coordinate: "))
            z2 = float(input(f"Point2 Z coordinate: "))

            gt_data = {
                'point1': np.array([x1, y1, z1]),
                'point2': np.array([x2, y2, z2]),
                'description': 'Custom user-defined rotation axis',
                'object_type': 'custom'
            }

            return gt_data

        except ValueError:
            print("Invalid input, using default GT keypoints")
            return create_default_gt_keypoints()

    else:
        print("Invalid choice, using default GT keypoints")
        return create_default_gt_keypoints()

def main():
    """主函数"""

    print("=== GT Keypoints Creation Tool ===")

    try:
        # 加载现有数据
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        cam_pc = point_cloud["points"]
        traj_prediction = pred_results["traj_prediction"][0]

        print(f"Loaded trajectory data: {traj_prediction.shape}")
        print(f"Loaded point cloud: {cam_pc.shape}")

        # 创建或加载GT keypoints
        gt_file_path = 'pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints.pkl'

        if os.path.exists(gt_file_path):
            print(f"\nFound existing GT keypoints at {gt_file_path}")
            overwrite = input("Overwrite existing GT keypoints? (y/n): ").lower()

            if overwrite != 'y':
                print("Keeping existing GT keypoints")
                return

        # 创建新的GT keypoints
        gt_data = interactive_create_gt_keypoints(traj_prediction)

        # 保存GT keypoints
        with open(gt_file_path, 'wb') as f:
            pickle.dump(gt_data, f)

        print(f"\n✅ GT keypoints saved to {gt_file_path}")
        print(f"Point1: {gt_data['point1']}")
        print(f"Point2: {gt_data['point2']}")
        print(f"Description: {gt_data['description']}")

        # 显示验证提示
        print(f"\nYou can now run: python verify_gt_trajectory.py")
        print("to verify the algorithm with these GT keypoints.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()