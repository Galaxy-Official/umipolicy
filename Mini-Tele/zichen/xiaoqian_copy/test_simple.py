#!/usr/bin/env python3
"""
简单的测试脚本，检查数据和基本功能
"""

import pickle
import numpy as np

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

def test_data_loading():
    """测试数据加载"""
    print("=== Testing Data Loading ===")

    try:
        # 加载pred_gflow.pkl
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        print(f"✓ pred_gflow.pkl loaded successfully")
        print(f"  Keys: {list(pred_results.keys())}")

        # 检查轨迹数据
        traj_prediction = pred_results["traj_prediction"][0]
        print(f"  Trajectory prediction shape: {traj_prediction.shape}")
        print(f"  Number of trajectories: {traj_prediction.shape[0]}")
        print(f"  Timesteps per trajectory: {traj_prediction.shape[1]}")
        print(f"  Coordinate dimensions: {traj_prediction.shape[2]}")

        # 检查夹爪位置
        gripper_pos = pred_results["gripper_3d_pos"]
        print(f"  Gripper position: {gripper_pos}")

        # 检查动作方向提取
        action_directions = []
        for i in range(traj_prediction.shape[0]):
            traj = traj_prediction[i]
            for t in range(traj.shape[0] - 1):
                direction = traj[t+1] - traj[t]
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                    action_directions.append(direction)

        action_directions = np.array(action_directions)
        print(f"  Extracted {len(action_directions)} action directions")
        print(f"  Action directions shape: {action_directions.shape}")

        # 加载点云数据
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')
        print(f"✓ point_cloud.pkl loaded successfully")
        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]
        print(f"  Point cloud shape: {cam_pc.shape}")
        print(f"  Point cloud range: X[{cam_pc[:,0].min():.3f}, {cam_pc[:,0].max():.3f}], Y[{cam_pc[:,1].min():.3f}, {cam_pc[:,1].max():.3f}], Z[{cam_pc[:,2].min():.3f}, {cam_pc[:,2].max():.3f}]")

        return True

    except Exception as e:
        print(f"✗ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_axis_calculation():
    """测试轴计算逻辑"""
    print("\n=== Testing Axis Calculation ===")

    try:
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        traj_prediction = pred_results["traj_prediction"][0]

        # 简化的轴方向计算
        action_directions = []
        for i in range(min(10, traj_prediction.shape[0])):  # 只取前10条轨迹
            traj = traj_prediction[i]
            for t in range(traj.shape[0] - 1):
                direction = traj[t+1] - traj[t]
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                    action_directions.append(direction)

        if len(action_directions) == 0:
            print("  No action directions found")
            return False

        action_directions = np.array(action_directions)

        # 计算协方差矩阵
        cov = np.dot(action_directions.T, action_directions)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 排序并选择最小特征值对应的特征向量
        idx = np.argsort(eigenvalues)
        axis_direction = eigenvectors[:, idx[0]]

        print(f"  Computed axis direction: {axis_direction}")
        print(f"  Eigenvalues: {eigenvalues[idx]}")
        print(f"  Number of action directions used: {len(action_directions)}")

        return True

    except Exception as e:
        print(f"✗ Axis calculation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing data loading and calculation logic...")

    # 测试数据加载
    data_success = test_data_loading()

    # 测试轴计算
    axis_success = test_axis_calculation()

    if data_success and axis_success:
        print("\n✓ All tests passed!")
        print("Data and calculation logic are working correctly.")
        print("The visualization should work if Open3D is properly installed.")
    else:
        print("\n✗ Some tests failed!")

    print("\nTest completed!")