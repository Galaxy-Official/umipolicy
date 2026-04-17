#!/usr/bin/env python3
"""
测试可视化功能的简化脚本
"""

import pickle
import numpy as np
import open3d as o3d
import sys
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

def test_basic_visualization():
    """测试基本的可视化功能"""
    print("=== Testing Basic Visualization ===")

    # 加载数据
    try:
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]
        traj_prediction = pred_results["traj_prediction"][0]
        gripper_pos = pred_results["gripper_3d_pos"]

        print(f"Point cloud shape: {cam_pc.shape}")
        print(f"Trajectory shape: {traj_prediction.shape}")
        print(f"Gripper position: {gripper_pos}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    # 创建简单的可视化
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Test Visualization", width=800, height=600)

        # 添加点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cam_pc)
        pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
        vis.add_geometry(pcd)

        # 添加夹爪位置
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(gripper_pos)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        vis.add_geometry(sphere)

        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # 运行可视化
        print("Running test visualization... (Press Q to close)")
        vis.update_renderer()
        vis.poll_events()
        vis.run()
        vis.destroy_window()

        print("✓ Basic visualization test successful!")
        return True

    except Exception as e:
        print(f"✗ Visualization error: {e}")
        return False

def test_trajectory_visualization():
    """测试轨迹可视化"""
    print("\n=== Testing Trajectory Visualization ===")

    try:
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]
        traj_prediction = pred_results["traj_prediction"][0]

        print(f"Visualizing {traj_prediction.shape[0]} trajectories...")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Trajectory Test", width=800, height=600)

        # 添加点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cam_pc)
        pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
        vis.add_geometry(pcd)

        # 添加少量轨迹
        max_trajs = min(10, traj_prediction.shape[0])
        for i in range(max_trajs):
            trajectory = traj_prediction[i]

            # 为每条轨迹创建线条
            for t in range(trajectory.shape[0] - 1):
                point1 = trajectory[t]
                point2 = trajectory[t + 1]

                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([point1, point2])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])  # 绿色
                vis.add_geometry(line)

        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        print("Running trajectory visualization... (Press Q to close)")
        vis.update_renderer()
        vis.poll_events()
        vis.run()
        vis.destroy_window()

        print("✓ Trajectory visualization test successful!")
        return True

    except Exception as e:
        print(f"✗ Trajectory visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing visualization functionality...")

    # 测试基本可视化
    basic_success = test_basic_visualization()

    # 测试轨迹可视化
    traj_success = test_trajectory_visualization()

    if basic_success and traj_success:
        print("\n✓ All visualization tests passed!")
    else:
        print("\n✗ Some visualization tests failed!")
        sys.exit(1)

    print("\nVisualization tests completed!")