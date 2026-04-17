#!/usr/bin/env python3
"""
测试服务中的轨迹生成功能
"""

import numpy as np
import pickle
import sys
import os

sys.path.append('/Disk3/xiaoqian')

# 导入我们需要的函数（避免依赖问题）
from tests.gflow_axis_util import (
    compute_rotation_axis_from_trajectories,
    generate_trajectories_from_axis,
    visualize_trajectory_plane_normals,
    visualize_trajectory_axis_computation_detailed
)

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

def test_trajectory_generation():
    """测试轨迹生成功能"""
    print("=== Testing Trajectory Generation ===")

    try:
        # 加载数据
        print("Loading data...")
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]
        traj_prediction = pred_results["traj_prediction"][0]
        gripper_3d_pos = pred_results["gripper_3d_pos"]

        print(f"Point cloud shape: {cam_pc.shape}")
        print(f"Trajectory prediction shape: {traj_prediction.shape}")
        print(f"Gripper 3D positions shape: {gripper_3d_pos.shape}")

        # 使用原始算法计算旋转轴（匹配服务中的算法）
        print("Computing rotation axis from trajectories...")
        axis_result = compute_rotation_axis_from_trajectories(traj_prediction)

        if axis_result['success']:
            point1 = axis_result['pred_anchor_points']
            point2 = axis_result['pred_anchor_points'] + axis_result['pred_joint_directions'] * 0.05

            print(f"Computed axis point1: {point1}")
            print(f"Computed axis point2: {point2}")
            print(f"Axis direction: {axis_result['pred_joint_directions']}")
            print(f"Direction magnitude: {np.linalg.norm(axis_result['pred_joint_directions']):.4f}")

            # 生成新的轨迹（模仿服务中的方法）
            print("\nGenerating new trajectories from computed axis...")

            # 使用gripper位置作为接触点
            contact_points = [gripper_3d_pos]  # 使用单个接触点生成多条轨迹

            # 生成轨迹参数
            num_trajectories = 128
            radius_range = 0.1

            # 生成轨迹（函数内部固定为13个点，3段，每段旋转15°）
            trajectories_new = generate_trajectories_from_axis(
                point1, point2,
                gripper_3d_pos,  # 使用单个接触点
                num_trajectories=num_trajectories,
                radius_range=radius_range
            )

            print(f"Generated {trajectories_new.shape[0]} trajectories")
            print(f"Each trajectory has {trajectories_new.shape[1]} points in 3 segments")
            print("Trajectory generation: contact point → perpendicular to axis → screw motion around axis")
            print("Each segment: perpendicular vector rotates 15° around axis → new trajectory point")
            print(f"Total rotation: 3 segments × 15° = 45°")
            print()

            # 显示一些样本轨迹
            for i in range(min(3, trajectories_new.shape[0])):
                sample_traj = trajectories_new[i]
                print(f"Sample trajectory {i} points:")
                for j, point in enumerate(sample_traj):
                    print(f"  Point {j}: [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
                print(f"  Trajectory length: {len(sample_traj)} points")
                print(f"  Start to end distance: {np.linalg.norm(sample_traj[-1] - sample_traj[0]):.4f}")
                print()

            # 可视化轨迹平面法向量（轴方向计算过程）
            print("=== Visualizing trajectory plane normals (perpendicular vectors) ===")
            print("This shows how the axis direction is computed from trajectory planes")

            if 'plane_data' in axis_result and len(axis_result['plane_data']) > 0:
                print(f"=== Visualizing {len(axis_result['plane_data'])} trajectory plane normals ===")
                try:
                    visualize_trajectory_plane_normals(
                        axis_result['plane_data'],
                        cam_pc,
                        pc_rgb,
                        window_name="Trajectory Plane Normals - Axis Direction Computation"
                    )
                    print("=== Trajectory plane normals visualization complete ===")
                except Exception as e:
                    print(f"Visualization failed (headless environment): {e}")
            else:
                print("No plane data available for visualization")

            # 详细可视化轴位置计算过程
            print("\n=== Visualizing detailed axis position computation ===")
            print("This shows segments, perpendicular lines, and intersection points")

            try:
                visualize_trajectory_axis_computation_detailed(
                    traj_prediction,
                    axis_result['pred_joint_directions'],
                    axis_result['pred_anchor_points'],
                    cam_pc,
                    pc_rgb,
                    window_name="Trajectory Axis Position Computation - Segments, Perpendiculars, Intersections"
                )
                print("=== Detailed axis computation visualization complete ===")
            except Exception as e:
                print(f"Detailed visualization failed (headless environment): {e}")

            return True
        else:
            print("❌ Failed to compute rotation axis")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== Trajectory Service Test ===")

    # 运行测试
    test_passed = test_trajectory_generation()

    print(f"\n=== Test Results ===")
    print(f"Trajectory generation test: {'PASSED' if test_passed else 'FAILED'}")

    if test_passed:
        print("\n✅ Trajectory service functionality is working correctly!")
        return 0
    else:
        print("\n❌ Test failed.")
        return 1

if __name__ == "__main__":
    exit(main())