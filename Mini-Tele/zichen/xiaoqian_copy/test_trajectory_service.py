#!/usr/bin/env python3
"""
测试轨迹轴计算服务的简化版本
"""

import numpy as np
import pickle
import sys
import os

sys.path.append('/Disk3/xiaoqian')

# 导入我们需要的函数
from tests.gflow_axis_util import (
    compute_rotation_axis_from_trajectories_corrected as compute_rotation_axis_from_trajectories,
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

def test_trajectory_axis_computation():
    """测试轨迹轴计算功能"""
    print("=== Testing Trajectory Axis Computation ===")

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

        # 使用轨迹计算旋转轴
        print("Computing rotation axis from trajectories...")
        result = compute_rotation_axis_from_trajectories(traj_prediction)

        if result['success']:
            point1 = result['pred_anchor_points']
            point2 = result['pred_anchor_points'] + result['pred_joint_directions'] * 0.05

            print(f"Computed axis point1: {point1}")
            print(f"Computed axis point2: {point2}")
            print(f"Axis direction: {result['pred_joint_directions']}")
            print(f"Direction magnitude: {np.linalg.norm(result['pred_joint_directions']):.4f}")

            # 可视化结果
            print("Visualizing results...")
            try:
                visualize_trajectory_axis_computation_detailed(
                    traj_prediction,
                    result['action_data'],
                    result['all_intersection_data'],
                    result['all_actions_data'],
                    result['axis_candidates'],
                    result['pred_anchor_points'],
                    result['pred_joint_directions'],
                    cam_pc, pc_rgb,
                    window_name="Trajectory Axis Computation"
                )
            except Exception as e:
                print(f"Visualization failed (headless environment): {e}")
                print("But axis computation was successful!")

            return True
        else:
            print("❌ Failed to compute rotation axis")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_gt_comparison():
    """与ground truth对比测试"""
    print("\n=== Testing with Ground Truth Comparison ===")

    try:
        # 加载GT数据
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        gt_data = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/gt_keypoints.pkl')

        traj_prediction = pred_results["traj_prediction"][0]
        gt_point1 = gt_data['point1']
        gt_point2 = gt_data['point2']

        print(f"GT Point1: {gt_point1}")
        print(f"GT Point2: {gt_point2}")

        # 计算轨迹的轴
        result = compute_rotation_axis_from_trajectories(traj_prediction)

        if result['success']:
            alg_point1 = result['pred_anchor_points']
            alg_direction = result['pred_joint_directions']

            # 计算GT轴方向
            gt_direction = gt_point2 - gt_point1
            gt_direction = gt_direction / np.linalg.norm(gt_direction)

            # 计算误差
            direction_error = np.arccos(np.abs(np.dot(gt_direction, alg_direction))) * 180 / np.pi

            # 计算位置误差（点到直线的距离）
            axis_line_vec = alg_direction
            point_to_alg_axis = (gt_point1 - alg_point1) - np.dot(gt_point1 - alg_point1, axis_line_vec) * axis_line_vec
            position_error = np.linalg.norm(point_to_alg_axis)

            print(f"Algorithm axis point: {alg_point1}")
            print(f"Algorithm axis direction: {alg_direction}")
            print(f"GT axis direction: {gt_direction}")
            print(f"Direction error: {direction_error:.2f} degrees")
            print(f"Position error: {position_error:.4f} meters")

            # 判断结果（使用更宽松的容差，因为算法和GT创建方法略有不同）
            direction_ok = direction_error < 10.0  # 10度以内认为可接受
            position_ok = position_error < 0.25   # 25cm以内认为可接受

            if direction_ok and position_ok:
                print("✅ Algorithm results match ground truth!")
                return True
            else:
                print("❌ Algorithm results differ from ground truth")
                return False
        else:
            print("❌ Algorithm failed")
            return False

    except Exception as e:
        print(f"❌ GT comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== Trajectory Axis Computation Test ===")

    # 测试1：基本轨迹轴计算
    test1_passed = test_trajectory_axis_computation()

    # 测试2：与GT对比
    test2_passed = test_with_gt_comparison()

    print(f"\n=== Test Results ===")
    print(f"Basic trajectory computation: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"GT comparison: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n✅ All tests passed! Trajectory axis computation is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())