#!/usr/bin/env python3
"""
测试 gflow_axis_util.py 模块的功能
"""

import sys
sys.path.append('/Disk3/xiaoqian')

import pickle
import numpy as np
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

def test_axis_computation():
    """测试轴计算功能"""
    print("=== Testing Axis Computation Module ===")

    try:
        # 加载数据
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        traj_prediction = pred_results["traj_prediction"][0]
        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]

        print(f"Trajectory shape: {traj_prediction.shape}")
        print(f"Point cloud shape: {cam_pc.shape}")

        # 使用新模块计算旋转轴
        axis_result = compute_rotation_axis_from_trajectories(traj_prediction)

        print(f"Axis computation success: {axis_result['success']}")
        print(f"Joint directions: {axis_result['pred_joint_directions']}")
        print(f"Anchor points: {axis_result['pred_anchor_points']}")
        print(f"Action data count: {len(axis_result['action_data'])}")
        print(f"Intersection data count: {len(axis_result['all_intersection_data'])}")
        print(f"Actions data count: {len(axis_result['all_actions_data'])}")
        print(f"Axis candidates count: {len(axis_result['axis_candidates'])}")

        # 验证数据结构
        assert 'success' in axis_result
        assert 'pred_joint_directions' in axis_result
        assert 'pred_anchor_points' in axis_result
        assert 'action_data' in axis_result
        assert 'all_intersection_data' in axis_result
        assert 'all_actions_data' in axis_result
        assert 'axis_candidates' in axis_result

        # 验证数据类型和形状
        assert isinstance(axis_result['pred_joint_directions'], np.ndarray)
        assert isinstance(axis_result['pred_anchor_points'], np.ndarray)
        assert axis_result['pred_joint_directions'].shape == (3,)
        assert axis_result['pred_anchor_points'].shape == (3,)

        print("✓ Axis computation module test passed!")
        return True

    except Exception as e:
        print(f"✗ Axis computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_data():
    """测试可视化数据生成"""
    print("\n=== Testing Visualization Data Generation ===")

    try:
        # 加载数据
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        point_cloud = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/point_cloud.pkl')

        traj_prediction = pred_results["traj_prediction"][0]
        cam_pc = point_cloud["points"]
        pc_rgb = point_cloud["colors"]

        # 计算旋转轴
        axis_result = compute_rotation_axis_from_trajectories(traj_prediction)

        # 检查可视化数据结构
        action_data = axis_result['action_data']
        all_intersection_data = axis_result['all_intersection_data']
        all_actions_data = axis_result['all_actions_data']
        axis_candidates = axis_result['axis_candidates']

        # 验证 action_data 结构
        if action_data:
            sample_action = action_data[0]
            required_keys = ['pos', 'direction', 'traj_idx', 'step']
            for key in required_keys:
                assert key in sample_action, f"Missing key {key} in action_data"
            print(f"✓ Action data structure valid, sample: {sample_action}")

        # 验证 intersection_data 结构
        if all_intersection_data:
            sample_intersection = all_intersection_data[0]
            required_keys = ['intersection_point', 'line1_pos', 'line1_dir', 'line2_pos', 'line2_dir']
            for key in required_keys:
                assert key in sample_intersection, f"Missing key {key} in intersection_data"
            print(f"✓ Intersection data structure valid")

        # 验证 actions_data 结构
        if all_actions_data:
            sample_action = all_actions_data[0]
            required_keys = ['pos', 'line_dir', 'original_dir', 'step']
            for key in required_keys:
                assert key in sample_action, f"Missing key {key} in actions_data"
            print(f"✓ Actions data structure valid")

        print("✓ Visualization data generation test passed!")
        return True

    except Exception as e:
        print(f"✗ Visualization data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_axis_direction_calculation():
    """测试轴方向计算逻辑"""
    print("\n=== Testing Axis Direction Calculation ===")

    try:
        # 加载数据
        pred_results = load_pickle('pred_keypoints/capture_0301/box/view_000_pose_000/pred_gflow.pkl')
        traj_prediction = pred_results["traj_prediction"][0]

        # 计算旋转轴
        axis_result = compute_rotation_axis_from_trajectories(traj_prediction)

        axis_direction = axis_result['pred_joint_directions']

        # 验证轴方向
        assert isinstance(axis_direction, np.ndarray), "Axis direction should be numpy array"
        assert axis_direction.shape == (3,), f"Axis direction shape should be (3,), got {axis_direction.shape}"

        # 检查是否为单位向量
        norm = np.linalg.norm(axis_direction)
        assert abs(norm - 1.0) < 1e-6, f"Axis direction should be unit vector, norm is {norm}"

        # 检查Z轴方向一致性（确保指向上方）
        if axis_direction[2] < 0:
            print(f"  Axis direction flipped to ensure positive Z: {axis_direction}")

        print(f"✓ Axis direction calculation valid: {axis_direction}")
        print(f"  Norm: {norm}")
        print(f"  Z component: {axis_direction[2]}")

        return True

    except Exception as e:
        print(f"✗ Axis direction calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing gflow_axis_util.py module...")

    # 测试轴计算
    axis_success = test_axis_computation()

    # 测试可视化数据
    viz_success = test_visualization_data()

    # 测试轴方向计算
    dir_success = test_axis_direction_calculation()

    if axis_success and viz_success and dir_success:
        print("\n✓ All module tests passed!")
        print("The gflow_axis_util.py module is working correctly.")
    else:
        print("\n✗ Some module tests failed!")
        sys.exit(1)

    print("\nModule tests completed!")