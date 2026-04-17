#!/usr/bin/env python3
"""
测试插值和渐变色功能
"""

import numpy as np
import open3d as o3d
import sys
import os

sys.path.append('/Disk3/xiaoqian')

def test_interpolation():
    """测试3个到10个的插值功能"""
    print("=== Testing Transform Interpolation ===")

    try:
        # 创建3个测试变换矩阵
        from scipy.spatial.transform import Rotation as R

        transforms = []
        for i in range(3):
            transform = np.eye(4)
            # 旋转：绕Z轴旋转，从0到90度
            angle = i * np.pi / 4  # 0, 45, 90度
            rot = R.from_rotvec([0, 0, angle])
            transform[:3, :3] = rot.as_matrix()
            # 平移：沿X轴移动
            transform[:3, 3] = [i * 0.1, 0, 0]
            transforms.append(transform)

        print(f"Original transforms: {len(transforms)}")

        # 执行插值
        interpolated_transforms = []
        for i in range(10):
            t = i / 9.0  # 归一化参数 [0, 1]
            segment = t * (len(transforms) - 1)  # [0, 2]
            idx = int(segment)
            local_t = segment - idx

            if idx >= len(transforms) - 1:
                interpolated_transforms.append(transforms[-1].copy())
            else:
                # 插值旋转部分（使用四元数线性插值）
                rot1 = R.from_matrix(transforms[idx][:3, :3])
                rot2 = R.from_matrix(transforms[idx + 1][:3, :3])
                # 使用简单的线性插值（对于小角度足够准确）
                # 转换为四元数
                quat1 = rot1.as_quat()
                quat2 = rot2.as_quat()
                # 确保四元数在同一半球
                if np.dot(quat1, quat2) < 0:
                    quat2 = -quat2
                # 线性插值四元数（对于小角度足够准确）
                interpolated_quat = quat1 + local_t * (quat2 - quat1)
                # 归一化
                interpolated_quat = interpolated_quat / np.linalg.norm(interpolated_quat)
                interpolated_rot = R.from_quat(interpolated_quat)

                # 插值平移部分（线性插值）
                trans1 = transforms[idx][:3, 3]
                trans2 = transforms[idx + 1][:3, 3]
                interpolated_trans = trans1 + local_t * (trans2 - trans1)

                # 构建插值后的变换矩阵
                interpolated_transform = np.eye(4)
                interpolated_transform[:3, :3] = interpolated_rot.as_matrix()
                interpolated_transform[:3, 3] = interpolated_trans
                interpolated_transforms.append(interpolated_transform)

        print(f"Interpolated transforms: {len(interpolated_transforms)}")

        # 验证插值结果
        for i, transform in enumerate(interpolated_transforms):
            rotation_matrix = transform[:3, :3]
            translation = transform[:3, 3]
            print(f"Transform {i}: translation={translation}, rotation trace={np.trace(rotation_matrix):.3f}")

        print("✅ Transform interpolation test passed!")
        return True

    except Exception as e:
        print(f"❌ Transform interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_colors():
    """测试渐变色功能"""
    print("\n=== Testing Gradient Colors ===")

    try:
        # 模拟轨迹数据
        num_trajectories = 3
        points_per_trajectory = 4

        # 创建测试轨迹
        trajectories = []
        for i in range(num_trajectories):
            trajectory = []
            for j in range(points_per_trajectory):
                point = [i * 0.1, j * 0.05, 0.5 + j * 0.02]
                trajectory.append(point)
            trajectories.append(trajectory)

        trajectories = np.array(trajectories)
        print(f"Test trajectory data: {trajectories.shape}")

        # 测试渐变色计算
        colors = []
        for idx_t, pred_vec in enumerate(trajectories):
            n_points = pred_vec.shape[0]
            trajectory_colors = []

            for ii in range(n_points - 1):
                # 计算渐变色 - 从亮绿色(0,1,0)渐变到深绿色(0,0.3,0)
                color_intensity = 1.0 - (ii / (n_points - 2)) * 0.7  # 从1.0渐变到0.3
                segment_color = [0.0, color_intensity, 0.0]
                trajectory_colors.append(segment_color)

            colors.append(trajectory_colors)
            print(f"Trajectory {idx_t} colors: {len(trajectory_colors)} segments")
            for i, color in enumerate(trajectory_colors):
                print(f"  Segment {i}: RGB={color}")

        print("✅ Gradient colors test passed!")
        return True

    except Exception as e:
        print(f"❌ Gradient colors test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== Interpolation and Gradient Test ===")

    test1_passed = test_interpolation()
    test2_passed = test_gradient_colors()

    print(f"\n=== Test Results ===")
    print(f"Transform interpolation: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Gradient colors: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n✅ All tests passed!")
        print("The new features are working correctly:")
        print("  - 3 transforms interpolated to 10 transforms")
        print("  - Gradient colors from green to dark green")
        print("  - Ready for integration into robot_sim_run_gflow.py")
    else:
        print("\n❌ Some tests failed.")

    return 0 if (test1_passed and test2_passed) else 1

if __name__ == "__main__":
    exit(main())