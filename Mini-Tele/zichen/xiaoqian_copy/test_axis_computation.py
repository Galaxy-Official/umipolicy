#!/usr/bin/env python3
"""
测试修复后的轴计算功能
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.append('/Disk3/xiaoqian')

# 测试导入
try:
    from tests.gflow_axis_util import (
        compute_trajectory_plane_normals,
        compute_rotation_axis_from_trajectories_corrected,
        visualize_trajectory_plane_normals
    )
    print("✅ 所有导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def create_test_trajectories():
    """创建测试轨迹数据"""
    # 创建简单的圆弧轨迹
    num_trajectories = 5
    points_per_trajectory = 4

    # 定义旋转轴
    axis_point = np.array([0, 0, 0])
    axis_direction = np.array([0, 0, 1])

    trajectories = np.zeros((num_trajectories, points_per_trajectory, 3))

    for i in range(num_trajectories):
        # 每个轨迹有不同的半径和起始角度
        radius = 0.1 + i * 0.02
        start_angle = i * np.pi / 8

        # 生成圆弧轨迹点
        for j in range(points_per_trajectory):
            angle = start_angle + j * np.pi / 12  # 每段15°

            # 在XY平面上的点
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.05 + j * 0.01  # 稍微向上倾斜

            point = np.array([x, y, z])
            trajectories[i, j] = point

    return trajectories

def test_plane_normal_computation():
    """测试平面法向量计算"""
    print("\n=== 测试平面法向量计算 ===")

    # 创建测试轨迹
    trajectories = create_test_trajectories()
    print(f"创建测试轨迹: {trajectories.shape}")

    # 计算平面法向量
    plane_normals, plane_data = compute_trajectory_plane_normals(trajectories)

    print(f"计算得到的平面法向量: {plane_normals.shape}")
    for i, normal in enumerate(plane_normals):
        print(f"  轨迹 {i}: 法向量 = {normal}, 模长 = {np.linalg.norm(normal):.4f}")

    # 验证法向量是否大致指向Z轴方向（因为我们创建了XY平面的圆弧）
    expected_normal = np.array([0, 0, 1])
    avg_normal = np.mean(plane_normals, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)

    print(f"\n平均法向量: {avg_normal}")
    print(f"期望法向量: {expected_normal}")

    # 计算角度差异
    cos_angle = np.dot(avg_normal, expected_normal)
    angle = np.arccos(np.abs(cos_angle)) * 180 / np.pi
    print(f"角度差异: {angle:.2f}°")

    if angle < 10.0:  # 小于10度认为正确
        print("✅ 平面法向量计算正确")
        return True
    else:
        print("❌ 平面法向量计算可能有误")
        return False

def test_corrected_axis_computation():
    """测试修正的轴计算"""
    print("\n=== 测试修正的轴计算 ===")

    # 创建测试轨迹
    trajectories = create_test_trajectories()

    # 使用修正的算法计算旋转轴
    result = compute_rotation_axis_from_trajectories_corrected(trajectories)

    if result['success']:
        axis_direction = result['pred_joint_directions']
        axis_position = result['pred_anchor_points']

        print(f"轴方向: {axis_direction}")
        print(f"轴位置: {axis_position}")
        print(f"轴方向模长: {np.linalg.norm(axis_direction):.4f}")

        # 验证轴方向是否接近Z轴
        expected_direction = np.array([0, 0, 1])
        cos_angle = np.dot(axis_direction, expected_direction)
        angle = np.arccos(np.abs(cos_angle)) * 180 / np.pi
        print(f"与期望方向角度差异: {angle:.2f}°")

        if angle < 15.0:  # 小于15度认为正确
            print("✅ 修正的轴计算正确")
            return True
        else:
            print("❌ 修正的轴计算可能有误")
            return False
    else:
        print("❌ 轴计算失败")
        return False

def main():
    """主测试函数"""
    print("=== 测试修复后的轴计算功能 ===")

    # 测试1：平面法向量计算
    test1_passed = test_plane_normal_computation()

    # 测试2：修正的轴计算
    test2_passed = test_corrected_axis_computation()

    print(f"\n=== 测试结果总结 ===")
    print(f"平面法向量计算: {'通过' if test1_passed else '失败'}")
    print(f"修正轴计算: {'通过' if test2_passed else '失败'}")

    if test1_passed and test2_passed:
        print("\n✅ 所有测试通过！轴计算功能修复成功。")
        return 0
    else:
        print("\n❌ 部分测试失败，需要进一步检查。")
        return 1

if __name__ == "__main__":
    exit(main())