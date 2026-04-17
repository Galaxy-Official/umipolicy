#!/usr/bin/env python3
"""
详细分析joint2tcp和tcp2joint过程中的数值误差
"""

import numpy as np
import sys
sys.path.append('/Disk3/xiaoqian')
from robot_sim import robot, read_joint_sequence, read_tcp_positions

def quaternion_distance(q1, q2):
    """计算两个四元数之间的距离，考虑双覆盖性质"""
    # q1和q2应该是[w, x, y, z]格式
    dot_product = np.abs(np.dot(q1, q2))
    # 四元数距离 = 2 * arccos(|dot_product|)
    return 2 * np.arccos(np.clip(dot_product, 0, 1))

def rotation_angle_from_quaternion_distance(quat_dist):
    """将四元数距离转换为旋转角度（度）"""
    return np.degrees(quat_dist)

def analyze_errors():
    """详细分析误差"""
    print("=== 详细数值误差分析 ===\n")

    # 读取数据
    orig_timestamps, orig_joints = read_joint_sequence('/Disk3/xiaoqian/pyroki/examples/joint_sequence.csv')
    comp_timestamps, comp_joints = read_joint_sequence('/Disk3/xiaoqian/pyroki/examples/joint_from_tcp.csv')
    tcp_timestamps, tcp_positions = read_tcp_positions('/Disk3/xiaoqian/pyroki/examples/tcp_positions.csv')

    print(f"分析配置数量: {min(len(orig_joints), len(comp_joints), len(tcp_positions))}")
    print()

    # 误差统计
    position_errors = []
    orientation_errors = []
    joint_errors = []
    quaternion_distances = []

    for i in range(min(10, len(orig_joints))):  # 分析前10个配置
        print(f"--- 配置 {i} ---")

        # 1. 关节角度误差
        joint_diff = np.abs(orig_joints[i] - comp_joints[i])
        joint_error = np.linalg.norm(joint_diff)
        joint_errors.append(joint_error)

        print(f"关节角度误差 (弧度): {joint_diff}")
        print(f"关节角度误差 (度): {np.degrees(joint_diff)}")
        print(f"总关节误差: {joint_error:.6f} 弧度 = {np.degrees(joint_error):.2f} 度")

        # 2. TCP位置误差
        robot.set_joints(orig_joints[i])
        orig_tcp = robot.get_tcp_catersian()  # [x,y,z,qx,qy,qz,qw]

        robot.set_joints(comp_joints[i])
        comp_tcp = robot.get_tcp_catersian()

        # 位置误差 (mm)
        pos_error = np.abs(np.array(orig_tcp[:3]) - np.array(comp_tcp[:3])) * 1000  # 转换为mm
        position_errors.append(pos_error)

        print(f"TCP位置误差 (mm): {pos_error}")
        print(f"TCP位置总误差: {np.linalg.norm(pos_error):.6f} mm")

        # 3. 姿态误差 (四元数)
        orig_quat = np.array([orig_tcp[6], orig_tcp[3], orig_tcp[4], orig_tcp[5]])  # [w,x,y,z]
        comp_quat = np.array([comp_tcp[6], comp_tcp[3], comp_tcp[4], comp_tcp[5]])

        # 四元数距离
        quat_dist = quaternion_distance(orig_quat, comp_quat)
        quaternion_distances.append(quat_dist)

        # 转换为旋转角度
        rot_angle = rotation_angle_from_quaternion_distance(quat_dist)
        orientation_errors.append(rot_angle)

        print(f"四元数距离: {quat_dist:.6f}")
        print(f"等效旋转角度: {rot_angle:.6f} 度")

        # 4. 相对误差
        orig_pos = np.array(orig_tcp[:3])
        comp_pos = np.array(comp_tcp[:3])
        relative_pos_error = np.linalg.norm(pos_error) / np.linalg.norm(orig_pos) * 100

        print(f"相对位置误差: {relative_pos_error:.6f}%")
        print()

    # 总体统计
    print("=== 总体误差统计 ===")

    position_errors = np.array(position_errors)
    orientation_errors = np.array(orientation_errors)
    joint_errors = np.array(joint_errors)
    quaternion_distances = np.array(quaternion_distances)

    print(f"关节角度误差统计:")
    print(f"  平均总误差: {np.mean(joint_errors):.6f} 弧度 = {np.degrees(np.mean(joint_errors)):.2f} 度")
    print(f"  最大总误差: {np.max(joint_errors):.6f} 弧度 = {np.degrees(np.max(joint_errors)):.2f} 度")
    print(f"  最小总误差: {np.min(joint_errors):.6f} 弧度 = {np.degrees(np.min(joint_errors)):.2f} 度")

    print(f"\nTCP位置误差统计 (mm):")
    print(f"  平均位置误差: {np.mean([np.linalg.norm(pe) for pe in position_errors]):.6f} mm")
    print(f"  最大位置误差: {np.max([np.linalg.norm(pe) for pe in position_errors]):.6f} mm")
    print(f"  最小位置误差: {np.min([np.linalg.norm(pe) for pe in position_errors]):.6f} mm")

    print(f"\n姿态误差统计:")
    print(f"  平均旋转角度误差: {np.mean(orientation_errors):.6f} 度")
    print(f"  最大旋转角度误差: {np.max(orientation_errors):.6f} 度")
    print(f"  最小旋转角度误差: {np.min(orientation_errors):.6f} 度")

    print(f"\n四元数距离统计:")
    print(f"  平均四元数距离: {np.mean(quaternion_distances):.6f}")
    print(f"  最大四元数距离: {np.max(quaternion_distances):.6f}")
    print(f"  最小四元数距离: {np.min(quaternion_distances):.6f}")

    # 评估误差是否可接受
    print(f"\n=== 误差评估 ===")

    # 机器人学中的典型精度要求
    pos_tolerance_mm = 0.1  # 0.1mm
    ori_tolerance_deg = 0.1  # 0.1度

    pos_acceptable = np.mean([np.linalg.norm(pe) for pe in position_errors]) < pos_tolerance_mm
    ori_acceptable = np.mean(orientation_errors) < ori_tolerance_deg

    print(f"位置精度要求: < {pos_tolerance_mm} mm")
    print(f"姿态精度要求: < {ori_tolerance_deg} 度")
    print()
    print(f"位置误差是否可接受: {'✓ 是' if pos_acceptable else '✗ 否'}")
    print(f"姿态误差是否可接受: {'✓ 是' if ori_acceptable else '✗ 否'}")

    if pos_acceptable and ori_acceptable:
        print("\n🎉 总体评估: 误差在可接受范围内")
    else:
        print("\n⚠️  总体评估: 误差超出可接受范围")
        if not pos_acceptable:
            print(f"   建议: 位置误差需要优化 (当前 {np.mean([np.linalg.norm(pe) for pe in position_errors]):.3f}mm)")
        if not ori_acceptable:
            print(f"   建议: 姿态误差需要优化 (当前 {np.mean(orientation_errors):.3f}°)")

if __name__ == "__main__":
    analyze_errors()