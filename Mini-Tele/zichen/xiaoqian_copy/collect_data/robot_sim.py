import time
import numpy as np
import torch
import transformations as tf
import os
import json
import sys
import csv
from flexiv import *
from utilities.pose_util import *
from utilities.flexivutils import parse_pt_states

#from frankx import Affine, JointMotion, Robot, Waypoint, WaypointMotion, Gripper, LinearRelativeMotion, LinearMotion, ImpedanceMotion

# sys.path.append('/home/rhos/xiaoyang/flexiv_api/lib_py')
sys.path.append('/Disk3/xiaoqian/flexiv_api/lib_py')
import flexivrdk

# 全局机器人实例，将在main()中根据参数初始化
robot = None

def read_joint_sequence(csv_file):
    """Read joint sequence from CSV file"""
    joint_sequences = []
    timestamps = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = int(row['timestamp'])
            joints = [
                float(row['joint_0']),
                float(row['joint_1']),
                float(row['joint_2']),
                float(row['joint_3']),
                float(row['joint_4']),
                float(row['joint_5']),
                float(row['joint_6'])
            ]
            timestamps.append(timestamp)
            joint_sequences.append(np.array(joints))

    return timestamps, joint_sequences

def read_tcp_positions(csv_file):
    """Read TCP positions from CSV file"""
    tcp_positions = []
    timestamps = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = int(row['timestamp'])
            tcp_pose = [
                float(row['pos_x']),
                float(row['pos_y']),
                float(row['pos_z']),
                float(row['quat_w']),
                float(row['quat_x']),
                float(row['quat_y']),
                float(row['quat_z'])
            ]
            timestamps.append(timestamp)
            tcp_positions.append(np.array(tcp_pose))

    return timestamps, tcp_positions

def convert_quat_wxyz_to_zyxw(tcp_pose):
    """Convert quaternion from wxyz to zyxw format for calc_ik"""
    # tcp_pose: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    # calc_ik expects: [pos_x, pos_y, pos_z, quat_z, quat_y, quat_x, quat_w]
    return np.array([
        tcp_pose[0],  # pos_x
        tcp_pose[1],  # pos_y
        tcp_pose[2],  # pos_z
        tcp_pose[6],  # quat_z
        tcp_pose[5],  # quat_y
        tcp_pose[4],  # quat_x
        tcp_pose[3]   # quat_w
    ])

def verify_tcp_consistency(joints, expected_tcp, tolerance=1e-4, visualize=True, sleep_time=0.0001):
    """Verify that computed TCP matches expected TCP"""
    robot.set_joints(joints)
    computed_tcp = robot.get_tcp_catersian()

    # Convert to same format: [x, y, z, qw, qx, qy, qz]
    computed_tcp_reordered = np.array([
        computed_tcp[0],  # pos_x
        computed_tcp[1],  # pos_y
        computed_tcp[2],  # pos_z
        computed_tcp[6],  # quat_w
        computed_tcp[3],  # quat_x
        computed_tcp[4],  # quat_y
        computed_tcp[5]   # quat_z
    ])

    # Check position difference
    pos_diff = np.abs(computed_tcp_reordered[:3] - expected_tcp[:3])
    max_pos_diff = np.max(pos_diff)

    # Check quaternion difference (consider double cover property)
    quat_diff1 = np.abs(computed_tcp_reordered[3:] - expected_tcp[3:])
    quat_diff2 = np.abs(computed_tcp_reordered[3:] + expected_tcp[3:])  # Antipodal case
    max_quat_diff = min(np.max(quat_diff1), np.max(quat_diff2))

    return max_pos_diff < tolerance and max_quat_diff < tolerance, max_pos_diff, max_quat_diff

def process_joint_to_tcp(visualize=True, sleep_time=0.0001):
    """Original main function: process joints to get TCP positions"""
    # Read joint sequences from CSV
    csv_input = '/Disk3/xiaoqian/pyroki/examples/joint_sequence.csv'
    timestamps, joint_sequences = read_joint_sequence(csv_input)

    print(f"Loaded {len(joint_sequences)} joint configurations")

    # Process each joint configuration
    tcp_positions = []

    for i, (_, joints) in enumerate(zip(timestamps, joint_sequences)):
        # Set robot joints
        robot.set_joints(joints)

        # Get flange cartesian position (no reordering needed)
        tcp_pos = robot.get_flange_catersian()
        tcp_positions.append(tcp_pos)

        # Optional: visualize and simulate
        if visualize:
            robot.visualize()
            time.sleep(sleep_time)
            if hasattr(robot, 'p'):
                robot.p.stepSimulation()

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(joint_sequences)} configurations")

    # Save TCP positions to CSV
    csv_output = '/Disk3/xiaoqian/pyroki/examples/tcp_positions.csv'
    save_tcp_positions(csv_output, timestamps, tcp_positions)

    print(f"TCP positions saved to {csv_output}")
    print(f"Total configurations processed: {len(tcp_positions)}")

    return timestamps, tcp_positions, joint_sequences

def compare_joint_sequences(original_file, computed_file, tolerance=1e-4):
    """Compare two joint sequence files to analyze differences"""
    print(f"\n=== Comparing joint sequences ===")
    print(f"Original: {original_file}")
    print(f"Computed: {computed_file}")
    print(f"Tolerance: {tolerance}")

    # Read original joints
    orig_timestamps, orig_joints = read_joint_sequence(original_file)

    # Read computed joints
    comp_timestamps, comp_joints = read_joint_sequence(computed_file)

    if len(orig_joints) != len(comp_joints):
        print(f"ERROR: Different number of configurations!")
        print(f"Original: {len(orig_joints)}, Computed: {len(comp_joints)}")
        return

    max_diff = 0
    max_diff_idx = 0
    total_diff = 0
    num_differences = 0

    for i, (orig_joint, comp_joint) in enumerate(zip(orig_joints, comp_joints)):
        # Calculate difference for each joint
        diff = np.abs(orig_joint - comp_joint)
        max_joint_diff = np.max(diff)

        if max_joint_diff > tolerance:
            num_differences += 1
            total_diff += np.sum(diff)

            if max_joint_diff > max_diff:
                max_diff = max_joint_diff
                max_diff_idx = i

            if num_differences <= 5:  # Show first 5 differences
                print(f"Config {i}: Max diff = {max_joint_diff:.6f}")
                print(f"  Original: {[f'{x:.6f}' for x in orig_joint]}")
                print(f"  Computed: {[f'{x:.6f}' for x in comp_joint]}")
                print(f"  Diff:     {[f'{x:.6f}' for x in diff]}")
                print()

    print(f"Summary:")
    print(f"  Total configurations: {len(orig_joints)}")
    print(f"  Configurations with differences > {tolerance}: {num_differences}")
    print(f"  Maximum difference: {max_diff:.6f} at config {max_diff_idx}")
    print(f"  Average total difference: {total_diff/len(orig_joints):.6f}")

    if num_differences == 0:
        print("✓ All configurations match within tolerance!")
    else:
        print(f"⚠ Found {num_differences} configurations with differences")

def check_joint_limits(joints):
    """Check if joints are within reasonable limits for Rizon 4"""
    # Approximate joint limits for Rizon 4 (in radians)
    # These are rough estimates - adjust based on actual robot specs
    joint_limits = [
        (-2.0, 2.0),    # joint 0
        (-1.5, 1.5),    # joint 1
        (-2.0, 2.0),    # joint 2
        (-0.5, 3.5),    # joint 3
        (-2.0, 2.0),    # joint 4
        (-1.5, 1.5),    # joint 5
        (-2.0, 2.0)     # joint 6
    ]

    violations = []
    for i, (joint, (min_limit, max_limit)) in enumerate(zip(joints, joint_limits)):
        if joint < min_limit or joint > max_limit:
            violations.append(f"Joint {i}: {joint:.3f} not in [{min_limit}, {max_limit}]")

    return len(violations) == 0, violations

def find_closest_ik_solution(tcp_pose_zyxw, reference_joints):
    """Find IK solution closest to reference joints"""
    # For now, just use the standard IK solver
    # The Flexiv API may not support initial guess
    joint_config = robot.calc_ik(tcp_pose_zyxw, is_tcp=True)

    if joint_config is not None:
        # Check joint limits
        valid, violations = check_joint_limits(joint_config)
        if not valid:
            print(f"IK solution violates joint limits: {violations}")
            # Try to find a valid solution by adjusting the pose slightly
            # This is a simplified approach
            return joint_config  # Return anyway for now

    return joint_config

def process_tcp_to_joint(visualize=True, sleep_time=0.0001):
    """Process TCP positions to get joint configurations using calc_ik"""
    # Read TCP positions from CSV
    csv_input = '/Disk3/xiaoqian/pyroki/examples/tcp_positions.csv'
    timestamps, tcp_positions = read_tcp_positions(csv_input)

    print(f"Loaded {len(tcp_positions)} TCP positions")

    # Process each TCP position
    joint_sequences = []
    ik_failures = 0

    for i, (_, tcp_pose) in enumerate(zip(timestamps, tcp_positions)):
        # Calculate inverse kinematics for flange pose
        joint_config = robot.calc_ik(tcp_pose, is_tcp=False)

        if joint_config is None:
            ik_failures += 1
            print(f"Warning: IK failed for configuration {i}")
            # Use reference joints as fallback
            # joint_config = reference_joints if reference_joints is not None else np.zeros(7)

        joint_sequences.append(joint_config)

        # # Check if solution is close to reference
        # if reference_joints is not None:
        #     diff = np.linalg.norm(joint_config - reference_joints)
        #     if diff > 0.1:  # More than ~5.7 degrees
        #         large_differences += 1
        #         if large_differences <= 3:  # Show first few large differences
        #             print(f"Config {i}: Large IK difference = {diff:.4f} radians")
        #             print(f"  Reference: {[f'{x:.4f}' for x in reference_joints]}")
        #             print(f"  IK Result: {[f'{x:.4f}' for x in joint_config]}")

        # Set robot joints to visualize
        robot.set_joints(joint_config)

        # Optional: visualize and simulate
        if visualize:
            robot.visualize()
            time.sleep(sleep_time)
            if hasattr(robot, 'p'):
                robot.p.stepSimulation()

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(tcp_positions)} TCP positions")

    # Save joint sequences to CSV
    csv_output = '/Disk3/xiaoqian/pyroki/examples/joint_from_tcp.csv'
    save_joint_sequence(csv_output, timestamps, joint_sequences)

    print(f"Joint sequences saved to {csv_output}")
    print(f"Total TCP positions processed: {len(joint_sequences)}")
    print(f"IK failures: {ik_failures}")

    # Compare with original
    original_file_path = '/Disk3/xiaoqian/pyroki/examples/joint_sequence.csv'
    compare_joint_sequences(original_file_path, csv_output)

def save_joint_sequence(csv_file, timestamps, joint_sequences):
    """Save joint sequences to CSV file"""
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['timestamp', 'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])

        # Write data
        for timestamp, joints in zip(timestamps, joint_sequences):
            # Handle both list and numpy array, round to 6 decimal places
            if hasattr(joints, 'tolist'):
                joint_values = [round(val, 6) for val in joints.tolist()]
            else:
                joint_values = [round(val, 6) for val in joints]
            row = [timestamp] + joint_values
            writer.writerow(row)

def save_tcp_positions(csv_file, timestamps, tcp_positions):
    """Save TCP positions to CSV file with same format as joint sequence"""
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['timestamp', 'pos_x','pos_y','pos_z','quat_x','quat_y','quat_z','quat_w'])

        # Write data - using TCP positions as the "joint" values for consistency with format
        for timestamp, tcp_pos in zip(timestamps, tcp_positions):
            # TCP position stored as: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
            # Round to 6 decimal places for consistency
            rounded_tcp_pos = [round(val, 6) for val in tcp_pos]
            row = [timestamp] + rounded_tcp_pos
            writer.writerow(row)

def save_joint_sequence(csv_file, timestamps, joint_sequences):
    """Save joint sequences to CSV file"""
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['timestamp', 'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])

        # Write data
        for timestamp, joints in zip(timestamps, joint_sequences):
            # Handle both list and numpy array, round to 6 decimal places
            if hasattr(joints, 'tolist'):
                joint_values = [round(val, 6) for val in joints.tolist()]
            else:
                joint_values = [round(val, 6) for val in joints]
            row = [timestamp] + joint_values
            writer.writerow(row)

def main():
    # Read joint sequences from CSV
    csv_input = '/Disk3/xiaoqian/pyroki/examples/joint_sequence.csv'
    timestamps, joint_sequences = read_joint_sequence(csv_input)

    print(f"Loaded {len(joint_sequences)} joint configurations")

    # Process each joint configuration
    tcp_positions = []

    for i, (_, joints) in enumerate(zip(timestamps, joint_sequences)):
        # Set robot joints
        robot.set_joints(joints)

        # Get TCP cartesian position
        tcp_pos = robot.get_tcp_catersian()
        # Convert from [x, y, z, qx, qy, qz, qw] to [x, y, z, qw, qx, qy, qz]
        # tcp_pos_reordered = np.array([
        #     tcp_pos[0],  # pos_x
        #     tcp_pos[1],  # pos_y
        #     tcp_pos[2],  # pos_z
        #     tcp_pos[6],  # quat_w
        #     tcp_pos[3],  # quat_x
        #     tcp_pos[4],  # quat_y
        #     tcp_pos[5]   # quat_z
        # ])
        tcp_positions.append(tcp_pos)

        # Optional: visualize and simulate
        if visualize:
            robot.visualize()
            time.sleep(sleep_time)
            if hasattr(robot, 'p'):
                robot.p.stepSimulation()

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(joint_sequences)} configurations")

    # Save TCP positions to CSV
    csv_output = '/Disk3/xiaoqian/pyroki/examples/tcp_positions.csv'
    save_tcp_positions(csv_output, timestamps, tcp_positions)

    print(f"TCP positions saved to {csv_output}")
    print(f"Total configurations processed: {len(tcp_positions)}")

def main():
    """Main function to choose between joint-to-tcp and tcp-to-joint processing"""
    import argparse

    parser = argparse.ArgumentParser(description='Robot simulation: joint-to-tcp or tcp-to-joint conversion')
    parser.add_argument('--mode', choices=['joint2tcp', 'tcp2joint', 'verify'], default='joint2tcp',
                        help='Processing mode: joint2tcp (default), tcp2joint, or verify (full cycle)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable visualization (default: True)')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                        help='Disable visualization')
    parser.add_argument('--save-images', action='store_true',
                        help='Save visualization images instead of displaying')
    parser.add_argument('--sleep', type=float, default=0.0001,
                        help='Sleep time between steps (default: 0.0001)')

    args = parser.parse_args()

    # 初始化机器人，根据visualize参数决定是否使用headless模式
    global robot
    robot = Rizon4(headless=not args.visualize)

    if args.mode == 'joint2tcp':
        print("Running joint-to-TCP conversion...")
        process_joint_to_tcp(visualize=args.visualize, sleep_time=args.sleep)
    elif args.mode == 'tcp2joint':
        print("Running TCP-to-joint conversion...")
        process_tcp_to_joint(visualize=args.visualize, sleep_time=args.sleep)
    elif args.mode == 'verify':
        print("Running full verification cycle...")
        # Step 1: joint -> TCP
        print("\n=== Step 1: Joint to TCP ===")
        timestamps, tcp_positions, original_joints = process_joint_to_tcp(visualize=args.visualize, sleep_time=args.sleep)

        # Step 2: TCP -> joint
        print("\n=== Step 2: TCP to Joint ===")
        process_tcp_to_joint(visualize=args.visualize, sleep_time=args.sleep)

        # Step 3: Compare results
        print("\n=== Step 3: Final Comparison ===")
        original_file = '/Disk3/xiaoqian/pyroki/examples/joint_sequence.csv'
        computed_file = '/Disk3/xiaoqian/pyroki/examples/joint_from_tcp.csv'
        compare_joint_sequences(original_file, computed_file)
    else:
        print("Invalid mode. Use 'joint2tcp', 'tcp2joint', or 'verify'")

if __name__ == "__main__":
    main()

# for i in range(100):
#     tcp_pose = np.array([0.780481,-0.100740,0.294439,-0.426692,-0.068826,-0.034667,-0.901108]) # quat wxyz -> zyxw
#     tcp_pose = np.concatenate([tcp_pose[:3], [tcp_pose[6], tcp_pose[5], tcp_pose[4], tcp_pose[3]]])
#     joint_next = robot.calc_ik(tcp_pose, is_tcp=True)
#     print("joint_next",joint_next)
#     robot.set_joints(joint_next)
#     robot.visualize()
#     time.sleep(0.05)
#     p.stepSimulation()