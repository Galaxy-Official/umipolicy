"""Extract TCP poses from data collection and convert to position+quaternion format.

This script reads TCP pose sequences from the data collection and converts them
to position and quaternion format that can be used in the IK GUI.
"""

import numpy as np
from pathlib import Path
import json


def transformation_matrix_to_position_quaternion(matrix):
    """Convert 4x4 transformation matrix to position and quaternion (w, x, y, z)."""
    # Extract position (translation)
    position = matrix[:3, 3]

    # Extract rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert rotation matrix to quaternion (w, x, y, z)
    # Using the method from scipy.spatial.transform.Rotation
    trace = np.trace(rotation_matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s

    quaternion = np.array([w, x, y, z])

    # Normalize quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)

    return position, quaternion


def extract_tcp_poses_from_data(data_path):
    """Extract TCP poses from the data collection directory."""
    data_path = Path(data_path)
    planning_dir = data_path / "planning"

    if not planning_dir.exists():
        print(f"Planning directory not found: {planning_dir}")
        return None

    # Load TCP poses
    tcp_poses_file = planning_dir / "tcp_poses.npy"
    if not tcp_poses_file.exists():
        print(f"TCP poses file not found: {tcp_poses_file}")
        return None

    tcp_poses = np.load(tcp_poses_file)
    print(f"Loaded TCP poses with shape: {tcp_poses.shape}")

    # Convert each pose to position and quaternion
    pose_sequence = []
    for i, pose_matrix in enumerate(tcp_poses):
        position, quaternion = transformation_matrix_to_position_quaternion(pose_matrix)
        pose_sequence.append({
            'timestamp': i,  # Simple index-based timestamp
            'position': position.tolist(),
            'quaternion': quaternion.tolist(),
            'matrix': pose_matrix.tolist()
        })

    return pose_sequence


def save_pose_sequence(pose_sequence, output_file):
    """Save pose sequence to JSON file."""
    output_data = {
        'pose_count': len(pose_sequence),
        'poses': pose_sequence
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved pose sequence to: {output_file}")


def print_pose_summary(pose_sequence):
    """Print a summary of the pose sequence."""
    if not pose_sequence:
        return

    print(f"\nPose Sequence Summary:")
    print(f"Number of poses: {len(pose_sequence)}")

    # First pose
    first_pose = pose_sequence[0]
    print(f"\nFirst pose:")
    print(f"  Position: [{first_pose['position'][0]:.4f}, {first_pose['position'][1]:.4f}, {first_pose['position'][2]:.4f}]")
    print(f"  Quaternion: [{first_pose['quaternion'][0]:.4f}, {first_pose['quaternion'][1]:.4f}, {first_pose['quaternion'][2]:.4f}, {first_pose['quaternion'][3]:.4f}]")

    # Last pose
    last_pose = pose_sequence[-1]
    print(f"\nLast pose:")
    print(f"  Position: [{last_pose['position'][0]:.4f}, {last_pose['position'][1]:.4f}, {last_pose['position'][2]:.4f}]")
    print(f"  Quaternion: [{last_pose['quaternion'][0]:.4f}, {last_pose['quaternion'][1]:.4f}, {last_pose['quaternion'][2]:.4f}, {last_pose['quaternion'][3]:.4f}]")

    # Position range
    positions = np.array([pose['position'] for pose in pose_sequence])
    print(f"\nPosition ranges:")
    print(f"  X: {positions[:, 0].min():.4f} to {positions[:, 0].max():.4f}")
    print(f"  Y: {positions[:, 1].min():.4f} to {positions[:, 1].max():.4f}")
    print(f"  Z: {positions[:, 2].min():.4f} to {positions[:, 2].max():.4f}")


def main():
    """Main function to extract and display TCP poses."""
    # Path to your data
    data_path = "/Disk3/xiaoqian/collect_data/data/kp/2025-10-26_19:29:53"

    # Extract pose sequence
    pose_sequence = extract_tcp_poses_from_data(data_path)

    if pose_sequence is None:
        return

    # Print summary
    print_pose_summary(pose_sequence)

    # Save to JSON file
    output_file = "tcp_pose_sequence.json"
    save_pose_sequence(pose_sequence, output_file)

    # Also create a simple CSV format for easy import
    csv_file = "tcp_pose_sequence.csv"
    with open(csv_file, 'w') as f:
        f.write("timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z\n")
        for pose in pose_sequence:
            pos = pose['position']
            quat = pose['quaternion']
            f.write(f"{pose['timestamp']},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f},{quat[0]:.6f},{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f}\n")

    print(f"Also saved CSV format to: {csv_file}")

    # Create a simple playback script snippet
    print(f"\nTo use these poses in your IK GUI, you can copy-paste these values:")
    print(f"\nFirst pose (Home position):")
    first_pose = pose_sequence[0]
    print(f"Position: {first_pose['position']}")
    print(f"Quaternion: {first_pose['quaternion']}")

    print(f"\nLast pose (Target position):")
    last_pose = pose_sequence[-1]
    print(f"Position: {last_pose['position']}")
    print(f"Quaternion: {last_pose['quaternion']}")


if __name__ == "__main__":
    main()