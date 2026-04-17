"""Convert original TCP data to standard CSV format

This script reads TCP data from the original 00000/tcp/ directory and converts it
to the same format as tcp_poses_from_joints.csv with timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z
"""

import numpy as np
import glob
from pathlib import Path

def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle representation to quaternion (w, x, y, z)."""
    if len(axis_angle) != 4:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    # Format: [axis_x, axis_y, axis_z, angle]
    axis = axis_angle[:3]
    angle = axis_angle[3]

    # Normalize axis
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = axis / axis_norm

    # Convert to quaternion
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    quaternion = np.array([w, xyz[0], xyz[1], xyz[2]])

    # Normalize
    quaternion = quaternion / np.linalg.norm(quaternion)

    return quaternion

def convert_original_tcp_to_csv():
    """Convert original TCP data to standard CSV format"""
    # Source directory
    tcp_dir = Path("/Disk3/xiaoqian/collect_data/data/kp/2025-10-26_19:29:53/00000/tcp")

    if not tcp_dir.exists():
        print(f"Original TCP directory not found: {tcp_dir}")
        return

    # Output file
    output_csv = "original_tcp_poses.csv"

    # Get all TCP files
    tcp_files = sorted(glob.glob(str(tcp_dir / "*.npy")))

    if not tcp_files:
        print(f"No TCP files found in {tcp_dir}")
        return

    print(f"Found {len(tcp_files)} TCP files")

    # Write CSV file
    with open(output_csv, 'w') as f:
        # Write header
        f.write("timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z\n")

        # Process each file
        for i, tcp_file in enumerate(tcp_files):
            # Extract timestamp from filename (assuming format 0000000000.npy)
            filename = Path(tcp_file).stem
            timestamp = int(filename)  # Convert to integer

            # Load TCP data
            tcp_data = np.load(tcp_file)

            if len(tcp_data) != 7:
                print(f"Warning: Unexpected data format in {tcp_file}: {tcp_data}")
                continue

            # Extract position (first 3 values)
            position = tcp_data[:3]

            # Extract rotation (last 4 values) - axis-angle format
            rotation_data = tcp_data[3:]

            # Convert axis-angle to quaternion
            quaternion = axis_angle_to_quaternion(rotation_data)

            # Write to CSV
            csv_line = f"{timestamp},{position[0]:.6f},{position[1]:.6f},{position[2]:.6f},{quaternion[0]:.6f},{quaternion[1]:.6f},{quaternion[2]:.6f},{quaternion[3]:.6f}\n"
            f.write(csv_line)

            # Print first few for verification
            if i < 3:
                print(f"File {i}: {tcp_file}")
                print(f"  Original data: {tcp_data}")
                print(f"  Position: {position}")
                print(f"  Axis-angle: {rotation_data}")
                print(f"  Quaternion: {quaternion}")
                print(f"  CSV line: {csv_line.strip()}")
                print()

    print(f"✓ Successfully converted {len(tcp_files)} TCP poses to {output_csv}")

    # Also create a summary
    print("\n=== Summary ===")
    print(f"Input: {tcp_dir}")
    print(f"Output: {output_csv}")
    print(f"Total poses: {len(tcp_files)}")
    print(f"Format: timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z")

if __name__ == "__main__":
    convert_original_tcp_to_csv()