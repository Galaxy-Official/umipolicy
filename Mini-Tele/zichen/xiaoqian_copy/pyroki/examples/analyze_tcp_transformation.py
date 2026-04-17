"""Analyze transformation between TCP datasets

This script loads both TCP datasets and analyzes the transformation relationship
between them, including solving for the transformation matrix and evaluating
fitting quality.
"""

import numpy as np
import glob
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_tcp_from_original():
    """Load TCP data from original 00000/tcp/ directory"""
    tcp_dir = Path("/Disk3/xiaoqian/collect_data/data/kp/2025-10-26_19:29:53/00000/tcp")

    if not tcp_dir.exists():
        print(f"Original TCP directory not found: {tcp_dir}")
        return None

    # Get all TCP files
    tcp_files = sorted(glob.glob(str(tcp_dir / "*.npy")))

    if not tcp_files:
        print(f"No TCP files found in {tcp_dir}")
        return None

    print(f"Found {len(tcp_files)} TCP files in original directory")

    # Load all TCP data
    tcp_data = []
    timestamps = []

    for i, file in enumerate(tcp_files):
        # Extract timestamp from filename
        filename = Path(file).stem
        timestamp = int(filename)

        # Load TCP data
        tcp = np.load(file)

        if len(tcp) != 7:
            print(f"Warning: Unexpected data format in {file}: {tcp}")
            continue

        # Extract position and quaternion (convert axis-angle to quaternion)
        position = tcp[:3]
        rotation_data = tcp[3:]
        quaternion = axis_angle_to_quaternion(rotation_data)

        tcp_data.append({
            'timestamp': timestamp,
            'position': position,
            'quaternion': quaternion
        })

    return tcp_data

def load_tcp_from_csv():
    """Load TCP data from our generated CSV"""
    csv_file = Path("tcp_poses_from_joints.csv")

    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        return None

    print(f"Loading from CSV: {csv_file}")

    tcp_data = []

    with open(csv_file, 'r') as f:
        lines = f.readlines()

    # Skip header, load all data
    for i, line in enumerate(lines[1:]):  # Skip header
        parts = line.strip().split(',')
        if len(parts) >= 8:  # timestamp + 7 values
            # Extract values (skip timestamp)
            timestamp = int(parts[0])
            values = [float(x) for x in parts[1:]]

            # Extract position and quaternion
            position = np.array(values[0:3])
            quaternion = np.array(values[3:7])

            tcp_data.append({
                'timestamp': timestamp,
                'position': position,
                'quaternion': quaternion
            })

    return tcp_data

def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle representation to quaternion (w, x, y, z)."""
    if len(axis_angle) != 4:
        return np.array([1.0, 0.0, 0.0, 0.0])

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
    quaternion = quaternion / np.linalg.norm(quaternion)

    return quaternion

def apply_transformation(point, rotation_matrix, translation):
    """Apply transformation to a point"""
    return rotation_matrix @ point + translation

def solve_rigid_transformation(source_points, target_points):
    """Solve for rigid transformation between two point sets using Kabsch algorithm"""
    if len(source_points) != len(target_points) or len(source_points) < 3:
        raise ValueError("Need at least 3 points for transformation solving")

    # Convert to numpy arrays
    src = np.array(source_points).T  # 3xN
    tgt = np.array(target_points).T  # 3xN

    # Calculate centroids
    src_centroid = np.mean(src, axis=1, keepdims=True)
    tgt_centroid = np.mean(tgt, axis=1, keepdims=True)

    # Center the points
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    # Calculate covariance matrix
    covariance = src_centered @ tgt_centered.T

    # SVD decomposition
    U, _, Vt = np.linalg.svd(covariance)

    # Calculate rotation matrix
    rotation = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    # Calculate translation
    translation = tgt_centroid - rotation @ src_centroid

    return rotation, translation.flatten()

def analyze_transformation(original_data, csv_data):
    """Analyze transformation between datasets"""
    if not original_data or not csv_data:
        print("Cannot analyze - missing data")
        return

    # Match by timestamp
    original_dict = {d['timestamp']: d for d in original_data}
    csv_dict = {d['timestamp']: d for d in csv_data}

    # Find common timestamps
    common_timestamps = sorted(set(original_dict.keys()) & set(csv_dict.keys()))
    print(f"Found {len(common_timestamps)} common timestamps")

    if len(common_timestamps) < 3:
        print("Need at least 3 common points for transformation analysis")
        return

    # Extract positions for transformation analysis
    original_positions = []
    csv_positions = []

    for ts in common_timestamps:
        original_positions.append(original_dict[ts]['position'])
        csv_positions.append(csv_dict[ts]['position'])

    # Solve transformation
    try:
        rotation_matrix, translation = solve_rigid_transformation(original_positions, csv_positions)

        print("\n=== Transformation Analysis ===")
        print(f"Rotation Matrix:")
        print(rotation_matrix)
        print(f"\nTranslation Vector: {translation}")

        # Calculate transformation quality
        errors = []
        for i, (orig_pos, csv_pos) in enumerate(zip(original_positions, csv_positions)):
            transformed_pos = apply_transformation(orig_pos, rotation_matrix, translation)
            error = np.linalg.norm(transformed_pos - csv_pos)
            errors.append(error)

        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)

        print(f"\nTransformation Quality:")
        print(f"  Mean position error: {mean_error:.6f} meters")
        print(f"  Std position error: {std_error:.6f} meters")
        print(f"  Max position error: {max_error:.6f} meters")

        # Analyze rotation difference
        angle_diff = np.arccos((np.trace(rotation_matrix) - 1) / 2)
        print(f"  Rotation angle: {np.degrees(angle_diff):.3f} degrees")

        # Check if it's close to identity
        identity_diff = np.linalg.norm(rotation_matrix - np.eye(3))
        print(f"  Distance from identity rotation: {identity_diff:.6f}")

        # Classification
        if mean_error < 0.001:  # 1mm
            print("  ✓ Excellent fit - likely same coordinate system")
        elif mean_error < 0.01:  # 1cm
            print("  ✓ Good fit - small calibration difference")
        elif mean_error < 0.1:  # 10cm
            print("  ⚠ Moderate fit - significant transformation needed")
        else:
            print("  ✗ Poor fit - major coordinate system difference")

        return rotation_matrix, translation, errors

    except Exception as e:
        print(f"Error solving transformation: {e}")
        return None, None, None

def visualize_errors(errors, timestamps):
    """Visualize transformation errors"""
    plt.figure(figsize=(12, 4))

    # Position error over time
    plt.subplot(1, 2, 1)
    plt.plot(timestamps, errors, 'b-', alpha=0.7)
    plt.xlabel('Timestamp')
    plt.ylabel('Position Error (m)')
    plt.title('Transformation Error Over Time')
    plt.grid(True)

    # Error histogram
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('tcp_transformation_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: tcp_transformation_analysis.png")

def main():
    print("=== TCP Transformation Analysis ===\n")

    # Load both datasets
    print("1. Loading original TCP data...")
    original_data = load_tcp_from_original()

    print("\n2. Loading CSV TCP data...")
    csv_data = load_tcp_from_csv()

    if not original_data or not csv_data:
        print("Failed to load required data")
        return

    print(f"\n3. Analyzing transformation...")
    rotation_matrix, translation, errors = analyze_transformation(original_data, csv_data)

    if errors is not None and len(errors) > 0:
        # Get timestamps for visualization
        original_dict = {d['timestamp']: d for d in original_data}
        csv_dict = {d['timestamp']: d for d in csv_data}
        common_timestamps = sorted(set(original_dict.keys()) & set(csv_dict.keys()))

        print(f"\n4. Creating visualization...")
        visualize_errors(errors, common_timestamps)

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()


"""
Rotation Matrix:
  [[ 0.99997549  0.00315093  0.00625238]
   [-0.00315363  0.99999494  0.00042245]
   [-0.00625102 -0.00044215  0.99998036]]

Translation Vector: [-0.002007, -0.001448,
  0.000174] meters
"""