"""Extract joint position sequence from data collection.

This script reads joint position sequences from the 00000/ directory and saves them
in a format that can be easily loaded for joint angle control visualization.
"""

import numpy as np
import glob
import json
from pathlib import Path


def load_joint_position_sequence(data_path):
    """Load joint position sequence from the 00000 directory."""
    data_path = Path(data_path)
    joint_dir = data_path / "00000" / "joint_position"

    if not joint_dir.exists():
        print(f"Joint position directory not found: {joint_dir}")
        return None

    # Get all joint position files
    jp_files = sorted(glob.glob(str(joint_dir / "*.npy")))

    if not jp_files:
        print(f"No joint position files found in {joint_dir}")
        return None

    print(f"Found {len(jp_files)} joint position files")

    # Load all joint positions
    joint_positions = []
    timestamps = []

    for i, file in enumerate(jp_files):
        jp = np.load(file)
        joint_positions.append(jp)
        timestamps.append(i)  # Simple index-based timestamp

        if i < 3:  # Print first few for verification
            print(f"Joint position {i}: {jp}")
            print(f"  Shape: {jp.shape}")
            print(f"  Range: [{jp.min():.3f}, {jp.max():.3f}]")

    return {
        'joint_positions': joint_positions,
        'timestamps': timestamps,
        'num_poses': len(joint_positions),
        'joint_dimension': len(joint_positions[0]) if joint_positions else 0
    }


def save_joint_sequence(joint_data, output_file):
    """Save joint sequence to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    joint_positions = [jp.tolist() for jp in joint_data['joint_positions']]

    output_data = {
        'num_poses': joint_data['num_poses'],
        'joint_dimension': joint_data['joint_dimension'],
        'timestamps': joint_data['timestamps'],
        'joint_positions': joint_positions
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved joint sequence to: {output_file}")


def save_joint_sequence_csv(joint_data, output_file):
    """Save joint sequence to CSV file."""
    joint_positions = joint_data['joint_positions']
    joint_dim = joint_data['joint_dimension']

    with open(output_file, 'w') as f:
        # Write header
        header = "timestamp," + ",".join([f"joint_{i}" for i in range(joint_dim)])
        f.write(header + "\n")

        # Write data
        for i, jp in enumerate(joint_positions):
            joint_values = ",".join([f"{val:.6f}" for val in jp])
            f.write(f"{i},{joint_values}\n")

    print(f"Saved joint sequence CSV to: {output_file}")


def print_joint_summary(joint_data):
    """Print a summary of the joint sequence."""
    if not joint_data:
        return

    joint_positions = joint_data['joint_positions']
    joint_dim = joint_data['joint_dimension']

    print(f"\nJoint Sequence Summary:")
    print(f"Number of poses: {len(joint_positions)}")
    print(f"Joint dimension: {joint_dim}")

    # Convert to numpy array for easier analysis
    joint_array = np.array(joint_positions)

    print(f"\nJoint ranges:")
    for i in range(joint_dim):
        min_val = joint_array[:, i].min()
        max_val = joint_array[:, i].max()
        mean_val = joint_array[:, i].mean()
        print(f"  Joint {i}: [{min_val:.4f}, {max_val:.4f}] (mean: {mean_val:.4f})")

    # First and last poses
    print(f"\nFirst joint pose: {joint_positions[0]}")
    print(f"Last joint pose: {joint_positions[-1]}")


def main():
    """Main function to extract joint positions."""
    # Path to your data
    data_path = "/Disk3/xiaoqian/collect_data/data/kp/2025-10-26_19:29:53"

    print("=== Loading Joint Position Data ===")
    # Load joint position sequence
    joint_data = load_joint_position_sequence(data_path)
    if joint_data is None:
        return

    print("\n=== Joint Sequence Summary ===")
    print_joint_summary(joint_data)

    # Save joint sequence
    save_joint_sequence(joint_data, "joint_sequence.json")
    save_joint_sequence_csv(joint_data, "joint_sequence.csv")

    # Create a simple playback script snippet
    print(f"\nTo use these joint positions in your visualization:")
    print(f"1. Load the joint sequence from joint_sequence.json")
    print(f"2. Use the joint positions to set robot configuration")
    print(f"3. Each pose has {joint_data['joint_dimension']} joints")


if __name__ == "__main__":
    main()