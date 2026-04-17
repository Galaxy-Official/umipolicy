"""Compare TCP data from different sources"""

import numpy as np
import glob
from pathlib import Path

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

    # Load first few for comparison
    tcp_data = []
    for i, file in enumerate(tcp_files[:5]):  # Load first 5 for comparison
        tcp = np.load(file)
        tcp_data.append(tcp)
        if i < 3:  # Print first 3 for inspection
            print(f"File {i}: {file}")
            print(f"  Data: {tcp}")
            print(f"  Shape: {tcp.shape}")
            # Parse as position + rotation format
            if len(tcp) == 7:
                position = tcp[:3]
                rotation = tcp[3:]
                print(f"  Position: {position}")
                print(f"  Rotation: {rotation}")
        print()

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

    # Skip header, load first few for comparison
    for i, line in enumerate(lines[1:6]):  # Load first 5 for comparison
        parts = line.strip().split(',')
        if len(parts) >= 8:  # timestamp + 7 values
            # Extract values (skip timestamp)
            values = [float(x) for x in parts[1:]]
            tcp_data.append(np.array(values))

            if i < 3:  # Print first 3 for inspection
                print(f"CSV row {i}: {parts[0]}")
                print(f"  Values: {values}")
                # Parse as position + quaternion
                if len(values) == 7:
                    position = values[0:3]
                    quaternion = values[3:7]
                    print(f"  Position: {position}")
                    print(f"  Quaternion: {quaternion}")
        print()

    return tcp_data

def compare_data(original_data, csv_data):
    """Compare the two data sources"""
    if original_data is None or csv_data is None:
        print("Cannot compare - missing data sources")
        return

    min_len = min(len(original_data), len(csv_data))
    print(f"\n=== Comparing first {min_len} poses ===")

    for i in range(min_len):
        orig = original_data[i]
        csv_vals = csv_data[i]

        print(f"\nPose {i}:")
        print(f"  Original: {orig}")
        print(f"  CSV:      {csv_vals}")

        # Check if they represent the same information
        if len(orig) == 7 and len(csv_vals) == 7:
            # Both have 7 values - compare position and rotation
            orig_pos = orig[:3]
            orig_rot = orig[3:]
            csv_pos = csv_vals[:3]
            csv_rot = csv_vals[3:]

            pos_diff = np.linalg.norm(orig_pos - csv_pos)
            rot_diff = np.linalg.norm(orig_rot - csv_rot)

            print(f"  Position difference: {pos_diff:.6f}")
            print(f"  Rotation difference: {rot_diff:.6f}")

            if pos_diff < 1e-6 and rot_diff < 1e-6:
                print("  ✓ Data matches perfectly!")
            elif pos_diff < 0.01 and rot_diff < 0.01:
                print("  ✓ Data matches closely (small numerical differences)")
            else:
                print("  ✗ Data differs significantly")

def main():
    print("=== Comparing TCP Data Sources ===\n")

    print("1. Loading from original TCP directory...")
    original_data = load_tcp_from_original()

    print("\n2. Loading from generated CSV...")
    csv_data = load_tcp_from_csv()

    print("\n3. Comparing data...")
    compare_data(original_data, csv_data)

    if original_data and csv_data:
        print(f"\n=== Summary ===")
        print(f"Original data: {len(original_data)} poses")
        print(f"CSV data: {len(csv_data)} poses")

        # Check if they represent the same type of data
        if len(original_data) > 0 and len(csv_data) > 0:
            orig_shape = original_data[0].shape if hasattr(original_data[0], 'shape') else len(original_data[0])
            csv_shape = csv_data[0].shape if hasattr(csv_data[0], 'shape') else len(csv_data[0])
            print(f"Original data format: {orig_shape} values per pose")
            print(f"CSV data format: {csv_shape} values per pose")

if __name__ == "__main__":
    main()