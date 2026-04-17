#!/usr/bin/env python3
"""Command-line tool for transforming TCP CSV files between coordinate systems.

Usage:
    python transform_tcp_csv.py input.csv output.csv --direction sensor_to_fk
    python transform_tcp_csv.py input.csv output.csv --direction fk_to_sensor --no-header

The input CSV should have the format: timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z
"""

import argparse
import sys
from pathlib import Path
from tcp_transform_utils import transform_csv_file, print_transformation_summary

def main():
    parser = argparse.ArgumentParser(
        description="Transform TCP CSV files between sensor and FK coordinate systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform from sensor to FK coordinates
  python transform_tcp_csv.py original_tcp_poses.csv transformed_fk.csv --direction sensor_to_fk

  # Transform from FK to sensor coordinates
  python transform_tcp_csv.py tcp_poses_from_joints.csv transformed_sensor.csv --direction fk_to_sensor

  # Transform without header row
  python transform_tcp_csv.py data.csv transformed.csv --direction sensor_to_fk --no-header

  # Show transformation information
  python transform_tcp_csv.py --info
        """
    )

    parser.add_argument('input_file', nargs='?', help='Input CSV file path')
    parser.add_argument('output_file', nargs='?', help='Output CSV file path')
    parser.add_argument(
        '--direction',
        choices=['sensor_to_fk', 'fk_to_sensor'],
        default='sensor_to_fk',
        help='Transformation direction (default: sensor_to_fk)'
    )
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Input file has no header row'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show transformation information and exit'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate transformation by performing round-trip'
    )

    args = parser.parse_args()

    # Show transformation info if requested
    if args.info:
        print_transformation_summary()
        return 0

    # Check input arguments
    if not args.input_file or not args.output_file:
        parser.error("input_file and output_file are required (unless using --info)")

    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found")
        return 1

    # Check if output directory exists
    output_path = Path(args.output_file)
    if not output_path.parent.exists():
        print(f"Error: Output directory '{output_path.parent}' does not exist")
        return 1

    try:
        print(f"Transforming TCP poses from {args.input_file} to {args.output_file}")
        print(f"Direction: {args.direction}")
        print(f"Preserve header: {not args.no_header}")

        # Perform transformation
        transform_csv_file(
            str(input_path),
            str(output_path),
            direction=args.direction,
            preserve_header=not args.no_header
        )

        if args.validate:
            print("\nValidating transformation with round-trip test...")
            from tcp_transform_utils import transform_csv_pose_data
            import csv

            # Read original data
            with open(input_path, 'r') as f:
                reader = csv.reader(f)
                original_rows = list(reader)

            # Skip header if present
            header = None
            data_start = 0
            if not args.no_header and len(original_rows) > 0:
                try:
                    float(original_rows[0][1])  # Check if first row is data
                except (ValueError, IndexError):
                    header = original_rows[0]
                    data_start = 1

            original_data = original_rows[data_start:]

            # Transform forward and back
            if args.direction == 'sensor_to_fk':
                forward_data = transform_csv_pose_data(original_data, 'sensor_to_fk')
                back_data = transform_csv_pose_data(forward_data, 'fk_to_sensor')
            else:
                forward_data = transform_csv_pose_data(original_data, 'fk_to_sensor')
                back_data = transform_csv_pose_data(forward_data, 'sensor_to_fk')

            # Calculate errors
            import numpy as np
            pos_errors = []
            quat_errors = []

            for orig, back in zip(original_data, back_data):
                pos_orig = np.array([float(x) for x in orig[1:4]])
                pos_back = np.array([float(x) for x in back[1:4]])
                quat_orig = np.array([float(x) for x in orig[4:8]])
                quat_back = np.array([float(x) for x in back[4:8]])

                pos_errors.append(np.linalg.norm(pos_orig - pos_back))
                quat_errors.append(np.linalg.norm(quat_orig - quat_back))

            print(f"Round-trip validation results:")
            print(f"  Mean position error: {np.mean(pos_errors):.2e} ± {np.std(pos_errors):.2e} meters")
            print(f"  Mean quaternion error: {np.mean(quat_errors):.2e} ± {np.std(quat_errors):.2e}")
            print(f"  Max position error: {np.max(pos_errors):.2e} meters")

            if np.max(pos_errors) < 1e-6:  # Less than 1 micrometer
                print("  ✅ Validation passed - excellent transformation accuracy")
            elif np.max(pos_errors) < 1e-3:  # Less than 1 millimeter
                print("  ✅ Validation passed - good transformation accuracy")
            else:
                print("  ⚠️  Validation warning - larger errors detected")

        print(f"\n✅ Transformation completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during transformation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())