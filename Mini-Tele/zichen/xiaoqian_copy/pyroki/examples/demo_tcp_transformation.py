"""Demonstration of TCP Coordinate Transformation

This script demonstrates practical usage of the TCP coordinate transformation functions
for converting between sensor and forward kinematics coordinate systems.
"""

import numpy as np
from tcp_transform_utils import (
    transform_pose_sensor_to_fk, transform_pose_fk_to_sensor,
    transform_csv_pose_data, transform_csv_file,
    get_transformation_matrix, get_transformation_info,
    s2fk, fk2s,  # Convenience aliases
    print_transformation_summary
)

def demo_basic_transformations():
    """Demonstrate basic pose transformations."""
    print("=== Basic Pose Transformations ===\n")

    # Example TCP pose from sensor data
    sensor_position = [0.515928, -0.011346, 0.450704]
    sensor_quaternion = [0.999987, 0.000290, -0.000130, 0.005006]

    print(f"Original sensor pose:")
    print(f"  Position: [{sensor_position[0]:.6f}, {sensor_position[1]:.6f}, {sensor_position[2]:.6f}]")
    print(f"  Quaternion: [{sensor_quaternion[0]:.6f}, {sensor_quaternion[1]:.6f}, {sensor_quaternion[2]:.6f}, {sensor_quaternion[3]:.6f}]")

    # Transform to FK coordinates
    fk_position, fk_quaternion = transform_pose_sensor_to_fk(sensor_position, sensor_quaternion)

    print(f"\nTransformed to FK coordinates:")
    print(f"  Position: [{fk_position[0]:.6f}, {fk_position[1]:.6f}, {fk_position[2]:.6f}]")
    print(f"  Quaternion: [{fk_quaternion[0]:.6f}, {fk_quaternion[1]:.6f}, {fk_quaternion[2]:.6f}, {fk_quaternion[3]:.6f}]")

    # Transform back to sensor coordinates
    back_position, back_quaternion = transform_pose_fk_to_sensor(fk_position, fk_quaternion)

    print(f"\nTransformed back to sensor coordinates:")
    print(f"  Position: [{back_position[0]:.6f}, {back_position[1]:.6f}, {back_position[2]:.6f}]")
    print(f"  Quaternion: [{back_quaternion[0]:.6f}, {back_quaternion[1]:.6f}, {back_quaternion[2]:.6f}, {back_quaternion[3]:.6f}]")

    # Calculate errors
    pos_error = np.linalg.norm(np.array(sensor_position) - back_position)
    quat_error = np.linalg.norm(np.array(sensor_quaternion) - back_quaternion)

    print(f"\nRound-trip transformation errors:")
    print(f"  Position error: {pos_error:.2e} meters")
    print(f"  Quaternion error: {quat_error:.2e}")

def demo_convenience_aliases():
    """Demonstrate convenience aliases."""
    print("\n\n=== Convenience Aliases ===\n")

    position = [0.5, -0.01, 0.45]
    quaternion = [0.999987, 0.000290, -0.000130, 0.005006]

    print(f"Original pose: pos={position}, quat={quaternion}")

    # Using convenience aliases
    from tcp_transform_utils import s2fk, fk2s

    fk_pos, fk_quat = s2fk(position, quaternion)
    print(f"Using s2fk(): pos={fk_pos.tolist()}, quat={fk_quat.tolist()}")

    back_pos, back_quat = fk2s(fk_pos, fk_quat)
    print(f"Using fk2s(): pos={back_pos.tolist()}, quat={back_quat.tolist()}")

def demo_csv_transformation():
    """Demonstrate CSV data transformation."""
    print("\n\n=== CSV Data Transformation ===\n")

    # Sample CSV data (timestamp, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
    csv_data = [
        [0, 0.515928, -0.011346, 0.450704, 0.999987, 0.000290, -0.000130, 0.005006],
        [1, 0.519129, -0.010797, 0.446710, 0.999987, 0.000290, -0.000129, 0.005007],
        [2, 0.522066, -0.010246, 0.443041, 0.999987, 0.000290, -0.000129, 0.005003]
    ]

    print("Original CSV data (first 3 rows):")
    for i, row in enumerate(csv_data):
        print(f"  Row {i}: timestamp={row[0]}, pos=[{row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}], quat=[{row[4]:.3f}, {row[5]:.3f}, {row[6]:.3f}, {row[7]:.3f}]")

    # Transform to FK coordinates
    fk_data = transform_csv_pose_data(csv_data, 'sensor_to_fk')

    print(f"\nTransformed to FK coordinates:")
    for i, row in enumerate(fk_data):
        print(f"  Row {i}: timestamp={row[0]}, pos=[{row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}], quat=[{row[4]:.3f}, {row[5]:.3f}, {row[6]:.3f}, {row[7]:.3f}]")

    # Transform back to sensor coordinates
    back_data = transform_csv_pose_data(fk_data, 'fk_to_sensor')

    print(f"\nTransformed back to sensor coordinates:")
    for i, row in enumerate(back_data):
        print(f"  Row {i}: timestamp={row[0]}, pos=[{row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}], quat=[{row[4]:.3f}, {row[5]:.3f}, {row[6]:.3f}, {row[7]:.3f}]")

    # Calculate average errors
    pos_errors = []
    quat_errors = []
    for orig, back in zip(csv_data, back_data):
        pos_orig = np.array(orig[1:4])
        pos_back = np.array(back[1:4])
        quat_orig = np.array(orig[4:8])
        quat_back = np.array(back[4:8])

        pos_errors.append(np.linalg.norm(pos_orig - pos_back))
        quat_errors.append(np.linalg.norm(quat_orig - quat_back))

    print(f"\nAverage round-trip errors:")
    print(f"  Position error: {np.mean(pos_errors):.2e} ± {np.std(pos_errors):.2e} meters")
    print(f"  Quaternion error: {np.mean(quat_errors):.2e} ± {np.std(quat_errors):.2e}")

def demo_transformation_matrices():
    """Demonstrate transformation matrices."""
    print("\n\n=== Transformation Matrices ===\n")

    # Get transformation matrices
    T_sensor_to_fk = get_transformation_matrix('sensor_to_fk')
    T_fk_to_sensor = get_transformation_matrix('fk_to_sensor')

    print("Transformation matrix (sensor → FK):")
    print(T_sensor_to_fk)

    print("\nTransformation matrix (FK → sensor):")
    print(T_fk_to_sensor)

    print("\nVerification: T_fk_to_sensor should be the inverse of T_sensor_to_fk")
    identity_check = T_fk_to_sensor @ T_sensor_to_fk
    print("T_fk_to_sensor @ T_sensor_to_fk =")
    print(identity_check)
    print(f"Max deviation from identity: {np.max(np.abs(identity_check - np.eye(4))):.2e}")

def demo_batch_transformations():
    """Demonstrate batch transformations."""
    print("\n\n=== Batch Transformations ===\n")

    # Multiple points
    points_sensor = [
        [0.515928, -0.011346, 0.450704],
        [0.519129, -0.010797, 0.446710],
        [0.522066, -0.010246, 0.443041],
        [0.525242, -0.009691, 0.439024]
    ]

    print("Original points (sensor coordinates):")
    for i, point in enumerate(points_sensor):
        print(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")

    # Transform to FK
    from tcp_transform_utils import transform_points_sensor_to_fk
    points_fk = transform_points_sensor_to_fk(points_sensor)

    print("\nTransformed points (FK coordinates):")
    for i, point in enumerate(points_fk):
        print(f"  Point {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")

def demo_transformation_info():
    """Demonstrate getting transformation information."""
    print("\n\n=== Transformation Information ===\n")

    info = get_transformation_info()

    print("Translation vector (sensor → FK):")
    print(f"  [{info['translation_sensor_to_fk'][0]:.6f}, {info['translation_sensor_to_fk'][1]:.6f}, {info['translation_sensor_to_fk'][2]:.6f}] meters")

    print("\nRotation matrix (sensor → FK):")
    R = np.array(info['rotation_sensor_to_fk'])
    for row in R:
        print(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")

    print("\nQuaternion (sensor → FK):")
    quat = info['quaternion_sensor_to_fk']
    print(f"  [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")

    # Calculate rotation angle
    angle_rad = np.arccos((np.trace(R) - 1) / 2)
    angle_deg = angle_rad * 180 / np.pi
    print(f"\nRotation angle: {angle_deg:.3f} degrees")

def main():
    """Main demonstration function."""
    print("TCP Coordinate Transformation Demonstration")
    print("=" * 50)

    # Print transformation summary
    print_transformation_summary()

    # Run all demonstrations
    demo_basic_transformations()
    demo_convenience_aliases()
    demo_csv_transformation()
    demo_transformation_matrices()
    demo_batch_transformations()
    demo_transformation_info()

    print("\n" + "=" * 50)
    print("Demonstration complete!")
    print("\nKey takeaways:")
    print("- Transformation is very small (sub-millimeter, sub-degree)")
    print("- Both coordinate systems are essentially aligned")
    print("- Round-trip transformations have negligible errors")
    print("- Functions support both individual poses and batch operations")
    print("- CSV data can be easily transformed between coordinate systems")

if __name__ == "__main__":
    main()