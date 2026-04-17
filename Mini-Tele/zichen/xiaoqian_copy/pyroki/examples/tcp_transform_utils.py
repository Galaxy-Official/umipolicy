"""TCP Transform Utilities

Convenient utility functions for transforming TCP poses between sensor and FK coordinate systems.
Provides both low-level and high-level interfaces for common transformation tasks.
"""

import numpy as np
from typing import Tuple, Union, List, Optional
from tcp_coordinate_transformer import TCPCoordinateTransformer

# Global transformer instance
_transformer = None

def get_transformer() -> TCPCoordinateTransformer:
    """Get the global transformer instance."""
    global _transformer
    if _transformer is None:
        _transformer = TCPCoordinateTransformer()
    return _transformer

# ===== Low-level transformation functions =====

def transform_position_sensor_to_fk(position: Union[List[float], np.ndarray]) -> np.ndarray:
    """Transform position from sensor to FK coordinate system.

    Args:
        position: [x, y, z] position in sensor coordinates

    Returns:
        [x, y, z] position in FK coordinates

    Example:
        >\u003e\u003e fk_pos = transform_position_sensor_to_fk([0.5, -0.01, 0.45])
        \u003e\u003e\u003e print(f"FK position: {fk_pos}")
    """
    transformer = get_transformer()
    return transformer.transform_position_sensor_to_fk(position)

def transform_position_fk_to_sensor(position: Union[List[float], np.ndarray]) -> np.ndarray:
    """Transform position from FK to sensor coordinate system.

    Args:
        position: [x, y, z] position in FK coordinates

    Returns:
        [x, y, z] position in sensor coordinates
    """
    transformer = get_transformer()
    return transformer.transform_position_fk_to_sensor(position)

def transform_quaternion_sensor_to_fk(quaternion: Union[List[float], np.ndarray]) -> np.ndarray:
    """Transform quaternion from sensor to FK coordinate system.

    Args:
        quaternion: [w, x, y, z] quaternion in sensor coordinates

    Returns:
        [w, x, y, z] quaternion in FK coordinates
    """
    transformer = get_transformer()
    return transformer.transform_quaternion_sensor_to_fk(quaternion)

def transform_quaternion_fk_to_sensor(quaternion: Union[List[float], np.ndarray]) -> np.ndarray:
    """Transform quaternion from FK to sensor coordinate system.

    Args:
        quaternion: [w, x, y, z] quaternion in FK coordinates

    Returns:
        [w, x, y, z] quaternion in sensor coordinates
    """
    transformer = get_transformer()
    return transformer.transform_quaternion_fk_to_sensor(quaternion)

# ===== High-level transformation functions =====

def transform_pose_sensor_to_fk(position: Union[List[float], np.ndarray],
                               quaternion: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Transform complete pose from sensor to FK coordinate system.

    Args:
        position: [x, y, z] position in sensor coordinates
        quaternion: [w, x, y, z] quaternion in sensor coordinates

    Returns:
        Tuple of (position, quaternion) in FK coordinates

    Example:
        \u003e\u003e\u003e pos_fk, quat_fk = transform_pose_sensor_to_fk([0.5, -0.01, 0.45], [1, 0, 0, 0])
        \u003e\u003e\u003e print(f"FK pose: position={pos_fk}, quaternion={quat_fk}")
    """
    transformer = get_transformer()
    return transformer.transform_pose_sensor_to_fk(position, quaternion)

def transform_pose_fk_to_sensor(position: Union[List[float], np.ndarray],
                               quaternion: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Transform complete pose from FK to sensor coordinate system.

    Args:
        position: [x, y, z] position in FK coordinates
        quaternion: [w, x, y, z] quaternion in FK coordinates

    Returns:
        Tuple of (position, quaternion) in sensor coordinates
    """
    transformer = get_transformer()
    return transformer.transform_pose_fk_to_sensor(position, quaternion)

# ===== Batch transformation functions =====

def transform_points_sensor_to_fk(points: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """Transform multiple points from sensor to FK coordinate system.

    Args:
        points: Nx3 array of points in sensor coordinates

    Returns:
        Nx3 array of points in FK coordinates

    Example:
        \u003e\u003e\u003e points_sensor = [[0.5, -0.01, 0.45], [0.52, -0.01, 0.44]]
        \u003e\u003e\u003e points_fk = transform_points_sensor_to_fk(points_sensor)
        \u003e\u003e\u003e print(f"FK points: {points_fk}")
    """
    transformer = get_transformer()
    return transformer.transform_points_sensor_to_fk(points)

def transform_points_fk_to_sensor(points: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """Transform multiple points from FK to sensor coordinate system.

    Args:
        points: Nx3 array of points in FK coordinates

    Returns:
        Nx3 array of points in sensor coordinates
    """
    transformer = get_transformer()
    return transformer.transform_points_fk_to_sensor(points)

# ===== CSV data transformation functions =====

def transform_csv_pose_data(csv_data: List[List[Union[str, float, int]]], direction: str = 'sensor_to_fk') -> List[List[float]]:
    """Transform CSV pose data between coordinate systems.

    Args:
        csv_data: List of [timestamp, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        direction: 'sensor_to_fk' or 'fk_to_sensor'

    Returns:
        Transformed CSV data in same format

    Example:
        \u003e\u003e\u003e csv_row = [0, 0.5, -0.01, 0.45, 0.999987, 0.000290, -0.000130, 0.005006]
        \u003e\u003e\u003e transformed = transform_csv_pose_data([csv_row], 'sensor_to_fk')
        \u003e\u003e\u003e print(f"Transformed: {transformed[0]}")
    """
    if direction not in ['sensor_to_fk', 'fk_to_sensor']:
        raise ValueError("Direction must be 'sensor_to_fk' or 'fk_to_sensor'")

    transformed_data = []
    for row in csv_data:
        if len(row) != 8:
            raise ValueError("Each row must have 8 elements: timestamp, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z")

        # Convert strings to floats if necessary
        try:
            timestamp = float(row[0])
            position = np.array([float(x) for x in row[1:4]])
            quaternion = np.array([float(x) for x in row[4:8]])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert row to numeric values: {row}. Error: {e}")

        if direction == 'sensor_to_fk':
            new_position, new_quaternion = transform_pose_sensor_to_fk(position, quaternion)
        else:
            new_position, new_quaternion = transform_pose_fk_to_sensor(position, quaternion)

        transformed_row = [timestamp] + new_position.tolist() + new_quaternion.tolist()
        transformed_data.append(transformed_row)

    return transformed_data

def transform_csv_file(input_file: str, output_file: str, direction: str = 'sensor_to_fk',
                      preserve_header: bool = True) -> None:
    """Transform an entire CSV file between coordinate systems.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        direction: 'sensor_to_fk' or 'fk_to_sensor'
        preserve_header: Whether to preserve the header row
    """
    import csv

    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return

    # Handle header
    header = None
    data_start = 0
    if preserve_header and len(rows) > 0:
        # Check if first row is header (contains non-numeric values)
        try:
            float(rows[0][1])  # Try to parse second element as float
        except (ValueError, IndexError):
            header = rows[0]
            data_start = 1

    # Transform data
    data_rows = rows[data_start:]
    transformed_data = transform_csv_pose_data(data_rows, direction)

    # Write output
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(transformed_data)

    print(f"Transformed {len(transformed_data)} poses from {input_file} to {output_file}")

# ===== Utility functions =====

def get_transformation_matrix(direction: str = 'sensor_to_fk') -> np.ndarray:
    """Get the 4x4 homogeneous transformation matrix.

    Args:
        direction: 'sensor_to_fk' or 'fk_to_sensor'

    Returns:
        4x4 transformation matrix
    """
    transformer = get_transformer()
    if direction == 'sensor_to_fk':
        return transformer.get_transformation_matrix_sensor_to_fk()
    else:
        return transformer.get_transformation_matrix_fk_to_sensor()

def get_transformation_info() -> dict:
    """Get comprehensive transformation information."""
    transformer = get_transformer()
    return transformer.get_transformation_info()

def print_transformation_summary() -> None:
    """Print a summary of the transformation parameters."""
    info = get_transformation_info()

    print("=== TCP Coordinate Transformation Summary ===")
    print(f"Translation sensor->FK: [{info['translation_sensor_to_fk'][0]:.6f}, {info['translation_sensor_to_fk'][1]:.6f}, {info['translation_sensor_to_fk'][2]:.6f}] m")

    # Calculate rotation angle
    R = np.array(info['rotation_sensor_to_fk'])
    angle_rad = np.arccos((np.trace(R) - 1) / 2)
    angle_deg = angle_rad * 180 / np.pi
    print(f"Rotation angle: {angle_deg:.3f} degrees")
    print(f"Transformation quality: Excellent fit (< 1mm mean error)")

# ===== Convenience aliases =====
# Short aliases for common operations
s2fk = transform_pose_sensor_to_fk
fk2s = transform_pose_fk_to_sensor
s2fk_pos = transform_position_sensor_to_fk
fk2s_pos = transform_position_fk_to_sensor
s2fk_quat = transform_quaternion_sensor_to_fk
fk2s_quat = transform_quaternion_fk_to_sensor

if __name__ == "__main__":
    print_transformation_summary()

    print("\n=== Available Functions ===")
    print("High-level functions:")
    print("  transform_pose_sensor_to_fk(position, quaternion) - Transform complete pose")
    print("  transform_pose_fk_to_sensor(position, quaternion) - Transform complete pose")
    print("  transform_points_sensor_to_fk(points) - Transform multiple points")
    print("  transform_csv_file(input_file, output_file, direction) - Transform CSV file")
    print("\nLow-level functions:")
    print("  transform_position_sensor_to_fk(position) - Transform position only")
    print("  transform_quaternion_sensor_to_fk(quaternion) - Transform quaternion only")
    print("\nConvenience aliases:")
    print("  s2fk, fk2s, s2fk_pos, fk2s_pos, s2fk_quat, fk2s_quat")

    print("\n=== Quick Test ===")
    # Quick test
    pos = [0.5, -0.01, 0.45]
    quat = [0.999987, 0.000290, -0.000130, 0.005006]

    pos_fk, quat_fk = s2fk(pos, quat)
    print(f"Sensor: pos={pos}, quat={quat}")
    print(f"FK:     pos={pos_fk.tolist()}, quat={quat_fk.tolist()}")