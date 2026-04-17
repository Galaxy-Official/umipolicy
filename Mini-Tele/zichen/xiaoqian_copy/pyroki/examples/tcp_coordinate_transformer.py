"""TCP Coordinate System Transformer

This script provides functions to transform TCP poses between the original sensor coordinate system
and the forward kinematics coordinate system, using the solved transformation matrix.

Supports both homogeneous transformation matrix and quaternion+position representations.
"""

import numpy as np
from typing import Tuple, Union
from scipy.spatial.transform import Rotation

class TCPCoordinateTransformer:
    """Transformer for TCP poses between sensor and FK coordinate systems."""

    def __init__(self):
        """Initialize with the solved transformation parameters."""
        # Rotation matrix (sensor -> FK)
        self.R_sensor_to_fk = np.array([
            [ 9.99975489e-01,  3.15093219e-03,  6.25238019e-03],
            [-3.15363484e-03,  9.99994938e-01,  4.22446660e-04],
            [-6.25101744e-03, -4.42154029e-04,  9.99980364e-01]
        ])

        # Translation vector (sensor -> FK)
        self.t_sensor_to_fk = np.array([-0.002007, -0.00144849, 0.00017366])

        # Inverse transformation (FK -> sensor)
        self.R_fk_to_sensor = self.R_sensor_to_fk.T
        self.t_fk_to_sensor = -self.R_sensor_to_fk.T @ self.t_sensor_to_fk

        # Quaternion representations
        self.quat_sensor_to_fk = self._rotation_to_quaternion(self.R_sensor_to_fk)
        self.quat_fk_to_sensor = self._rotation_to_quaternion(self.R_fk_to_sensor)

    def _rotation_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        rotation = Rotation.from_matrix(rotation_matrix)
        return rotation.as_quat()  # Returns [x, y, z, w], we'll reorder

    def _quaternion_to_rotation(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        # Input: [w, x, y, z], scipy expects [x, y, z, w]
        quaternion_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        rotation = Rotation.from_quat(quaternion_xyzw)
        return rotation.as_matrix()

    def transform_position_sensor_to_fk(self, position: np.ndarray) -> np.ndarray:
        """Transform position from sensor to FK coordinate system."""
        position = np.asarray(position)
        return self.R_sensor_to_fk @ position + self.t_sensor_to_fk

    def transform_position_fk_to_sensor(self, position: np.ndarray) -> np.ndarray:
        """Transform position from FK to sensor coordinate system."""
        position = np.asarray(position)
        return self.R_fk_to_sensor @ position + self.t_fk_to_sensor

    def transform_quaternion_sensor_to_fk(self, quaternion: np.ndarray) -> np.ndarray:
        """Transform quaternion from sensor to FK coordinate system.

        Args:
            quaternion: [w, x, y, z] quaternion

        Returns:
            [w, x, y, z] transformed quaternion
        """
        quaternion = np.asarray(quaternion)
        # Convert to rotation matrix
        R_input = self._quaternion_to_rotation(quaternion)
        # Apply transformation
        R_transformed = self.R_sensor_to_fk @ R_input
        # Convert back to quaternion
        return self._rotation_to_quaternion(R_transformed)

    def transform_quaternion_fk_to_sensor(self, quaternion: np.ndarray) -> np.ndarray:
        """Transform quaternion from FK to sensor coordinate system.

        Args:
            quaternion: [w, x, y, z] quaternion

        Returns:
            [w, x, y, z] transformed quaternion
        """
        quaternion = np.asarray(quaternion)
        # Convert to rotation matrix
        R_input = self._quaternion_to_rotation(quaternion)
        # Apply transformation
        R_transformed = self.R_fk_to_sensor @ R_input
        # Convert back to quaternion
        return self._rotation_to_quaternion(R_transformed)

    def transform_pose_sensor_to_fk(self, position: np.ndarray, quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform complete pose from sensor to FK coordinate system.

        Args:
            position: [x, y, z] position
            quaternion: [w, x, y, z] quaternion

        Returns:
            Tuple of (transformed_position, transformed_quaternion)
        """
        new_position = self.transform_position_sensor_to_fk(position)
        new_quaternion = self.transform_quaternion_sensor_to_fk(quaternion)
        return new_position, new_quaternion

    def transform_pose_fk_to_sensor(self, position: np.ndarray, quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform complete pose from FK to sensor coordinate system.

        Args:
            position: [x, y, z] position
            quaternion: [w, x, y, z] quaternion

        Returns:
            Tuple of (transformed_position, transformed_quaternion)
        """
        new_position = self.transform_position_fk_to_sensor(position)
        new_quaternion = self.transform_quaternion_fk_to_sensor(quaternion)
        return new_position, new_quaternion

    def get_transformation_matrix_sensor_to_fk(self) -> np.ndarray:
        """Get homogeneous transformation matrix from sensor to FK coordinates.

        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = self.R_sensor_to_fk
        T[:3, 3] = self.t_sensor_to_fk
        return T

    def get_transformation_matrix_fk_to_sensor(self) -> np.ndarray:
        """Get homogeneous transformation matrix from FK to sensor coordinates.

        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = self.R_fk_to_sensor
        T[:3, 3] = self.t_fk_to_sensor
        return T

    def transform_points_sensor_to_fk(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points from sensor to FK coordinate system.

        Args:
            points: Nx3 array of points

        Returns:
            Nx3 array of transformed points
        """
        points = np.asarray(points)
        if points.ndim == 1 and len(points) == 3:
            return self.transform_position_sensor_to_fk(points)
        elif points.ndim == 2 and points.shape[1] == 3:
            return np.array([self.transform_position_sensor_to_fk(p) for p in points])
        else:
            raise ValueError("Points must be 3D vector or Nx3 array")

    def transform_points_fk_to_sensor(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points from FK to sensor coordinate system.

        Args:
            points: Nx3 array of points

        Returns:
            Nx3 array of transformed points
        """
        points = np.asarray(points)
        if points.ndim == 1 and len(points) == 3:
            return self.transform_position_fk_to_sensor(points)
        elif points.ndim == 2 and points.shape[1] == 3:
            return np.array([self.transform_position_fk_to_sensor(p) for p in points])
        else:
            raise ValueError("Points must be 3D vector or Nx3 array")

    def get_transformation_info(self) -> dict:
        """Get detailed transformation information."""
        return {
            'rotation_sensor_to_fk': self.R_sensor_to_fk.tolist(),
            'translation_sensor_to_fk': self.t_sensor_to_fk.tolist(),
            'rotation_fk_to_sensor': self.R_fk_to_sensor.tolist(),
            'translation_fk_to_sensor': self.t_fk_to_sensor.tolist(),
            'quaternion_sensor_to_fk': self.quat_sensor_to_fk.tolist(),
            'quaternion_fk_to_sensor': self.quat_fk_to_sensor.tolist(),
            'transformation_matrix_sensor_to_fk': self.get_transformation_matrix_sensor_to_fk().tolist(),
            'transformation_matrix_fk_to_sensor': self.get_transformation_matrix_fk_to_sensor().tolist()
        }


def create_example_usage():
    """Create example usage of the transformer."""
    print("=== TCP Coordinate Transformer Example Usage ===\n")

    # Initialize transformer
    transformer = TCPCoordinateTransformer()

    # Example 1: Transform a single pose
    print("1. Transforming a single TCP pose:")
    position_sensor = np.array([0.5, -0.01, 0.45])
    quaternion_sensor = np.array([0.999987, 0.000290, -0.000130, 0.005006])

    print(f"   Sensor coordinates: position={position_sensor}, quaternion={quaternion_sensor}")

    position_fk, quaternion_fk = transformer.transform_pose_sensor_to_fk(position_sensor, quaternion_sensor)
    print(f"   FK coordinates: position={position_fk}, quaternion={quaternion_fk}")

    # Transform back
    position_back, quaternion_back = transformer.transform_pose_fk_to_sensor(position_fk, quaternion_fk)
    print(f"   Back to sensor: position={position_back}, quaternion={quaternion_back}")
    print(f"   Round-trip error: position={np.linalg.norm(position_sensor - position_back):.2e}, quaternion={np.linalg.norm(quaternion_sensor - quaternion_back):.2e}")

    # Example 2: Use transformation matrix
    print("\n2. Using transformation matrix:")
    T_sensor_to_fk = transformer.get_transformation_matrix_sensor_to_fk()
    print(f"   Transformation matrix:\n{T_sensor_to_fk}")

    # Transform point using matrix
    point_sensor = np.array([0.5, -0.01, 0.45, 1.0])  # Homogeneous coordinates
    point_fk = T_sensor_to_fk @ point_sensor
    print(f"   Point in FK: {point_fk[:3]} (using matrix)")

    # Example 3: Transform multiple points
    print("\n3. Transforming multiple points:")
    points_sensor = np.array([
        [0.5, -0.01, 0.45],
        [0.52, -0.01, 0.44],
        [0.55, -0.01, 0.42]
    ])

    points_fk = transformer.transform_points_sensor_to_fk(points_sensor)
    print(f"   Points in sensor coords:\n{points_sensor}")
    print(f"   Points in FK coords:\n{points_fk}")

    # Example 4: Get transformation info
    print("\n4. Transformation information:")
    info = transformer.get_transformation_info()
    print(f"   Translation sensor->FK: {info['translation_sensor_to_fk']}")
    print(f"   Rotation angle: {np.arccos((np.trace(transformer.R_sensor_to_fk) - 1) / 2) * 180 / np.pi:.3f} degrees")


if __name__ == "__main__":
    create_example_usage()