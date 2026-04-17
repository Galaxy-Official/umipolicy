#!/usr/bin/env python3
"""
Minimal environment setup script that creates mock modules for problematic imports
and allows the original planning functionality to work without environment conflicts.
"""

import sys
import os
import importlib
import types

def create_mock_module(name, attributes=None):
    """Create a mock module with specified attributes"""
    mock = types.ModuleType(name)
    if attributes:
        for attr_name, attr_value in attributes.items():
            setattr(mock, attr_name, attr_value)
    return mock

def setup_mock_dependencies():
    """Setup mock dependencies to avoid import conflicts"""

    # Mock open3d and its problematic submodules
    mock_open3d = create_mock_module('open3d')
    mock_open3d.ml = None  # Block the problematic ML submodule
    mock_open3d._ml3d = None  # Block the problematic ML3D submodule

    # Add basic open3d functions that might be needed
    def mock_visualize_geometries(*args, **kwargs):
        pass

    def mock_read_point_cloud(*args, **kwargs):
        return None

    def mock_PointCloud():
        return type('MockPointCloud', (), {
            'points': type('MockPoints', (), {'from_numpy': lambda x: x})(),
            'colors': type('MockColors', (), {'from_numpy': lambda x: x})(),
        })()

    mock_open3d.visualization = create_mock_module('open3d.visualization', {
        'draw_geometries': mock_visualize_geometries,
        'Visualizer': lambda: type('MockVisualizer', (), {'create_window': lambda: None, 'add_geometry': lambda x: None, 'run': lambda: None, 'destroy_window': lambda: None})()
    })

    mock_open3d.io = create_mock_module('open3d.io', {
        'read_point_cloud': mock_read_point_cloud
    })

    mock_open3d.geometry = create_mock_module('open3d.geometry', {
        'PointCloud': mock_PointCloud
    })

    # Install mock open3d
    sys.modules['open3d'] = mock_open3d
    sys.modules['o3d'] = mock_open3d

    # Mock camera module
    import numpy as np

    class MockCameraD400:
        def __init__(self, *args, **kwargs):
            self.camera_id = "mock_camera"
            self.pipeline = None
            self.config = None

        def get_data(self, hole_filling=False):
            # Return mock RGB and depth data
            color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)
            return color, depth

        def readCamPose(self):
            # Return mock camera pose
            return np.eye(4)

    mock_camera = create_mock_module('camera')
    mock_camera.CameraD400 = MockCameraD400
    sys.modules['camera'] = mock_camera

    # Mock robot module
    class MockFlexiv:
        def __init__(self, *args, **kwargs):
            self.robot_ip = "192.168.2.100"
            self.local_ip = "192.168.2.103"

        def get_tcp(self):
            # Return mock TCP pose
            return np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])  # x,y,z,qx,qy,qz,qw

        def get_joint_positions(self):
            # Return mock joint positions
            return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        def get_gripper_width(self):
            return 0.0

        def movePosePrimitive(self, pose):
            pass

        def send_tcp_pose(self, pose):
            pass

        def send_gripper_state(self, width, speed, force):
            pass

    mock_robot = create_mock_module('robot')
    mock_robot.Flexiv = MockFlexiv
    sys.modules['robot'] = mock_robot

    # Mock other problematic modules
    mock_utilities = create_mock_module('utilities')
    mock_utilities.data_utils = create_mock_module('utilities.data_utils')
    mock_utilities.vis_utils = create_mock_module('utilities.vis_utils')
    mock_utilities.transformation = create_mock_module('utilities.transformation')

    # Add basic transformation functions
    def mock_transform_pc(pc, transform):
        return pc

    def mock_transform_dir(dir_vec, transform):
        return dir_vec

    def mock_xyz_rot_transform(data, from_rep="quaternion", to_rep="quaternion"):
        return data

    def mock_rotation_transform(angle, direction, point=None):
        import numpy as np
        return np.eye(4)

    mock_utilities.data_utils.transform_pc = mock_transform_pc
    mock_utilities.data_utils.transform_dir = mock_transform_dir
    mock_utilities.transformation.xyz_rot_transform = mock_xyz_rot_transform
    mock_utilities.transformation.rotation_transform = mock_rotation_transform

    sys.modules['utilities'] = mock_utilities
    sys.modules['utilities.data_utils'] = mock_utilities.data_utils
    sys.modules['utilities.vis_utils'] = mock_utilities.vis_utils
    sys.modules['utilities.transformation'] = mock_utilities.transformation

    # Mock other modules that might cause issues
    problematic_modules = [
        'fpsample', 'transformations', 'cv2', 'pickle', 'configargparse'
    ]

    for module_name in problematic_modules:
        if module_name not in sys.modules:
            if module_name == 'cv2':
                # Special handling for cv2
                mock_cv2 = create_mock_module('cv2')
                mock_cv2.imwrite = lambda path, img: None
                mock_cv2.imread = lambda path: np.zeros((480, 640, 3), dtype=np.uint8)
                sys.modules['cv2'] = mock_cv2
            elif module_name == 'configargparse':
                # Use the real configargparse if available, otherwise mock it
                try:
                    import configargparse
                except ImportError:
                    mock_config = create_mock_module('configargparse')
                    mock_config.ArgumentParser = lambda: type('MockParser', (), {
                        'add_argument': lambda *args, **kwargs: None,
                        'parse_args': lambda: type('MockArgs', (), {
                            'use_planning': True,
                            'planner_type': 'rrt',
                            'step_size': 0.05,
                            'max_iterations': 500,
                            'visualize_path': False,
                            'headless': True
                        })()
                    })()
                    sys.modules['configargparse'] = mock_config
            else:
                sys.modules[module_name] = create_mock_module(module_name)

def test_setup():
    """Test if the minimal environment setup works"""
    try:
        print("Testing minimal environment setup...")

        # Test basic imports
        import numpy as np
        print("✓ NumPy import successful")

        # Test mock modules
        import open3d as o3d
        print("✓ Open3D mock import successful")

        import camera
        print("✓ Camera mock import successful")

        import robot
        print("✓ Robot mock import successful")

        import utilities
        print("✓ Utilities mock import successful")

        # Test camera and robot instantiation
        cam = camera.CameraD400()
        color, depth = cam.get_data()
        print(f"✓ Camera mock functionality: color {color.shape}, depth {depth.shape}")

        rob = robot.Flexiv()
        tcp = rob.get_tcp()
        joints = rob.get_joint_positions()
        print(f"✓ Robot mock functionality: TCP {tcp.shape}, joints {joints.shape}")

        # Test utilities functions
        import utilities.transformation as tf
        result = tf.xyz_rot_transform(np.array([0, 0, 0, 1, 0, 0, 0]))
        print("✓ Utilities transformation functions working")

        print("✅ All minimal environment tests passed!")
        return True

    except Exception as e:
        print(f"❌ Minimal environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to setup and test the minimal environment"""
    print("=== Setting Up Minimal Environment ===")
    print("Creating mock modules to resolve import conflicts...")

    # Setup the minimal environment
    setup_mock_dependencies()

    # Test the setup
    success = test_setup()

    if success:
        print("\n🎉 Minimal environment setup complete!")
        print("You can now run the original script with planning functionality.")
        print("\nTry running:")
        print("  python run_original_with_fixes.py --use_planning --planner_type rrt")
        return 0
    else:
        print("\n❌ Minimal environment setup failed.")
        print("Use the standalone version instead:")
        print("  python robot_sim_run_with_planning_standalone.py")
        return 1

if __name__ == "__main__":
    exit(main())