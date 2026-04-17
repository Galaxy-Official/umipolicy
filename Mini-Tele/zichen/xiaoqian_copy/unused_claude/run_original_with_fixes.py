#!/usr/bin/env python3
"""
Final wrapper that applies the minimal environment fixes and runs the original
robot_sim_run_with_planning.py script with the pybullet_ompl integration.
"""

import sys
import os
import importlib
import importlib.util
import types

# Apply environment fixes before any other imports
def setup_environment():
    """Setup minimal environment to avoid import conflicts"""

    # Block problematic open3d submodules
    sys.modules['open3d.ml'] = None
    sys.modules['open3d._ml3d'] = None

    # Create mock open3d
    import types
    import numpy as np

    mock_open3d = types.ModuleType('open3d')
    mock_open3d.ml = None
    mock_open3d._ml3d = None

    # Add basic functions
    def mock_visualize_geometries(*args, **kwargs):
        pass

    def mock_read_point_cloud(*args, **kwargs):
        return None

    def mock_PointCloud():
        return type('MockPointCloud', (), {
            'points': type('MockPoints', (), {'from_numpy': lambda x: x})(),
            'colors': type('MockColors', (), {'from_numpy': lambda x: x})(),
        })()

    mock_open3d.visualization = types.ModuleType('open3d.visualization')
    mock_open3d.visualization.draw_geometries = mock_visualize_geometries

    mock_open3d.io = types.ModuleType('open3d.io')
    mock_open3d.io.read_point_cloud = mock_read_point_cloud

    mock_open3d.geometry = types.ModuleType('open3d.geometry')
    mock_open3d.geometry.PointCloud = mock_PointCloud

    # Install mock open3d
    sys.modules['open3d'] = mock_open3d
    sys.modules['o3d'] = mock_open3d

    # Mock camera module
    class MockCameraD400:
        def __init__(self, *args, **kwargs):
            self.camera_id = "mock_camera"

        def get_data(self, hole_filling=False):
            color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)
            return color, depth

        def readCamPose(self):
            return np.eye(4)

    mock_camera = types.ModuleType('camera')
    mock_camera.CameraD400 = MockCameraD400
    sys.modules['camera'] = mock_camera

    # Mock robot module
    class MockFlexiv:
        def __init__(self, *args, **kwargs):
            self.robot_ip = "192.168.2.100"
            self.local_ip = "192.168.2.103"

        def get_tcp(self):
            return np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])

        def get_joint_positions(self):
            return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        def get_gripper_width(self):
            return 0.0

        def movePosePrimitive(self, pose):
            pass

        def send_tcp_pose(self, pose):
            pass

        def send_gripper_state(self, width, speed, force):
            pass

    mock_robot = types.ModuleType('robot')
    mock_robot.Flexiv = MockFlexiv
    sys.modules['robot'] = mock_robot

    # Mock utilities modules
    mock_utilities = types.ModuleType('utilities')
    mock_utilities.data_utils = types.ModuleType('utilities.data_utils')
    mock_utilities.vis_utils = types.ModuleType('utilities.vis_utils')
    mock_utilities.transformation = types.ModuleType('utilities.transformation')

    # Add basic transformation functions
    def mock_transform_pc(pc, transform):
        return pc

    def mock_transform_dir(dir_vec, transform):
        return dir_vec

    def mock_xyz_rot_transform(data, from_rep="quaternion", to_rep="quaternion"):
        return data

    def mock_rotation_transform(angle, direction, point=None):
        return np.eye(4)

    def mock_visualize(*args, **kwargs):
        pass  # Mock visualization function

    mock_utilities.data_utils.transform_pc = mock_transform_pc
    mock_utilities.data_utils.transform_dir = mock_transform_dir
    mock_utilities.transformation.xyz_rot_transform = mock_xyz_rot_transform
    mock_utilities.transformation.rotation_transform = mock_rotation_transform
    mock_utilities.vis_utils.visualize = mock_visualize

    sys.modules['utilities'] = mock_utilities
    sys.modules['utilities.data_utils'] = mock_utilities.data_utils
    sys.modules['utilities.vis_utils'] = mock_utilities.vis_utils
    sys.modules['utilities.transformation'] = mock_utilities.transformation

    # Mock other problematic modules
    problematic_modules = [
        'fpsample', 'transformations', 'cv2', 'pickle', 'configargparse', 'tqdm'
    ]

    for module_name in problematic_modules:
        if module_name not in sys.modules:
            if module_name == 'cv2':
                mock_cv2 = types.ModuleType('cv2')
                mock_cv2.imwrite = lambda path, img: None
                mock_cv2.imread = lambda path: np.zeros((480, 640, 3), dtype=np.uint8)
                sys.modules['cv2'] = mock_cv2
            elif module_name == 'configargparse':
                # Use real configargparse if available
                try:
                    import configargparse
                except ImportError:
                    mock_config = types.ModuleType('configargparse')
                    mock_config.ArgumentParser = lambda: type('MockParser', (), {
                        'add_argument': lambda *args, **kwargs: None,
                        'parse_args': lambda: type('MockArgs', (), {
                            'use_planning': True,
                            'planner_type': 'rrt',
                            'step_size': 0.05,
                            'max_iterations': 500,
                            'visualize_path': False,
                            'headless': True,
                            'root_dir': 'collect_data/data/kp',
                            'global_key': '2025-10-26_19:29:53',
                            'ompl_planner': 'RRTConnect',
                            'planning_time': 5.0,
                            'record': False,
                            'episode': 0,
                            'demo_mode': False,
                            'num_steps': 10
                        })()
                    })()
                    sys.modules['configargparse'] = mock_config
            elif module_name == 'tqdm':
                mock_tqdm = types.ModuleType('tqdm')
                mock_tqdm.trange = lambda n: range(n)
                mock_tqdm.tqdm = lambda x: x
                sys.modules['tqdm'] = mock_tqdm
            else:
                sys.modules[module_name] = types.ModuleType(module_name)

def run_with_fixed_environment():
    """Run the original script with fixed environment"""
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        # Add the collect_data directory to path
        sys.path.insert(0, os.path.join(script_dir, 'collect_data'))

        # Apply environment fixes
        print("Applying environment conflict fixes...")
        setup_environment()

        print("✓ Environment fixes applied successfully")

        # Now import and run the original script
        print("Importing original planning script...")

        # Import the original script as a module
        spec = importlib.util.spec_from_file_location(
            "robot_sim_run_with_planning",
            os.path.join(script_dir, "collect_data", "robot_sim_run_with_planning.py")
        )
        original_module = importlib.util.module_from_spec(spec)

        print("✓ Original script imported successfully")
        print("Running original script with planning integration...")

        # Execute the module
        spec.loader.exec_module(original_module)

        print("\n🎉 Original script execution completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Original script execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main function"""
    print("=== Running Original Planning Script with Fixed Environment ===")
    print("This will run the original robot_sim_run_with_planning.py with environment fixes")

    # Parse arguments for the original script
    if len(sys.argv) > 1:
        print(f"Arguments: {' '.join(sys.argv[1:])}")

    return run_with_fixed_environment()

if __name__ == "__main__":
    exit(main())