#!/usr/bin/env python3
"""
Wrapper script to run the planning functionality with fixed environment conflicts.
This script resolves import conflicts before running the main planning code.
"""

import sys
import os

# Fix environment conflicts before any other imports
def fix_environment():
    """Apply environment fixes"""
    # Block problematic modules
    sys.modules['open3d.ml'] = None
    sys.modules['open3d._ml3d'] = None

    # Create mock modules for missing dependencies
    class MockModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    # Mock problematic imports
    sys.modules['open3d'] = MockModule()
    sys.modules['o3d'] = MockModule()

    # Mock camera and robot modules for basic functionality
    import numpy as np

    class MockCamera:
        def __init__(self, *args, **kwargs):
            pass
        def get_data(self, *args, **kwargs):
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640))

    class MockRobot:
        def __init__(self, *args, **kwargs):
            pass
        def get_tcp(self):
            return np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])
        def get_joint_positions(self):
            return np.zeros(7)
        def get_gripper_width(self):
            return 0.0

    # Create mock module objects
    camera_mock = type('MockModule', (), {'CameraD400': MockCamera})()
    robot_mock = type('MockModule', (), {'Flexiv': MockRobot})()

    sys.modules['camera'] = camera_mock
    sys.modules['robot'] = robot_mock

def main():
    """Main function that applies fixes and runs the planning script"""
    print("=== Running Planning with Fixed Environment ===")

    # Apply environment fixes
    print("Applying environment conflict fixes...")
    fix_environment()

    # Now we can safely import and run the standalone version
    print("Importing planning modules...")

    try:
        # Import the standalone version
        from robot_sim_run_with_planning_standalone import main as standalone_main

        print("✓ Planning modules imported successfully")
        print("Starting planning execution...")

        # Run the standalone main function
        result = standalone_main()

        if result == 0:
            print("\n🎉 Planning execution completed successfully!")
        else:
            print(f"\n⚠ Planning execution returned code: {result}")

        return result

    except Exception as e:
        print(f"\n❌ Planning execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Save original arguments for the standalone script
    original_argv = sys.argv.copy()

    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        exit(1)
    except SystemExit as e:
        # Pass through SystemExit from the standalone script
        exit(e.code if hasattr(e, 'code') else 1)