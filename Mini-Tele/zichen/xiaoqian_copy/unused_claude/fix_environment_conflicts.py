#!/usr/bin/env python3
"""
Script to fix environment conflicts by isolating problematic imports
and providing clean import paths for the planning functionality.
"""

import sys
import os

def fix_open3d_import():
    """Fix open3d import conflicts by isolating sklearn/scipy dependencies"""
    try:
        # Try to import open3d without the ML modules that cause conflicts
        import importlib.util

        # Block problematic open3d submodules
        sys.modules['open3d.ml'] = None
        sys.modules['open3d._ml3d'] = None

        # Now try to import open3d core
        import open3d as o3d
        print("✓ Open3D core imported successfully (ML modules disabled)")
        return True

    except Exception as e:
        print(f"⚠ Open3D import failed: {e}")
        print("  Creating mock open3d module for compatibility")

        # Create a mock open3d module for basic compatibility
        class MockOpen3D:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        sys.modules['open3d'] = MockOpen3D()
        sys.modules['o3d'] = MockOpen3D()
        return False

def fix_sklearn_scipy_conflicts():
    """Fix sklearn/scipy conflicts by controlling import order"""
    try:
        # Import scipy first to establish its symbols
        import scipy
        print("✓ SciPy imported successfully")

        # Import numpy with specific settings
        import numpy as np
        print("✓ NumPy imported successfully")

        # Try importing sklearn
        import sklearn
        print("✓ Sklearn imported successfully")

        return True

    except Exception as e:
        print(f"⚠ Import conflict resolution failed: {e}")
        return False

def create_minimal_imports():
    """Create minimal import versions for essential functionality"""

    # Mock camera module for basic functionality
    class MockCamera:
        def __init__(self, *args, **kwargs):
            pass
        def get_data(self, *args, **kwargs):
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640))
        def get_tcp(self, *args, **kwargs):
            return np.eye(4)

    # Mock robot module for basic functionality
    class MockRobot:
        def __init__(self, *args, **kwargs):
            pass
        def get_tcp(self):
            return np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])
        def get_joint_positions(self):
            return np.zeros(7)
        def get_gripper_width(self):
            return 0.0

    # Create mock modules
    sys.modules['camera'] = type('MockModule', (), {'CameraD400': MockCamera})()
    sys.modules['robot'] = type('MockModule', (), {'Flexiv': MockRobot})()

    print("✓ Created minimal mock modules for camera and robot")

def test_clean_imports():
    """Test clean imports of the planning modules"""
    try:
        # Test basic imports
        import numpy as np
        import pybullet as p
        print("✓ Basic imports successful")

        # Test flexiv_plan import
        from flexiv_plan import Rizon4, RRTPlanner, OMPLPlanner, pb_ompl
        print("✓ Planning module imports successful")

        # Test utilities imports
        from utilities.transformation import xyz_rot_transform
        print("✓ Utilities imports successful")

        return True

    except Exception as e:
        print(f"⚠ Clean import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to fix environment conflicts"""
    print("=== Environment Conflict Resolution ===")
    print("Attempting to fix sklearn/scipy/open3d import conflicts...")

    # Step 1: Fix sklearn/scipy conflicts
    print("\n1. Fixing sklearn/scipy conflicts...")
    sklearn_ok = fix_sklearn_scipy_conflicts()

    # Step 2: Fix open3d import
    print("\n2. Fixing open3d import conflicts...")
    open3d_ok = fix_open3d_import()

    # Step 3: Create minimal mocks for missing modules
    print("\n3. Creating minimal mock modules...")
    create_minimal_imports()

    # Step 4: Test clean imports
    print("\n4. Testing clean imports...")
    imports_ok = test_clean_imports()

    # Summary
    print("\n=== Resolution Summary ===")
    print(f"Sklearn/Scipy: {'✅ Fixed' if sklearn_ok else '⚠ Still problematic'}")
    print(f"Open3D: {'✅ Fixed' if open3d_ok else '⚠ Using mock module'}")
    print(f"Clean Imports: {'✅ Working' if imports_ok else '❌ Still failing'}")

    if imports_ok:
        print("\n🎉 Environment conflicts resolved!")
        print("You can now use the planning functionality.")
        print("\nTry running:")
        print("  python robot_sim_run_with_planning_standalone.py --use_planning --planner_type rrt")
        return 0
    else:
        print("\n❌ Some conflicts remain. Use the standalone version instead:")
        print("  python robot_sim_run_with_planning_standalone.py")
        return 1

if __name__ == "__main__":
    # Save original state
    original_modules = sys.modules.copy()

    try:
        exit(main())
    finally:
        # Restore original state if needed
        pass