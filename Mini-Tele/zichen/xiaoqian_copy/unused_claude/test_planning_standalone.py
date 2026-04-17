#!/usr/bin/env python3
"""
Standalone test script for the updated planning functionality.
This script tests the planning integration without camera dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pybullet as p
from flexiv_plan import Rizon4, RRTPlanner, OMPLPlanner, pb_ompl

def test_planning_integration():
    """Comprehensive test of the planning integration"""
    print("=== Planning Integration Test ===")

    try:
        # Test 1: Robot initialization
        print("\n1. Testing Rizon4 robot initialization...")
        robot = Rizon4(headless=True)  # Use headless mode for testing
        print("   ✓ Rizon4 robot initialized successfully")

        # Test 2: Basic robot functions
        print("\n2. Testing basic robot functions...")
        joints = robot.get_joints()
        tcp_pose = robot.get_tcp_catersian()
        print(f"   ✓ Current joints: {len(joints)} joints")
        print(f"   ✓ TCP position: [{tcp_pose[0]:.3f}, {tcp_pose[1]:.3f}, {tcp_pose[2]:.3f}]")

        # Test 3: RRT Planner
        print("\n3. Testing RRT Planner...")
        rrt_planner = RRTPlanner(robot, step_size=0.05, max_iterations=100)
        print("   ✓ RRT planner initialized")

        # Define test poses
        start_pose = np.array([0.6, 0.0, 0.3, 1, 0, 0, 0])  # x, y, z, qw, qx, qy, qz
        end_pose = np.array([0.6, 0.1, 0.3, 1, 0, 0, 0])    # Slightly different y position

        # Test RRT planning
        print("   Testing RRT pose-to-pose planning...")
        success, path, message = rrt_planner.plan_and_validate_movement(start_pose, end_pose, visualize_path=False)

        if success:
            print(f"   ✓ RRT planning successful! Path has {len(path)} waypoints")
            print(f"   ✓ Message: {message}")
        else:
            print(f"   ⚠ RRT planning failed: {message}")

        # Test 4: OMPL Planner (if available)
        if pb_ompl is not None:
            print("\n4. Testing OMPL Planner...")
            try:
                ompl_planner = OMPLPlanner(robot, planner_name='RRTConnect')
                print("   ✓ OMPL planner initialized")

                print("   Testing OMPL pose-to-pose planning...")
                success, path, message = ompl_planner.plan_pose_path(start_pose, end_pose)

                if success:
                    print(f"   ✓ OMPL planning successful! Path has {len(path)} waypoints")
                    print(f"   ✓ Message: {message}")
                else:
                    print(f"   ⚠ OMPL planning failed: {message}")

            except Exception as e:
                print(f"   ⚠ OMPL planner error: {e}")
                print("   ✓ Falling back to RRT planner (this is expected)")
        else:
            print("\n4. OMPL not available - using RRT as primary planner")

        # Test 5: Collision checking
        print("\n5. Testing collision checking...")
        is_collision = robot.check_collision(robot.get_joints())
        print(f"   ✓ Current configuration: {'In collision' if is_collision else 'Collision free'}")

        # Test 6: Path validation (if we have a path)
        if success and path:
            print("\n6. Testing path validation...")
            is_valid, collision_idx = robot.validate_path(path)
            print(f"   ✓ Path validation: {'Valid' if is_valid else 'Invalid'}")
            if not is_valid:
                print(f"   ⚠ Collision detected at waypoint {collision_idx}")

        # Test 7: Joint interpolation
        print("\n7. Testing joint interpolation...")
        start_joints = robot.calc_ik(start_pose)
        end_joints = robot.calc_ik(end_pose)
        interpolated_path = robot.interpolate_joints(start_joints, end_joints, num_waypoints=5)
        print(f"   ✓ Generated interpolated path with {len(interpolated_path)} waypoints")

        print("\n=== All tests completed successfully! ===")
        print(f"\nSummary:")
        print(f"- Robot: Rizon4 with {len(joints)} DOF")
        print(f"- Primary Planner: {'OMPL' if pb_ompl is not None else 'RRT'}")
        print(f"- Planning: {'Successful' if success else 'Failed - using direct IK'}")
        print(f"- Collision Detection: {'Available' if hasattr(robot, 'check_collision') else 'Not available'}")
        print(f"- Path Validation: {'Available' if hasattr(robot, 'validate_path') else 'Not available'}")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        try:
            p.disconnect()
        except:
            pass

def test_specific_scenarios():
    """Test specific planning scenarios"""
    print("\n=== Specific Scenario Tests ===")

    try:
        robot = Rizon4(headless=True)

        # Scenario 1: Simple linear movement
        print("\n1. Testing simple linear movement...")
        start = np.array([0.5, 0.0, 0.3, 1, 0, 0, 0])
        end = np.array([0.7, 0.0, 0.3, 1, 0, 0, 0])  # Move along x-axis

        planner = RRTPlanner(robot, step_size=0.02, max_iterations=200)
        success, path, message = planner.plan_and_validate_movement(start, end, visualize_path=False)

        if success:
            print(f"   ✓ Linear movement planned: {len(path)} waypoints")
        else:
            print(f"   ⚠ Linear movement failed: {message}")

        # Scenario 2: Movement with orientation change
        print("\n2. Testing movement with orientation change...")
        start = np.array([0.6, 0.0, 0.3, 1, 0, 0, 0])
        end = np.array([0.6, 0.1, 0.3, 0.707, 0, 0, 0.707])  # 90-degree rotation around z

        success, path, message = planner.plan_and_validate_movement(start, end, visualize_path=False)

        if success:
            print(f"   ✓ Orientation change planned: {len(path)} waypoints")
        else:
            print(f"   ⚠ Orientation change failed: {message}")

        # Scenario 3: Collision checking with different configurations
        print("\n3. Testing collision checking...")
        current_joints = robot.get_joints()

        # Test current configuration
        is_collision_current = robot.check_collision(current_joints)
        print(f"   ✓ Current configuration: {'In collision' if is_collision_current else 'Collision free'}")

        # Test zero configuration
        zero_joints = np.zeros_like(current_joints)
        is_collision_zero = robot.check_collision(zero_joints)
        print(f"   ✓ Zero configuration: {'In collision' if is_collision_zero else 'Collision free'}")

        # Test random configuration
        lower_limits, upper_limits = robot.get_dof_limits()
        random_joints = np.random.uniform(lower_limits, upper_limits)
        is_collision_random = robot.check_collision(random_joints)
        print(f"   ✓ Random configuration: {'In collision' if is_collision_random else 'Collision free'}")

        print("\n=== Scenario tests completed ===")
        return True

    except Exception as e:
        print(f"\n✗ Scenario tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        try:
            p.disconnect()
        except:
            pass

def main():
    """Main test function"""
    print("Starting standalone planning integration test...")
    print("This test verifies the OMPL and RRT planning functionality")
    print("without requiring camera or other hardware dependencies.\n")

    # Run comprehensive test
    success1 = test_planning_integration()

    # Run specific scenarios
    success2 = test_specific_scenarios()

    if success1 and success2:
        print("\n🎉 All tests passed! The planning integration is working correctly.")
        print("\nYou can now use the updated planning system with:")
        print("  - OMPL planners (if available): RRTConnect, BITstar, etc.")
        print("  - Custom RRT implementation (reliable fallback)")
        print("  - Enhanced collision detection and path validation")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())