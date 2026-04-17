#!/usr/bin/env python3
"""Test script for screw motion trajectory planning."""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyroki as pk
import yourdfpy
from pathlib import Path
from jaxlie import SE3, SO3


def test_screw_motion_basic():
    """Basic test of screw motion functionality."""
    print("=== Basic Screw Motion Test ===")

    # Try to load robot URDF
    urdf_path = None
    possible_paths = [
        "~/robot_models/panda.urdf",
        "~/robot_data/example-robot-data/robots/panda_description/urdf/panda.urdf",
        "./panda.urdf",
        "../panda.urdf",
    ]

    for path in possible_paths:
        expanded_path = Path(path).expanduser()
        if expanded_path.exists():
            urdf_path = str(expanded_path)
            print(f"Found URDF file: {urdf_path}")
            break

    if urdf_path is None:
        print("❌ No panda URDF file found!")
        print("Please download panda.urdf first:")
        print("wget -O ~/robot_models/panda.urdf https://raw.githubusercontent.com/Gepetto/example-robot-data/master/robots/panda_description/urdf/panda.urdf")
        return False

    try:
        # Load robot
        urdf = yourdfpy.URDF.load(urdf_path)
        robot = pk.Robot.from_urdf(urdf)
        print(f"✓ Robot loaded: {len(robot.joints.lower_limits)} joints")

        # Test screw motion parameters
        rotation_axis = np.array([0.0, 0.0, 1.0])  # Z-axis
        start_position = np.array([0.5, 0.0, 0.3])
        rotation_angle = np.pi / 4  # 45 degrees
        translation_distance = 0.05  # 5cm

        print(f"\nTest parameters:")
        print(f"  Rotation axis: {rotation_axis}")
        print(f"  Start position: {start_position}")
        print(f"  Rotation angle: {np.degrees(rotation_angle):.1f}°")
        print(f"  Translation distance: {translation_distance}m")

        # Generate screw poses (basic test)
        from jaxlie import SE3, SO3
        start_pose = SE3.from_rotation_and_translation(
            SO3(np.array([0.0, 0.0, 0.0, 1.0])), start_position
        )

        target_positions, target_quaternions = generate_screw_poses_test(
            start_pose, rotation_axis, rotation_angle, translation_distance, 5
        )

        print(f"\n✓ Generated {len(target_positions)} target poses")
        print(f"  Positions shape: {target_positions.shape}")
        print(f"  Quaternions shape: {target_quaternions.shape}")

        # Test IK for first waypoint
        target_link_index = robot.links.names.index("panda_hand")
        success, cfg = solve_ik_waypoint_test(
            robot=robot,
            target_link_index=target_link_index,
            target_position=target_positions[0],
            target_wxyz=target_quaternions[0],
            initial_cfg=robot.joints.default_factory(),
        )

        if success:
            print(f"✓ IK solve successful for first waypoint")
            print(f"  Joint config: {cfg}")
            print(f"  Joint limits check: {is_config_valid_test(robot, cfg)}")
        else:
            print(f"❌ IK solve failed for first waypoint")

        return success

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_screw_poses_test(
    start_pose: SE3,
    rotation_axis: np.ndarray,
    rotation_angle: float,
    translation_distance: float,
    num_waypoints: int,
) -> tuple:
    """Test version of screw pose generation."""
    positions = []
    quaternions = []

    for i in range(num_waypoints):
        t = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0

        # Rotation
        angle = t * rotation_angle
        axis_angle = angle * rotation_axis
        rotation = SO3.exp(axis_angle)

        # Translation
        translation = t * translation_distance * rotation_axis

        # Combine
        relative_transform = SE3.from_rotation_and_translation(rotation, translation)
        waypoint_pose = start_pose @ relative_transform

        positions.append(waypoint_pose.translation())
        quaternions.append(waypoint_pose.rotation().wxyz)

    return np.array(positions), np.array(quaternions)


def solve_ik_waypoint_test(
    robot: pk.Robot,
    target_link_index: int,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    initial_cfg: np.ndarray,
) -> tuple:
    """Test version of IK solve for single waypoint."""
    import jaxls
    from jax import numpy as jnp

    joint_var = robot.joint_var_cls(0)

    # Basic IK factors
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            SE3.from_rotation_and_translation(SO3(target_wxyz), target_position),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.limit_cost(robot, joint_var, jnp.array(100.0)),
    ]

    try:
        solution = (
            jaxls.LeastSquaresProblem(factors, [joint_var])
            .analyze()
            .solve(
                initial_vals=jaxls.VarValues.make((joint_var.with_value(initial_cfg),)),
                verbose=False,
            )
        )

        result_cfg = solution[joint_var]

        # Basic validation
        if is_config_valid_test(robot, result_cfg):
            return True, result_cfg
        else:
            return False, initial_cfg

    except Exception as e:
        print(f"IK solve exception: {e}")
        return False, initial_cfg


def is_config_valid_test(robot: pk.Robot, cfg: np.ndarray) -> bool:
    """Basic validation of joint configuration."""
    lower_limits = robot.joints.lower_limits
    upper_limits = robot.joints.upper_limits

    for q, lower, upper in zip(cfg, lower_limits, upper_limits):
        if q < lower - 0.01 or q > upper + 0.01:
            return False
    return True


def test_with_manual_urdf():
    """Test with manually specified URDF path."""
    print("\n=== Manual URDF Test ===")

    # Import the simple screw motion module
    try:
        from examples import pyroki_snippets as pks
        from examples._13_screw_motion_simple import plan_screw_motion

        urdf_path = "~/robot_models/panda.urdf"
        rotation_axis = np.array([0.0, 0.0, 1.0])
        start_position = np.array([0.5, 0.0, 0.3])

        success, joint_trajectory, message = plan_screw_motion(
            urdf_path=urdf_path,
            rotation_axis=rotation_axis,
            start_position=start_position,
            rotation_angle=np.pi/3,
            translation_distance=0.08,
            num_waypoints=10,
        )

        print(f"Success: {success}")
        print(f"Message: {message}")

        if success:
            print(f"Joint trajectory shape: {joint_trajectory.shape}")
            print(f"First config: {joint_trajectory[0]}")
            print(f"Last config: {joint_trajectory[-1]}")

        return success

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting screw motion trajectory planning tests...\n")

    # Test 1: Basic functionality
    test1_passed = test_screw_motion_basic()

    # Test 2: With manual URDF (if basic test passed)
    if test1_passed:
        test2_passed = test_with_manual_urdf()
    else:
        print("\n⏭️  Skipping manual URDF test due to basic test failure")
        test2_passed = False

    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Basic functionality test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Manual URDF test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Screw motion trajectory planning is ready.")
        print("\nYou can now run:")
        print("  python examples/13_screw_motion_simple.py")
        print("  python examples/13_screw_motion_trajopt.py")
    else:
        print("\n⚠️  Some tests failed. Please check the setup.")

    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)