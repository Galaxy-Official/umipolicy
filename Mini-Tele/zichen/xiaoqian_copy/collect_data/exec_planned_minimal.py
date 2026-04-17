"""
Execute Planned Trajectory (Minimal Version)

This script loads and executes the pre-planned trajectory without full dependencies.
"""

import os
import time
import numpy as np
import pickle
import threading

# Try to import configargparse, fallback to argparse if not available
try:
    import configargparse as argparse
except ImportError:
    import argparse as argparse


def config_parse():
    parser = argparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str, required=True,
                       help='Global key for the data session')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--planning_dt', type=float, default=0.1,
                       help='Time step between waypoints (seconds)')

    args = parser.parse_args()
    return args


def load_planning_results(root_dir, global_key):
    """Load previously planned trajectory and parameters"""
    try:
        planning_dir = f"{root_dir}/{global_key}/planning"

        if not os.path.exists(planning_dir):
            print(f"Error: Planning directory not found at {planning_dir}")
            return None

        # Load trajectory
        trajectory_path = f"{planning_dir}/trajectory.npy"
        if not os.path.exists(trajectory_path):
            print(f"Error: Trajectory file not found at {trajectory_path}")
            return None

        trajectory = np.load(trajectory_path)

        # Load poses
        start_pose = np.load(f"{planning_dir}/start_pose.npy")
        target_pose = np.load(f"{planning_dir}/target_pose.npy")

        # Load motion parameters
        motion_params_path = f"{planning_dir}/motion_params.npz"
        if os.path.exists(motion_params_path):
            motion_params = dict(np.load(motion_params_path))
        else:
            motion_params = {}

        # Load metadata
        metadata_path = f"{planning_dir}/metadata.npz"
        if os.path.exists(metadata_path):
            metadata = dict(np.load(metadata_path))
        else:
            metadata = {}

        return {
            'trajectory': trajectory,
            'start_pose': start_pose,
            'target_pose': target_pose,
            'motion_params': motion_params,
            'metadata': metadata
        }

    except Exception as e:
        print(f"Error loading planning results: {e}")
        return None


def validate_trajectory(trajectory_data):
    """Validate the planned trajectory before execution"""
    try:
        trajectory = trajectory_data['trajectory']
        metadata = trajectory_data['metadata']

        print("=== Trajectory Validation ===")

        # Basic checks
        if trajectory is None or len(trajectory) == 0:
            print("Error: Empty trajectory")
            return False

        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Number of waypoints: {len(trajectory)}")

        # Check metadata
        if metadata:
            print(f"Planning success: {metadata.get('success', 'Unknown')}")
            print(f"Joint limits valid: {metadata.get('joint_limits_valid', 'Unknown')}")

        return True

    except Exception as e:
        print(f"Error validating trajectory: {e}")
        return False


def simulate_robot_execution(trajectory, dt):
    """Simulate robot execution (placeholder for actual robot control)"""
    try:
        print("=== Simulating Robot Execution ===")
        print(f"Trajectory: {len(trajectory)} waypoints, dt={dt}s")

        # Validate trajectory length
        if len(trajectory) == 0:
            print("Error: Empty trajectory")
            return False

        # Simulate execution waypoint by waypoint
        for i, waypoint in enumerate(trajectory):
            try:
                print(f"Moving to waypoint {i+1}/{len(trajectory)}: {waypoint}")

                # In real implementation, this would be:
                # robot.move_joints(waypoint, dt) or similar

                # Simulate movement time
                time.sleep(dt)

            except Exception as e:
                print(f"Failed to execute waypoint {i}: {e}")
                return False

        print("✓ Trajectory execution completed successfully")
        return True

    except Exception as e:
        print(f"Trajectory execution failed: {e}")
        return False


def print_execution_summary(trajectory_data, success):
    """Print summary of execution results"""
    print("\n" + "="*50)
    print("EXECUTION SUMMARY")
    print("="*50)

    if success:
        print(f"✓ Execution Status: SUCCESS")
        trajectory = trajectory_data['trajectory']
        print(f"  Waypoints Executed: {len(trajectory)}")
        print(f"  Total Duration: {len(trajectory) * 0.1:.2f} seconds")
        print(f"  Start Position: {trajectory[0]}")
        print(f"  End Position: {trajectory[-1]}")
        print(f"  Position Change: {np.linalg.norm(trajectory[-1] - trajectory[0]):.4f}")
    else:
        print(f"✗ Execution Status: FAILED")

    # Print motion parameters
    motion_params = trajectory_data.get('motion_params', {})
    if motion_params:
        print(f"\nMotion Parameters:")
        print(f"  Joint Type: {'Rotation' if motion_params.get('joint_type', 0) == 0 else 'Prismatic'}")
        print(f"  Joint Direction: {motion_params.get('joint_direction', 'Unknown')}")
        print(f"  Joint Base: {motion_params.get('joint_base', 'Unknown')}")

    print("="*50)


def main():
    args = config_parse()

    try:
        print(f"=== Executing Planned Trajectory for {args.global_key} ===")

        # Load planning results
        print("=== Loading planning results ===")
        trajectory_data = load_planning_results(args.root_dir, args.global_key)
        if trajectory_data is None:
            print("Error: Could not load planning results")
            return False

        # Validate trajectory
        if not validate_trajectory(trajectory_data):
            print("Error: Trajectory validation failed")
            return False

        print("\n=== Starting trajectory execution ===")

        # Execute trajectory (simulated for now)
        success = simulate_robot_execution(
            trajectory=trajectory_data['trajectory'],
            dt=args.planning_dt
        )

        if success:
            print("\n=== Execution completed successfully ===")

            # Save execution summary
            execution_summary = {
                'success': True,
                'trajectory_length': len(trajectory_data['trajectory']),
                'total_duration': len(trajectory_data['trajectory']) * args.planning_dt,
                'dt': args.planning_dt,
                'timestamp': time.time()
            }

            summary_path = f"{args.root_dir}/{args.global_key}/planning/execution_summary.npz"
            np.savez(summary_path, **execution_summary)
            print(f"Execution summary saved to {summary_path}")

        else:
            print("\n=== Execution failed ===")
            return False

        # Print final summary
        print_execution_summary(trajectory_data, success)

        return True

    except Exception as e:
        print(f"Error in trajectory execution: {e}")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)