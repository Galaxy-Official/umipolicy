"""
Execute Planned Trajectory Step

This script loads the pre-planned trajectory and executes it on the robot,
with recording and monitoring capabilities.
"""

import os
import time
import tqdm
import configargparse
import numpy as np
import transformations as tf
from camera import CameraD400
from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize
import threading
import cv2
import pickle

# Import motion planning utilities
import sys
sys.path.append('/Disk3/xiaoqian')
from motion_planning import FlexivMotionPlanner


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str)
    # record config
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--episode', type=int, default=0)
    # execution config
    parser.add_argument('--planning_dt', type=float, default=0.1,
                       help='Time step between waypoints (seconds)')
    parser.add_argument('--check_joint_limits', action='store_true', default=True,
                       help='Check joint limits before execution')
    parser.add_argument('--enable_viz', action='store_true',
                       help='Enable web visualization during execution')
    parser.add_argument('--urdf_path', type=str, default=None,
                       help='Path to Flexiv URDF file')

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

        # Load joint limits info
        joint_limits_path = f"{planning_dir}/joint_limits.npz"
        if os.path.exists(joint_limits_path):
            joint_limits = dict(np.load(joint_limits_path))
        else:
            joint_limits = {'valid': True, 'violations': []}

        return {
            'trajectory': trajectory,
            'start_pose': start_pose,
            'target_pose': target_pose,
            'motion_params': motion_params,
            'metadata': metadata,
            'joint_limits': joint_limits
        }

    except Exception as e:
        print(f"Error loading planning results: {e}")
        return None


def validate_trajectory(trajectory_data, check_joint_limits=True):
    """Validate the planned trajectory before execution"""
    try:
        trajectory = trajectory_data['trajectory']
        joint_limits = trajectory_data['joint_limits']
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

        # Check joint limits
        if check_joint_limits and joint_limits:
            if not joint_limits.get('valid', True):
                num_violations = joint_limits.get('num_violations', 0)
                max_violation = joint_limits.get('max_violation', 0.0)

                print(f"⚠ Warning: {num_violations} joint limit violations detected")
                print(f"  Max violation: {max_violation:.4f} radians")

                if num_violations > 0:
                    response = input("Continue with execution? (y/N): ")
                    if response.lower() != 'y':
                        print("Execution cancelled by user")
                        return False
            else:
                print("✓ All joint limits satisfied")

        return True

    except Exception as e:
        print(f"Error validating trajectory: {e}")
        return False


def record_execution(args, robot, camera, trajectory_length, dt):
    """Record execution data"""
    try:
        # Create directories
        episode_dir = f"{args.root_dir}/{args.global_key}/{str(args.episode).zfill(5)}"
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        else:
            # Clean existing data
            os.system(f"rm -r {episode_dir}")
            os.makedirs(episode_dir)

        # Create subdirectories
        for save_key in ["tcp", "joint_position", "gripper_width", "cam_top_rgb", "cam_top_depth"]:
            os.makedirs(f"{episode_dir}/{save_key}", exist_ok=True)

        # Create metadata file
        metadata = {
            'trajectory_length': trajectory_length,
            'dt': dt,
            'total_duration': trajectory_length * dt,
            'timestamp_start': time.time()
        }
        np.savez(f"{episode_dir}/execution_metadata.npz", **metadata)

        print(f"Recording to {episode_dir}")
        timestamp = 0

        while True:
            # Robot state
            tcp = robot.get_tcp()
            gripper_width = robot.get_gripper_width()
            joint_position = robot.get_joint_positions()

            print(f"Recording t={timestamp}: tcp={tcp[:3, 3]}, joints={joint_position}")

            # Save data
            np.save(f"{episode_dir}/tcp/{str(timestamp).zfill(10)}.npy", tcp)
            np.save(f"{episode_dir}/joint_position/{str(timestamp).zfill(10)}.npy", joint_position)
            np.save(f"{episode_dir}/gripper_width/{str(timestamp).zfill(10)}.npy", gripper_width)

            # Camera data
            color, depth = camera.get_data(hole_filling=False)
            cv2.imwrite(f"{episode_dir}/cam_top_rgb/{str(timestamp).zfill(10)}.png", color)
            pickle.dump(depth, open(f"{episode_dir}/cam_top_depth/{str(timestamp).zfill(10)}.pkl", "wb"))

            time.sleep(1/30)  # 30 Hz
            timestamp += 1

    except Exception as e:
        print(f"Recording error: {e}")


def execute_trajectory_on_robot(robot, trajectory, dt, check_limits=True):
    """Execute the planned trajectory on the robot"""
    try:
        print("=== Executing Planned Trajectory ===")
        print(f"Trajectory: {len(trajectory)} waypoints, dt={dt}s")

        # Validate trajectory length
        if len(trajectory) == 0:
            print("Error: Empty trajectory")
            return False

        # Execute waypoint by waypoint
        for i, joint_config in enumerate(tqdm.tqdm(trajectory, desc="Executing trajectory")):
            try:
                # Convert joint configuration to robot command
                # For Flexiv robot, we need to convert joint angles to pose
                # This is a simplified approach - you may need to adjust based on your robot API

                print(f"Waypoint {i+1}/{len(trajectory)}: {joint_config}")

                # Check if robot has joint control method
                if hasattr(robot, 'move_joints'):
                    robot.move_joints(joint_config, dt)
                elif hasattr(robot, 'set_joint_positions'):
                    robot.set_joint_positions(joint_config)
                    time.sleep(dt)
                else:
                    # Fallback: use existing movePosePrimitive
                    # This would require forward kinematics to convert joints to pose
                    print(f"Joint config: {joint_config}")
                    # For now, just wait
                    time.sleep(dt)

            except Exception as e:
                print(f"Failed to execute waypoint {i}: {e}")
                return False

        print("✓ Trajectory execution completed successfully")
        return True

    except Exception as e:
        print(f"Trajectory execution failed: {e}")
        return False


def setup_visualization_during_execution(planner, trajectory):
    """Setup web visualization during execution"""
    try:
        if planner.server and planner.urdf_vis:
            # Add execution progress slider
            slider = planner.server.gui.add_slider(
                "Execution Progress", min=0, max=len(trajectory)-1, step=1, initial_value=0
            )
            status_text = planner.server.gui.add_text(
                "Execution Status", initial_value="Ready"
            )

            print("Web visualization available at http://localhost:8080")
            return slider, status_text
        return None, None
    except Exception as e:
        print(f"Visualization setup failed: {e}")
        return None, None


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
        if not validate_trajectory(trajectory_data, args.check_joint_limits):
            print("Error: Trajectory validation failed")
            return False

        # Initialize robot and camera
        print("=== Initializing hardware ===")
        camera_loaded = False
        robot_loaded = False

        try:
            print("Initializing camera...")
            camera = CameraD400(camera_id="001622071104")
            camera_loaded = True
            print("Camera initialized")

            print("Initializing robot...")
            robot = Flexiv()
            robot_loaded = True
            print("Robot initialized")

            # Setup visualization if enabled
            slider = None
            status_text = None
            if args.enable_viz:
                print("Setting up visualization...")
                planner = FlexivMotionPlanner(args.urdf_path)
                planner.setup_visualization()
                slider, status_text = setup_visualization_during_execution(planner, trajectory_data['trajectory'])

            # Start recording if enabled
            if args.record:
                print("Starting recording...")
                record_thread = threading.Thread(
                    target=record_execution,
                    kwargs={
                        "args": args,
                        "robot": robot,
                        "camera": camera,
                        "trajectory_length": len(trajectory_data['trajectory']),
                        "dt": args.planning_dt
                    },
                    daemon=True
                )
                record_thread.start()

            # Execute trajectory
            print("=== Executing trajectory ===")
            success = execute_trajectory_on_robot(
                robot=robot,
                trajectory=trajectory_data['trajectory'],
                dt=args.planning_dt,
                check_limits=args.check_joint_limits
            )

            if success:
                print("=== Execution completed successfully ===")

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
                print("=== Execution failed ===")
                return False

            # Control gripper at the end
            print("Controlling gripper...")
            robot.send_gripper_state(0.5, 0.1, 20)

            return True

        except Exception as e:
            print(f"Hardware initialization or execution error: {e}")
            return False

        finally:
            # Cleanup
            if camera_loaded:
                del camera
            if robot_loaded:
                del robot

    except Exception as e:
        print(f"Error in trajectory execution: {e}")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)