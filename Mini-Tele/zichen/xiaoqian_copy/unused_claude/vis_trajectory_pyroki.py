"""
Pyroki Trajectory Visualization

This script visualizes planned trajectories using Pyroki for 3D TCP pose display.
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

# Import motion planning
try:
    # Try to import full Pyroki version
    import sys
    sys.path.append('/Disk3/xiaoqian')
    from motion_planning import create_motion_planner
    PYROKI_AVAILABLE = True
except ImportError:
    # Fallback to simplified version
    print("Pyroki not available, using simplified motion planner")
    from motion_planning_simple import create_simple_motion_planner
    def create_motion_planner(urdf_path=None, visualization=True):
        return create_simple_motion_planner(urdf_path=urdf_path, visualization=visualization)
    PYROKI_AVAILABLE = False

# Import URDF and viser extras for robot visualization
try:
    import yourdfpy
    from viser.extras import ViserUrdf
    URDF_AVAILABLE = True
except ImportError:
    URDF_AVAILABLE = False
    print("URDF visualization not available")


def config_parse():
    parser = argparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str, required=True,
                       help='Global key for the data session')
    parser.add_argument('--trajectory_file', type=str, default="tcp_poses.npy",
                       help='Trajectory file to visualize')
    parser.add_argument('--enable_interactive', action='store_true', default=True,
                       help='Enable interactive trajectory visualization')

    args = parser.parse_args()
    return args


def load_trajectory_data(root_dir, global_key, trajectory_file):
    """Load trajectory data for visualization"""
    try:
        trajectory_path = f"{root_dir}/{global_key}/planning/{trajectory_file}"
        if not os.path.exists(trajectory_path):
            print(f"Error: Trajectory file not found at {trajectory_path}")
            return None

        tcp_poses = np.load(trajectory_path)
        print(f"Loaded {len(tcp_poses)} TCP poses from {trajectory_path}")
        return tcp_poses

    except Exception as e:
        print(f"Error loading trajectory data: {e}")
        return None


def create_simple_visualization(tcp_poses):
    """Create simple text-based visualization of TCP poses"""
    print("\n=== TCP Pose Trajectory Visualization ===")
    print(f"Total waypoints: {len(tcp_poses)}")

    print("\nTrajectory waypoints:")
    print("Waypoint | Position (X, Y, Z)")
    print("-" * 40)

    for i, pose in enumerate(tcp_poses):
        pos = pose[:3, 3]
        print(f"   {i+1:2d}    | ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # Show start and end comparison
    if len(tcp_poses) > 1:
        start_pos = tcp_poses[0][:3, 3]
        end_pos = tcp_poses[-1][:3, 3]
        delta = end_pos - start_pos
        distance = np.linalg.norm(delta)

        print(f"\nTrajectory summary:")
        print(f"  Start position: {start_pos}")
        print(f"  End position:   {end_pos}")
        print(f"  Total movement: {distance:.3f} meters")
        print(f"  Movement vector: {delta}")

    print("=" * 50)

def _create_simple_robot_visualization(server, num_joints=7):
    """Create a simple robot arm visualization using basic shapes when URDF is not available"""
    try:
        print(f"Creating simplified robot visualization with {num_joints} joints")

        # Create base link
        server.scene.add_box(
            "/robot_base",
            position=np.array([0.0, 0.0, 0.0]),
            size=np.array([0.1, 0.1, 0.05]),
            color=np.array([0.5, 0.5, 0.5])
        )

        # Create simple link representations
        for i in range(num_joints):
            # Simple cylindrical links
            server.scene.add_icosphere(
                f"/robot_joint_{i}",
                radius=0.02,
                position=np.array([0.0, 0.0, 0.1 + i * 0.08]),
                color=np.array([0.2, 0.2, 0.8])
            )

            # Link between joints
            if i > 0:
                server.scene.add_line_segments(
                    f"/robot_link_{i-1}_{i}",
                    np.array([0.0, 0.0, 0.1 + (i-1) * 0.08, 0.0, 0.0, 0.1 + i * 0.08], dtype=np.float32).reshape(1, 6),
                    color=np.array([0.3, 0.3, 0.3]),
                    line_width=3.0
                )

        print("✓ Simplified robot visualization created")
        return True
    except Exception as e:
        print(f"Failed to create simplified robot visualization: {e}")
        return False

def _estimate_joint_angles_from_tcp(tcp_pose, num_joints=7):
    """
    Estimate joint angles from TCP pose using a simplified heuristic approach.
    This is a placeholder - in reality you would use proper inverse kinematics.
    """
    try:
        # Extract position and orientation from TCP pose
        position = tcp_pose[:3, 3]
        rotation_matrix = tcp_pose[:3, :3]

        # Simple heuristic: map TCP position to joint angles
        # This is a very simplified approach for visualization purposes

        # Base joint (rotation around Z) based on X-Y position
        base_angle = np.arctan2(position[1], position[0])

        # Shoulder joint based on vertical position
        shoulder_angle = np.arctan2(position[2] - 0.3, np.sqrt(position[0]**2 + position[1]**2))

        # Elbow joint based on distance from base
        distance_from_base = np.linalg.norm(position[:2])
        elbow_angle = max(-np.pi/2, min(np.pi/2, distance_from_base - 0.5))

        # Wrist joints based on orientation (simplified)
        # Extract rotation angles (very rough approximation)
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Create joint angle array
        joint_angles = np.array([
            base_angle,           # Joint 1: base rotation
            shoulder_angle,       # Joint 2: shoulder
            elbow_angle,          # Joint 3: elbow
            elbow_angle * 0.5,    # Joint 4: forearm
            pitch * 0.3,          # Joint 5: wrist pitch
            roll * 0.3,           # Joint 6: wrist roll
            yaw * 0.3             # Joint 7: wrist yaw
        ])

        # Apply joint limits (Flexiv Rizon 4 approximate limits)
        joint_limits = np.array([
            [-2.9671, 2.9671],   # J1
            [-1.7453, 1.7453],   # J2
            [-2.9671, 2.9671],   # J3
            [-3.4907, 1.7453],   # J4
            [-2.9671, 2.9671],   # J5
            [-0.7854, 3.4907],   # J6
            [-2.9671, 2.9671]    # J7
        ])

        # Clamp joint angles to limits
        for i in range(len(joint_angles)):
            joint_angles[i] = np.clip(joint_angles[i], joint_limits[i][0], joint_limits[i][1])

        return joint_angles

    except Exception as e:
        print(f"Error estimating joint angles: {e}")
        return np.zeros(num_joints)  # Return home position on error

def visualize_with_pyroki_interactive(tcp_poses):
    """Create interactive Pyroki visualization with robot arm motion following TCP trajectory"""
    if not PYROKI_AVAILABLE:
        print("Pyroki not available for interactive visualization")
        return False

    try:
        print("=== Setting up Pyroki interactive visualization with robot arm motion ===")

        # Create motion planner with visualization
        planner = create_motion_planner(visualization=True)

        if not hasattr(planner, 'server') or not planner.server:
            print("Pyroki server not available")
            return False

        server = planner.server
        print(f"✓ Pyroki visualization server started at http://localhost:8080")

        # Add grid for reference
        server.scene.add_grid("/ground", width=2, height=2)

        # Load and visualize robot URDF if available
        urdf_vis = None
        robot_joints = None
        num_robot_joints = 7  # Flexiv Rizon 4 has 7 joints

        if URDF_AVAILABLE:
            try:
                # Try to load Flexiv robot URDF
                flexiv_urdf_path = "/Disk3/xiaoqian/flexiv_rizon4_kinematics/urdf/flexiv_rizon4.urdf"
                if os.path.exists(flexiv_urdf_path):
                    print(f"Loading Flexiv URDF from: {flexiv_urdf_path}")
                    urdf = yourdfpy.URDF.load(flexiv_urdf_path)
                    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

                    # Initialize robot joints to home position (all zeros)
                    robot_joints = np.zeros(num_robot_joints)
                    urdf_vis.update_cfg(robot_joints)
                    print(f"✓ Robot URDF loaded successfully with {num_robot_joints} joints")
                    print(f"✓ Robot initialized at home position: {robot_joints}")
                else:
                    print("Flexiv URDF not found, using simplified robot visualization")
                    # Create a simple robot representation using basic shapes
                    self._create_simple_robot_visualization(server, num_robot_joints)
            except Exception as e:
                print(f"Could not load robot URDF: {e}")
                print("Falling back to simplified robot visualization")
                _create_simple_robot_visualization(server, num_robot_joints)
        else:
            print("URDF visualization not available, using simplified robot visualization")
            _create_simple_robot_visualization(server, num_robot_joints)

        # Add trajectory controls
        slider = server.gui.add_slider(
            "Trajectory Waypoint", min=0, max=len(tcp_poses)-1, step=1, initial_value=0
        )

        info_text = server.gui.add_text(
            "Waypoint Info",
            initial_value=f"Total waypoints: {len(tcp_poses)} | Robot joints: {num_robot_joints}DOF"
        )

        # Add robot joint info display
        joint_info_text = server.gui.add_text(
            "Robot Joints",
            initial_value="Joints: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]"
        )

        playing = server.gui.add_checkbox("Auto Play", initial_value=False)
        playback_speed = server.gui.add_slider(
            "Playback Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
        )

        def update_visualization():
            """Update visualization based on slider"""
            try:
                current_waypoint = int(slider.value)

                if current_waypoint < len(tcp_poses):
                    current_pose = tcp_poses[current_waypoint]

                    # Debug info
                    print(f"Updating visualization for waypoint {current_waypoint}, pose type: {type(current_pose)}")

                    # Ensure current_pose is numpy array and 4x4 matrix
                    if isinstance(current_pose, (list, np.ndarray)):
                        current_pose = np.array(current_pose, dtype=np.float32)
                        if current_pose.shape != (4, 4):
                            print(f"Warning: Expected 4x4 matrix, got shape {current_pose.shape}")
                            return
                    else:
                        print(f"Error: current_pose is not list or ndarray, type: {type(current_pose)}")
                        return

                    # Add TCP frame - ensure all parameters are correctly formatted
                    position_array = current_pose[:3, 3]
                    if isinstance(position_array, np.ndarray):
                        position_np = position_array.astype(np.float32)
                    else:
                        position_np = np.array([float(position_array[0]), float(position_array[1]), float(position_array[2])], dtype=np.float32)

                    print(f"Adding frame at position: {position_np}")

                    # Update robot arm position based on TCP pose
                    if urdf_vis is not None:
                        # Estimate joint angles from TCP pose
                        joint_angles = _estimate_joint_angles_from_tcp(current_pose, num_robot_joints)
                        urdf_vis.update_cfg(joint_angles)

                        # Update joint info display
                        joint_str = "["
                        for i, angle in enumerate(joint_angles):
                            joint_str += f"{angle:.2f}"
                            if i < len(joint_angles) - 1:
                                joint_str += ", "
                        joint_str += "]"
                        joint_info_text.value = f"Joints: {joint_str}"
                        print(f"Updated robot joints: {joint_angles}")

                    server.scene.add_frame(
                        f"/tcp_waypoint_{current_waypoint}",
                        position=position_np,  # Must be numpy array, not list
                        wxyz=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # Identity quaternion as numpy array
                        axes_length=0.05,
                        axes_radius=0.01
                    )

                    # Add trajectory path
                    if current_waypoint > 0:
                        # Draw line from previous to current position
                        prev_pose = tcp_poses[current_waypoint-1]
                        if isinstance(prev_pose, (list, np.ndarray)):
                            prev_pose = np.array(prev_pose, dtype=np.float32)
                            if prev_pose.shape != (4, 4):
                                print(f"Warning: Expected 4x4 matrix for prev pose, got shape {prev_pose.shape}")
                                return
                        else:
                            print(f"Error: prev_pose is not list or ndarray, type: {type(prev_pose)}")
                            return

                        # Prepare line points - ensure proper numpy array format
                        prev_pos = prev_pose[:3, 3]
                        curr_pos = current_pose[:3, 3]

                        # Convert to numpy arrays if not already
                        if isinstance(prev_pos, np.ndarray):
                            prev_pos_np = prev_pos.astype(np.float32)
                        else:
                            prev_pos_np = np.array([float(prev_pos[0]), float(prev_pos[1]), float(prev_pos[2])], dtype=np.float32)

                        if isinstance(curr_pos, np.ndarray):
                            curr_pos_np = curr_pos.astype(np.float32)
                        else:
                            curr_pos_np = np.array([float(curr_pos[0]), float(curr_pos[1]), float(curr_pos[2])], dtype=np.float32)

                        print(f"Adding line from {prev_pos_np} to {curr_pos_np}")

                        # Use icospheres to represent trajectory points instead of add_line
                        # Since add_line doesn't exist, we'll use icospheres to represent the trajectory
                        server.scene.add_icosphere(
                            f"/trajectory_point_prev_{current_waypoint}",
                            radius=0.01,
                            position=prev_pos_np,
                            color=np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        )
                        server.scene.add_icosphere(
                            f"/trajectory_point_curr_{current_waypoint}",
                            radius=0.01,
                            position=curr_pos_np,
                            color=np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        )

                        # Use add_line_segments to create trajectory lines
                        # This method expects a numpy array of line segments
                        line_segments = np.array([
                            prev_pos_np[0], prev_pos_np[1], prev_pos_np[2],
                            curr_pos_np[0], curr_pos_np[1], curr_pos_np[2]
                        ], dtype=np.float32).reshape(1, 6)  # Shape: (1, 6) for one line segment

                        server.scene.add_line_segments(
                            f"/trajectory_line_{current_waypoint}",
                            line_segments,
                            color=np.array([1.0, 0.2, 0.2], dtype=np.float32),
                            line_width=2.0
                        )

                    # Update info - ensure position is properly formatted
                    pos = current_pose[:3, 3]
                    if isinstance(pos, np.ndarray):
                        pos_list = pos.tolist()
                    else:
                        pos_list = [float(pos[0]), float(pos[1]), float(pos[2])]
                    info_text.value = f"Waypoint {current_waypoint + 1}/{len(tcp_poses)}: ({pos_list[0]:.3f}, {pos_list[1]:.3f}, {pos_list[2]:.3f})"
                    print(f"Updated info text: {info_text.value}")

                    # Clean up previous elements (except the first few)
                    if current_waypoint > 2:
                        try:
                            server.scene.remove(f"/tcp_waypoint_{current_waypoint-3}")
                            server.scene.remove(f"/trajectory_point_prev_{current_waypoint-2}")
                            server.scene.remove(f"/trajectory_point_curr_{current_waypoint-2}")
                            # Clean up trajectory line segments
                            server.scene.remove(f"/trajectory_line_{current_waypoint-2}")
                        except Exception as cleanup_e:
                            print(f"Cleanup warning: {cleanup_e}")

            except Exception as e:
                print(f"Visualization update error: {e}")
                import traceback
                traceback.print_exc()

        # Set up update loop
        def visualization_loop():
            sleep_time = 0.1 / playback_speed.value
            while True:
                if playing.value:
                    slider.value = (slider.value + 1) % len(tcp_poses)
                update_visualization()
                time.sleep(sleep_time)

        # Start visualization thread
        viz_thread = threading.Thread(target=visualization_loop, daemon=True)
        viz_thread.start()

        print("✓ Pyroki interactive visualization setup complete!")
        print("Web interface: http://localhost:8080")
        print("Use the slider to navigate waypoints")
        print("Robot arm will follow the TCP trajectory in real-time!")
        if urdf_vis is not None:
            print("Robot URDF model is loaded and will move with trajectory")
        else:
            print("Using simplified robot visualization")
        print("Press Ctrl+C to stop visualization...")

        # Keep visualization running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down visualization...")

        return True

    except Exception as e:
        print(f"Pyroki interactive visualization failed: {e}")
        return False

def main():
    args = config_parse()

    try:
        print(f"=== Pyroki Trajectory Visualization for {args.global_key} ===")

        # Load trajectory data
        tcp_poses = load_trajectory_data(args.root_dir, args.global_key, args.trajectory_file)
        if tcp_poses is None:
            return False

        print(f"Loaded trajectory with {len(tcp_poses)} waypoints")

        # Create basic visualization
        create_simple_visualization(tcp_poses)

        # Create interactive Pyroki visualization if available
        if args.enable_interactive and PYROKI_AVAILABLE:
            success = visualize_with_pyroki_interactive(tcp_poses)
            if success:
                print("✓ Interactive visualization completed successfully!")
            else:
                print("✗ Interactive visualization failed")
                return False
        else:
            print("Note: Interactive visualization requires Pyroki installation")
            print("To enable: pip install pyroki viser")

        return True

    except Exception as e:
        print(f"Error in trajectory visualization: {e}")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)