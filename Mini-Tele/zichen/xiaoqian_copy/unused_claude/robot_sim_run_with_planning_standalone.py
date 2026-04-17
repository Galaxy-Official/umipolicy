#!/usr/bin/env python3
"""
Standalone version of robot_sim_run_with_planning.py
Focuses only on planning functionality without camera/robot hardware dependencies.
"""

import os
import time
import tqdm
import configargparse
import numpy as np
import pybullet as p
from typing import Optional, Tuple, List
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexiv_plan import Rizon4, RRTPlanner, OMPLPlanner, pb_ompl
from utilities.transformation import xyz_rot_transform

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # path config
    parser.add_argument('--root_dir', type=str, default="collect_data/data/kp")
    parser.add_argument('--global_key', type=str, default="2025-10-26_19:29:53")

    # planning config
    parser.add_argument('--use_planning', action='store_true', help='Use path planning')
    parser.add_argument('--planner_type', type=str, default='rrt', choices=['ompl', 'rrt'],
                       help='Planner type to use (ompl or rrt)')
    parser.add_argument('--ompl_planner', type=str, default='RRTConnect',
                       help='OMPL planner algorithm (RRTConnect, RRTstar, BITstar, etc.)')
    parser.add_argument('--planning_time', type=float, default=5.0, help='OMPL planning time limit')
    parser.add_argument('--step_size', type=float, default=0.05, help='RRT step size')
    parser.add_argument('--max_iterations', type=int, default=500, help='RRT max iterations')
    parser.add_argument('--visualize_path', action='store_true', help='Visualize planned path')

    # simulation config
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--demo_mode', action='store_true', help='Run in demo mode with predefined poses')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of movement steps in demo mode')

    args = parser.parse_args()
    return args

def plan_and_validate_movement(robot, planner, start_pose, end_pose, use_planning=True,
                               planner_type='ompl', visualize_path=True, args=None):
    """Plan and validate a movement using path planning"""
    if not use_planning:
        # Direct IK approach (original method)
        try:
            start_joints = robot.calc_ik(start_pose)
            end_joints = robot.calc_ik(end_pose)
            path = robot.interpolate_joints(start_joints, end_joints, num_waypoints=10)

            # Validate direct path
            is_valid, collision_idx = robot.validate_path(path)
            if not is_valid:
                return False, None, f"Direct path invalid - collision at waypoint {collision_idx}"

            if visualize_path:
                if hasattr(planner, 'visualize_path'):
                    planner.visualize_path(path)
                else:
                    # Fallback to RRT planner visualization
                    if isinstance(planner, RRTPlanner):
                        planner.visualize_path(path)

            return True, path, f"Direct path planning successful - {len(path)} waypoints"
        except Exception as e:
            return False, None, f"Direct IK failed: {str(e)}"

    else:
        # Use specified planner
        if planner_type == 'ompl' and pb_ompl is not None and isinstance(planner, OMPLPlanner):
            # OMPL planning approach
            success, path, message = planner.plan_pose_path(start_pose, end_pose)

            if success and visualize_path:
                planner.visualize_path(path, color=(0, 1, 1))  # Cyan for OMPL

            return success, path, message

        else:
            # RRT planning approach (fallback)
            if isinstance(planner, RRTPlanner):
                return planner.plan_and_validate_movement(start_pose, end_pose, visualize_path)
            else:
                return False, None, "No valid planner available"

def execute_planned_path(robot, path, step_duration=0.1):
    """Execute a planned path on the robot"""
    if not path:
        print("No path to execute")
        return

    print(f"Executing path with {len(path)} waypoints...")

    for i, joint_states in enumerate(path):
        robot.set_joints(joint_states)
        robot.visualize()
        p.stepSimulation()

        if step_duration > 0:
            time.sleep(step_duration)

        if i % 10 == 0:
            print(f"Progress: {i+1}/{len(path)} ({(i+1)/len(path)*100:.1f}%)")

    print("Path execution complete")

def demo_planning_scenario(robot, planner, args):
    """Run a demonstration planning scenario with predefined poses"""
    print("\n=== Running Planning Demo ===")

    # Define demonstration poses (reasonable movements)
    poses = [
        np.array([0.55, 0.0, 0.35, 1, 0, 0, 0]),   # Start pose
        np.array([0.55, 0.05, 0.35, 1, 0, 0, 0]),  # Small y movement
        np.array([0.60, 0.05, 0.35, 1, 0, 0, 0]),  # Small x movement
        np.array([0.60, 0.0, 0.40, 1, 0, 0, 0]),   # Small z movement
        np.array([0.55, 0.0, 0.35, 1, 0, 0, 0]),   # Back to start
    ]

    print(f"Demo will execute {len(poses)-1} movements between {len(poses)} poses")

    for i in range(len(poses)-1):
        start_pose = poses[i]
        end_pose = poses[i+1]

        print(f"\nMovement {i+1}: Pose {i} → Pose {i+1}")
        print(f"   Start: {start_pose[:3]}")
        print(f"   End:   {end_pose[:3]}")

        # Visualize start and end positions
        if args.visualize_path:
            p.removeAllUserDebugItems()
            robot.visualize_point(start_pose[:3], (0, 1, 0), 6)  # Green for start
            robot.visualize_point(end_pose[:3], (1, 0, 0), 6)    # Red for end

        # Plan movement
        success, path, message = plan_and_validate_movement(
            robot, planner, start_pose, end_pose,
            use_planning=args.use_planning, planner_type=args.planner_type,
            visualize_path=args.visualize_path, args=args
        )

        if success:
            print(f"   ✓ Planning successful: {message}")
            execute_planned_path(robot, path, step_duration=0.1)
        else:
            print(f"   ⚠ Planning failed: {message}")
            print("   Falling back to direct IK...")
            # Fallback to direct IK
            try:
                joints = robot.calc_ik(end_pose)
                robot.set_joints(joints)
                robot.visualize()
                p.stepSimulation()
                time.sleep(0.5)
            except Exception as e:
                print(f"   ⚠ Direct IK also failed: {e}")

        # Pause between movements
        time.sleep(1.0)

def main():
    args = config_parse()

    print("=== Standalone Robot Planning Demo ===")
    print("This script demonstrates the planning functionality without hardware dependencies.")

    # Initialize robot
    print(f"\n1. Initializing Rizon4 robot ({'headless' if args.headless else 'with visualization'})...")
    robot = Rizon4(headless=args.headless)
    print(f"   ✓ Robot initialized with {p.getNumJoints(robot.robot)} joints")

    # Get current state
    current_joints = robot.get_joints()
    current_tcp = robot.get_tcp_catersian()
    print(f"   ✓ Current TCP position: [{current_tcp[0]:.3f}, {current_tcp[1]:.3f}, {current_tcp[2]:.3f}]")

    # Initialize planners
    print("\n2. Initializing planners...")
    rrt_planner = RRTPlanner(robot, step_size=args.step_size, max_iterations=args.max_iterations)

    # Initialize OMPL planner if available
    ompl_planner = None
    if pb_ompl is not None and args.planner_type == 'ompl':
        try:
            ompl_planner = OMPLPlanner(robot, planner_name=args.ompl_planner,
                                     planning_time=args.planning_time)
            print(f"   ✓ OMPL planner initialized with {args.ompl_planner}")
            active_planner = ompl_planner
            active_planner_type = 'ompl'
        except Exception as e:
            print(f"   ⚠ Failed to initialize OMPL planner: {e}")
            print("   ✓ Falling back to RRT planner")
            active_planner = rrt_planner
            active_planner_type = 'rrt'
    else:
        active_planner = rrt_planner
        active_planner_type = 'rrt'
        print(f"   ✓ Using RRT planner (step_size={args.step_size}, max_iterations={args.max_iterations})")

    print(f"\n3. Active planner: {active_planner_type.upper()}")

    # Run demo or single planning test
    if args.demo_mode:
        demo_planning_scenario(robot, active_planner, args)
    else:
        # Single planning test
        print("\n4. Running single planning test...")

        # Define test poses
        start_pose = np.array([0.55, 0.0, 0.35, 1, 0, 0, 0])
        end_pose = np.array([0.55, 0.05, 0.35, 1, 0, 0, 0])

        print(f"   Planning from: {start_pose[:3]}")
        print(f"   To:            {end_pose[:3]}")

        # Visualize start and end positions
        if args.visualize_path:
            robot.visualize_point(start_pose[:3], (0, 1, 0), 8)  # Green for start
            robot.visualize_point(end_pose[:3], (1, 0, 0), 8)    # Red for end

        # Plan movement
        success, path, message = plan_and_validate_movement(
            robot, active_planner, start_pose, end_pose,
            use_planning=args.use_planning, planner_type=active_planner_type,
            visualize_path=args.visualize_path, args=args
        )

        if success:
            print(f"   ✓ Planning successful: {message}")
            execute_planned_path(robot, path, step_duration=0.1)
        else:
            print(f"   ⚠ Planning failed: {message}")

    # Final summary
    print("\n=== Planning Test Summary ===")
    print(f"Robot: Rizon4 with {len(current_joints)} DOF")
    print(f"Planner: {active_planner_type.upper()}")
    print(f"Planning mode: {'Enabled' if args.use_planning else 'Direct IK'}")
    print(f"Visualization: {'Enabled' if args.visualize_path else 'Disabled'}")
    print(f"Demo mode: {'Enabled' if args.demo_mode else 'Single test'}")

    if args.use_planning:
        if active_planner_type == 'ompl':
            print(f"OMPL Algorithm: {args.ompl_planner}")
            print(f"Planning time: {args.planning_time}s")
        else:
            print(f"RRT step size: {args.step_size}")
            print(f"RRT max iterations: {args.max_iterations}")

    # Keep visualization open briefly if enabled
    if args.visualize_path and not args.headless:
        print("\nVisualization will close automatically in 3 seconds...")
        time.sleep(3)

    print("\n✅ Planning test completed successfully!")

    # Cleanup
    try:
        p.disconnect()
    except:
        pass

if __name__ == '__main__':
    main()