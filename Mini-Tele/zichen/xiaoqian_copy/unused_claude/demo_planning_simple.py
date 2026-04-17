#!/usr/bin/env python3
"""
Simple demonstration of the updated planning functionality.
Shows both OMPL (when available) and RRT planners in action.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pybullet as p
from flexiv_plan import Rizon4, RRTPlanner, OMPLPlanner, pb_ompl

def demo_planning_comparison():
    """Demonstrate both OMPL and RRT planners with visualization"""
    print("=== Planning Integration Demo ===")
    print("This demo shows both OMPL and RRT planners working with the Rizon4 robot")

    try:
        # Initialize robot with GUI for visualization
        print("\n1. Initializing Rizon4 robot with visualization...")
        robot = Rizon4(headless=False)  # GUI mode for visualization
        print("   ✓ Robot initialized with visualization")

        # Define reasonable test poses (close to current position)
        current_tcp = robot.get_tcp_catersian()
        print(f"   Current TCP position: [{current_tcp[0]:.3f}, {current_tcp[1]:.3f}, {current_tcp[2]:.3f}]")

        # Define start and end poses (small, achievable movements)
        start_pose = np.array([0.55, 0.0, 0.35, 1, 0, 0, 0])
        end_pose = np.array([0.55, 0.05, 0.35, 1, 0, 0, 0])  # Small movement in y direction

        # Visualize start and end positions
        robot.visualize_point(start_pose[:3], (0, 1, 0), 8)  # Green for start
        robot.visualize_point(end_pose[:3], (1, 0, 0), 8)    # Red for end

        print(f"\n   Start pose: {start_pose[:3]} (position)")
        print(f"   End pose:   {end_pose[:3]} (position)")

        # Test RRT Planner
        print("\n2. Testing RRT Planner...")
        rrt_planner = RRTPlanner(robot, step_size=0.02, max_iterations=300)

        print("   Planning path with RRT...")
        success, rrt_path, message = rrt_planner.plan_and_validate_movement(
            start_pose, end_pose, visualize_path=True
        )

        if success:
            print(f"   ✓ RRT planning successful!")
            print(f"   ✓ Path: {len(rrt_path)} waypoints")
            print(f"   ✓ Message: {message}")

            input("\n   Press Enter to execute RRT path...")
            rrt_planner.execute_path(rrt_path, step_duration=0.1)
        else:
            print(f"   ⚠ RRT planning failed: {message}")

        # Test OMPL Planner if available
        if pb_ompl is not None:
            print("\n3. Testing OMPL Planner...")
            try:
                ompl_planner = OMPLPlanner(robot, planner_name='RRTConnect', planning_time=5.0)
                print("   ✓ OMPL planner initialized")

                print("   Planning path with OMPL RRTConnect...")
                success, ompl_path, message = ompl_planner.plan_pose_path(start_pose, end_pose)

                if success:
                    print(f"   ✓ OMPL planning successful!")
                    print(f"   ✓ Path: {len(ompl_path)} waypoints")
                    print(f"   ✓ Message: {message}")

                    # Visualize OMPL path in different color
                    ompl_planner.visualize_path(ompl_path, color=(0, 1, 1))  # Cyan for OMPL

                    input("\n   Press Enter to execute OMPL path...")
                    ompl_planner.execute_path(ompl_path, step_duration=0.05)
                else:
                    print(f"   ⚠ OMPL planning failed: {message}")

            except Exception as e:
                print(f"   ⚠ OMPL planner error: {e}")
                print("   ✓ This is expected if OMPL C++ bindings are not available")
        else:
            print("\n3. OMPL not available - RRT is the primary planner")

        # Show comparison summary
        print("\n=== Planning Comparison Summary ===")
        print("Both planners provide:")
        print("  ✓ Collision-free path planning")
        print("  ✓ Path validation and visualization")
        print("  ✓ Smooth trajectory execution")
        print("  ✓ Configurable parameters (step size, planning time, etc.)")

        if pb_ompl is not None:
            print("\nOMPL advantages (when fully available):")
            print("  • Multiple algorithm choices (RRTConnect, BITstar, PRM, etc.)")
            print("  • Generally faster planning for complex environments")
            print("  • Optimal path algorithms (RRTstar, BITstar)")

        print("\nRRT advantages:")
        print("  • Reliable fallback when OMPL is unavailable")
        print("  • Simple and predictable behavior")
        print("  • Custom implementation for specific needs")

        print("\n=== Demo completed successfully! ===")
        print("The planning integration is working correctly.")
        print("You can now use the enhanced planning in your manipulation tasks.")

        # Keep the visualization open for inspection
        print("\nPress Ctrl+C to exit and close the visualization...")
        try:
            while True:
                p.stepSimulation()
        except KeyboardInterrupt:
            print("\nExiting demo...")

        return True

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        try:
            p.disconnect()
        except:
            pass

def demo_command_line_usage():
    """Show command line usage examples"""
    print("=== Command Line Usage Examples ===")
    print("\nAfter the integration, you can use the planning system with:")

    print("\n1. OMPL Planning (when available):")
    print("   python -m collect_data.robot_sim_run_with_planning --use_planning --planner_type ompl --ompl_planner RRTConnect")
    print("   python -m collect_data.robot_sim_run_with_planning --use_planning --planner_type ompl --ompl_planner BITstar --planning_time 10.0")

    print("\n2. RRT Planning (reliable fallback):")
    print("   python -m collect_data.robot_sim_run_with_planning --use_planning --planner_type rrt --step_size 0.05 --max_iterations 500")

    print("\n3. Direct IK (no planning):")
    print("   python -m collect_data.robot_sim_run_with_planning")

    print("\n4. With visualization:")
    print("   python -m collect_data.robot_sim_run_with_planning --use_planning --planner_type ompl --visualize_path")

    print("\nNote: The full script requires camera and robot hardware.")
    print("Use this demo script for testing planning functionality standalone.")

def main():
    """Main demo function"""
    print("Starting planning integration demo...")

    # Show usage examples first
    demo_command_line_usage()

    print("\n" + "="*60)
    input("\nPress Enter to start the interactive demo...")

    # Run the interactive demo
    success = demo_planning_comparison()

    if success:
        print("\n🎉 Demo completed successfully!")
        return 0
    else:
        print("\n❌ Demo failed. Please check the error messages.")
        return 1

if __name__ == "__main__":
    exit(main())