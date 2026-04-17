"""Debug IK accuracy and precision issues

This script helps identify why IK solutions are not meeting accuracy requirements.
"""

import numpy as np
import sys
sys.path.append('/Disk3/xiaoqian/pyroki/examples')

def debug_ik_accuracy():
    """Debug the IK accuracy issues."""

    print("🔍 DEBUGGING IK ACCURACY ISSUES")
    print("=" * 60)

    try:
        # Import required modules
        import pyroki as pk
        import yourdfpy
        from pathlib import Path
        import jaxls
        import jaxlie
        import jax

        # Load robot
        cache_dir = Path.home() / ".cache" / "robot_descriptions"
        if cache_dir.exists():
            panda_files = list(cache_dir.glob("**/panda*.urdf"))
            if panda_files:
                urdf = yourdfpy.URDF.load(str(panda_files[0]))
            else:
                from robot_descriptions.loaders.yourdfpy import load_robot_description
                urdf = load_robot_description("panda_description")
        else:
            from robot_descriptions.loaders.yourdfpy import load_robot_description
            urdf = load_robot_description("panda_description")

        robot = pk.Robot.from_urdf(urdf)
        print(f"✅ Robot loaded with {robot.joints.num_actuated_joints} actuated joints")

        # Test target pose
        target_position = np.array([0.5, 0.0, 0.5])
        target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        print(f"\n🎯 Target pose:")
        print(f"  Position: {target_position}")
        print(f"  Quaternion: {target_quaternion}")

        # Create optimization variable
        joint_var = robot.joint_var_cls(0)

        # Create target pose as SE(3) transformation
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_quaternion),
            target_position
        )

        # Get target link index
        target_link_index = robot.links.names.index("tcp")
        target_link_index_jax = jax.numpy.array(target_link_index)

        # Create pose match cost with different weight configurations
        print(f"\n⚙️  Testing different cost weights:")

        weight_configs = [
            (50.0, 10.0),   # Original weights
            (100.0, 20.0),  # Higher weights
            (200.0, 50.0),  # Much higher weights
            (500.0, 100.0), # Very high weights
        ]

        for pos_weight, ori_weight in weight_configs:
            print(f"\n📊 Testing weights: pos={pos_weight}, ori={ori_weight}")

            try:
                # Create pose match cost
                pose_cost = pk.costs.pose_cost_analytic_jac(
                    robot=robot,
                    joint_var=joint_var,
                    target_pose=target_pose,
                    target_link_index=target_link_index_jax,
                    pos_weight=pos_weight,
                    ori_weight=ori_weight
                )

                # Create joint limit cost
                joint_limit_cost = pk.costs.limit_cost(
                    robot=robot,
                    joint_var=joint_var,
                    weight=100.0
                )

                # Create optimization problem
                problem = jaxls.LeastSquaresProblem(
                    costs=[pose_cost, joint_limit_cost],
                    variables=[joint_var]
                )

                # Solve with different configurations
                solution = problem.analyze().solve(
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0)
                )

                # Extract solution
                optimal_joints = solution[joint_var]

                # Compute forward kinematics to verify
                fk_result = robot.forward_kinematics(optimal_joints)
                tcp_pose_data = fk_result[target_link_index]

                # Extract final pose
                final_position = tcp_pose_data[4:7]
                final_quaternion = np.array([tcp_pose_data[3], tcp_pose_data[0], tcp_pose_data[1], tcp_pose_data[2]])

                # Calculate errors
                position_error = np.linalg.norm(final_position - target_position)
                quaternion_error = np.linalg.norm(final_quaternion - target_quaternion)

                print(f"  📍 Position error: {position_error:.8f} meters ({position_error*1000:.3f} mm)")
                print(f"  🔄 Quaternion error: {quaternion_error:.8f}")
                print(f"  🎯 Final position: {final_position}")
                print(f"  🎯 Final quaternion: {final_quaternion}")

                # Check if solution meets different tolerance levels
                tolerances = [
                    (0.001, 0.01),   # 1mm, 0.01 rad (original)
                    (0.005, 0.05),   # 5mm, 0.05 rad
                    (0.010, 0.10),   # 1cm, 0.1 rad
                    (0.050, 0.50),   # 5cm, 0.5 rad
                ]

                for pos_tol, quat_tol in tolerances:
                    if position_error < pos_tol and quaternion_error < quat_tol:
                        print(f"  ✅ Meets tolerance: {pos_tol*1000:.1f}mm, {quat_tol:.3f}rad")
                        break
                else:
                    print(f"  ❌ Does not meet any tolerance levels")

                # Show joint angles
                print(f"  🔧 Joint angles: {optimal_joints}")

                # Test with better initial guess if first attempt fails
                if position_error > 0.010:  # More than 1cm error
                    print(f"  🔄 Trying with better initial guess...")
                    # Use current solution as initial guess for next iteration
                    joint_var = robot.joint_var_cls(0)  # Reset variable

                    # Create new costs with same weights but better initial guess
                    pose_cost = pk.costs.pose_cost_analytic_jac(
                        robot=robot,
                        joint_var=joint_var,
                        target_pose=target_pose,
                        target_link_index=target_link_index_jax,
                        pos_weight=pos_weight * 2,  # Increase weights
                        ori_weight=ori_weight * 2
                    )

                    joint_limit_cost = pk.costs.limit_cost(
                        robot=robot,
                        joint_var=joint_var,
                        weight=100.0
                    )

                    problem = jaxls.LeastSquaresProblem(
                        costs=[pose_cost, joint_limit_cost],
                        variables=[joint_var]
                    )

                    solution = problem.analyze().solve(
                        verbose=False,
                        linear_solver="dense_cholesky",
                        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1)  # Smaller initial trust region
                    )

                    optimal_joints = solution[joint_var]

                    # Re-verify
                    fk_result = robot.forward_kinematics(optimal_joints)
                    tcp_pose_data = fk_result[target_link_index]
                    final_position = tcp_pose_data[4:7]
                    final_quaternion = np.array([tcp_pose_data[3], tcp_pose_data[0], tcp_pose_data[1], tcp_pose_data[2]])

                    position_error = np.linalg.norm(final_position - target_position)
                    quaternion_error = np.linalg.norm(final_quaternion - target_quaternion)

                    print(f"  📍 After refinement - Position error: {position_error:.8f} meters ({position_error*1000:.3f} mm)")
                    print(f"  🔄 After refinement - Quaternion error: {quaternion_error:.8f}")

                break  # Only test first successful configuration

            except Exception as e:
                print(f"  ❌ Failed with weights {pos_weight}/{ori_weight}: {e}")
                continue

        # Test different optimization parameters
        print(f"\n⚙️  Testing optimization parameters:")

        # Test with more iterations or different solver
        try:
            # Reset variable
            joint_var = robot.joint_var_cls(0)

            pose_cost = pk.costs.pose_cost_analytic_jac(
                robot=robot,
                joint_var=joint_var,
                target_pose=target_pose,
                target_link_index=target_link_index_jax,
                pos_weight=200.0,
                ori_weight=50.0
            )

            joint_limit_cost = pk.costs.limit_cost(
                robot=robot,
                joint_var=joint_var,
                weight=50.0  # Reduced joint limit weight
            )

            problem = jaxls.LeastSquaresProblem(
                costs=[pose_cost, joint_limit_cost],
                variables=[joint_var]
            )

            # Try with different trust region settings
            solution = problem.analyze().solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(
                    lambda_initial=0.1,  # Smaller initial trust region
                    lambda_factor=2.0,   # More conservative trust region updates
                    lambda_max=1e6       # Higher maximum trust region size
                )
            )

            optimal_joints = solution[joint_var]

            # Verify
            fk_result = robot.forward_kinematics(optimal_joints)
            tcp_pose_data = fk_result[target_link_index]
            final_position = tcp_pose_data[4:7]
            final_quaternion = np.array([tcp_pose_data[3], tcp_pose_data[0], tcp_pose_data[1], tcp_pose_data[2]])

            position_error = np.linalg.norm(final_position - target_position)
            quaternion_error = np.linalg.norm(final_quaternion - target_quaternion)

            print(f"  📍 With refined parameters - Position error: {position_error:.8f} meters ({position_error*1000:.3f} mm)")
            print(f"  🔄 With refined parameters - Quaternion error: {quaternion_error:.8f}")

        except Exception as e:
            print(f"  ❌ Failed with refined parameters: {e}")

    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🔍 IK ACCURACY DEBUG COMPLETE!")
    print("=" * 60)
    print("\n💡 Recommendations:")
    print("   1. Increase pose cost weights for better accuracy")
    print("   2. Use smaller trust region initial values")
    print("   3. Reduce joint limit cost weight relative to pose cost")
    print("   4. Consider multiple optimization iterations")
    print("   5. Adjust tolerance levels based on application needs")

if __name__ == "__main__":
    debug_ik_accuracy()