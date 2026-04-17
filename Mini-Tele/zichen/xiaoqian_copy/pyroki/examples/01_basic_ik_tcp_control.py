"""Basic Robot Control with Joint Angles and TCP Pose

Robot visualization and control example using PyRoki with both joint angle input and TCP pose control.
Users can control individual joints through sliders or input desired TCP pose (position + quaternion)
and the robot will solve inverse kinematics to reach that pose.
"""

import time
import json
from pathlib import Path

import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf
import jaxlie


def compute_tcp_pose_from_joints(robot, joint_angles, target_link_name="tcp"):
    """Compute TCP pose from joint angles using forward kinematics."""
    try:
        # Get target link index
        target_link_index = robot.links.names.index(target_link_name)

        # Handle different joint counts - pad or truncate to match robot DOF
        if len(joint_angles) < robot.joints.num_actuated_joints:
            # Pad with zeros if data has fewer joints than robot
            cfg = np.zeros(robot.joints.num_actuated_joints)
            cfg[:len(joint_angles)] = joint_angles
        elif len(joint_angles) > robot.joints.num_actuated_joints:
            # Truncate if data has more joints than robot
            cfg = joint_angles[:robot.joints.num_actuated_joints]
        else:
            cfg = joint_angles

        # Compute forward kinematics
        fk_result = robot.forward_kinematics(cfg)

        # Get TCP pose data - format is [quat_x, quat_y, quat_z, quat_w, pos_x, pos_y, pos_z]
        tcp_pose_data = fk_result[target_link_index]

        # Extract quaternion (w, x, y, z format for our convention)
        quaternion = np.array([tcp_pose_data[3], tcp_pose_data[0], tcp_pose_data[1], tcp_pose_data[2]])

        # Extract position
        position = tcp_pose_data[4:7]

        return position, quaternion

    except Exception as e:
        print(f"Error computing TCP pose: {e}")
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])  # Default pose


def solve_inverse_kinematics(robot, target_position, target_quaternion, initial_joints=None, target_link_name="tcp"):
    """Solve inverse kinematics to find joint angles for target TCP pose."""
    try:
        import jaxls
        import pyroki  # Import pyroki here to ensure it's available
        import jaxlie

        # Get target link index
        target_link_index = robot.links.names.index(target_link_name)

        # Default initial joints if not provided
        if initial_joints is None:
            initial_joints = np.zeros(robot.joints.num_actuated_joints)

        # Ensure target quaternion is normalized
        target_quaternion = target_quaternion / np.linalg.norm(target_quaternion)

        # Create the optimization variable using robot's joint variable class
        joint_var = robot.joint_var_cls(0)

        # Create target pose as SE(3) transformation
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_quaternion),
            target_position
        )

        # Convert target_link_index to JAX array (required by the cost function)
        import jax
        target_link_index_jax = jax.numpy.array(target_link_index)

        # Create pose match cost using analytical Jacobian for better performance
        # Increased orientation weight to prevent 180-degree flips
        pose_cost = pyroki.costs.pose_cost_analytic_jac(
            robot=robot,
            joint_var=joint_var,
            target_pose=target_pose,
            target_link_index=target_link_index_jax,
            pos_weight=100.0,  # Higher weight for position accuracy
            ori_weight=100.0   # Increased orientation weight to match position importance
        )

        # Create joint limit cost for regularization (reduced weight to prioritize pose accuracy)
        joint_limit_cost = pyroki.costs.limit_cost(
            robot=robot,
            joint_var=joint_var,
            weight=10.0  # Reduced weight to prioritize pose matching over joint limits
        )

        # Create optimization problem
        problem = jaxls.LeastSquaresProblem(
            costs=[pose_cost, joint_limit_cost],
            variables=[joint_var]
        )

        # Solve the optimization problem (analyze first, then solve)
        solution = problem.analyze().solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0)
        )

        # Extract the solution - access by variable
        optimal_joints = solution[joint_var]

        # Verify the solution
        final_position, final_quaternion = compute_tcp_pose_from_joints(robot, optimal_joints, target_link_name)

        position_error = np.linalg.norm(final_position - target_position)
        quaternion_error = np.linalg.norm(final_quaternion - target_quaternion)

        print(f"IK Solution Found:")
        print(f"  Position error: {position_error:.6f} meters")
        print(f"  Quaternion error: {quaternion_error:.6f}")
        print(f"  Joint angles: {optimal_joints}")

        # More reasonable tolerance levels for practical applications
        # Position: 5mm tolerance, Orientation: 0.1 radian (~5.7 degrees) tolerance
        if position_error < 0.005 and quaternion_error < 0.1:  # 5mm and 0.1 radian tolerance
            return optimal_joints, True
        else:
            print(f"Warning: IK solution may not be accurate enough")
            print(f"  Required: position < 5mm, quaternion < 0.1rad")
            print(f"  Actual: position = {position_error*1000:.3f}mm, quaternion = {quaternion_error:.3f}rad")
            return optimal_joints, False

    except Exception as e:
        print(f"Error solving inverse kinematics: {e}")
        import traceback
        traceback.print_exc()
        return initial_joints, False


def load_joint_sequence(json_file):
    """Load joint sequence from JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading joint sequence: {e}")
        return None


def main():
    """Main function for joint angle and TCP pose control."""

    # Use cached robot description - should be in ~/.cache/robot_descriptions/
    cache_dir = Path.home() / ".cache" / "robot_descriptions"
    if cache_dir.exists():
        print(f"Cache directory exists: {cache_dir}")
        # Try to find panda URDF in cache
        panda_files = list(cache_dir.glob("**/panda*.urdf"))
        if panda_files:
            print(f"Found panda URDF in cache: {panda_files[0]}")
            urdf = yourdfpy.URDF.load(str(panda_files[0]))
        else:
            print("No panda URDF found in cache, using load_robot_description anyway")
            from robot_descriptions.loaders.yourdfpy import load_robot_description
            urdf = load_robot_description("panda_description")
    else:
        print("Cache directory not found, using load_robot_description")
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        urdf = load_robot_description("panda_description")

    # Create robot
    robot = pk.Robot.from_urdf(urdf)
    print(f"Robot has {robot.joints.num_actuated_joints} actuated joints")

    # Set up visualizer
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Current joint configuration
    current_joints = np.zeros(robot.joints.num_actuated_joints)

    # Flag to prevent recursive callbacks
    updating_gui = False

    # Load joint sequence if available
    joint_sequence = load_joint_sequence("joint_sequence.json")
    current_pose_index = 0

    # Create GUI for joint control
    control_folder = server.gui.add_folder("Joint Control")

    # Joint sliders
    joint_handles = []
    joint_names = [f"Joint {i}" for i in range(robot.joints.num_actuated_joints)]

    # Try to get actual joint names if available
    try:
        if hasattr(robot.joints, 'names'):
            joint_names = robot.joints.names[:robot.joints.num_actuated_joints]
    except:
        pass

    # Create joint sliders
    for i in range(robot.joints.num_actuated_joints):
        # Default range for joints (-π to π)
        joint_handle = server.gui.add_slider(
            f"{joint_names[i]}",
            min=-np.pi,
            max=np.pi,
            step=0.01,
            initial_value=0.0
        )
        joint_handles.append(joint_handle)

    # Joint value inputs (for precise control)
    input_folder = server.gui.add_folder("Joint Input")
    joint_input_handles = []

    for i in range(robot.joints.num_actuated_joints):
        joint_input = server.gui.add_number(
            f"{joint_names[i]} (rad)",
            initial_value=0.0,
            step=0.01
        )
        joint_input_handles.append(joint_input)

    # CSV Input for joint angles
    csv_folder = server.gui.add_folder("CSV Joint Input")
    csv_input = server.gui.add_text(
        "Joint Angles CSV",
        initial_value="0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        disabled=False
    )

    def parse_csv_joints():
        """Parse CSV input and apply joint angles."""
        nonlocal updating_gui
        if updating_gui:
            return

        try:
            # Parse CSV input
            csv_text = csv_input.value.strip()
            if not csv_text:
                return

            print(f"Parsing CSV input: {csv_text}")

            # Split by comma and convert to float
            joint_values = [float(x.strip()) for x in csv_text.split(',')]
            print(f"Parsed joint values: {joint_values}")

            # Apply to joints (handle different lengths)
            updating_gui = True
            try:
                for i in range(min(len(joint_values), robot.joints.num_actuated_joints)):
                    # Clamp values to valid range
                    value = np.clip(joint_values[i], -np.pi, np.pi)
                    print(f"Setting joint {i}: {value}")
                    joint_handles[i].value = value
                    joint_input_handles[i].value = value

                # Zero out remaining joints if CSV has fewer values
                for i in range(len(joint_values), robot.joints.num_actuated_joints):
                    print(f"Setting joint {i}: 0.0 (default)")
                    joint_handles[i].value = 0.0
                    joint_input_handles[i].value = 0.0

                print("Calling update_robot_configuration...")
                update_robot_configuration()
                print(f"✓ Applied CSV joints: {joint_values[:min(len(joint_values), robot.joints.num_actuated_joints)]}")

            finally:
                updating_gui = False

        except ValueError as e:
            print(f"Error parsing CSV: {e}")
            print("Please enter comma-separated numbers like: 0.1,-0.5,0.2,1.0,0.0,0.8,-0.2")
        except Exception as e:
            print(f"Error applying CSV joints: {e}")
            import traceback
            traceback.print_exc()

    apply_csv_button = server.gui.add_button("Apply CSV Joints")
    apply_csv_button.on_click(lambda _: parse_csv_joints())

    # Add example format hint
    csv_example = server.gui.add_text(
        "CSV Format Example",
        "0.780481,-0.100740,0.294439,-0.426692,-0.068826,-0.034667,-0.901108",
        disabled=True
    )

    # Load from CSV file
    def load_from_csv_file():
        """Load joint angles from CSV file."""
        try:
            # Try to load from joint_sequence.csv
            csv_file = "joint_sequence.csv"
            if not Path(csv_file).exists():
                print(f"CSV file not found: {csv_file}")
                return

            # Read the first data row (skip header)
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    print("CSV file is empty or has no data")
                    return

                # Get the first data row
                data_line = lines[1].strip()  # Skip header line
                parts = data_line.split(',')

                # Extract joint values (skip timestamp which is first column)
                if len(parts) > 1:
                    joint_values_str = ','.join(parts[1:])  # Skip timestamp
                    csv_input.value = joint_values_str
                    print(f"Loaded joint values from CSV: {joint_values_str}")
                else:
                    print("Invalid CSV format")

        except Exception as e:
            print(f"Error loading from CSV file: {e}")

    load_csv_file_button = server.gui.add_button("Load from CSV File")
    load_csv_file_button.on_click(lambda _: load_from_csv_file())

    # Load specific row from CSV
    csv_row_input = server.gui.add_number(
        "CSV Row Index",
        initial_value=0,
        min=0,
        max=1000,  # Will be updated when CSV is loaded
        step=1
    )

    def load_specific_csv_row():
        """Load specific row from CSV file."""
        try:
            csv_file = "joint_sequence.csv"
            if not Path(csv_file).exists():
                print(f"CSV file not found: {csv_file}")
                return

            row_index = int(csv_row_input.value)

            with open(csv_file, 'r') as f:
                lines = f.readlines()
                if row_index + 1 >= len(lines):  # +1 because first line is header
                    print(f"Row {row_index} not found in CSV file")
                    return

                # Get the specific data row
                data_line = lines[row_index + 1].strip()  # +1 to skip header
                parts = data_line.split(',')

                # Extract joint values (skip timestamp which is first column)
                if len(parts) > 1:
                    joint_values_str = ','.join(parts[1:])  # Skip timestamp
                    csv_input.value = joint_values_str
                    print(f"Loaded row {row_index}: {joint_values_str}")
                    # Automatically apply the loaded values
                    parse_csv_joints()
                else:
                    print("Invalid CSV format")

        except ValueError:
            print("Invalid row index")
        except Exception as e:
            print(f"Error loading CSV row: {e}")

    load_csv_row_button = server.gui.add_button("Load CSV Row")
    load_csv_row_button.on_click(lambda _: load_specific_csv_row())

    # Update max value for row index when we know the CSV size
    def update_csv_row_range():
        try:
            csv_file = "joint_sequence.csv"
            if Path(csv_file).exists():
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    max_rows = max(0, len(lines) - 1)  # -1 for header
                    csv_row_input.max = max_rows
                    print(f"CSV has {max_rows} data rows")
        except Exception as e:
            print(f"Error checking CSV size: {e}")

    # Call this on startup
    update_csv_row_range()

    # Preset poses
    presets_folder = server.gui.add_folder("Presets")

    # Zero position
    def set_zero_position():
        """Set all joints to zero."""
        for i in range(robot.joints.num_actuated_joints):
            joint_handles[i].value = 0.0
            joint_input_handles[i].value = 0.0
        update_robot_configuration()

    zero_button = server.gui.add_button("Zero Position")
    zero_button.on_click(lambda _: set_zero_position())

    # Home position (example configuration)
    def set_home_position():
        """Set to home position."""
        # Example home position for Panda (you may need to adjust these values)
        home_position = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0, 0.0])
        # Take only the first 7 joints if robot has 8
        if len(home_position) > robot.joints.num_actuated_joints:
            home_position = home_position[:robot.joints.num_actuated_joints]

        for i in range(min(len(home_position), robot.joints.num_actuated_joints)):
            joint_handles[i].value = home_position[i]
            joint_input_handles[i].value = home_position[i]
        update_robot_configuration()

    home_button = server.gui.add_button("Home Position")
    home_button.on_click(lambda _: set_home_position())

    # Joint sequence playback (if loaded)
    if joint_sequence:
        sequence_folder = server.gui.add_folder("Sequence Playback")

        pose_count = joint_sequence['num_poses']
        joint_dimension = joint_sequence['joint_dimension']

        pose_index_handle = server.gui.add_number(
            "Pose Index",
            initial_value=0,
            min=0,
            max=pose_count-1,
            step=1
        )

        def load_pose():
            """Load a specific pose from the sequence."""
            nonlocal current_pose_index
            current_pose_index = int(pose_index_handle.value)
            if 0 <= current_pose_index < pose_count:
                pose_joints = joint_sequence['joint_positions'][current_pose_index]

                # Apply to joint sliders and inputs
                for i in range(min(joint_dimension, robot.joints.num_actuated_joints)):
                    joint_handles[i].value = pose_joints[i]
                    joint_input_handles[i].value = pose_joints[i]

                # Zero out remaining joints if robot has more DOF than data
                for i in range(joint_dimension, robot.joints.num_actuated_joints):
                    joint_handles[i].value = 0.0
                    joint_input_handles[i].value = 0.0

                update_robot_configuration()
                print(f"Loaded pose {current_pose_index}")

        load_button = server.gui.add_button("Load Pose")
        load_button.on_click(lambda _: load_pose())

        # Auto-playback
        auto_play = server.gui.add_checkbox("Auto Play", initial_value=False)
        play_speed = server.gui.add_number("Play Speed", initial_value=1.0, step=0.1, min=0.1, max=5.0)

        # Previous/Next buttons
        def previous_pose():
            if current_pose_index > 0:
                pose_index_handle.value = current_pose_index - 1
                load_pose()

        def next_pose():
            if current_pose_index < pose_count - 1:
                pose_index_handle.value = current_pose_index + 1
                load_pose()

        prev_button = server.gui.add_button("Previous")
        prev_button.on_click(lambda _: previous_pose())

        next_button = server.gui.add_button("Next")
        next_button.on_click(lambda _: next_pose())

    # ===== NEW: TCP Pose Control Section =====
    tcp_control_folder = server.gui.add_folder("TCP Pose Control")

    # TCP Position input (XYZ)
    tcp_position_input_folder = server.gui.add_folder("TCP Position Input")
    tcp_position_handles = []
    position_labels = ["X (m)", "Y (m)", "Z (m)"]
    position_defaults = [0.5, 0.0, 0.5]  # Reasonable default position

    for i, (label, default) in enumerate(zip(position_labels, position_defaults)):
        pos_handle = server.gui.add_number(
            label,
            initial_value=default,
            step=0.01,
            min=-1.0,
            max=1.0
        )
        tcp_position_handles.append(pos_handle)

    # TCP Quaternion input (WXYZ)
    tcp_quaternion_input_folder = server.gui.add_folder("TCP Quaternion Input")
    tcp_quaternion_handles = []
    quaternion_labels = ["W", "X", "Y", "Z"]
    quaternion_defaults = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion

    for i, (label, default) in enumerate(zip(quaternion_labels, quaternion_defaults)):
        quat_handle = server.gui.add_number(
            label,
            initial_value=default,
            step=0.01,
            min=-1.0,
            max=1.0
        )
        tcp_quaternion_handles.append(quat_handle)

    # Normalize quaternion button
    def normalize_quaternion():
        """Normalize the quaternion inputs."""
        quaternion = np.array([h.value for h in tcp_quaternion_handles])
        normalized = quaternion / np.linalg.norm(quaternion)
        for i, handle in enumerate(tcp_quaternion_handles):
            handle.value = normalized[i]
        print(f"Normalized quaternion: {normalized}")

    normalize_button = server.gui.add_button("Normalize Quaternion")
    normalize_button.on_click(lambda _: normalize_quaternion())

    # Apply TCP pose button
    def apply_tcp_pose():
        """Apply the TCP pose and solve inverse kinematics."""
        nonlocal updating_gui
        if updating_gui:
            return

        try:
            # Get target position and quaternion from inputs
            target_position = np.array([h.value for h in tcp_position_handles])
            target_quaternion = np.array([h.value for h in tcp_quaternion_handles])

            # Normalize quaternion
            target_quaternion = target_quaternion / np.linalg.norm(target_quaternion)

            print(f"Applying TCP pose:")
            print(f"  Target position: {target_position}")
            print(f"  Target quaternion: {target_quaternion}")

            # Solve inverse kinematics
            current_joints_array = np.array([joint_handles[i].value for i in range(robot.joints.num_actuated_joints)])
            solution_joints, success = solve_inverse_kinematics(
                robot, target_position, target_quaternion, current_joints_array
            )

            if success:
                print("✅ IK solution found successfully!")

                # Update GUI with solution
                updating_gui = True
                try:
                    for i in range(robot.joints.num_actuated_joints):
                        joint_handles[i].value = solution_joints[i]
                        joint_input_handles[i].value = solution_joints[i]

                    # Update robot configuration
                    update_robot_configuration()

                finally:
                    updating_gui = False

            else:
                print("❌ IK solution not accurate enough or failed")

        except Exception as e:
            print(f"Error applying TCP pose: {e}")
            import traceback
            traceback.print_exc()

    apply_tcp_button = server.gui.add_button("Apply TCP Pose")
    apply_tcp_button.on_click(lambda _: apply_tcp_pose())

    # Get current TCP pose button
    def get_current_tcp_pose():
        """Get the current TCP pose and populate the input fields."""
        current_joints_array = np.array([joint_handles[i].value for i in range(robot.joints.num_actuated_joints)])
        position, quaternion = compute_tcp_pose_from_joints(robot, current_joints_array)

        # Convert JAX arrays to regular numpy arrays if needed
        if hasattr(position, 'tolist'):
            position = np.array(position.tolist())
        if hasattr(quaternion, 'tolist'):
            quaternion = np.array(quaternion.tolist())

        # Update input fields
        for i, handle in enumerate(tcp_position_handles):
            handle.value = float(position[i])
        for i, handle in enumerate(tcp_quaternion_handles):
            handle.value = float(quaternion[i])

        print(f"Current TCP pose loaded:")
        print(f"  Position: {position}")
        print(f"  Quaternion: {quaternion}")

    get_current_button = server.gui.add_button("Get Current TCP Pose")
    get_current_button.on_click(lambda _: get_current_tcp_pose())

    # Preset TCP poses
    tcp_presets_folder = server.gui.add_folder("TCP Presets")

    def set_tcp_preset(pose_name, position, quaternion):
        """Set TCP to a preset pose."""
        for i, handle in enumerate(tcp_position_handles):
            handle.value = position[i]
        for i, handle in enumerate(tcp_quaternion_handles):
            handle.value = quaternion[i]
        print(f"Set TCP preset: {pose_name}")

    # Define some reasonable preset poses
    tcp_presets = {
        "Home": ([0.5, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]),
        "Forward": ([0.6, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0]),
        "Up": ([0.4, 0.0, 0.6], [1.0, 0.0, 0.0, 0.0]),
        "Side": ([0.5, 0.2, 0.4], [0.707, 0.0, 0.707, 0.0]),  # 90° rotation around Z
        "Down": ([0.5, 0.0, 0.3], [0.707, 0.707, 0.0, 0.0]),  # 90° rotation around X
    }

    for pose_name, (position, quaternion) in tcp_presets.items():
        preset_button = server.gui.add_button(f"Preset: {pose_name}")
        preset_button.on_click(lambda _, name=pose_name, pos=position, quat=quaternion: set_tcp_preset(name, pos, quat))

    # Additional controls
    misc_folder = server.gui.add_folder("Misc Controls")

    # Random configuration
    def random_configuration():
        """Set random joint configuration."""
        random_joints = np.random.uniform(-np.pi, np.pi, robot.joints.num_actuated_joints)
        for i in range(robot.joints.num_actuated_joints):
            joint_handles[i].value = random_joints[i]
            joint_input_handles[i].value = random_joints[i]
        update_robot_configuration()

    random_button = server.gui.add_button("Random Configuration")
    random_button.on_click(lambda _: random_configuration())

    # Save current configuration
    def save_configuration():
        """Save current joint configuration."""
        config = {
            'joint_values': current_joints.tolist(),
            'timestamp': time.time()
        }
        with open('saved_joint_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved joint configuration: {current_joints}")

    save_button = server.gui.add_button("Save Configuration")
    save_button.on_click(lambda _: save_configuration())

    # Load saved configuration
    def load_configuration():
        """Load saved joint configuration."""
        try:
            with open('saved_joint_config.json', 'r') as f:
                config = json.load(f)
            saved_joints = np.array(config['joint_values'])

            for i in range(min(len(saved_joints), robot.joints.num_actuated_joints)):
                joint_handles[i].value = saved_joints[i]
                joint_input_handles[i].value = saved_joints[i]

            update_robot_configuration()
            print(f"Loaded joint configuration: {saved_joints}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

    load_config_button = server.gui.add_button("Load Saved Config")
    load_config_button.on_click(lambda _: load_configuration())

    # Info display
    info_folder = server.gui.add_folder("Info")
    joint_count_info = server.gui.add_text("Joint Count", str(robot.joints.num_actuated_joints), disabled=True)

    if joint_sequence:
        sequence_info = server.gui.add_text("Sequence Poses", str(joint_sequence['num_poses']), disabled=True)

    # TCP Pose display
    tcp_folder = server.gui.add_folder("TCP Pose")
    tcp_position_info = server.gui.add_text("Position (XYZ)", "[0.0000, 0.0000, 0.0000]", disabled=True)
    tcp_quaternion_info = server.gui.add_text("Quaternion (WXYZ)", "[1.0000, 0.0000, 0.0000, 0.0000]", disabled=True)

    # Add 3D TCP visualization
    tcp_visualizer = server.scene.add_transform_controls(
        "/tcp_marker", scale=0.1, position=(0.5, 0.0, 0.5), wxyz=(1, 0, 0, 0)
    )
    tcp_visualizer.visible = True

    # Update function for robot configuration
    def update_robot_configuration():
        """Update robot configuration based on current joint values."""
        try:
            # Get current joint values from sliders
            for i in range(robot.joints.num_actuated_joints):
                current_joints[i] = joint_handles[i].value

            # Update visualizer
            urdf_vis.update_cfg(current_joints)

            # Compute and display TCP pose
            tcp_position, tcp_quaternion = compute_tcp_pose_from_joints(robot, current_joints)

            # Convert JAX arrays to regular numpy arrays if needed for GUI
            if hasattr(tcp_position, 'tolist'):
                tcp_position = np.array(tcp_position.tolist())
            if hasattr(tcp_quaternion, 'tolist'):
                tcp_quaternion = np.array(tcp_quaternion.tolist())

            # Update TCP pose display
            if hasattr(update_robot_configuration, 'tcp_position_info'):
                update_robot_configuration.tcp_position_info.value = f"[{tcp_position[0]:.4f}, {tcp_position[1]:.4f}, {tcp_position[2]:.4f}]"
                update_robot_configuration.tcp_quaternion_info.value = f"[{tcp_quaternion[0]:.4f}, {tcp_quaternion[1]:.4f}, {tcp_quaternion[2]:.4f}, {tcp_quaternion[3]:.4f}]"

            # Also update the 3D TCP visualization if available
            if hasattr(update_robot_configuration, 'tcp_visualizer'):
                update_robot_configuration.tcp_visualizer.position = tuple(float(x) for x in tcp_position)
                update_robot_configuration.tcp_visualizer.wxyz = tuple(float(x) for x in tcp_quaternion)

        except Exception as e:
            print(f"Error in update_robot_configuration: {e}")
            import traceback
            traceback.print_exc()

    # Store references to TCP display elements in the update function
    update_robot_configuration.tcp_position_info = tcp_position_info
    update_robot_configuration.tcp_quaternion_info = tcp_quaternion_info
    update_robot_configuration.tcp_visualizer = tcp_visualizer

    # Initialize with zero configuration
    update_robot_configuration()

    print(f"Joint and TCP control interface ready!")
    print(f"- Use sliders or input fields to control individual joints")
    print(f"- Use TCP Pose Control section to input desired end-effector pose")
    print(f"- Robot has {robot.joints.num_actuated_joints} actuated joints")
    if joint_sequence:
        print(f"- Loaded {joint_sequence['num_poses']} poses from sequence")
    print(f"- Server running at http://localhost:8080")

    # Main loop - just keep the server running
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()