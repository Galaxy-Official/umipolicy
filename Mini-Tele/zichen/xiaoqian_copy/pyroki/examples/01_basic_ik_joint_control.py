"""Basic Robot Control with Joint Angles

Robot visualization and control example using PyRoki with direct joint angle input.
This allows users to control individual joints through sliders or input fields.
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
    """Main function for joint angle control."""

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

    # Update function
    def update_robot_configuration():
        """Update robot configuration based on current joint values."""
        try:
            # Get current joint values from sliders
            for i in range(robot.joints.num_actuated_joints):
                current_joints[i] = joint_handles[i].value

            # Debug: print current joint values (optional - can be commented out for production)
            # print(f"Updating robot configuration: {current_joints}")

            # Update visualizer
            urdf_vis.update_cfg(current_joints)

            # Compute and display TCP pose
            tcp_position, tcp_quaternion = compute_tcp_pose_from_joints(robot, current_joints)

            # Update TCP pose display
            if hasattr(update_robot_configuration, 'tcp_position_info'):
                update_robot_configuration.tcp_position_info.value = f"[{tcp_position[0]:.4f}, {tcp_position[1]:.4f}, {tcp_position[2]:.4f}]"
                update_robot_configuration.tcp_quaternion_info.value = f"[{tcp_quaternion[0]:.4f}, {tcp_quaternion[1]:.4f}, {tcp_quaternion[2]:.4f}, {tcp_quaternion[3]:.4f}]"

            # Also update the 3D TCP visualization if available
            if hasattr(update_robot_configuration, 'tcp_visualizer'):
                update_robot_configuration.tcp_visualizer.position = tuple(tcp_position)
                update_robot_configuration.tcp_visualizer.wxyz = tuple(tcp_quaternion)

        except Exception as e:
            print(f"Error in update_robot_configuration: {e}")
            import traceback
            traceback.print_exc()

    # Callback functions for joint control
    def on_joint_slider_change(_):
        """Handle joint slider changes."""
        nonlocal updating_gui
        if updating_gui:
            return

        updating_gui = True
        try:
            # Update input fields to match sliders
            for i in range(robot.joints.num_actuated_joints):
                joint_input_handles[i].value = joint_handles[i].value
            update_robot_configuration()
        finally:
            updating_gui = False

    def on_joint_input_change(_):
        """Handle joint input changes."""
        nonlocal updating_gui
        if updating_gui:
            return

        updating_gui = True
        try:
            # Update sliders to match input fields
            for i in range(robot.joints.num_actuated_joints):
                joint_handles[i].value = joint_input_handles[i].value
            update_robot_configuration()
        finally:
            updating_gui = False

    # Connect callbacks
    for i in range(robot.joints.num_actuated_joints):
        joint_handles[i].on_update(on_joint_slider_change)
        joint_input_handles[i].on_update(on_joint_input_change)

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

    # Store references to TCP display elements in the update function
    update_robot_configuration.tcp_position_info = tcp_position_info
    update_robot_configuration.tcp_quaternion_info = tcp_quaternion_info
    update_robot_configuration.tcp_visualizer = tcp_visualizer

    # Initialize with zero configuration
    update_robot_configuration()

    print(f"Joint control interface ready!")
    print(f"- Use sliders or input fields to control individual joints")
    print(f"- Robot has {robot.joints.num_actuated_joints} actuated joints")
    if joint_sequence:
        print(f"- Loaded {joint_sequence['num_poses']} poses from sequence")
    print(f"- Server running at http://localhost:8080")

    # Main loop - just keep the server running
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()