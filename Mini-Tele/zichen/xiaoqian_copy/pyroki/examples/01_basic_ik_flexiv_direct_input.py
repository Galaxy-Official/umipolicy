"""Basic IK with Direct Input

Inverse Kinematics Example using PyRoki with direct position and quaternion input through GUI.
This allows users to specify exact target position and orientation values.
"""

import time

import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf

import pyroki_snippets as pks


def main():
    """Main function for basic IK with direct input."""

    # Use cached robot description - should be in ~/.cache/robot_descriptions/
    from pathlib import Path
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

    target_link_name = "tcp"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Create GUI input fields for direct position and quaternion input
    gui_folder = server.gui.add_folder("Target Input")

    # Position input fields
    pos_x_handle = server.gui.add_number("Position X", initial_value=0.61, step=0.01)
    pos_y_handle = server.gui.add_number("Position Y", initial_value=0.0, step=0.01)
    pos_z_handle = server.gui.add_number("Position Z", initial_value=0.56, step=0.01)

    # Quaternion input fields (w, x, y, z)
    quat_w_handle = server.gui.add_number("Quaternion W", initial_value=0.0, step=0.01)
    quat_x_handle = server.gui.add_number("Quaternion X", initial_value=0.0, step=0.01)
    quat_y_handle = server.gui.add_number("Quaternion Y", initial_value=1.0, step=0.01)
    quat_z_handle = server.gui.add_number("Quaternion Z", initial_value=0.0, step=0.01)

    # Update button
    update_button = server.gui.add_button("Update Target")

    # Auto-update toggle
    auto_update_handle = server.gui.add_checkbox("Auto Update", initial_value=False)

    def update_ik_target():
        """Update the IK target based on GUI input values."""
        # Get position values
        pos_x = pos_x_handle.value
        pos_y = pos_y_handle.value
        pos_z = pos_z_handle.value

        # Get quaternion values
        quat_w = quat_w_handle.value
        quat_x = quat_x_handle.value
        quat_y = quat_y_handle.value
        quat_z = quat_z_handle.value

        # Update the transform controls
        ik_target.position = (pos_x, pos_y, pos_z)
        ik_target.wxyz = (quat_w, quat_x, quat_y, quat_z)

        print(f"Updated target: pos=({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}), "
              f"quat=({quat_w:.3f}, {quat_x:.3f}, {quat_y:.3f}, {quat_z:.3f})")

    def on_gui_change(_):
        """Handle GUI changes for auto-update mode."""
        if auto_update_handle.value:
            update_ik_target()

    # Register callbacks
    update_button.on_click(lambda _: update_ik_target())

    # Add change listeners for auto-update
    pos_x_handle.on_update(on_gui_change)
    pos_y_handle.on_update(on_gui_change)
    pos_z_handle.on_update(on_gui_change)
    quat_w_handle.on_update(on_gui_change)
    quat_x_handle.on_update(on_gui_change)
    quat_y_handle.on_update(on_gui_change)
    quat_z_handle.on_update(on_gui_change)

    # Add preset buttons for common positions
    presets_folder = server.gui.add_folder("Presets")

    def set_preset_position(position, quaternion):
        """Set a preset position and quaternion."""
        pos_x_handle.value = position[0]
        pos_y_handle.value = position[1]
        pos_z_handle.value = position[2]
        quat_w_handle.value = quaternion[0]
        quat_x_handle.value = quaternion[1]
        quat_y_handle.value = quaternion[2]
        quat_z_handle.value = quaternion[3]
        update_ik_target()

    # Add some common preset positions
    home_button = server.gui.add_button("Home Position")
    home_button.on_click(lambda _: set_preset_position(
        (0.61, 0.0, 0.56), (0.0, 0.0, 1.0, 0.0)
    ))

    forward_button = server.gui.add_button("Forward Reach")
    forward_button.on_click(lambda _: set_preset_position(
        (0.8, 0.0, 0.4), (0.0, 0.0, 1.0, 0.0)
    ))

    side_button = server.gui.add_button("Side Reach")
    side_button.on_click(lambda _: set_preset_position(
        (0.5, 0.3, 0.4), (0.0, 0.707, 0.707, 0.0)
    ))

    up_button = server.gui.add_button("Up Reach")
    up_button.on_click(lambda _: set_preset_position(
        (0.5, 0.0, 0.8), (0.0, 0.0, 1.0, 0.0)
    ))

    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()