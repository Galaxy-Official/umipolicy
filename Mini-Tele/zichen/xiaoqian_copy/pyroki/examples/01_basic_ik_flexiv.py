"""Basic IK with Local URDF

Simplest Inverse Kinematics Example using PyRoki with local URDF file.
This avoids downloading robot descriptions from the internet.
"""

import time
from pathlib import Path

import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf

import pyroki_snippets as pks


def main():
    """Main function for basic IK with local URDF."""

    # Option 1: Use a local URDF file if you have one
    # urdf_path = Path("/path/to/your/panda.urdf")
    # if urdf_path.exists():
    #     urdf = yourdfpy.URDF.load(str(urdf_path))
    # else:
    #     print(f"URDF file not found at {urdf_path}")
    #     return

    # Use cached robot description - should be in ~/.cache/robot_descriptions/
    # try:
    #     from robot_descriptions.loaders.yourdfpy import load_robot_description
    #     urdf = load_robot_description("panda_description")
    #     print("Using cached robot description from ~/.cache/robot_descriptions/")
    # except Exception as e:
    #     print(f"Could not load cached robot description: {e}")
    #     # Fallback: try to load from cache directory directly
    import os
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
            urdf = load_robot_description("panda_description")
    else:
        print("Cache directory not found, using load_robot_description")
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