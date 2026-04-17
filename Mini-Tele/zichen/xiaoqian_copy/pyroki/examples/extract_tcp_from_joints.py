"""Extract TCP poses from joint positions using forward kinematics.

This script reads joint position sequences from the 00000/ directory and computes
TCP poses using PyRoki forward kinematics, then converts to position+quaternion format.
"""

import numpy as np
import glob
import json
from pathlib import Path
import pyroki as pk
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
import jaxlie


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle representation to quaternion (w, x, y, z)."""
    if len(axis_angle) == 4:
        # Format: [axis_x, axis_y, axis_z, angle]
        axis = axis_angle[:3]
        angle = axis_angle[3]
    else:
        return None

    # Normalize axis
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    axis = axis / axis_norm

    # Convert to quaternion
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    quaternion = np.array([w, xyz[0], xyz[1], xyz[2]])

    # Normalize
    quaternion = quaternion / np.linalg.norm(quaternion)

    return quaternion


def load_joint_position_sequence(data_path):
    """Load joint position sequence from the 00000 directory."""
    data_path = Path(data_path)
    joint_dir = data_path / "00000" / "joint_position"

    if not joint_dir.exists():
        print(f"Joint position directory not found: {joint_dir}")
        return None

    # Get all joint position files
    jp_files = sorted(glob.glob(str(joint_dir / "*.npy")))

    if not jp_files:
        print(f"No joint position files found in {joint_dir}")
        return None

    print(f"Found {len(jp_files)} joint position files")

    # Load all joint positions
    joint_positions = []
    for i, file in enumerate(jp_files):
        jp = np.load(file)
        joint_positions.append(jp)
        if i < 3:  # Print first few for verification
            print(f"Joint position {i}: {jp}")

    return joint_positions


def load_tcp_data_sequence(data_path):
    """Load TCP data sequence from the 00000 directory."""
    data_path = Path(data_path)
    tcp_dir = data_path / "00000" / "tcp"

    if not tcp_dir.exists():
        print(f"TCP directory not found: {tcp_dir}")
        return None

    # Get all TCP files
    tcp_files = sorted(glob.glob(str(tcp_dir / "*.npy")))

    if not tcp_files:
        print(f"No TCP files found in {tcp_dir}")
        return None

    print(f"Found {len(tcp_files)} TCP files")

    # Load all TCP data
    tcp_data = []
    for i, file in enumerate(tcp_files):
        tcp = np.load(file)
        tcp_data.append(tcp)
        if i < 3:  # Print first few for verification
            print(f"TCP data {i}: {tcp}")

    return tcp_data


def compute_tcp_poses_from_joints(joint_positions, robot, target_link_name="tcp"):
    """Compute TCP poses from joint positions using forward kinematics."""
    target_link_index = robot.links.names.index(target_link_name)

    tcp_poses = []

    for i, jp in enumerate(joint_positions):
        # Handle different joint counts - pad or truncate to match robot DOF
        if len(jp) < robot.joints.num_actuated_joints:
            # Pad with zeros if data has fewer joints than robot
            cfg = np.zeros(robot.joints.num_actuated_joints)
            cfg[:len(jp)] = jp
        elif len(jp) > robot.joints.num_actuated_joints:
            # Truncate if data has more joints than robot
            cfg = jp[:robot.joints.num_actuated_joints]
        else:
            cfg = jp

        # Compute forward kinematics
        fk_result = robot.forward_kinematics(cfg)

        # Debug: check what we get
        if i == 0:
            print(f"FK result shape: {fk_result.shape}")
            print(f"FK result type: {type(fk_result)}")
            print(f"TCP index: {target_link_index}")
            print(f"TCP pose data: {fk_result[target_link_index]}")

        # Get TCP pose data - format is [quat_x, quat_y, quat_z, quat_w, pos_x, pos_y, pos_z]
        tcp_pose_data = fk_result[target_link_index]

        # Extract quaternion (w, x, y, z format for our convention)
        quaternion = np.array([tcp_pose_data[3], tcp_pose_data[0], tcp_pose_data[1], tcp_pose_data[2]])

        # Extract position
        position = tcp_pose_data[4:7]

        # Debug
        if i == 0:
            print(f"Position: {position}")
            print(f"Quaternion: {quaternion}")

        tcp_poses.append({
            'timestamp': i,
            'joint_position': jp.tolist(),
            'position': position.tolist(),
            'quaternion': quaternion.tolist()
        })

        if i < 3:  # Print first few for verification
            print(f"Computed TCP pose {i}:")
            print(f"  Position: {position}")
            print(f"  Quaternion: {quaternion}")

    return tcp_poses


def rotation_matrix_to_quaternion(rotation_matrix):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    # Convert to numpy array if it's a JAX array
    if hasattr(rotation_matrix, 'numpy'):
        rotation_matrix = rotation_matrix.numpy()
    else:
        rotation_matrix = np.array(rotation_matrix)

    # Use the method from scipy.spatial.transform.Rotation
    trace = np.trace(rotation_matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s

    quaternion = np.array([w, x, y, z])

    # Normalize
    quaternion = quaternion / np.linalg.norm(quaternion)

    return quaternion


def parse_tcp_data_format(tcp_data):
    """Parse the TCP data format and convert to position+quaternion."""
    parsed_poses = []

    for i, tcp in enumerate(tcp_data):
        # First 3 elements are position
        position = tcp[:3]

        # Last 4 elements are rotation (try different formats)
        rot_part = tcp[3:]

        # Try axis-angle format
        quaternion = axis_angle_to_quaternion(rot_part)

        parsed_poses.append({
            'timestamp': i,
            'position': position.tolist(),
            'quaternion': quaternion.tolist(),
            'raw_tcp': tcp.tolist()
        })

        if i < 3:  # Print first few for verification
            print(f"Parsed TCP pose {i}:")
            print(f"  Position: {position}")
            print(f"  Quaternion: {quaternion}")
            print(f"  Raw rotation: {rot_part}")

    return parsed_poses


def save_pose_sequence(pose_sequence, output_file, format_type):
    """Save pose sequence to JSON file."""
    output_data = {
        'pose_count': len(pose_sequence),
        'format_type': format_type,
        'poses': pose_sequence
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {format_type} pose sequence to: {output_file}")


def print_pose_summary(pose_sequence, title):
    """Print a summary of the pose sequence."""
    if not pose_sequence:
        return

    print(f"\n{title} Summary:")
    print(f"Number of poses: {len(pose_sequence)}")

    # First pose
    first_pose = pose_sequence[0]
    print(f"\nFirst pose:")
    print(f"  Position: [{first_pose['position'][0]:.4f}, {first_pose['position'][1]:.4f}, {first_pose['position'][2]:.4f}]")
    print(f"  Quaternion: [{first_pose['quaternion'][0]:.4f}, {first_pose['quaternion'][1]:.4f}, {first_pose['quaternion'][2]:.4f}, {first_pose['quaternion'][3]:.4f}]")

    # Last pose
    last_pose = pose_sequence[-1]
    print(f"\nLast pose:")
    print(f"  Position: [{last_pose['position'][0]:.4f}, {last_pose['position'][1]:.4f}, {last_pose['position'][2]:.4f}]")
    print(f"  Quaternion: [{last_pose['quaternion'][0]:.4f}, {last_pose['quaternion'][1]:.4f}, {last_pose['quaternion'][2]:.4f}, {last_pose['quaternion'][3]:.4f}]")

    # Position range
    positions = np.array([pose['position'] for pose in pose_sequence])
    print(f"\nPosition ranges:")
    print(f"  X: {positions[:, 0].min():.4f} to {positions[:, 0].max():.4f}")
    print(f"  Y: {positions[:, 1].min():.4f} to {positions[:, 1].max():.4f}")
    print(f"  Z: {positions[:, 2].min():.4f} to {positions[:, 2].max():.4f}")


def main():
    """Main function to extract TCP poses from joint positions."""
    # Path to your data
    data_path = "/Disk3/xiaoqian/collect_data/data/kp/2025-10-26_19:29:53"

    print("=== Loading Robot Model ===")
    # Load robot model for forward kinematics
    try:
        urdf = load_robot_description("panda_description")
    except:
        # Fallback to cached version
        cache_dir = Path.home() / ".cache" / "robot_descriptions"
        panda_files = list(cache_dir.glob("**/panda*.urdf"))
        if panda_files:
            urdf = yourdfpy.URDF.load(str(panda_files[0]))
        else:
            print("Could not load robot description")
            return

    robot = pk.Robot.from_urdf(urdf)
    print(f"Robot has {robot.joints.num_actuated_joints} actuated joints")
    print(f"Available links: {robot.links.names}")

    print("\n=== Loading Joint Position Data ===")
    # Load joint position sequence
    joint_positions = load_joint_position_sequence(data_path)
    if joint_positions is None:
        return

    print("\n=== Computing TCP Poses from Joints (Forward Kinematics) ===")
    # Compute TCP poses using forward kinematics
    tcp_poses_fk = compute_tcp_poses_from_joints(joint_positions, robot)

    print("\n=== Loading Raw TCP Data ===")
    # Also load the raw TCP data for comparison
    tcp_data_raw = load_tcp_data_sequence(data_path)
    if tcp_data_raw:
        print("\n=== Parsing Raw TCP Data ===")
        tcp_poses_raw = parse_tcp_data_format(tcp_data_raw)

    print("\n=== Results Summary ===")
    # Print summaries
    print_pose_summary(tcp_poses_fk, "Forward Kinematics TCP Poses")

    if tcp_data_raw:
        print_pose_summary(tcp_poses_raw, "Raw TCP Data (Parsed)")

    # Save forward kinematics results
    save_pose_sequence(tcp_poses_fk, "tcp_poses_from_joints.json", "forward_kinematics")

    # Also create CSV format
    csv_file = "tcp_poses_from_joints.csv"
    with open(csv_file, 'w') as f:
        f.write("timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z\n")
        for pose in tcp_poses_fk:
            pos = pose['position']
            quat = pose['quaternion']
            f.write(f"{pose['timestamp']},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f},{quat[0]:.6f},{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f}\n")

    print(f"\nSaved CSV format to: {csv_file}")

    # Create a simple playback script snippet
    print(f"\nTo use these poses in your IK GUI, you can copy-paste these values:")
    print(f"\nFirst pose (Home position):")
    first_pose = tcp_poses_fk[0]
    print(f"Position: {first_pose['position']}")
    print(f"Quaternion: {first_pose['quaternion']}")

    print(f"\nLast pose (Target position):")
    last_pose = tcp_poses_fk[-1]
    print(f"Position: {last_pose['position']}")
    print(f"Quaternion: {last_pose['quaternion']}")


if __name__ == "__main__":
    main()