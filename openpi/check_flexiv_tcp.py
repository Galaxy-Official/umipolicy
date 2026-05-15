import sys
import time
import argparse
import numpy as np
import scipy.spatial.transform as st

try:
    import flexivrdk
except ImportError:
    print("Error: flexivrdk not found. Please run this script in the correct conda environment (e.g. umi).")
    sys.exit(1)

def pose7_to_mat(pose7):
    # pose7: [x, y, z, qw, qx, qy, qz]
    # 注意: Flexiv 的四元数顺序可能是 [qw, qx, qy, qz]
    pos = np.array(pose7[:3], dtype=np.float64)
    qw, qx, qy, qz = pose7[3:]
    rot = st.Rotation.from_quat([qx, qy, qz, qw])
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = pos
    return mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_sn", type=str, default="Rizon4-062339", help="Serial Number of the robot")
    parser.add_argument("--local_ip", type=str, default="192.168.2.102", help="IP of the local PC")
    args = parser.parse_args()

    print(f"Connecting to Flexiv robot {args.robot_sn} via {args.local_ip}...")
    try:
        robot = flexivrdk.Robot(args.robot_sn, [args.local_ip])
    except Exception as e:
        print(f"Failed to connect to the robot: {e}")
        sys.exit(1)

    print("Connected successfully! Fetching robot states...")
    # 稍微等一下让数据同步
    time.sleep(0.5)
    
    states = robot.states()
    
    flange_pose = list(states.flange_pose)
    tcp_pose = list(states.tcp_pose)
    
    print("\n" + "="*50)
    print("Raw Pose Data (X, Y, Z, Qw, Qx, Qy, Qz)")
    print(f"Flange Pose : {['{:.4f}'.format(x) for x in flange_pose]}")
    print(f"TCP Pose    : {['{:.4f}'.format(x) for x in tcp_pose]}")
    print("="*50)

    # 计算相对变换矩阵 T_{flange_tcp}
    flange_mat = pose7_to_mat(flange_pose)
    tcp_mat = pose7_to_mat(tcp_pose)
    
    # flange_to_tcp = inv(flange_mat) @ tcp_mat
    flange_to_tcp_mat = np.linalg.inv(flange_mat) @ tcp_mat
    
    offset_xyz = flange_to_tcp_mat[:3, 3]
    rot_matrix = flange_to_tcp_mat[:3, :3]
    euler_xyz = st.Rotation.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
    
    # 检查是否为零偏移
    offset_norm = np.linalg.norm(offset_xyz)
    angle_norm = np.linalg.norm(euler_xyz)
    
    print("\n--- Analysis Result ---")
    if offset_norm < 1e-4 and angle_norm < 1e-4:
        print("✅ TCP is EXACTLY THE SAME as Flange (No Tool Offset).")
        print("The robot is NOT using any active gripper offset in its controller.")
        print("Your hypothesis is CONFIRMED: The robot is effectively using EEF/Flange control right now!")
    else:
        print("❌ A Tool Offset IS ACTIVE in the robot controller!")
        print(f"The currently active TCP differs from the Flange by:")
        print(f"  Translation (X, Y, Z) : {offset_xyz[0]*1000:.2f} mm, {offset_xyz[1]*1000:.2f} mm, {offset_xyz[2]*1000:.2f} mm")
        print(f"  Rotation (Rx, Ry, Rz) : {euler_xyz[0]:.2f} deg, {euler_xyz[1]:.2f} deg, {euler_xyz[2]:.2f} deg")
        if abs(euler_xyz[0]) > 90 or abs(euler_xyz[1]) > 90:
            print("\n⚠️ WARNING: Large rotation detected! This explains why moving in Z direction might be reversed!")
    print("==================================================\n")

if __name__ == "__main__":
    main()
