import os
import sys
import time
import socket
import numpy as np
import scipy.spatial.transform as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "lib_py")))
import flexivrdk

def get_local_ip(robot_ip):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((robot_ip, 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.2.102"

def main():
    robot_ip = os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100")
    robot_sn = os.environ.get("FLEXIV_ROBOT_SN", "Rizon4-062339")
    local_ip = get_local_ip(robot_ip)

    print("Connecting to robot...")
    try:
        robot = flexivrdk.Robot(robot_sn, [local_ip])
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("\n--- Z-Axis Dynamic Margin Calibration ---")
    print("Please enable free-drive (drag mode) on your robot.")
    print("Rotate the gripper around the X axis (Roll) and carefully lower it until the edge touches the safe boundary.")
    print("Press Ctrl+C to exit when you are done observing.\n")
    
    Z_MIN_BASE = 0.0715

    try:
        while True:
            time.sleep(0.2)
            states = robot.states()
            pose = states.tcp_pose
            
            x, y, z = pose[0], pose[1], pose[2]
            qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
            
            rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
            roll = rot.as_euler('xyz')[0]
            roll_deg = np.degrees(roll)
            
            # The current gap/margin from the absolute base limit (0.0715)
            current_margin = z - Z_MIN_BASE
            
            # Use ANSI escape to clear the line and update in place
            sys.stdout.write(f"\r[Live] Z: {z:.4f}m | Roll: {roll_deg:>6.1f}° | Required Z_ROT_MARGIN: {current_margin:+.4f}m   ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nCalibration ended.")
        print(f"Update `Z_ROT_MARGIN` in `flexiv_safety.py` to the maximum 'Required Z_ROT_MARGIN' you observed when the gripper was fully horizontal (Roll ~90° or -90°).")

if __name__ == "__main__":
    main()
