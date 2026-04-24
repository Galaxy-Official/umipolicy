import os
import sys
import time
import socket
import numpy as np
import scipy.spatial.transform as st
import select
import tty
import termios

import flexivrdk

# --- Calibration Settings ---
STEP_ROT_DEG = 5.0      # 每次旋转 5 度
STEP_Z_M = 0.005        # 每次移动 0.5 cm
# ----------------------------

def get_local_ip(robot_ip):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((robot_ip, 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.2.102"

def getch():
    """Reads a single character from standard input without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    ch = None
    try:
        tty.setraw(sys.stdin.fileno())
        # Non-blocking read with 0.05s timeout
        if select.select([sys.stdin], [], [], 0.05)[0]:
            ch = sys.stdin.read(1)
            # Drain the buffer to prevent insane accumulation when holding keys
            while select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    robot_ip = os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100")
    robot_sn = os.environ.get("FLEXIV_ROBOT_SN", "Rizon4-062339")
    local_ip = get_local_ip(robot_ip)

    print("Connecting to robot...")
    try:
        robot = flexivrdk.Robot(robot_sn, [local_ip])
        model = flexivrdk.Model(robot)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Enabling robot...")
    if robot.fault():
        robot.ClearFault()
        time.sleep(2)
    robot.Enable()
    while not robot.operational():
        time.sleep(1)

    robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)

    # Initial Pose (Z_UPPER joint positions)
    z_upper_q = [0.2510, -0.6796, -0.1615, 1.8408, 0.1169, 0.9518, 0.1125]
    print("Moving to initial Z_UPPER boundary...")
    robot.SendJointPosition(z_upper_q, [0]*7, [0.1]*7, [0.1]*7)
    
    while True:
        curr_q = np.array(robot.states().q)
        if np.max(np.abs(curr_q - np.array(z_upper_q))) < 0.005:
            break
        time.sleep(0.5)
    
    # Wait for the physical movement to completely finish
    time.sleep(2)
    
    # After reaching the initial pose, switch to Cartesian mode for easier rotation!
    robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
    time.sleep(1) # Settle
    time.sleep(1) # Settle
    
    # Read Initial TCP Cartesian Pose
    init_pose = robot.states().tcp_pose
    x, y, z = init_pose[0], init_pose[1], init_pose[2]
    qw, qx, qy, qz = init_pose[3], init_pose[4], init_pose[5], init_pose[6]
    
    target_pos = np.array([x, y, z])
    target_rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
    accum_roll_deg = 0.0
    
    print("\n================= CALIBRATION CONTROLS =================")
    print(f" [A] / [D] : Rotate Tool X-axis (Roll) left/right by {STEP_ROT_DEG}°")
    print(f" [W] / [S] : Move Z-axis UP/DOWN by {STEP_Z_M*100} cm")
    print(" [P]       : Print current Required Z_ROT_MARGIN")
    print(" [Q]       : Quit Calibration")
    print("========================================================")
    print(f"Initial State: Z = {z:.4f}m | Roll = 0.0°\n")

    exit_app = False
    while not exit_app:
        key = getch()
        pose_changed = False
        
        if key:
            key = key.lower()
            if key == 'w':
                target_pos[2] += STEP_Z_M
                pose_changed = True
            elif key == 's':
                target_pos[2] -= STEP_Z_M
                pose_changed = True
            elif key == 'a':
                # Local rotation around Tool X axis
                delta = st.Rotation.from_euler('x', -STEP_ROT_DEG, degrees=True)
                target_rot = target_rot * delta
                accum_roll_deg -= STEP_ROT_DEG
                pose_changed = True
            elif key == 'd':
                delta = st.Rotation.from_euler('x', STEP_ROT_DEG, degrees=True)
                target_rot = target_rot * delta
                accum_roll_deg += STEP_ROT_DEG
                pose_changed = True
            elif key == 'p':
                z_val = target_pos[2]
                margin = z_val - 0.1
                print(f"\n[Recorded] Z: {z_val:.4f} | Roll: {accum_roll_deg:.1f}° | Required Margin: {margin:+.4f}m\n")
            elif key == 'q' or key == '\x1b': # 'q' or ESC
                exit_app = True
        
        if pose_changed:
            try:
                # Use scalar_first=False for compatibility, target_tcp wants w,x,y,z
                new_quat = target_rot.as_quat(scalar_first=False) # x, y, z, w
                # Must cast to python float because PyBind11 rejects np.float64
                target_tcp = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), 
                              float(new_quat[3]), float(new_quat[0]), float(new_quat[1]), float(new_quat[2])]
                
                # Directly command cartesian motion instead of doing IK ourselves!
                robot.SendCartesianMotionForce(target_tcp, max_linear_vel=0.1, max_angular_vel=0.5)
                
                margin = target_pos[2] - 0.1
                sys.stdout.write(f"\r[Moving] Z: {target_pos[2]:.4f}m | Roll: {accum_roll_deg:>6.1f}° | Margin: {margin:+.4f}m       ")
                sys.stdout.flush()
                time.sleep(0.1) # Throttle movement
                
            except Exception as e:
                sys.stdout.write(f"\r[WARNING] Motion Failed! {e}       ")
                sys.stdout.flush()
                # Revert change
                curr_p = robot.states().tcp_pose
                target_pos[2] = curr_p[2]
                target_rot = st.Rotation.from_quat([curr_p[4], curr_p[5], curr_p[6], curr_p[3]], scalar_first=False)

    print("\nCalibration Ended.")
    robot.SwitchMode(flexivrdk.Mode.NRT_PLAN_EXECUTION) # safe idle
    
if __name__ == "__main__":
    main()
