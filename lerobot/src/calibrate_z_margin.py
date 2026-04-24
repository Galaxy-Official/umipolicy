import os
import sys
import time
import socket
import numpy as np
import scipy.spatial.transform as st

try:
    from pynput import keyboard
except ImportError:
    print("Please install pynput: pip install pynput")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "lib_py")))
import flexivrdk

# --- Calibration Settings ---
STEP_ROT_DEG = 5.0      # 每次旋转 5 度
STEP_Z_M = 0.005        # 每次移动 0.5 cm
# ----------------------------

target_pose = None  # [x, y, z, roll, pitch, yaw]
pose_changed = False
exit_app = False

def get_local_ip(robot_ip):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((robot_ip, 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.2.102"

def on_press(key):
    global target_pose, pose_changed, exit_app
    if target_pose is None:
        return
        
    try:
        if key == keyboard.Key.up:
            target_pose[2] += STEP_Z_M
            pose_changed = True
        elif key == keyboard.Key.down:
            target_pose[2] -= STEP_Z_M
            pose_changed = True
        elif key == keyboard.Key.left:
            target_pose[3] -= np.radians(STEP_ROT_DEG)
            pose_changed = True
        elif key == keyboard.Key.right:
            target_pose[3] += np.radians(STEP_ROT_DEG)
            pose_changed = True
        elif key == keyboard.Key.enter:
            # Print current Z and Roll
            z = target_pose[2]
            roll = np.degrees(target_pose[3])
            margin = z - 0.0715
            print(f"\n[Recorded] Z: {z:.4f} | Roll: {roll:.1f}° | Required Margin: {margin:+.4f}m")
        elif key == keyboard.Key.esc:
            exit_app = True
    except Exception:
        pass

def main():
    global target_pose, pose_changed, exit_app
    
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
        if np.max(np.abs(curr_q - np.array(z_upper_q))) < 0.05:
            break
        time.sleep(0.5)
    time.sleep(1) # Settle
    
    # Read Initial TCP Cartesian Pose
    init_pose = robot.states().tcp_pose
    x, y, z = init_pose[0], init_pose[1], init_pose[2]
    qw, qx, qy, qz = init_pose[3], init_pose[4], init_pose[5], init_pose[6]
    
    rot = st.Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
    roll, pitch, yaw = rot.as_euler('xyz')
    
    target_pose = [x, y, z, roll, pitch, yaw]
    
    print("\n================= CALIBRATION CONTROLS =================")
    print(f" [LEFT/RIGHT] : Rotate X-axis (Roll) by {STEP_ROT_DEG}°")
    print(f" [UP/DOWN]    : Move Z-axis by {STEP_Z_M*100} cm")
    print(" [ENTER]      : Print current Required Z_ROT_MARGIN")
    print(" [ESC]        : Exit Calibration")
    print("========================================================")
    print(f"Initial State: Z = {z:.4f}m | Roll = {np.degrees(roll):.1f}°")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not exit_app:
        if pose_changed:
            pose_changed = False
            
            # Construct new TCP 7D list
            tx, ty, tz, troll, tpitch, tyaw = target_pose
            new_rot = st.Rotation.from_euler('xyz', [troll, tpitch, tyaw])
            new_quat = new_rot.as_quat(scalar_first=False) # x, y, z, w
            
            target_tcp = [tx, ty, tz, new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
            
            # Use strict IK (same as inference script)
            result = model.reachable(target_tcp, robot.states().q, True)
            if result[0]:
                robot.SendJointPosition(result[1], [0]*7, [0.1]*7, [0.1]*7)
                margin = tz - 0.0715
                sys.stdout.write(f"\r[Moving] Z: {tz:.4f}m | Roll: {np.degrees(troll):>6.1f}° | Margin: {margin:+.4f}m       ")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r[WARNING] IK Unreachable! Z: {tz:.4f}m | Roll: {np.degrees(troll):>6.1f}°       ")
                sys.stdout.flush()
                # Revert change
                # (You might need to press the opposite key to fully clear the error state in your mind,
                # but target_pose remains at the unreachable value. We revert it to match robot state)
                curr_p = robot.states().tcp_pose
                target_pose[2] = curr_p[2]
                rot2 = st.Rotation.from_quat([curr_p[4], curr_p[5], curr_p[6], curr_p[3]], scalar_first=False)
                target_pose[3] = rot2.as_euler('xyz')[0]
                
        time.sleep(0.05)

    listener.stop()
    print("\nCalibration Ended.")
    robot.SwitchMode(flexivrdk.Mode.NRT_PLAN_EXECUTION) # safe idle
    
if __name__ == "__main__":
    main()
