import os
import sys
import time
import socket
import numpy as np

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

    print("Connecting to robot for Forward Kinematics computation...")
    robot = flexivrdk.Robot(robot_sn, [local_ip])
    model = flexivrdk.Model(robot)
    
    # We will just print the Cartesian poses for these joint sets.
    # Note: flexivrdk usually does not require moving the robot to compute IK/FK if the Model API is used.
    
    joint_bounds = {
        "Z_UPPER": [0.2510, -0.6796, -0.1615, 1.8408, 0.1169, 0.9518, 0.1125],
        "Z_LOWER": [0.2343, -0.1154, -0.2446, 1.6448, -0.1649, 1.1261, 0.2185],
        "X_UPPER": [0.1990, 0.4214, -0.2038, 2.6006, -0.0526, 0.5814, 0.0766],
        "X_LOWER": [0.1166, -1.3262, -0.1235, 0.5007, 0.0600, 0.8002, -0.0273],
        "Y_LOWER": [-0.0179, -0.6123, -0.4847, 1.7435, 0.0350, 0.9849, -0.1486],
        "Y_UPPER": [0.4299, -0.6353, 0.2477, 1.6003, 0.1313, 0.9242, 0.4712]
    }

    print("\n--- Cartesian Coordinates (TCP) ---")
    # In some RDK versions, model.forward might not be exposed to Python. 
    # If this fails, we will need to move the robot. Let's try to see if model has a method.
    for name, q in joint_bounds.items():
        try:
            # Let's see if we can get TCP position from model. 
            # If `model` doesn't have FK, we might need to actually move the arm. 
            # We will ask the user to be ready just in case.
            pass 
        except Exception:
            pass

    # Since we aren't 100% sure of the Python FK API, I will provide a safe movement script.
    print("To get exact Cartesian limits, the robot will slowly move to each boundary pose.")
    print("Please ensure the workspace is CLEAR.")
    confirm = input("Type 'yes' to proceed with physical movement, or 'no' to abort: ")
    if confirm.strip().lower() != 'yes':
        print("Aborted.")
        return

    robot.Enable()
    while not robot.operational():
        time.sleep(1)
    
    robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
    
    for name, q in joint_bounds.items():
        print(f"\nMoving to {name}...")
        robot.SendJointPosition(q, [0]*7, [0.1]*7, [0.1]*7)
        
        # Wait until reached
        while True:
            curr_q = np.array(robot.states().q)
            if np.max(np.abs(curr_q - np.array(q))) < 0.05:
                break
            time.sleep(0.5)
            
        time.sleep(1.0) # Settle
        pose = robot.states().tcp_pose
        x, y, z = pose[0], pose[1], pose[2]
        print(f"[{name}] Cartesian TCP: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

    print("\nDone! Please copy the output above and provide it to the AI.")

if __name__ == "__main__":
    main()
