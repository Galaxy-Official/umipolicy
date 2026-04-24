import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "lib_py")))

import flexivrdk
import socket
import time

robot_ip = "192.168.2.100"
robot_sn = os.environ.get("FLEXIV_ROBOT_SN", robot_ip)

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((robot_ip, 1))
    actual_local_ip = s.getsockname()[0]
    s.close()
except Exception:
    actual_local_ip = "192.168.2.102"

try:
    robot = flexivrdk.Robot(robot_sn, [actual_local_ip])
    time.sleep(0.5)
    states = robot.states()
    joints = states.q
    ee_pose = states.tcp_pose
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
