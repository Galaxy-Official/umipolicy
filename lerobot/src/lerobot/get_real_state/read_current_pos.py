import sys
import os
import time
import socket

import flexivrdk

def main():
    robot_ip = os.environ.get("FLEXIV_ROBOT_IP", "192.168.2.100")
    local_ip = os.environ.get("FLEXIV_LOCAL_IP", "192.168.2.102")
    robot_sn = os.environ.get("FLEXIV_ROBOT_SN", "Rizon4-062339")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((robot_ip, 1))
        actual_local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        actual_local_ip = local_ip

    # Initialize Robot interface (native RDK)
    try:
        robot = flexivrdk.Robot(robot_sn, [actual_local_ip])
    except Exception as e:
        print(f"Failed to connect to robot: {e}", file=sys.stderr)
        sys.exit(1)

    # Allow a moment for UDP state packets to arrive
    time.sleep(0.5)
    states = robot.states()

    # Export Joints
    joints = states.q
    if joints is not None and len(joints) > 0:
        joints_str = ",".join(["%.4f" % j for j in joints])
        print('export FLEXIV_INIT_POSE="[%s]"' % joints_str)
    else:
        print('export FLEXIV_INIT_POSE="[]"')

    # Export TCP Pose (EEF)
    ee_pose = states.tcp_pose
    if ee_pose is not None and len(ee_pose) > 0:
        ee_pose_str = ",".join(["%.4f" % p for p in ee_pose])
        print('export FLEXIV_INIT_EEF_POSE="[%s]"' % ee_pose_str)
    else:
        print('export FLEXIV_INIT_EEF_POSE="[]"')

if __name__ == "__main__":
    main()
