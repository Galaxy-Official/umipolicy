import flexivrdk
import numpy as np

robot = flexivrdk.Robot("192.168.2.100", ["192.168.2.102"])
robot.enable()
import time
while not robot.isOperational():
    time.sleep(1)

model = flexivrdk.Model(robot)
states = flexivrdk.RobotStates()
robot.getRobotStates(states)

print("Current TCP Pose:", states.tcp_pose)
print("Current Joints:", states.q)

# test IK
target_pose = list(states.tcp_pose)
target_pose[0] += 0.05  # move 5cm in X
res, ik_joints = model.reachable(target_pose, states.q, False)
print("Reachable:", res)
print("IK Joints:", ik_joints)
