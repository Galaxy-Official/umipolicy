import pybullet as pb

pb.connect(pb.DIRECT)
robot = pb.loadURDF("urdf/assets/low_cost_robot_description/urdf/low_cost_robot.urdf", useFixedBase=True)

# Test with 0 angles
data = [0, 0, 0, 0, 0, 0]
for i, joint in enumerate(data):
    pb.resetJointState(robot, i, joint)

pos, orn = pb.getLinkState(robot, 4)[:2]
print("0 angles -> pos:", pos)

# Test with [0, 90, -90, -90, 90, 0] (The Mini-Tele map for raw 0s)
data2 = [0, 90, -90, -90, 90, 0]
data2 = [x * 3.14159/180 for x in data2]
for i, joint in enumerate(data2):
    pb.resetJointState(robot, i, joint)

pos2, orn2 = pb.getLinkState(robot, 4)[:2]
print("Mini-Tele map -> pos:", pos2)
