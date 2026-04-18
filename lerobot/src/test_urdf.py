import pybullet as pb

pb.connect(pb.DIRECT)
robot = pb.loadURDF("flexiv_rizon4.urdf", useFixedBase=True)
print("=== FLEXIV RIZON 4 ===")
for i in range(pb.getNumJoints(robot)):
    info = pb.getJointInfo(robot, i)
    print(f"Index {i} | JointName: {info[1].decode()} | LinkName: {info[12].decode()}")

robot2 = pb.loadURDF("low_cost_robot.urdf", useFixedBase=True)
print("\n=== LOW COST ROBOT (KOCH) ===")
for i in range(pb.getNumJoints(robot2)):
    info = pb.getJointInfo(robot2, i)
    print(f"Index {i} | JointName: {info[1].decode()} | LinkName: {info[12].decode()}")
