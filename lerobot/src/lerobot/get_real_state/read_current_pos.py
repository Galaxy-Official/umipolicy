from scipy.spatial.transform import Rotation as R
import sys
import os

# from umi.common.pose_util import *
# from umi.real_world.flexiv import *
sys.path.append(os.path.join(os.path.dirname(__file__), "./../../lib_py"))
from lerobot.umi.real_world.flexiv_controller import FlexivInterface

sys.path.append("lib_py")

import flexivrdk


robot = FlexivInterface("192.168.2.100", "192.168.2.102", move_home=False, init_offset=None, init_qpos=None)
print(robot.get_ee_pose())
print(robot.get_joint_positions())

joints = robot.get_joint_positions()
joints_str = ",".join(["%.4f"%j for j in joints])
print('export FLEXIV_INIT_POSE="[%s]"'%joints_str)


### Set to specific joint positio
# print(robot.send_joint_position([-1.3136, 0.0514, 1.3010, 2.2039, 0.2174, 1.2170, -0.18627563]))


### Set to flexiv home
# robot = FlexivInterface("192.168.2.100", "192.168.2.103", move_home=True, init_offset=None, init_qpos=None)
