

import os
import time
import tqdm
import numpy as np
import transformations as tf
from camera import CameraD400
from robot import Flexiv
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize

import numpy as np

robot = Flexiv()
base_joint_base = np.array([ 0.82071282 ,-0.14327244,  0.18064141])
base_joint_direction = np.array([-0.02972793 , 0.9995213 , -0.00856835])
rotation_angle = 30 * np.pi / 180
delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=base_joint_direction, point=base_joint_base)
point1 = np.array([[-0.73799321250429, -0.09549912555174345, 0.6680164184482673, 0.391803115606308], [-0.046102514734295635, 0.9947577090307628, 0.09127791879224047, -0.00643054349347949], [-0.6732314434373756, 0.03656524774592858, -0.7385271871938818, 0.36939406394958496], [0.0, 0.0, 0.0, 1.0]])
point2 = np.array([[-0.9607515195093781, -0.0068475216203268165, 0.27732585383999236, 0.48352834582328796], [0.04115933616819667, 0.9851120454744393, 0.1669136510515543, 0.00028473243582993746], [-0.27433998397356346, 0.1717770919209713, -0.9461660551322688, 0.5206602811813354], [0.0, 0.0, 0.0, 1.0]])
point3 = np.array([[-0.9894351180610128, -0.11110781635142362, 0.09313001821758601, 0.5475508570671082], [-0.012501433743757617, 0.7053758254432096, 0.7087232598374776, -0.25283142924308777], [-0.14443635727177567, 0.7000714235175786, -0.6993126201293174, 0.3959086239337921], [0.0, 0.0, 0.0, 1.0]])
robot.movePosePrimitive(point3)
print(delta_pose @ point1)
tcppose = robot.readPose()
print("pose: ", tcppose)
print("Next pose: ", delta_pose @ tcppose)

# robot.send_gripper_state(0.07,0.1,20)