import numpy as np
import scipy.spatial.transform as st
import sys
sys.path.append("/Users/macbookpro/Desktop/workspace/umipolicy/lerobot/src")
from lerobot.datasets.pose_utils import mat_to_pose10d, pose10d_to_mat

mat = np.eye(4)
mat[0, 3] = 1.0
mat[1, 3] = 2.0
mat[2, 3] = 3.0
print("mat_to_pose10d:", mat_to_pose10d(mat).shape)
