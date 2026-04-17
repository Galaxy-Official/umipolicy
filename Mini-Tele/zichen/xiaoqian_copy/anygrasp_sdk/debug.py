import h5py
from PIL import Image
import numpy as np

# 打开HDF5文件
with h5py.File('/home/rhos/xiaoyang/calib_out/capture_0301/clapper/view_000_pose_002/depth.h5', 'r') as f:
    first_frame = f['00000_depth.png'][:]
    image = Image.fromarray(first_frame)
    image.save('depth.png')