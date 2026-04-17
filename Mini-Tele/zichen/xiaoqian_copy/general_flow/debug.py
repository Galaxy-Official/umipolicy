# import h5py
# from PIL import Image
# import numpy as np

# # 打开HDF5文件
# with h5py.File('/home/rhos/xiaoyang/calib_out/capture/box033/view_000_pose_000/depth.h5', 'r') as f:
#     # 获取第一帧的数据
#     first_frame = f['00000_depth.png'][:]
    
#     # 将数据转换为图像
#     image = Image.fromarray(first_frame)
    
#     # 保存为PNG文件
#     image.save('dep.png')
import os
import glob
import h5py
from PIL import Image
import numpy as np

# 定义根目录
root_dir = "/home/rhos/xiaoyang/calib_out/capture_0305/door"
h5_files = glob.glob(os.path.join(root_dir, "view_*", "depth.h5"))

# 遍历每个文件
for h5_file in h5_files:
    try:
        # 打开 HDF5 文件
        with h5py.File(h5_file, 'r') as f:
            # 获取第一帧的数据（假设数据集名称为 '00000_depth.png'）
            first_frame = f['00000_depth.png'][:]
            
            # 创建图像并保存
            image = Image.fromarray(first_frame)
            
            # 生成输出文件名
            output_dir = os.path.dirname(h5_file)  # 获取文件所在目录
            output_file = os.path.join(output_dir, "dep.png")  # 输出文件路径
            
            # 保存图像
            image.save(output_file)
            print(f"Processed and saved: {output_file}")
    
    except Exception as e:
        print(f"Error processing {h5_file}: {e}")
# import pickle

# # 加载 data.pkl 文件
# with open('/home/rhos/xiaoyang/general_flow/demo/output/robot_exec.pkl', 'rb') as file:
#     data = pickle.load(file)

# # 输出数据
# print(data)

# # 'gripper_3d_pos': array([0.0888476 , 0.03886898, 0.377     ])