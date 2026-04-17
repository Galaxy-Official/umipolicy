import pyrealsense2 as rs
import cv2
import os
import numpy as np

# 设置 .bag 文件路径和输出文件夹
bag_file = "/home/rhos/xiaoyang/data/20250306_225750.bag"  # 替换为你的 .bag 文件路径
output_folder = "output_frames4"  # 输出文件夹

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 配置 Realsense 管道
pipeline = rs.pipeline()
config = rs.config()

# 启用 .bag 文件作为输入
config.enable_device_from_file(bag_file)

# 启动管道
pipeline.start(config)

# 初始化帧计数器
frame_count = 0
try:
    while True:
        # 等待下一帧
        frames = pipeline.wait_for_frames()

        # 获取 RGB 帧
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 将帧转换为 NumPy 数组
        color_image = np.asanyarray(color_frame.get_data())

        # 将 BGR 转换为 RGB
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 保存为 .jpg 文件
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, color_image_rgb)
        print(f"Saved {frame_filename}")

        # 增加帧计数器
        frame_count += 1

except RuntimeError:
    # 当 .bag 文件读取完毕时，会抛出 RuntimeError
    print("Finished processing the .bag file.")

finally:
    # 停止管道
    pipeline.stop()
# try:
#     while True:
#         # 等待下一帧
#         frames = pipeline.wait_for_frames()

#         # 获取 RGB 帧
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             continue

#         # 将帧转换为 NumPy 数组
#         color_image = np.asanyarray(color_frame.get_data())

#         # 保存为 .jpg 文件
#         frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
#         cv2.imwrite(frame_filename, color_image)
#         print(f"Saved {frame_filename}")

#         # 增加帧计数器
#         frame_count += 1

# except RuntimeError:
#     # 当 .bag 文件读取完毕时，会抛出 RuntimeError
#     print("Finished processing the .bag file.")

# finally:
#     # 停止管道
#     pipeline.stop()