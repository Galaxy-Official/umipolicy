import cv2
import glob
import os

# 设置图片所在的文件夹路径
folder_path = '/home/rhos/xiaoyang/output_frames4'  # 替换为你的文件夹路径

# 获取所有 .jpg 文件并按文件名排序
images = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))

# 读取第一张图片以获取图像的尺寸
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# 设置视频输出的格式（MP4，帧率30）
output_video = 'door_v2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# 将每一张图片作为一帧写入视频
for image in images:
    frame = cv2.imread(image)
    video_writer.write(frame)

# 释放视频写入器
video_writer.release()

print(f"视频已保存为 {output_video}")
