import cv2
import os

def save_middle_frame(video_path, output_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return False
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("错误：视频没有帧数据")
        return False
    
    # 计算中间帧位置
    middle_frame_index = total_frames // 3
    
    # 设置读取位置到中间帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    
    # 读取中间帧
    ret, frame = cap.read()
    
    if not ret:
        print("错误：无法读取中间帧")
        return False
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为JPG文件（质量为95%）
    cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # 释放视频资源
    cap.release()
    
    print(f"成功保存中间帧至: {output_path}")
    return True

# 使用示例
if __name__ == "__main__":
    video_file = "/home/yushun/Workspace/Mini-Tele/test_output_5s.mp4"  # 替换为你的MP4文件路径
    output_file = "/home/yushun/Workspace/Mini-Tele/lerobot/scripts/test/ori.jpg"  # 替换为输出路径
    
    save_middle_frame(video_file, output_file)