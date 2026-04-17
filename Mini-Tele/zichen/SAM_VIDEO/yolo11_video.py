from ultralytics import YOLO

# 加载模型 (n, s, m, l, x 不同大小可选)
model = YOLO("/home/yushun/Workspace/Mini-Tele/zichen/ckpt/yolo11x-seg.pt") 

# 预测视频
results = model.predict("/home/yushun/Workspace/Mini-Tele/zichen/SAM_VIDEO/episode_000001.mp4", save=True, show=True)