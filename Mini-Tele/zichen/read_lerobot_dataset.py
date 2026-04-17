from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import torch

# 1. 指定本地数据集路径
# 注意：路径应当指向包含 info.json 的那个文件夹
repo_id = "test/test" 
root_path = Path("/home/yushun/Workspace/Mini-Tele/zichen/lerobot_data/multi-objects/01-19-15-17-44/lerobot/test") # 替换为你的实际路径

# 2. 加载数据集
# 如果数据集在本地，LeRobot 会自动识别并加载
dataset = LeRobotDataset(
    repo_id=repo_id,
    root=root_path,
    local_files_only=True
)

print(f"数据集加载成功！包含 {len(dataset)} 帧数据。")
print(f"数据特征格式: {dataset.features}")

# 3. 读取其中的一帧数据
# idx 是帧的索引
frame_idx = 0
data = dataset[frame_idx]

# data 是一个字典，通常包含：
# - 'observation.images.laptop' (图像数据)
# - 'observation.state' (机器人状态/关节位置)
# - 'action' (指令数据)
# - 'next.reward' 等等

print("\n第一帧的状态数据 (observation.state):")
print(data['observation.state'])

print("\n第一帧的操作指令 (action):")
print(data['action'])

print("\n第一帧的手腕相机位姿:")
print(data['observation.extrinsics.wrist'])  # 应该是 (16,) 的 numpy 数组

print("\n第一帧的手腕相机内参:")
print(data['observation.intrinsics.wrist'])  # 应该是 (9,) 的 numpy 数组

# 4. 遍历数据集的小例子
# 使用 DataLoader 批量读取
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

for batch in dataloader:
    # batch 是一个字典，里面的 tensor 维度为 (Batch, ...)
    images = batch['observation.images.wrist'] 
    actions = batch['action']
    depths = batch.get('observation.depth.wrist', None)  # 如果有深度图的话
    print(f"Batch images shape: {images.shape}") # (8, 3, 480, 640)
    print(f"Batch extrinsics shape: {batch['observation.extrinsics.wrist'].shape}")
    if depths is not None:
        print(f"Batch depths dtype: {depths.dtype}")
        print(depths/1000.0)  # 转换为米
    break # 只演示一组