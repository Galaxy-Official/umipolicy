# 修改为你要指定的显卡槽位 (目前针对你的配置：2 张 H200 则设为 0,1)
export CUDA_VISIBLE_DEVICES=0,1

# 屏蔽烦人的 torchvision pyav 弃用警告刷屏问题
export PYTHONWARNINGS="ignore"


# # 强制 Hugging Face 进入离线模式，不联网检查数据集
# export HF_HUB_OFFLINE=1
# # 强制读取本地文件，不尝试在线下载
# export TRANSFORMERS_OFFLINE=1
# source /root/miniconda3/bin/activate
# conda activate umipolicy

# 【核心硬件优化提升说明】：
# 1. --mixed_precision=bf16 针对 Hopper 架构开启更快的 bfloat16 半精度加速计算。
# 2. --batch_size=128：2 张显卡组成双卡并行则总 Global Batch Size 为 256，保障扩散模型最好的收敛速度与稳定度。
# 3. --training.num_workers=20：压榨你的 80 核 CPU 和 200内存，保证 GPU 高速读图不掉速。

accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=inductor -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lihongcs/tool_to_cups_0415 \
  --dataset.root=Data/tool_to_cups_0415 \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --batch_size=128 \
  --num_workers=20 \
  --policy.use_tactile=false \
  --optimizer.type=adamw \
  --output_dir=outputs/train/tool_to_cups_0415_1  \
  --job_name=dp_tool_to_cups_0415_handcap \
  --policy.device=cuda \
  --wandb.enable=false \
  --use_handcap=true \
  --steps=200000 \
  --wandb.mode="offline" \
  --policy.repo_id=tool_to_cups_0415/dp

# 注意: 上面的 --batch_size=128 代表 "单卡 Batch Size"。
# LeRobot 3.0 中的总有效 Batch Size = batch_size * num_processes。
# 当使用 2 卡且每卡 `batch_size=128` 时，系统总吞吐量将会是 256。