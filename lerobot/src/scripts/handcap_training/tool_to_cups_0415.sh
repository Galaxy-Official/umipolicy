# 修改为你要指定的显卡槽位 (例如使用 4 张卡则是 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1


# # 强制 Hugging Face 进入离线模式，不联网检查数据集
# export HF_HUB_OFFLINE=1
# # 强制读取本地文件，不尝试在线下载
# export TRANSFORMERS_OFFLINE=1
# source /root/miniconda3/bin/activate
# conda activate umipolicy
accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=no --dynamo_backend=no -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lihongcs/tool_to_cups_0415 \
  --dataset.root=Data/tool_to_cups_0415 \
  --policy.type=diffusion \
  --batch_size=256 \
  --policy.use_tactile=false \
  --optimizer.type=adamw \
  --output_dir=outputs/train/tool_to_cups_0415 \
  --job_name=dp_tool_to_cups_0415_handcap \
  --policy.device=cuda \
  --wandb.enable=false \
  --use_handcap=true \
  --steps=200000 \
  --wandb.mode="offline" \
  --policy.repo_id=tool_to_cups_0415/dp

# 注意: 上面的 --batch_size=64 代表 "单卡 Batch Size"。
# LeRobot 3.0 中的总有效 Batch Size = batch_size * num_processes。
# 如果想保持以前 256 的总训练吞吐量，当使用 4 卡时，则应把 batch_size 设为 64 (64 * 4 = 256)。