# 修改为你要指定的显卡槽位 (例如使用 4 张卡则是 0,1,2,3)
export CUDA_VISIBLE_DEVICES=2,3

# source /root/miniconda3/bin/activate
# conda activate umipolicy

accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=no --dynamo_backend=no -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lihongcs/block_stack_handcap \
  --dataset.root=Data/block_stack_0414 \
  --policy.type=diffusion \
  --batch_size=256 \
  --policy.use_tactile=false \
  --optimizer.type=adamw \
  --output_dir=outputs/train/block_stack_0414 \
  --job_name=dp_block_stack_0409_handcap \
  --policy.device=cuda \
  --wandb.enable=false \
  --use_handcap=true \
  --steps=200000 \
  --wandb.mode="offline" \
  --policy.repo_id=block_stack_0409/dp

# 注意: 上面的 --batch_size=64 代表 "单卡 Batch Size"。
# LeRobot 3.0 中的总有效 Batch Size = batch_size * num_processes。
# 如果想保持以前 256 的总训练吞吐量，当使用 4 卡时，则应把 batch_size 设为 64 (64 * 4 = 256)。