# 修改为你要指定的显卡槽位 (例如使用 4 张卡则是 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1

source /root/miniconda3/bin/activate
conda activate robopolicy

# 1. 使用 accelerate launch 来启动分布式训练任务
# 2. 指定 --num_processes 的数量（与你上面分配的显卡数量一致）
# 3. 将原来的 train 替换为了 v3.0 中的 lerobot_train
accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=no --dynamo_backend=no -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lihongcs/block_to_pot_handcap \
  --dataset.root=Data/handcap2603/simple_sorting_0409 \
  --policy.type=diffusion \
  --batch_size=64 \
  --policy.use_tactile=false \
  --optimizer.type=adamw \
  --output_dir=outputs/train/simple_sorting_0409 \
  --job_name=dp_simple_sorting_0409_handcap \
  --policy.device=cuda \
  --wandb.enable=true \
  --use_handcap=true \
  --steps=200000 \
  --wandb.mode="offline" \
  --policy.repo_id=simple_sorting_0409/dp

# 注意: 上面的 --batch_size=64 代表 "单卡 Batch Size"。
# LeRobot 3.0 中的总有效 Batch Size = batch_size * num_processes。
# 如果想保持以前 256 的总训练吞吐量，当使用 4 卡时，则应把 batch_size 设为 64 (64 * 4 = 256)。