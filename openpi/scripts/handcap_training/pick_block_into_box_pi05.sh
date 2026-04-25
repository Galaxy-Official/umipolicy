#!/bin/bash
set -e

# 修改为你要指定的显卡槽位 (例如使用 4 张卡则是 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 在训练前自动计算并生成规范化统计信息 (Norm stats)
# 注意：第一次跑新数据前必须算一次。
python scripts/compute_norm_stats.py --config-name pi05_pick_block_into_box

# 启动训练
python scripts/train.py \
  pi05_pick_block_into_box \
  --exp-name pick_block_into_box_0425_handcap_pi05 \
  --batch-size 128 \
  --num-train-steps 200000 \
  --save-interval 10000 \
  --num-workers 8 \
  --no-wandb-enabled \
  --fsdp-devices 1 \
  --overwrite
  
# 若开启 tactile 触觉传感器，可将上面的 config-name 替换为 pi05_pick_block_into_box_tactile
# 若开启大规模 FSDP 训练节省显存，可修改 --fsdp-devices 到你的显卡数量 (例如 2)
