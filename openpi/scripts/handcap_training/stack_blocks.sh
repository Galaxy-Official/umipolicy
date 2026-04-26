#!/bin/bash
set -e

# =========================================================================
# 自动保存终端日志到输出文件夹 (logs/)，方便容器被回收后排查报错原因
# =========================================================================
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1



# 修改为你要指定的显卡槽位 (例如使用 4 张卡则是 0,1,2,3)
export CUDA_VISIBLE_DEVICES=2,3

# OpenPI(JAX环境)不需要 accelerate，直接调用 python script 即可跑 DDP
# conda activate umipolicy

# 若你之前单卡 64，2 张卡其实总归是 128；如果 4 卡则总计 256。
# 此处我们设成全局 128 (对应原有2卡 * 64)。

# 在训练前自动计算并生成规范化统计信息 (Norm stats)
python scripts/compute_norm_stats.py --config-name pi0_stack_blocks

python scripts/train.py \
  pi0_stack_blocks \
  --exp-name dp_stack_blocks_0409_handcap \
  --batch-size 128 \
  --num-train-steps 200000 \
  --no-wandb-enabled \
  --fsdp-devices 1 \
  --overwrite
  
# 若开启 tactile 触觉传感器，可将上面的 config-name 替换为 pi0_stack_blocks_tactile
# 若开启大规模 FSDP 训练节省显存，可修改 --fsdp-devices 到你的显卡数量 (例如 2)
