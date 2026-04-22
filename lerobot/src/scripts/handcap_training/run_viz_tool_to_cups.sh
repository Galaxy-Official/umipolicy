#!/bin/bash

# =========================================================
# 离线轨迹与视觉评估可视化脚本 (Offline Eval Viz)
# 用于加载训练好的 Checkpoint 并随机测试生成 10 个测试视频
# =========================================================

# 1. 指定使用的显卡 (推断只需要 1 张卡)
export CUDA_VISIBLE_DEVICES=0

# 如果服务器处于内网或者你想避免联网校验，取消注释下面两行：
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# 2. 执行评测脚本
# 请根据实际情况修改 --policy-path, --repo-id, --root 等参数
python -m lerobot.scripts.lerobot_eval_offline_viz \
  --policy-path outputs/train/tool_to_cups_0415_1/checkpoints/090000 \
  --repo-id lihongcs/tool_to_cups_0415 \
  --root Data/tool_to_cups_0415 \
  --num-episodes 10 \
  --output-video outputs/eval_viz_tool_to_cups \
  --fps 10

echo "Finished evaluation! Check the output directory for MP4 videos."
