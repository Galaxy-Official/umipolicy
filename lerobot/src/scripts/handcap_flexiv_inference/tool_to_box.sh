#!/bin/bash

# 获取目前所在的工作空间目录，并设置 PYTHONPATH
# 将 lerobot/src 和 lingbot-va 所在的根目录加入 PYTHONPATH
export PYTHONPATH="$(pwd)/lerobot/src:$(pwd):$PYTHONPATH"

# （可选）如果有需要，可以指定好 CUDA 环境
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-$CUDA_VERSION/lib64"
# export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
# export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"

# 设定真实的机器人连接参数（安全保护下的初始归零点）
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_INIT_POSE="[0.2532,-0.2683,-0.1907,2.0341,0.0946,0.7968,0.0414]"

echo "Checking CUDA version..."
echo "$(nvcc -V || echo 'nvcc not locally found')"

echo "Starting Real Robot Inference..."
python -m lerobot.scripts.lerobot_flexiv \
    --pretrained_model_name_or_path "ckpt/010000/pretrained_model" \
    --task_name "tool_to_box" \
    --ctrl_freq 20 \
    --obs_horizon 2
