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
export FLEXIV_INIT_POSE="[-0.0009,-0.1701,-0.0133,2.0214,-0.0058,0.6921,-0.0012]"

echo "Checking CUDA version..."
echo "$(nvcc -V || echo 'nvcc not locally found')"

echo "Starting Real Robot Inference..."
python -m lerobot.scripts.lerobot_flexiv \
    --pretrained_model_name_or_path "ckpt/050000/pretrained_model" \
    --task_name "simple_sorting" \
    --ctrl_freq 1 \
    --steps_per_inference 2 \
    --obs_horizon 2
