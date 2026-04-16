#!/bin/bash

# 获取目前所在的工作空间目录，并设置 PYTHONPATH
export PYTHONPATH="$(pwd)/lerobot/src/lib_py:$(pwd)/lerobot/src:$(pwd):$PYTHONPATH"

# (可选) 设置 CUDA 环境
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-$CUDA_VERSION/lib64"
# export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
# export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"

python -m lerobot.get_real_state.read_current_pos
