#!/bin/bash

# 获取当前脚本所在路径，然后推导出 lerobot/src 的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LEROBOT_SRC_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 设置 PYTHONPATH
export PYTHONPATH="$LEROBOT_SRC_DIR:$PYTHONPATH"
export FLEXIV_ROBOT_SN="Rizon4-062339"

# (可选) 设置 CUDA 环境
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-$CUDA_VERSION/lib64"
# export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
# export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"

python -m lerobot.get_real_state.read_current_pos
