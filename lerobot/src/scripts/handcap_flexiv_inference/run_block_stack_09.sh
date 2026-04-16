#!/bin/bash

# 获取当前脚本所在路径，然后推导出 lerobot/src 的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LEROBOT_SRC_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 设置 PYTHONPATH，确保优先加载 lib_py 目录下的 flexivrdk 0.9 环境
export PYTHONPATH="$LEROBOT_SRC_DIR/lib_py:$LEROBOT_SRC_DIR:$PYTHONPATH"

# 设定真实的机器人连接参数（安全保护下的初始归零点）
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_INIT_POSE="[0.0007,-0.2421,-0.0005,2.1403,0.0057,1.0728,0.0085]"

echo "Starting Real Robot Inference (RDK 0.9) ..."
python -m lerobot.scripts.lerobot_flexiv_09 \
    --pretrained_model_name_or_path "./ckpt/lingbot-va-base/transformer" \
    --task_name "block_stack" \
    --ctrl_freq 20 \
    --obs_horizon 2
