#!/bin/bash

# 获取目前所在的工作空间目录，并设置 PYTHONPATH
# 将 lerobot/src 和 当前运行所在的根目录加入 PYTHONPATH
export PYTHONPATH="$(pwd)/lerobot/src/lib_py:$(pwd)/lerobot/src:$(pwd):$PYTHONPATH"

# 设定真实的机器人连接参数（安全保护下的初始归零点）
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_INIT_POSE="[0.0007,-0.2421,-0.0005,2.1403,0.0057,1.0728,0.0085]"

echo "Starting Real Robot Inference (RDK 0.9) ..."
python -m lerobot.scripts.lerobot_flexiv_09 \
    --pretrained_model_name_or_path "./ckpt/lingbot-va-base/transformer" \
    --task_name "block_stack" \
    --ctrl_freq 20 \
    --obs_horizon 2
