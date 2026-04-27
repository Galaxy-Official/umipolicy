#!/bin/bash

# ==========================================
# UMI 真机闭环推理执行脚本 (Flexiv + MVS Camera)
# ==========================================

# 退出遇到错误时终止脚本
set -e

# 设置机械臂 RDK 通信参数
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_LOCAL_IP="192.168.2.102"
export FLEXIV_ROBOT_SN="Rizon4-062339"
export FLEXIV_GRIPPER_NAME="Flexiv-GN01"

# 设置机械臂初始安全位姿 [关节弧度制]
# 启动推理前，机械臂会自动运动到此位置
export FLEXIV_INIT_POSE="[-0.0009,-0.4753,-0.0133,1.7838,-0.0058,0.6921,-0.0012]"

# 预训练模型 Checkpoint 路径
CKPT_PATH="ckpt/epoch=0090-train_loss=0.014.ckpt"

# 设置数据保存输出路径
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="data_local/eval_recordings/eval_${TIMESTAMP}"

echo "=========================================="
echo "🚀 准备启动 UMI 真机推理 (Policy Inference)"
echo "📂 模型路径: ${CKPT_PATH}"
echo "💾 数据保存路径: ${OUTPUT_DIR}"
echo "▶️ 机械臂初始位姿: ${FLEXIV_INIT_POSE}"
echo "=========================================="

# 运行推理脚本
python scripts_real/eval_flexiv.py \
    --input "${CKPT_PATH}" \
    --output "${OUTPUT_DIR}" \
    --robot_ip "${FLEXIV_ROBOT_IP}" \
    --local_ip "${FLEXIV_LOCAL_IP}" \
    --frequency 10 \
    --steps_per_inference 6
