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
export FLEXIV_INIT_POSE="[-0.0009,-0.3386,-0.0174,1.8982,-0.0058,0.6271,-0.0012]"

# 预训练模型 Checkpoint 路径
CKPT_PATH="ckpt/erase_320.ckpt"

# Policy 预测夹爪宽度偏置，单位与夹爪命令一致。
# 正数会比预测值更宽，负数会比预测值更窄。
# 夹爪命令会在 offset 后裁剪到 [GRIPPER_WIDTH_MIN, GRIPPER_WIDTH_MAX]。
# 可以用位置参数或环境变量覆盖：
#   bash run_scripts/inference/erase_flexiv.sh 0.01 0.1 0.9
#   GRIPPER_WIDTH_OFFSET=0.01 GRIPPER_WIDTH_MIN=0.1 GRIPPER_WIDTH_MAX=0.9 bash run_scripts/inference/erase_flexiv.sh
GRIPPER_WIDTH_OFFSET="${1:-${GRIPPER_WIDTH_OFFSET:-0.0}}"
GRIPPER_WIDTH_MIN="${2:-${GRIPPER_WIDTH_MIN:-0.1}}"
GRIPPER_WIDTH_MAX="${3:-${GRIPPER_WIDTH_MAX:-0.9}}"

# 机械臂执行速度限制。数值越小，执行越慢、更柔和。
ARM_MAX_LINEAR_VEL="${ARM_MAX_LINEAR_VEL:-0.02}"
ARM_MAX_ANGULAR_VEL="${ARM_MAX_ANGULAR_VEL:-0.15}"
ARM_MAX_LINEAR_ACC="${ARM_MAX_LINEAR_ACC:-0.04}"
ARM_MAX_ANGULAR_ACC="${ARM_MAX_ANGULAR_ACC:-0.25}"
GRIPPER_MOVE_VELOCITY="${GRIPPER_MOVE_VELOCITY:-0.02}"

# 设置数据保存输出路径
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="data_local/eval_recordings/erase_board/eval_${TIMESTAMP}"

echo "=========================================="
echo "🚀 准备启动 UMI 真机推理 (Policy Inference)"
echo "📂 模型路径: ${CKPT_PATH}"
echo "💾 数据保存路径: ${OUTPUT_DIR}"
echo "▶️ 机械臂初始位姿: ${FLEXIV_INIT_POSE}"
echo "🤏 夹爪宽度偏置: ${GRIPPER_WIDTH_OFFSET}"
echo "🛡️ 夹爪安全范围: [${GRIPPER_WIDTH_MIN}, ${GRIPPER_WIDTH_MAX}]"
echo "执行限速: linear_vel=${ARM_MAX_LINEAR_VEL}, angular_vel=${ARM_MAX_ANGULAR_VEL}, linear_acc=${ARM_MAX_LINEAR_ACC}, angular_acc=${ARM_MAX_ANGULAR_ACC}, gripper_vel=${GRIPPER_MOVE_VELOCITY}"
echo "=========================================="

# 运行推理脚本
python scripts_real/eval_flexiv.py \
    --input "${CKPT_PATH}" \
    --output "${OUTPUT_DIR}" \
    --robot_ip "${FLEXIV_ROBOT_IP}" \
    --local_ip "${FLEXIV_LOCAL_IP}" \
    --frequency 3 \
    --steps_per_inference 6 \
    --gripper-width-offset "${GRIPPER_WIDTH_OFFSET}" \
    --gripper-width-min "${GRIPPER_WIDTH_MIN}" \
    --gripper-width-max "${GRIPPER_WIDTH_MAX}" \
    --arm-max-linear-vel "${ARM_MAX_LINEAR_VEL}" \
    --arm-max-angular-vel "${ARM_MAX_ANGULAR_VEL}" \
    --arm-max-linear-acc "${ARM_MAX_LINEAR_ACC}" \
    --arm-max-angular-acc "${ARM_MAX_ANGULAR_ACC}" \
    --gripper-move-velocity "${GRIPPER_MOVE_VELOCITY}"
