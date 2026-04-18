#!/bin/bash
# ==============================================================================
# Cosmos Policy: 通用训练启动脚本
# ==============================================================================

# 1. 设置系统所使用的 GPU 数量 (单机多卡)
NUM_GPUS=2

# 2. 设置您的预处理数据集所在的根目录
# (注: 下一级的目录应该是您转换好的类似于 ALOHA 或者 Handcap 的全量数据集)
export BASE_DATASETS_DIR="/Users/macbookpro/Desktop/simple_sorting_409/handcapcosmos"

# 3. 指定您想要跑的实验配置名称 (来源于 cosmos_policy/config/experiment/)
# 我们创建了新的 cosmos_predict2_handcap 用于处理类似 ALOHA HDF5 结构的 handcapcosmos:
EXPERIMENT_NAME="cosmos_predict2_handcap"

# 4. 指定开放的内部主节点通讯端口 (防止与其他训练任务冲突)
MASTER_PORT=12341

echo "================================================================================"
echo "🚀 正在启动 Cosmos Policy 训练..."
echo "📂 数据集基准路径: $BASE_DATASETS_DIR"
echo "🔧 GPU 调度数量: $NUM_GPUS"
echo "🧪 加载的配置卡: $EXPERIMENT_NAME"
echo "================================================================================"

# 5. 直接拉起 torchrun (因为您已经在 cosmos-policy 的 conda 环境中了)
torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="${EXPERIMENT_NAME}"
