#!/usr/bin/env bash

# 指定要使用的显卡，例如使用第 1 张卡就是 0
export CUDA_VISIBLE_DEVICES=0



# 定义刚才生成好的 Zarr 数据集路径
DATASET_PATH="/Users/macbookpro/Desktop/workspace/universal_manipulation_interface-main/data/umi_0425_dataset/combined_data.zarr"

echo "=========================================="
echo "🚀 准备开始 UMI Diffusion Policy 训练流水线"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "=========================================="

# 启动训练
# --config-name 指定了我们刚才修改了本地 resnet 权重的那个配置文件
python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset.dataset_path="${DATASET_PATH}" \
    training.device="cuda:0" \
    training.batch_size=128 \
    training.num_epochs=100
