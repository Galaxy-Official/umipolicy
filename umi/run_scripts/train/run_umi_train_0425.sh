#!/usr/bin/env bash

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "${TRAIN_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

# 指定要使用的显卡，例如使用第 1 张卡就是 0
export CUDA_VISIBLE_DEVICES=0



# 定义刚才生成好的 Zarr 数据集路径
DATASET_PATH="data/yellow_umi_428.zarr"
# 定义训练输出路径 (存放 Checkpoints, Logs 等)
OUTPUT_PATH="outputs/yellow_to_pink_block_train_0428"
# 定义视觉主干网络 (Vision Backbone)
VISION_BACKBONE="vit_base_patch16_224"
VISION_CKPT="ckpt/vit_b_16.pth"

echo "=========================================="
echo "🚀 准备开始 UMI Diffusion Policy 训练流水线"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "👁️ 视觉主干网络: ${VISION_BACKBONE}"
echo "📂 视觉权重路径: ${VISION_CKPT:-从头训练}"
echo "📂 输出路径: ${OUTPUT_PATH}"
echo "=========================================="

# 打开 Hydra 详细报错开关，以便看到具体是因为哪一行报错
export HYDRA_FULL_ERROR=1

# 启动训练
# --config-name 指定 UMI 的 timm 视觉编码器训练配置
python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    hydra.run.dir="${OUTPUT_PATH}" \
    policy.obs_encoder.model_name="${VISION_BACKBONE}" \
    policy.obs_encoder.checkpoint_path="${VISION_CKPT}" \
    policy.obs_encoder.feature_aggregation=null \
    task.dataset.dataset_path="${DATASET_PATH}" \
    training.device="cuda:0" \
    dataloader.batch_size=64 \
    training.num_epochs=500 \
    logging.mode=offline
