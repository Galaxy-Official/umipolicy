#!/usr/bin/bash
set -eo pipefail

# ==========================================
# Unified Lingbot-VA Extraction & Training Script
# ==========================================

# 1. 配置默认参数 (可以通过环境变量覆盖)
DATASET_PATH=${DATASET_PATH:-"/Users/macbookpro/Desktop/handcap_simple_sorting_phone_409"}
CKPT_PATH=${CKPT_PATH:-"./ckpt/lingbot-va-base"}
CONFIG_NAME=${CONFIG_NAME:-"handcap_train"}
NGPU=${NGPU:-"8"}

echo "============================================================"
echo "🎯 Target Dataset: $DATASET_PATH"
echo "============================================================"

# ==========================================
# 第一阶段: 智能判断并提取潜变量特征 (Latent Extraction)
# ==========================================
LATENT_DIR="${DATASET_PATH}/latents"

# 判断依据：如果目录存在，并且里面有文件，则认为特征已经提取完毕
if [ -d "$LATENT_DIR" ] && [ "$(ls -A $LATENT_DIR 2>/dev/null)" ]; then
    echo "✅ [检测通过] 发现已有的 latents 特征目录: $LATENT_DIR"
    echo "⏭️  自动跳过特征提取阶段，直接进入训练..."
else
    echo "⚠️ [拦截提示] 没有找到特征，或者特征目录为空: $LATENT_DIR"
    echo "🚀 [启动特征提取...] 正在调用 SD VAE 进行并行特征抽取..."
    
    NUM_WORKERS_PER_GPU=4
    TOTAL_SHARDS=$(( NUM_WORKERS_PER_GPU * 2 ))

    echo "Launching $NUM_WORKERS_PER_GPU extractions on GPU 0..."
    for (( i=0; i<$NUM_WORKERS_PER_GPU; i++ )); do
        CUDA_VISIBLE_DEVICES=0 python wan_va/extract_latents.py \
            --dataset_path "$DATASET_PATH" \
            --ckpt_path "$CKPT_PATH" \
            --chunk_size 2 \
            --height 256 \
            --width 320 \
            --shard_id $i \
            --num_shards $TOTAL_SHARDS &
    done

    echo "Launching $NUM_WORKERS_PER_GPU extractions on GPU 1..."
    for (( i=$NUM_WORKERS_PER_GPU; i<$TOTAL_SHARDS; i++ )); do
        CUDA_VISIBLE_DEVICES=1 python wan_va/extract_latents.py \
            --dataset_path "$DATASET_PATH" \
            --ckpt_path "$CKPT_PATH" \
            --chunk_size 2 \
            --height 256 \
            --width 320 \
            --shard_id $i \
            --num_shards $TOTAL_SHARDS &
    done

    echo "⏳ 正在 2 张 GPU 上并发运行 $TOTAL_SHARDS 个线程... (进度取决于视频数量，请耐心等待)"
    wait
    echo "✅ 特征提取全部完成！"
fi

echo ""
echo "============================================================"
echo "🚀 第二阶段: 启动分布式训练 (FSDP / DDP)..."
echo "============================================================"

umask 007
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# 如果有额外的命令行参数，传给 overrides
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# 禁用 wandb 默认设置
export WANDB_MODE="disabled"
export TOKENIZERS_PARALLELISM=false

PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --local-ranks-filter=${LOG_RANK} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.train_handcap \
    --config-name ${CONFIG_NAME} \
    --dataset_path "$DATASET_PATH" \
    $overrides
