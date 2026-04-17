# 修改为你要指定的显卡槽位 (目前针对你的配置：2 张 H200 则设为 0,1)
export CUDA_VISIBLE_DEVICES=0,1

# 屏蔽烦人的 torchvision pyav 弃用警告刷屏问题
export PYTHONWARNINGS="ignore"


# # 强制 Hugging Face 进入离线模式，不联网检查数据集
# export HF_HUB_OFFLINE=1
# # 强制读取本地文件，不尝试在线下载
# export TRANSFORMERS_OFFLINE=1
# source /root/miniconda3/bin/activate
# conda activate umipolicy

# 【核心硬件优化提升说明】：
# 1. --mixed_precision=bf16 针对 Hopper 架构开启更快的 bfloat16 半精度加速计算。
# 2. --batch_size=128：2 张显卡组成双卡并行则总 Global Batch Size 为 256，保障扩散模型最好的收敛速度与稳定度。
# 3. --training.num_workers=20：压榨你的 80 核 CPU 和 200内存，保证 GPU 高速读图不掉速。
# ==========================================
# 交互式训练控制逻辑：只有发现旧存档时才提示断点续训 (Resume)
# ==========================================
RESUME_PARAM="--resume=false"
OUTPUT_DIR="outputs/train/tool_to_cups_0415_1"

if [ -d "$OUTPUT_DIR/checkpoints" ]; then
    read -p "🤔 发现曾经的训练存档！是否从上一个断点恢复训练？[输入 r 继续训练(Resume) / 直接回车重新开始并且覆盖旧档案]: " choice < /dev/tty
    if [[ "$choice" == "r" || "$choice" == "R" ]]; then
        echo "▶️ 准备从断点继续训练 (Resume)..."
        RESUME_PARAM="--resume=true"
    else
        echo "🔄 准备重新开始全新训练（原有存档可能被覆盖）..."
    fi
else
    echo "🆕 未检测到历史存档（或者历史存档未保存成功），自动开始全新训练..."
    RESUME_PARAM="--resume=false"
fi

# ==========================================
# LeRobot 防止覆盖保护的自动擦除机制
# ==========================================
# 如果最终决定重新开始（resume=false），但该文件夹已经存在（比如上次在第一步崩溃了残留了 config），
# LeRobot 底层会强制报错 FileExistsError。为了让你能跑起来，我们在这里把旧的残骸删掉：
if [ "$RESUME_PARAM" == "--resume=false" ] && [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️ 检测到空壳残骸文件夹，为了顺利重新开始，已自动清理过期文件结构：$OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi
echo "=========================================="

accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=inductor -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lihongcs/tool_to_cups_0415 \
  --dataset.root=Data/tool_to_cups_0415 \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --batch_size=256 \
  --num_workers=10 \
  --policy.use_tactile=false \
  --policy.use_force=false \
  --optimizer.type=adamw \
  --output_dir=${OUTPUT_DIR}  \
  --job_name=dp_tool_to_cups_0415_handcap \
  --policy.device=cuda \
  --wandb.enable=true \
  --use_handcap=true \
  --steps=100000 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --log_freq=100 \
  --wandb.mode="offline" \
  --policy.repo_id=tool_to_cups_0415/dp \
  ${RESUME_PARAM}

# 注意: 上面的 --batch_size=128 代表 "单卡 Batch Size"。
# LeRobot 3.0 中的总有效 Batch Size = batch_size * num_processes。
# 当使用 2 卡且每卡 `batch_size=128` 时，系统总吞吐量将会是 256。