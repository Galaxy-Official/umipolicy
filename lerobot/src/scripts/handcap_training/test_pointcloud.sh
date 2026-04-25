#!/usr/bin/env bash
set -e

# 修改为你要指定的显卡槽位 (例如使用 4 张卡则是 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1

# 屏蔽烦人的 torchvision pyav 弃用警告刷屏问题
export PYTHONWARNINGS="ignore"

# ==========================================
# 交互式训练控制逻辑：只有发现旧存档时才提示断点续训 (Resume)
# ==========================================
RESUME_PARAM="--resume=false"
OUTPUT_DIR="outputs/train/test_point_xyz_only_1"

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
if [ "$RESUME_PARAM" == "--resume=false" ] && [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️ 检测到空壳残骸文件夹，为了顺利重新开始，已自动清理过期文件结构：$OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi
echo "=========================================="

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=inductor \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lihongcs/block_to_pot_handcap \
  --dataset.root=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/lerobot/src/Data/handcap_simple_sorting_phone_409 \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --policy.pointbert_repo_root=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/third_party/Point-BERT    \
  --batch_size=16 \
  --num_workers=10 \
  --policy.point_pretrained_ckpt=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/third_party/point-bert_ckpt/Point-BERT.pth    \
  --policy.use_tactile=false \
  --policy.use_force=false \
  --policy.use_point=true \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights="ckpt/resnet18-f37072fd.pth" \
  --policy.use_group_norm=false \
  --policy.point_camera_fx=420  \
  --policy.point_camera_fy=420  \
  --policy.point_camera_cx=320  \
  --policy.point_camera_cy=240  \
  --policy.point_use_rgb=false \
  --optimizer.type=adamw \
  --output_dir=${OUTPUT_DIR} \
  --job_name=dp_test_point_xyz_only \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.mode=offline \
  --use_handcap=true \
  --steps=200000 \
  --save_freq=2000 \
  --policy.push_to_hub=false \
  --eval_freq=50 \
  --log_freq=10 \
  --resume=true  \
  --config_path=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/lerobot/src/outputs/train/test_point_xyz_only/checkpoints/023000/pretrained_model \
  --policy.repo_id=test_point_xyz_only/dp \
  ${RESUME_PARAM}