#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lihongcs/block_to_pot_handcap \
  --dataset.root=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/lerobot/src/Data/handcap_simple_sorting_phone_409 \
  --policy.type=diffusion \
  --policy.pointbert_repo_root=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/third_party/Point-BERT    \
  --batch_size=16 \
  --policy.point_pretrained_ckpt=/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/third_party/point-bert_ckpt/Point-BERT.pth    \
  --policy.use_tactile=false \
  --policy.use_point=true \
  --policy.point_camera_fx=420  \
  --policy.point_camera_fy=420  \
  --policy.point_camera_cx=320  \
  --policy.point_camera_cy=240  \
  --policy.point_use_rgb=false \
  --optimizer.type=adamw \
  --output_dir=outputs/train/test_point_xyz_only \
  --job_name=dp_test_point_xyz_only \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.mode=offline \
  --use_handcap=true \
  --steps=500 \
  --policy.repo_id=test_point_xyz_only/dp