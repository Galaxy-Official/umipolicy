#!/usr/bin/bash

# Configure GPU devices if needed (e.g. 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0

python wan_va/extract_latents.py \
    --dataset_path ./Data/handcap_lingbot \
    --ckpt_path ./ckpt/lingbot-va-base \
    --chunk_size 2 \
    --height 256 \
    --width 320
