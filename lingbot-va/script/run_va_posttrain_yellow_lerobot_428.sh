#!/usr/bin/bash

set -eo pipefail
set -x

umask 007

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"yellow_lerobot_428_train"}
DATASET_PATH=${DATASET_PATH:-"Data/yellow_lerobot_428"}
CKPT_PATH=${CKPT_PATH:-"./ckpt/lingbot-va-base"}
REBUILD_EMPTY_EMB=${REBUILD_EMPTY_EMB:-"1"}
AUTO_EXTRACT_LATENTS=${AUTO_EXTRACT_LATENTS:-"1"}
EXTRACT_WORKERS_PER_GPU=${EXTRACT_WORKERS_PER_GPU:-"1"}
EXTRACT_VIDEO_KEYS=${EXTRACT_VIDEO_KEYS:-"observation.images.wrist"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# export WANDB_API_KEY="your key"
# export WANDB_BASE_URL="your url"
# export WANDB_TEAM_NAME="your team name"
# export WANDB_PROJECT="your project"
export WANDB_MODE=${WANDB_MODE:-"disabled"}
export CUDA_VISIBLE_DEVICES

num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
dataset_path=${DATASET_PATH}
ckpt_path=${CKPT_PATH}

export TOKENIZERS_PARALLELISM=false
if [ "${REBUILD_EMPTY_EMB}" = "1" ] || [ ! -f "${dataset_path}/empty_emb.pt" ]; then
    python wan_va/create_empty_emb.py \
        --dataset_path "${dataset_path}" \
        --ckpt_path "${ckpt_path}" \
        --force
fi

latent_probe=$(find "${dataset_path}/latents" -path "*/observation.images.wrist/*.pth" -print -quit 2>/dev/null || true)
if [ "${AUTO_EXTRACT_LATENTS}" = "1" ] && [ -z "${latent_probe}" ]; then
    IFS=',' read -r -a extract_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    total_shards=$(( ${#extract_gpu_ids[@]} * EXTRACT_WORKERS_PER_GPU ))
    shard_id=0
    pids=()

    echo "No wrist latents found under ${dataset_path}/latents. Extracting latents with ${total_shards} shards..."
    for gpu_id in "${extract_gpu_ids[@]}"; do
        for (( worker_id=0; worker_id<EXTRACT_WORKERS_PER_GPU; worker_id++ )); do
            CUDA_VISIBLE_DEVICES="${gpu_id}" python wan_va/extract_latents.py \
                --dataset_path "${dataset_path}" \
                --ckpt_path "${ckpt_path}" \
                --chunk_size 2 \
                --height 256 \
                --width 320 \
                --video_keys "${EXTRACT_VIDEO_KEYS}" \
                --shard_id "${shard_id}" \
                --num_shards "${total_shards}" &
            pids+=("$!")
            shard_id=$(( shard_id + 1 ))
        done
    done

    extract_failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || extract_failed=1
    done
    if [ "${extract_failed}" -ne 0 ]; then
        echo "Latent extraction failed. Stop before training." >&2
        exit 1
    fi
fi

latent_probe=$(find "${dataset_path}/latents" -path "*/observation.images.wrist/*.pth" -print -quit 2>/dev/null || true)
if [ -z "${latent_probe}" ]; then
    echo "No wrist latent files found. Run with AUTO_EXTRACT_LATENTS=1 or check dataset video key/path." >&2
    exit 1
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train_handcap \
    --config-name ${config_name} \
    --dataset_path "${dataset_path}" \
    $overrides
