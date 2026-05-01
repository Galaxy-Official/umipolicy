#!/usr/bin/env bash

set -eo pipefail
set -x

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "${TRAIN_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29531"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"towel_hanging_430_train"}
DATASET_PATH=${DATASET_PATH:-"Data/430_towel_hanging_lerobot"}
CKPT_PATH=${CKPT_PATH:-"${LINGBOT_VA_BASE_CKPT:-./ckpt/lingbot-va-base}"}
SAVE_ROOT=${SAVE_ROOT:-"train_out/430_towel_hanging_lerobot"}

REBUILD_EMPTY_EMB=${REBUILD_EMPTY_EMB:-"1"}
AUTO_EXTRACT_LATENTS=${AUTO_EXTRACT_LATENTS:-"1"}
EXTRACT_WORKERS_PER_GPU=${EXTRACT_WORKERS_PER_GPU:-"4"}
EXTRACT_CHUNK_SIZE=${EXTRACT_CHUNK_SIZE:-"4"}
EXTRACT_VIDEO_KEYS=${EXTRACT_VIDEO_KEYS:-"observation.images.wrist"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-"2"}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-"4"}
LOAD_WORKER=${LOAD_WORKER:-"18"}
NUM_STEPS=${NUM_STEPS:-"50000"}
SAVE_INTERVAL=${SAVE_INTERVAL:-"1000"}
GC_INTERVAL=${GC_INTERVAL:-"50"}
LEARNING_RATE=${LEARNING_RATE:-"1e-5"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-"disabled"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"2"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"2"}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-"1"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"${PYTORCH_CUDA_ALLOC_CONF}"}

num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
dataset_path=${DATASET_PATH}
ckpt_path=${CKPT_PATH}
save_root=${SAVE_ROOT}

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
                --chunk_size "${EXTRACT_CHUNK_SIZE}" \
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

TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train_handcap \
    --config-name ${config_name} \
    --dataset_path "${dataset_path}" \
    --save-root "${save_root}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --load-worker "${LOAD_WORKER}" \
    --num-steps "${NUM_STEPS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --gc-interval "${GC_INTERVAL}" \
    --learning-rate "${LEARNING_RATE}" \
    $overrides
