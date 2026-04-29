#!/usr/bin/env bash
set -euo pipefail

# Flexiv/Cosmos Policy training launcher.
# This uses UMIDataset with current-EE-relative pose10d actions. It does not use
# the ALOHA robot train/eval scripts.

NUM_GPUS="${NUM_GPUS:-4}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-data/flexiv_cosmos}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cosmos_predict2_flexiv}"
TASK_PROMPT="${TASK_PROMPT:-manipulate object}"
MASTER_PORT="${MASTER_PORT:-12341}"
T5_CKPT_DIR="${T5_CKPT_DIR:-ckpt/google-t5/t5-11b}"

export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export COSMOS_DISABLE_MANUAL_GC=1
export COSMOS_DISABLE_TOKENIZER_COMPILE=1
export TORCHDYNAMO_DISABLE=1
export PYTHONFAULTHANDLER=1

echo "================================================================================"
echo "Starting Flexiv Cosmos Policy training"
echo "Dataset: ${BASE_DATASETS_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Task prompt: ${TASK_PROMPT}"
echo "================================================================================"

if [[ ! -f "${BASE_DATASETS_DIR}/t5_embeddings.pkl" ]]; then
    echo "Missing ${BASE_DATASETS_DIR}/t5_embeddings.pkl; precomputing T5 embedding."
    PYTHONPATH=. python scripts/flexiv_training/precompute_t5.py \
        --prompt "${TASK_PROMPT}" \
        --data_dir "${BASE_DATASETS_DIR}" \
        --model_name "${T5_CKPT_DIR}"
fi

PYTHONPATH=. python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py -- \
    experiment="${EXPERIMENT_NAME}" \
    dataloader_train.dataset.data_dir="${BASE_DATASETS_DIR}" \
    dataloader_train.dataset.default_command="${TASK_PROMPT}" \
    dataloader_train.dataset.t5_text_embeddings_path="${BASE_DATASETS_DIR}/t5_embeddings.pkl" \
    job.wandb_mode="${WANDB_MODE}" \
    trainer.callbacks.manual_gc.enabled=false \
    trainer.callbacks.manual_gc.every_n=0 \
    trainer.callbacks.compile_tokenizer.enabled=false \
    trainer.callbacks.compile_tokenizer.compile_after_iterations=0 \
    model.config.use_torch_compile=false
