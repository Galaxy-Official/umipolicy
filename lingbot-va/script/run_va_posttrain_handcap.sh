#!/usr/bin/bash

set -x

umask 007

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs}"
mkdir -p "${TRAIN_LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "${TRAIN_LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"handcap_train"} # handcap_train, robotwin_train

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# export WANDB_API_KEY="your key"
# export WANDB_BASE_URL="your url"
# export WANDB_TEAM_NAME="your team name"
# export WANDB_PROJECT="your project"
export WANDB_MODE="disabled"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train_handcap --config-name ${config_name} $overrides
