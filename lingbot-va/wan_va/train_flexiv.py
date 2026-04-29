# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from train_handcap import Trainer
from utils import init_logger, logger


def run(args):
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(config.dataset_path, 'empty_emb.pt')

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train WAN model for Flexiv robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='flexiv_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override the dataset_path from config",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    init_logger()
    main()
