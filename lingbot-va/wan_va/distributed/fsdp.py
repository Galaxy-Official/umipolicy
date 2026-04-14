# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    # Shard all explicit BF16 submodules to easily separate them from the root's managed FP32 parameters
    if hasattr(model, 'patch_embedding_mlp'): fully_shard(model.patch_embedding_mlp, **fsdp_config)
    if hasattr(model, 'action_embedder'): fully_shard(model.action_embedder, **fsdp_config)
    if hasattr(model, 'condition_embedder'):
        fully_shard(model.condition_embedder.text_embedder, **fsdp_config)
        fully_shard(model.condition_embedder, **fsdp_config)
    if hasattr(model, 'condition_embedder_action'):
        fully_shard(model.condition_embedder_action.text_embedder, **fsdp_config)
        fully_shard(model.condition_embedder_action, **fsdp_config)
    if hasattr(model, 'proj_out'): fully_shard(model.proj_out, **fsdp_config)
    if hasattr(model, 'action_proj_out'): fully_shard(model.action_proj_out, **fsdp_config)
    
    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
