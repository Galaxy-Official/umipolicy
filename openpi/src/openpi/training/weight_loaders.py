import dataclasses
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        
        # EXPERIMENTAL: Auto-copy vision weights to tactile encoder if tac does not exist in the checkpoint
        has_tac = any(k.startswith("PaliGemma/tac/") for k in flat_loaded)
        if not has_tac:
            new_flat_loaded = dict(flat_loaded)
            for k, v in flat_loaded.items():
                if k.startswith("PaliGemma/img/"):
                    tac_key = k.replace("PaliGemma/img/", "PaliGemma/tac/", 1)
                    new_flat_loaded[tac_key] = v
            loaded_params = flax.traverse_util.unflatten_dict(new_flat_loaded, sep="/")
            logger.info("Automatically copied PaliGemma/img/ weights to PaliGemma/tac/ for the tactile encoder.")
            
        # Add newly introduced adaptation weights that are not present in base checkpoints.
        return _merge_params(loaded_params, params, missing_regex=".*(lora|PaliGemma/tac|fusion_|tacfilm_).*")


@dataclasses.dataclass(frozen=True)
class T3TactileConcatWeightLoader(WeightLoader):
    """Loads a base OpenPI checkpoint and local T3 encoder/trunk PyTorch weights."""

    base_params_path: str
    t3_encoder_path: str
    t3_trunk_path: str

    def load(self, params: at.Params) -> at.Params:
        params = CheckpointWeightLoader(self.base_params_path).load(params)
        if not self.t3_encoder_path or not self.t3_trunk_path:
            logger.warning("T3 encoder/trunk paths are empty; keeping randomly initialized T3 tactile weights.")
            return params

        encoder_path = pathlib.Path(download.maybe_download(self.t3_encoder_path))
        trunk_path = pathlib.Path(download.maybe_download(self.t3_trunk_path))
        if not encoder_path.exists():
            raise FileNotFoundError(f"T3 encoder checkpoint not found: {encoder_path}")
        if not trunk_path.exists():
            raise FileNotFoundError(f"T3 trunk checkpoint not found: {trunk_path}")

        return _merge_t3_tactile_params(params, encoder_path, trunk_path)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _load_torch_state_dict(path: pathlib.Path) -> dict[str, np.ndarray]:
    import torch

    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected a PyTorch state dict at {path}, got {type(state_dict)}.")
    return {
        key.removeprefix("module."): value.detach().cpu().numpy()
        for key, value in state_dict.items()
        if hasattr(value, "detach")
    }


def _merge_t3_tactile_params(params: at.Params, encoder_path: pathlib.Path, trunk_path: pathlib.Path) -> at.Params:
    encoder_state = _load_torch_state_dict(encoder_path)
    trunk_state = _load_torch_state_dict(trunk_path)

    flat = flax.traverse_util.flatten_dict(params, sep="/")
    loaded = 0

    def set_suffix(suffix: str, value: np.ndarray):
        nonlocal loaded
        matches = [key for key in flat if key.endswith(suffix)]
        if not matches:
            logger.warning("T3 loader could not find target param suffix: %s", suffix)
            return
        if len(matches) > 1:
            logger.warning("T3 loader found multiple params for suffix %s: %s", suffix, matches)
            return
        key = matches[0]
        ref = flat[key]
        if ref.shape != value.shape:
            logger.warning("Skipping T3 param %s: checkpoint shape %s != model shape %s", key, value.shape, ref.shape)
            return
        flat[key] = value.astype(ref.dtype) if value.dtype != ref.dtype else value
        loaded += 1

    def set_block(prefix: str, state: dict[str, np.ndarray], block_index: int):
        source_prefix = f"blocks.{block_index}"
        target_prefix = f"{prefix}/blocks_{block_index}"
        set_suffix(f"{target_prefix}/norm1/scale", state[f"{source_prefix}.norm1.weight"])
        set_suffix(f"{target_prefix}/norm1/bias", state[f"{source_prefix}.norm1.bias"])
        set_suffix(f"{target_prefix}/attn/qkv/kernel", state[f"{source_prefix}.attn.qkv.weight"].T)
        set_suffix(f"{target_prefix}/attn/qkv/bias", state[f"{source_prefix}.attn.qkv.bias"])
        set_suffix(f"{target_prefix}/attn/proj/kernel", state[f"{source_prefix}.attn.proj.weight"].T)
        set_suffix(f"{target_prefix}/attn/proj/bias", state[f"{source_prefix}.attn.proj.bias"])
        set_suffix(f"{target_prefix}/norm2/scale", state[f"{source_prefix}.norm2.weight"])
        set_suffix(f"{target_prefix}/norm2/bias", state[f"{source_prefix}.norm2.bias"])
        set_suffix(f"{target_prefix}/mlp/fc1/kernel", state[f"{source_prefix}.mlp.fc1.weight"].T)
        set_suffix(f"{target_prefix}/mlp/fc1/bias", state[f"{source_prefix}.mlp.fc1.bias"])
        set_suffix(f"{target_prefix}/mlp/fc2/kernel", state[f"{source_prefix}.mlp.fc2.weight"].T)
        set_suffix(f"{target_prefix}/mlp/fc2/bias", state[f"{source_prefix}.mlp.fc2.bias"])

    set_suffix("PaliGemma/tac/encoder/cls_token", encoder_state["cls_token"])
    set_suffix("PaliGemma/tac/encoder/pos_embed", encoder_state["pos_embed"])
    set_suffix("PaliGemma/tac/encoder/patch_embed_proj/kernel", np.transpose(encoder_state["patch_embed.proj.weight"], (2, 3, 1, 0)))
    set_suffix("PaliGemma/tac/encoder/patch_embed_proj/bias", encoder_state["patch_embed.proj.bias"])

    encoder_depth = sum(1 for key in encoder_state if re.fullmatch(r"blocks\.\d+\.norm1\.weight", key))
    trunk_depth = sum(1 for key in trunk_state if re.fullmatch(r"blocks\.\d+\.norm1\.weight", key))
    for i in range(encoder_depth):
        set_block("PaliGemma/tac/encoder", encoder_state, i)
    for i in range(trunk_depth):
        set_block("PaliGemma/tac/trunk", trunk_state, i)

    set_suffix("PaliGemma/tac/trunk/norm/scale", trunk_state["norm.weight"])
    set_suffix("PaliGemma/tac/trunk/norm/bias", trunk_state["norm.bias"])

    logger.info("Loaded %d T3 tactile encoder/trunk tensors from %s and %s", loaded, encoder_path, trunk_path)
    return flax.traverse_util.unflatten_dict(flat, sep="/")
