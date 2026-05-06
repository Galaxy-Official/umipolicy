import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    
    use_tactile: bool = False
    tactile_pretrained_ckpt: str = ""
    tactile_variant: str = "B/16"
    camera_keys: tuple = ("head_0_rgb", "wrist_0_rgb", "side_0_rgb")
    fusion_method: str = "concat"
    force_predict: bool = False
    force_guide: bool = False
    force_align: bool = False
    action_base_dim: int = 10
    force_dim: int = 2
    force_range: tuple[float, float] = (0.0, 10.0)
    min_vision_weight: float = 0.2
    default_tactile_weight: float = 0.5
    force_align_weight: float = 0.05
    force_align_temperature: float = 0.07
    force_align_camera_key: str = "wrist_0_rgb"
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.fusion_method not in ("concat", "linear", "film"):
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")
        if self.force_dim <= 0:
            raise ValueError("force_dim must be positive")
        if self.action_base_dim <= 0:
            raise ValueError("action_base_dim must be positive")
        if not 0.0 <= self.min_vision_weight <= 1.0:
            raise ValueError("min_vision_weight must be in [0, 1]")
        if not 0.0 <= self.default_tactile_weight <= 1.0:
            raise ValueError("default_tactile_weight must be in [0, 1]")
        if self.force_range[1] <= self.force_range[0]:
            raise ValueError("force_range max must be greater than min")
        if self.force_align:
            if not self.use_tactile:
                raise ValueError("force_align requires use_tactile=True")
            if self.force_align_weight < 0.0:
                raise ValueError("force_align_weight must be non-negative")
            if self.force_align_temperature <= 0.0:
                raise ValueError("force_align_temperature must be positive")
            if self.force_align_camera_key not in self.camera_keys:
                raise ValueError("force_align_camera_key must be included in camera_keys")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={k: image_spec for k in self.camera_keys},
                tactile_images={
                    "left_tactile_0_rgb": image_spec,
                    "right_tactile_0_rgb": image_spec,
                },
                image_masks={k: image_mask_spec for k in self.camera_keys},
                tactile_image_masks={
                    "left_tactile_0_rgb": image_mask_spec,
                    "right_tactile_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                force=jax.ShapeDtypeStruct([batch_size, self.force_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
