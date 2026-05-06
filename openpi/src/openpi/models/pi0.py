import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        if self.config.use_tactile:
            logger.info("Using tactile encoder.")
            tac = nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=paligemma_config.width,
                    variant=config.tactile_variant,
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                )
            )
            tac.lazy_init(next(iter(config.fake_obs().tactile_images.values())), train=True, rngs=rngs)
            self.PaliGemma = nnx.Dict(llm=llm, img=img, tac=tac)
        else:
            logger.info("Not using tactile encoder.")
            self.PaliGemma = nnx.Dict(llm=llm, img=img)

        if self.config.use_tactile and self.config.fusion_method == "film":
            self.fusion_gamma = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.fusion_beta = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)

        if self.config.force_align:
            self.force_align_vision_proj = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.force_align_tactile_proj = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.force_align_force_proj = nnx.Linear(config.force_dim, paligemma_config.width, rngs=rngs)

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _force_modality_weights(self, obs: _model.Observation) -> tuple[at.Float[at.Array, "b 1 1"], at.Float[at.Array, "b 1 1"]]:
        batch_size = obs.state.shape[0]
        if self.config.force_guide and obs.force is not None:
            force = jnp.asarray(obs.force, dtype=jnp.float32)
            force = jnp.nan_to_num(force, nan=0.0, posinf=self.config.force_range[1], neginf=0.0)
            force_mag = jnp.linalg.norm(force, axis=-1, keepdims=True)
            force_min, force_max = self.config.force_range
            alpha = jnp.clip((force_mag - force_min) / (force_max - force_min), 0.0, 1.0)
            tactile_weight = alpha * (1.0 - self.config.min_vision_weight)
        else:
            tactile_weight = jnp.full((batch_size, 1), self.config.default_tactile_weight, dtype=jnp.float32)

        vision_weight = 1.0 - tactile_weight
        return vision_weight[:, None, :], tactile_weight[:, None, :]

    def _masked_mean(self, tokens, mask):
        mask = jnp.asarray(mask, dtype=jnp.float32)
        denom = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        return jnp.sum(jnp.asarray(tokens, dtype=jnp.float32) * mask[:, :, None], axis=1) / denom

    def _force_align_loss(
        self,
        vision_tokens,
        vision_mask,
        tactile_tokens: list[at.Array],
        tactile_masks: list[at.Array],
        obs: _model.Observation,
    ) -> at.Float[at.Array, "b"]:
        batch_size = obs.state.shape[0]
        zeros = jnp.zeros((batch_size,), dtype=jnp.float32)
        if not self.config.force_align or vision_tokens is None or vision_mask is None or not tactile_tokens:
            return zeros

        tactile_tokens = jnp.concatenate(tactile_tokens, axis=1)
        tactile_mask = jnp.concatenate(tactile_masks, axis=1)
        vision_context = self._masked_mean(vision_tokens, vision_mask)
        tactile_context = self._masked_mean(tactile_tokens, tactile_mask)

        if obs.force is None:
            force = jnp.zeros((batch_size, self.config.force_dim), dtype=jnp.float32)
        else:
            force = jnp.asarray(obs.force[..., : self.config.force_dim], dtype=jnp.float32)
        force_min, force_max = self.config.force_range
        force = jnp.nan_to_num(force, nan=force_min, posinf=force_max, neginf=force_min)
        force = jnp.clip((force - force_min) / (force_max - force_min), 0.0, 1.0)
        force = force * 2.0 - 1.0

        vision_embed = self.force_align_vision_proj(vision_context)
        tactile_force_embed = self.force_align_tactile_proj(tactile_context + self.force_align_force_proj(force))
        vision_embed = vision_embed / jnp.maximum(jnp.linalg.norm(vision_embed, axis=-1, keepdims=True), 1e-6)
        tactile_force_embed = tactile_force_embed / jnp.maximum(
            jnp.linalg.norm(tactile_force_embed, axis=-1, keepdims=True), 1e-6
        )

        logits = jnp.matmul(vision_embed, tactile_force_embed.T) / self.config.force_align_temperature
        vision_valid = jnp.sum(jnp.asarray(vision_mask, dtype=jnp.float32), axis=1) > 0.0
        tactile_valid = jnp.sum(jnp.asarray(tactile_mask, dtype=jnp.float32), axis=1) > 0.0
        valid = jnp.logical_and(vision_valid, tactile_valid)
        valid_pair = jnp.logical_and(valid[:, None], valid[None, :])
        logits = jnp.where(valid_pair, logits, -1e9)

        image_to_touch = -jnp.diag(jax.nn.log_softmax(logits, axis=-1))
        touch_to_image = -jnp.diag(jax.nn.log_softmax(logits, axis=0))
        per_sample_loss = 0.5 * (image_to_touch + touch_to_image)
        valid_float = valid.astype(jnp.float32)
        valid_count = jnp.sum(valid_float)
        return jnp.where(valid_count > 1.0, per_sample_loss * valid_float, zeros)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, compute_force_align_loss: bool = False
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        vision_tokens = []
        vision_masks = []
        force_align_vision_tokens = None
        force_align_vision_mask = None
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            vision_tokens.append(image_tokens)
            vision_masks.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            if compute_force_align_loss and name == self.config.force_align_camera_key:
                force_align_vision_tokens = image_tokens
                force_align_vision_mask = vision_masks[-1]

        tactile_tokens = []
        tactile_masks = []
        if self.config.use_tactile:
            for name in obs.tactile_images:
                tactile_image_tokens, _ = self.PaliGemma.tac(obs.tactile_images[name], train=False)
                tactile_tokens.append(tactile_image_tokens)
                tactile_masks.append(
                    einops.repeat(
                        obs.tactile_image_masks[name],
                        "b -> b s",
                        s=tactile_image_tokens.shape[1],
                    )
                )

        force_align_loss = self._force_align_loss(
            force_align_vision_tokens,
            force_align_vision_mask,
            tactile_tokens,
            tactile_masks,
            obs,
        )

        if self.config.use_tactile and tactile_tokens and self.config.fusion_method in ("linear", "film"):
            vision_weight, tactile_weight = self._force_modality_weights(obs)
            if self.config.fusion_method == "film" and vision_tokens:
                tactile_context = jnp.mean(jnp.concatenate(tactile_tokens, axis=1), axis=1)
                gamma = jnp.tanh(self.fusion_gamma(tactile_context))[:, None, :] * tactile_weight
                beta = self.fusion_beta(tactile_context)[:, None, :] * tactile_weight
                vision_tokens = [image_tokens * (1.0 + gamma) + beta for image_tokens in vision_tokens]

            vision_tokens = [image_tokens * vision_weight for image_tokens in vision_tokens]
            tactile_tokens = [tactile_image_tokens * tactile_weight for tactile_image_tokens in tactile_tokens]
            tactile_active = tactile_weight[:, 0, 0] > 0.0
            tactile_masks = [
                jnp.logical_and(mask, einops.repeat(tactile_active, "b -> b s", s=mask.shape[1]))
                for mask in tactile_masks
            ]

        for image_tokens, mask in zip(vision_tokens, vision_masks, strict=True):
            tokens.append(image_tokens)
            input_mask.append(mask)
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        for tactile_image_tokens, mask in zip(tactile_tokens, tactile_masks, strict=True):
            tokens.append(tactile_image_tokens)
            input_mask.append(mask)
            # tactile image tokens attend to each other
            ar_mask += [False] * tactile_image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, force_align_loss

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, force_align_loss = self.embed_prefix(
            observation, compute_force_align_loss=True
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if self.config.force_align and self.config.force_align_weight > 0.0:
            flow_loss = flow_loss + self.config.force_align_weight * force_align_loss[:, None]
        return flow_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, _ = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
