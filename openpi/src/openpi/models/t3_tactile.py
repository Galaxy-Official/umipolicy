"""T3 tactile encoder for TactileConcat fusion."""

from __future__ import annotations

import dataclasses
from typing import Literal

import flax.linen as nn
import jax.numpy as jnp


T3Variant = Literal["tiny", "small", "medium", "large"]


@dataclasses.dataclass(frozen=True)
class T3Spec:
    embed_dim: int
    num_heads: int
    encoder_depth: int = 3
    trunk_depth: int = 9
    patch_size: int = 16
    mlp_ratio: float = 4.0


def get_t3_spec(variant: T3Variant) -> T3Spec:
    match variant:
        case "tiny":
            return T3Spec(embed_dim=192, num_heads=3)
        case "small":
            return T3Spec(embed_dim=384, num_heads=6)
        case "medium":
            return T3Spec(embed_dim=768, num_heads=12)
        case "large":
            return T3Spec(embed_dim=1024, num_heads=16)
        case _:
            raise ValueError(f"Unsupported T3 variant: {variant}")


class T3Attention(nn.Module):
    embed_dim: int
    num_heads: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        batch, tokens, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {channels}.")

        qkv = nn.Dense(3 * self.embed_dim, dtype=self.dtype_mm, name="qkv")(x)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.embed_dim // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (self.embed_dim // self.num_heads) ** -0.5
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = nn.softmax(attn, axis=-1)
        y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(batch, tokens, channels)
        return nn.Dense(self.embed_dim, dtype=self.dtype_mm, name="proj")(y)


class T3Mlp(nn.Module):
    embed_dim: int
    mlp_ratio: float = 4.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        hidden_dim = int(self.embed_dim * self.mlp_ratio)
        x = nn.Dense(hidden_dim, dtype=self.dtype_mm, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim, dtype=self.dtype_mm, name="fc2")(x)
        return x


class T3Block(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype_mm, name="norm1")(x)
        x = x + T3Attention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dtype_mm=self.dtype_mm,
            name="attn",
        )(y)
        y = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype_mm, name="norm2")(x)
        x = x + T3Mlp(
            embed_dim=self.embed_dim,
            mlp_ratio=self.mlp_ratio,
            dtype_mm=self.dtype_mm,
            name="mlp",
        )(y)
        return x


class T3ViTEncoder(nn.Module):
    spec: T3Spec
    image_size: int = 224
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image):
        image = jnp.asarray(image, jnp.float32)
        x = nn.Conv(
            self.spec.embed_dim,
            kernel_size=(self.spec.patch_size, self.spec.patch_size),
            strides=(self.spec.patch_size, self.spec.patch_size),
            padding="VALID",
            dtype=jnp.float32,
            name="patch_embed_proj",
        )(image)
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height * width, channels).astype(self.dtype_mm)

        cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.spec.embed_dim), self.dtype_mm)
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, (self.image_size // self.spec.patch_size) ** 2 + 1, self.spec.embed_dim),
            self.dtype_mm,
        )
        x = jnp.concatenate([jnp.tile(cls_token, (batch, 1, 1)), x], axis=1)
        x = x + pos_embed

        for i in range(self.spec.encoder_depth):
            x = T3Block(
                embed_dim=self.spec.embed_dim,
                num_heads=self.spec.num_heads,
                mlp_ratio=self.spec.mlp_ratio,
                dtype_mm=self.dtype_mm,
                name=f"blocks_{i}",
            )(x)
        return x


class T3TransformerTrunk(nn.Module):
    spec: T3Spec
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        for i in range(self.spec.trunk_depth):
            x = T3Block(
                embed_dim=self.spec.embed_dim,
                num_heads=self.spec.num_heads,
                mlp_ratio=self.spec.mlp_ratio,
                dtype_mm=self.dtype_mm,
                name=f"blocks_{i}",
            )(x)
        return nn.LayerNorm(epsilon=1e-6, dtype=self.dtype_mm, name="norm")(x)


class Module(nn.Module):  # pylint: disable=invalid-name
    """T3 encoder + trunk + two-layer projection for TactileConcat."""

    num_classes: int
    variant: T3Variant = "tiny"
    image_size: int = 224
    projector_hidden_dim: int | None = None
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):  # noqa: ARG002
        spec = get_t3_spec(self.variant)
        x = T3ViTEncoder(spec=spec, image_size=self.image_size, dtype_mm=self.dtype_mm, name="encoder")(image)
        x = T3TransformerTrunk(spec=spec, dtype_mm=self.dtype_mm, name="trunk")(x)

        # T3 keeps a CLS token. For VLM prefix tokens we use spatial tactile patch tokens.
        x = x[:, 1:, :]

        hidden_dim = self.projector_hidden_dim or max(spec.embed_dim, self.num_classes)
        x = nn.Dense(hidden_dim, dtype=self.dtype_mm, name="projector_in")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="projector_out")(x)
        return x, {"tokens": x}
