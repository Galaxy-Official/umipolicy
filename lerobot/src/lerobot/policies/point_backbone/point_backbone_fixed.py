
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PointIdentityProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate=0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).view(b, n, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(b, 1, m)
    return dist


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    gathered = torch.gather(
        points,
        dim=1,
        index=idx[..., None].expand(-1, -1, points.size(-1)),
    )
    return gathered.reshape(*raw_size, points.size(-1))


def _furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def _fps_points(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    idx = _furthest_point_sample(xyz, npoint)
    return _index_points(xyz, idx)


def _knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    sqrdists = _square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k=nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = xyz.shape
        center = _fps_points(xyz, self.num_group)
        idx = _knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointTransformerBackbone(nn.Module):
    """
    Self-contained Point-BERT-style encoder that avoids the original repo's
    runtime dependency chain on knn_cuda / pointnet2_ops / builder imports.

    It mirrors the public PointTransformer backbone structure closely enough
    to load standard Point-BERT checkpoints with strict=False.
    """

    def __init__(
        self,
        trans_dim: int = 384,
        depth: int = 12,
        drop_path_rate: float = 0.1,
        num_heads: int = 6,
        group_size: int = 32,
        num_group: int = 64,
        encoder_dims: int = 256,
    ) -> None:
        super().__init__()
        self.trans_dim = int(trans_dim)
        self.depth = int(depth)
        self.drop_path_rate = float(drop_path_rate)
        self.num_heads = int(num_heads)
        self.group_size = int(group_size)
        self.num_group = int(num_group)
        self.encoder_dims = int(encoder_dims)

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    @property
    def feature_dim(self) -> int:
        return self.trans_dim * 2

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(dim=1)[0]], dim=-1)
        return concat_f


class PointBERTEncoderPooled(nn.Module):
    """
    Point-BERT encoder wrapper for LeRobot / diffusion-policy style usage.

    This version keeps the external interface stable but replaces the brittle
    dynamic import of the original Point-BERT repo with a self-contained
    Point-BERT-style backbone. That avoids the repo's required runtime
    extensions (pointnet2_ops, knn_cuda, builder imports) during training.

    Expected inputs
    ---------------
    xyz:
        [B, N, 3] or [B, T, N, 3]
    rgb:
        optional, same leading dims as xyz, last dim = 3
    mask:
        optional, same leading dims as xyz without the xyz channel
    """

    def __init__(
        self,
        pointbert_repo_root: str,
        pointbert_ckpt: Optional[str] = None,
        output_dim: int = 64,
        num_points: int = 1024,
        use_rgb: bool = True,
        use_mask: bool = True,
        freeze_backbone: bool = True,
        normalize_xyz: bool = True,
        center_xyz: bool = True,
        l2_normalize_feature: bool = False,
        strict_ckpt: bool = False,
    ) -> None:
        super().__init__()

        self.pointbert_repo_root = str(pointbert_repo_root)
        self.pointbert_ckpt = pointbert_ckpt
        self.output_dim = int(output_dim)
        self.feature_dim = self.output_dim
        self.num_points = int(num_points)
        self.use_rgb = bool(use_rgb)
        self.use_mask = bool(use_mask)
        self.freeze_backbone = bool(freeze_backbone)
        self.normalize_xyz = bool(normalize_xyz)
        self.center_xyz = bool(center_xyz)
        self.l2_normalize_feature = bool(l2_normalize_feature)
        self.strict_ckpt = bool(strict_ckpt)

        self.backbone, self.backbone_feature_dim = self._build_pointbert_backbone()

        if self.use_rgb:
            self.rgb_mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.GELU(),
                nn.Linear(64, self.backbone_feature_dim),
            )
            self.rgb_fusion_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.rgb_mlp = None
            self.register_parameter("rgb_fusion_scale", None)

        if self.backbone_feature_dim == self.output_dim:
            self.mapping_dim = _PointIdentityProjector()
        else:
            self.mapping_dim = nn.Linear(self.backbone_feature_dim, self.output_dim)

        if self.freeze_backbone:
            self._freeze_module(self.backbone)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _freeze_module(self, module: nn.Module) -> None:
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _build_pointbert_backbone(self) -> Tuple[nn.Module, int]:
        backbone = PointTransformerBackbone(
            trans_dim=384,
            depth=12,
            drop_path_rate=0.1,
            num_heads=6,
            group_size=32,
            num_group=64,
            encoder_dims=256,
        )
        self._load_checkpoint_if_needed(backbone)
        return backbone, backbone.feature_dim

    def _load_checkpoint_if_needed(self, backbone: nn.Module) -> None:
        if not self.pointbert_ckpt:
            return

        ckpt_path = os.path.abspath(self.pointbert_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Point-BERT checkpoint not found: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")  
              
        if isinstance(ckpt, dict):
            if "base_model" in ckpt and isinstance(ckpt["base_model"], dict):
                state_dict = ckpt["base_model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state_dict = ckpt["model"]
            elif "module" in ckpt and isinstance(ckpt["module"], dict):
                state_dict = ckpt["module"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        cleaned = {}
        for k, v in state_dict.items():
            nk = str(k)
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("base_model."):
                nk = nk[len("base_model."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            if nk.startswith("transformer_q.") and not nk.startswith("transformer_q.cls_head"):
                nk = nk[len("transformer_q."):]
            if nk.startswith("cls_head_finetune"):
                continue
            cleaned[nk] = v

        incompatible = backbone.load_state_dict(cleaned, strict=self.strict_ckpt)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        print("\nPoint-BERT checkpoint loaded into self-contained backbone.")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")

    def _flatten_bt(
        self,
        xyz: torch.Tensor,
        rgb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], bool, Tuple[int, ...]]:
        if xyz.ndim == 3:
            return xyz, rgb, mask, False, xyz.shape[:2]
        if xyz.ndim == 4:
            b, t, n, c = xyz.shape
            xyz_flat = xyz.reshape(b * t, n, c)
            rgb_flat = rgb.reshape(b * t, n, rgb.shape[-1]) if rgb is not None else None
            mask_flat = mask.reshape(b * t, n) if mask is not None else None
            return xyz_flat, rgb_flat, mask_flat, True, (b, t, n)
        raise ValueError(f"xyz must be [B,N,3] or [B,T,N,3], got {tuple(xyz.shape)}")

    def _restore_bt(self, feat: torch.Tensor, had_time_dim: bool, shape_info: Tuple[int, ...]) -> torch.Tensor:
        if not had_time_dim:
            return feat
        b, t, _ = shape_info
        return feat.reshape(b, t, -1)

    def _validate_inputs(
        self,
        xyz: torch.Tensor,
        rgb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> None:
        if xyz.shape[-1] != 3:
            raise ValueError(f"xyz last dim must be 3, got {tuple(xyz.shape)}")
        if rgb is not None:
            if rgb.shape[:-1] != xyz.shape[:-1] or rgb.shape[-1] != 3:
                raise ValueError(
                    f"rgb must match xyz leading dims and end with 3, got xyz={tuple(xyz.shape)}, rgb={tuple(rgb.shape)}"
                )
        if mask is not None:
            if mask.shape != xyz.shape[:-1]:
                raise ValueError(
                    f"mask must have shape xyz.shape[:-1], got xyz={tuple(xyz.shape)}, mask={tuple(mask.shape)}"
                )

    def _apply_mask_and_sample(
        self,
        xyz: torch.Tensor,
        rgb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, n, _ = xyz.shape
        out_xyz = []
        out_rgb = [] if rgb is not None else None

        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=xyz.device)
        else:
            mask = mask.to(dtype=torch.bool, device=xyz.device)

        for i in range(b):
            valid_idx = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                valid_idx = torch.arange(n, device=xyz.device)

            if valid_idx.numel() >= self.num_points:
                sel = valid_idx[: self.num_points]
            else:
                repeat_count = self.num_points - valid_idx.numel()
                pad_idx = valid_idx[torch.randint(0, valid_idx.numel(), (repeat_count,), device=xyz.device)]
                sel = torch.cat([valid_idx, pad_idx], dim=0)

            out_xyz.append(xyz[i, sel])
            if rgb is not None:
                out_rgb.append(rgb[i, sel])

        xyz_out = torch.stack(out_xyz, dim=0)
        rgb_out = torch.stack(out_rgb, dim=0) if out_rgb is not None else None
        return xyz_out, rgb_out

    def _normalize_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.center_xyz:
            xyz = xyz - xyz.mean(dim=1, keepdim=True)
        if self.normalize_xyz:
            scale = xyz.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
            xyz = xyz / scale.unsqueeze(-1)
        return xyz

    def forward(
        self,
        xyz: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(xyz, rgb, mask)

        xyz = xyz.to(self.device)
        rgb = rgb.to(self.device) if rgb is not None else None
        mask = mask.to(self.device) if mask is not None else None

        xyz_flat, rgb_flat, mask_flat, had_time_dim, shape_info = self._flatten_bt(xyz, rgb, mask)
        xyz_flat, rgb_flat = self._apply_mask_and_sample(xyz_flat, rgb_flat, mask_flat if self.use_mask else None)
        xyz_flat = self._normalize_xyz(xyz_flat)

        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(xyz_flat)
        else:
            feat = self.backbone(xyz_flat)

        if rgb_flat is not None and self.use_rgb:
            rgb_flat = rgb_flat.float()
            if rgb_flat.max() > 1.0:
                rgb_flat = rgb_flat / 255.0
            rgb_feat = self.rgb_mlp(rgb_flat).mean(dim=1)
            feat = feat + self.rgb_fusion_scale * rgb_feat

        feat = self.mapping_dim(feat)
        if self.l2_normalize_feature:
            feat = F.normalize(feat, dim=-1)

        return self._restore_bt(feat, had_time_dim, shape_info)


class PointBERTEncoderTokens(nn.Module):
    def __init__(
        self,
        pointbert_repo_root: str,
        pointbert_ckpt: Optional[str] = None,
        output_dim: int = 256,
        num_points: int = 1024,
        freeze_backbone: bool = True,
        strict_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.pooled = PointBERTEncoderPooled(
            pointbert_repo_root=pointbert_repo_root,
            pointbert_ckpt=pointbert_ckpt,
            output_dim=output_dim,
            num_points=num_points,
            use_rgb=False,
            use_mask=True,
            freeze_backbone=freeze_backbone,
            normalize_xyz=True,
            center_xyz=True,
            l2_normalize_feature=False,
            strict_ckpt=strict_ckpt,
        )

    @property
    def device(self) -> torch.device:
        return self.pooled.device

    def forward(
        self,
        xyz: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.pooled(xyz=xyz, rgb=rgb, mask=mask)
        if feat.ndim == 2:
            return feat.unsqueeze(1)
        if feat.ndim == 3:
            return feat.unsqueeze(2)
        raise ValueError(f"Unexpected pooled feature shape: {tuple(feat.shape)}")


__all__ = [
    "PointBERTEncoderPooled",
    "PointBERTEncoderTokens",
]
