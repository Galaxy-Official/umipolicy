from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PointIdentityProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PointBERTEncoderPooled(nn.Module):
    """
    Point-BERT encoder wrapper for LeRobot / diffusion-policy style usage.

    Design goals:
    - mimic the role of `CLIPPretrainedTactileEncoderPooled`
    - accept xyz + optional rgb + optional mask
    - support both [B, N, C] and [B, T, N, C] inputs
    - return a pooled feature with shape [B, D] or [B, T, D]
    - keep Point-BERT-specific details inside this module

    Expected inputs
    ---------------
    xyz:
        [B, N, 3] or [B, T, N, 3]
    rgb:
        optional, same leading dims as xyz, last dim = 3
    mask:
        optional, same leading dims as xyz without the xyz channel, i.e.
        [B, N] or [B, T, N]. True/1 means valid point.

    Notes
    -----
    1) The upstream Point-BERT repo is not packaged as a stable Python library.
       This wrapper therefore performs a best-effort dynamic import and expects
       `pointbert_repo_root` to point at the cloned repository.
    2) The official Point-BERT repo is built around xyz-only point clouds. When
       rgb is provided here, we fuse it with xyz through a lightweight MLP before
       feeding the points to the final projection head. The backbone itself still
       runs on xyz geometry.
    3) This module is intentionally conservative: it exposes a stable pooled
       feature interface first, and leaves token-level fusion for later.
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

    def _ensure_repo_on_path(self) -> None:
        repo_root = os.path.abspath(self.pointbert_repo_root)
        if not os.path.isdir(repo_root):
            raise FileNotFoundError(
                f"Point-BERT repo root does not exist: {repo_root}. "
                "Please clone Julie-tang00/Point-BERT and pass its path here."
            )
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _build_pointbert_backbone(self) -> Tuple[nn.Module, int]:
        """
        Best-effort loader for the official Point-BERT repo.

        We try a few common import paths because different forks expose slightly
        different APIs. The wrapper expects the created backbone to accept xyz of
        shape [B, N, 3] and to return either:
          - [B, C]
          - [B, M, C]
          - a tuple/list whose first tensor is one of the above
          - a dict containing a tensor under a common key
        """
        self._ensure_repo_on_path()

        import importlib

        errors = []

        # Candidate 1: common fine-tuning class used by many Point-BERT forks.
        try:
            mod = importlib.import_module("models.Point_BERT")
            if hasattr(mod, "PointTransformer"):
                backbone = mod.PointTransformer(config=None)
                feature_dim = int(getattr(backbone, "trans_dim", 384))
                self._load_checkpoint_if_needed(backbone)
                return backbone, feature_dim
        except Exception as exc:
            errors.append(f"models.Point_BERT.PointTransformer failed: {exc}")

        # Candidate 2: builder-style loading used by some forks.
        try:
            mod = importlib.import_module("tools.builder")
            if hasattr(mod, "model_builder"):
                backbone = mod.model_builder(None)
                feature_dim = int(getattr(backbone, "trans_dim", 384))
                self._load_checkpoint_if_needed(backbone)
                return backbone, feature_dim
        except Exception as exc:
            errors.append(f"tools.builder.model_builder failed: {exc}")

        # Candidate 3: manual fallback placeholder.
        raise ImportError(
            "Unable to construct Point-BERT backbone from the provided repo. "
            "Tried common import paths in the official repo/forks, but none succeeded. "
            f"Repo root: {self.pointbert_repo_root}. Errors: {errors}"
        )

    def _load_checkpoint_if_needed(self, backbone: nn.Module) -> None:
        if not self.pointbert_ckpt:
            return
        ckpt_path = os.path.abspath(self.pointbert_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Point-BERT checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = None
            for key in ["state_dict", "base_model", "model", "module"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is None:
                state_dict = ckpt
        else:
            state_dict = ckpt

        cleaned = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ["module.", "model.", "base_model."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v

        missing, unexpected = backbone.load_state_dict(cleaned, strict=self.strict_ckpt)
        print("\nPoint-BERT checkpoint loaded.")
        if len(missing) > 0:
            print(f"  missing keys: {len(missing)}")
        if len(unexpected) > 0:
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
        """
        Convert variable valid-point counts into a fixed number of points per item.
        """
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
                # fully invalid: fall back to all points, then zeros if needed
                valid_idx = torch.arange(n, device=xyz.device)

            if valid_idx.numel() >= self.num_points:
                sel = valid_idx[: self.num_points]
            else:
                repeat_count = self.num_points - valid_idx.numel()
                pad_idx = valid_idx[torch.randint(0, valid_idx.numel(), (repeat_count,), device=xyz.device)]
                sel = torch.cat([valid_idx, pad_idx], dim=0)

            chosen_xyz = xyz[i, sel]
            out_xyz.append(chosen_xyz)

            if rgb is not None:
                chosen_rgb = rgb[i, sel]
                out_rgb.append(chosen_rgb)

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

    def _extract_backbone_feature(self, raw_output) -> torch.Tensor:
        if isinstance(raw_output, torch.Tensor):
            feat = raw_output
        elif isinstance(raw_output, (tuple, list)):
            tensor_candidates = [x for x in raw_output if isinstance(x, torch.Tensor)]
            if len(tensor_candidates) == 0:
                raise TypeError("Point-BERT output tuple/list does not contain a tensor.")
            feat = tensor_candidates[0]
        elif isinstance(raw_output, dict):
            for key in ["x", "feat", "features", "global_feat", "cls_token", "logits"]:
                if key in raw_output and isinstance(raw_output[key], torch.Tensor):
                    feat = raw_output[key]
                    break
            else:
                raise TypeError("Point-BERT output dict does not contain a recognized tensor field.")
        else:
            raise TypeError(f"Unsupported Point-BERT output type: {type(raw_output)}")

        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            # token-wise output -> mean pool over tokens
            return feat.mean(dim=1)
        raise ValueError(f"Unsupported Point-BERT feature shape: {tuple(feat.shape)}")

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
                raw_feat = self.backbone(xyz_flat)
        else:
            raw_feat = self.backbone(xyz_flat)

        feat = self._extract_backbone_feature(raw_feat)

        if rgb_flat is not None and self.use_rgb:
            # rgb is optional side information. We keep the backbone geometry-only,
            # then inject pooled color information at the feature level.
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
    """
    Optional token-returning version.

    Use this only if you later decide to do token-level fusion / cross-attention.
    For the current diffusion-policy integration, `PointBERTEncoderPooled` is the
    recommended default because the policy consumes global conditioning vectors.
    """

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
        # Placeholder token interface: currently returns pooled feature with a
        # singleton token dimension to keep the outward contract simple.
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
