import copy
import logging
import math
from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
import torchvision

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.timm_obs_encoder import AttentionPool2d


logger = logging.getLogger(__name__)


class TimmObsEncoderTactile(ModuleAttrMixin):
    """Timm RGB encoder with an optional separate tactile backbone.

    Tactile images are still regular ``type: rgb`` observations in shape_meta.
    Keys listed in ``tactile_keys`` use ``tactile_model_name`` while every other
    RGB key uses ``model_name``.
    """

    def __init__(
            self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            use_group_norm: bool = False,
            share_rgb_model: bool = False,
            imagenet_norm: bool = False,
            feature_aggregation: Optional[str] = 'spatial_embedding',
            downsample_ratio: int = 32,
            position_encording: str = 'learnable',
            checkpoint_path: str = '',
            tactile_keys: Optional[list] = None,
            tactile_model_name: Optional[str] = None,
            tactile_pretrained: Optional[bool] = None,
            tactile_checkpoint_path: Optional[str] = None,
            tactile_feature_aggregation: Optional[str] = None,
            tactile_downsample_ratio: Optional[int] = None,
            use_tactile: bool = True,
            fusion_method: str = 'concat',
            fusion_output_dim: Optional[int] = None,
            force_guide: bool = False,
            force_keys: Optional[list] = None,
            force_range: Tuple[float, float] = (0.0, 10.0),
            min_vision_weight: float = 0.2,
            default_tactile_weight: float = 0.5,
            force_use_abs: bool = True,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        assert global_pool == ''

        rgb_keys = list()
        low_dim_keys = list()
        key_shape_map = dict()

        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if obs_type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        if image_shape is None:
            raise RuntimeError("TimmObsEncoderTactile requires at least one rgb key.")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        if tactile_keys is None:
            tactile_keys = [key for key in rgb_keys if 'tactile' in key]
        tactile_keys = sorted(set(tactile_keys))
        unknown_tactile_keys = sorted(set(tactile_keys) - set(rgb_keys))
        if unknown_tactile_keys:
            raise ValueError(
                f"tactile_keys must be rgb obs keys. Unknown keys: {unknown_tactile_keys}")
        if not use_tactile:
            rgb_keys = [key for key in rgb_keys if key not in tactile_keys]
            tactile_keys = []
        if not rgb_keys:
            raise RuntimeError(
                "TimmObsEncoderTactile requires at least one active rgb key.")

        fusion_method = fusion_method.lower()
        if fusion_method in ['linear_fusion', 'weighted_linear']:
            fusion_method = 'linear'
        if fusion_method in ['film_fusion', 'film']:
            fusion_method = 'film'
        if fusion_method not in ['concat', 'linear', 'film']:
            raise ValueError(
                "fusion_method must be one of: concat, linear, film.")
        if fusion_method in ['linear', 'film'] and fusion_output_dim is None:
            fusion_output_dim = 1024
        if not 0.0 <= min_vision_weight < 1.0:
            raise ValueError("min_vision_weight must be in [0, 1).")
        if not 0.0 <= default_tactile_weight <= 1.0:
            raise ValueError("default_tactile_weight must be in [0, 1].")

        tactile_model_name = tactile_model_name or model_name
        tactile_pretrained = pretrained if tactile_pretrained is None else tactile_pretrained
        tactile_downsample_ratio = (
            downsample_ratio if tactile_downsample_ratio is None
            else tactile_downsample_ratio)
        if tactile_feature_aggregation is None:
            tactile_feature_aggregation = feature_aggregation
        if tactile_checkpoint_path is None:
            tactile_checkpoint_path = (
                checkpoint_path if tactile_model_name == model_name else '')

        transform = self._build_transform(
            transforms=transforms,
            image_shape=image_shape,
            imagenet_norm=imagenet_norm)

        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_to_model_key = dict()
        key_model_name_map = dict()
        key_feature_aggregation_map = dict()

        key_attention_map = nn.ModuleDict()
        key_attention_pool_2d_map = nn.ModuleDict()
        key_aggregation_transformer_map = nn.ModuleDict()
        key_spatial_embedding_map = nn.ParameterDict()
        key_position_embedding_map = nn.ParameterDict()

        specs = dict()
        for key in rgb_keys:
            is_tactile = key in tactile_keys
            this_model_name = tactile_model_name if is_tactile else model_name
            this_pretrained = tactile_pretrained if is_tactile else pretrained
            this_checkpoint_path = (
                tactile_checkpoint_path if is_tactile else checkpoint_path)
            this_downsample_ratio = (
                tactile_downsample_ratio if is_tactile else downsample_ratio)
            this_aggregation = (
                tactile_feature_aggregation if is_tactile else feature_aggregation)
            this_aggregation = self._resolve_aggregation(
                model_name=this_model_name,
                aggregation=this_aggregation,
                key=key)

            specs[key] = (
                this_model_name,
                bool(this_pretrained),
                this_checkpoint_path or '',
                int(this_downsample_ratio),
                this_aggregation)
            key_model_name_map[key] = this_model_name
            key_feature_aggregation_map[key] = this_aggregation

        if share_rgb_model:
            unique_model_specs = {
                (spec[0], spec[1], spec[2], spec[3])
                for spec in specs.values()
            }
            if len(unique_model_specs) != 1:
                raise ValueError(
                    "share_rgb_model=True requires wrist and tactile rgb keys "
                    "to use the same backbone, checkpoint, and downsample ratio.")
            spec = next(iter(specs.values()))
            model, feature_dim, feature_map_shape = self._build_model(
                model_name=spec[0],
                pretrained=spec[1],
                frozen=frozen,
                checkpoint_path=spec[2],
                downsample_ratio=spec[3],
                image_shape=image_shape,
                use_group_norm=use_group_norm)
            key_model_map['rgb'] = model
            for key in rgb_keys:
                key_to_model_key[key] = 'rgb'
                self._build_aggregation_modules(
                    key=key,
                    model_name=specs[key][0],
                    aggregation=specs[key][4],
                    feature_dim=feature_dim,
                    feature_map_shape=feature_map_shape,
                    position_encording=position_encording,
                    key_attention_map=key_attention_map,
                    key_spatial_embedding_map=key_spatial_embedding_map,
                    key_position_embedding_map=key_position_embedding_map,
                    key_aggregation_transformer_map=key_aggregation_transformer_map,
                    key_attention_pool_2d_map=key_attention_pool_2d_map)
        else:
            for key in rgb_keys:
                spec = specs[key]
                model, feature_dim, feature_map_shape = self._build_model(
                    model_name=spec[0],
                    pretrained=spec[1],
                    frozen=frozen,
                    checkpoint_path=spec[2],
                    downsample_ratio=spec[3],
                    image_shape=image_shape,
                    use_group_norm=use_group_norm)
                key_model_map[key] = model
                key_to_model_key[key] = key
                self._build_aggregation_modules(
                    key=key,
                    model_name=spec[0],
                    aggregation=spec[4],
                    feature_dim=feature_dim,
                    feature_map_shape=feature_map_shape,
                    position_encording=position_encording,
                    key_attention_map=key_attention_map,
                    key_spatial_embedding_map=key_spatial_embedding_map,
                    key_position_embedding_map=key_position_embedding_map,
                    key_aggregation_transformer_map=key_aggregation_transformer_map,
                    key_attention_pool_2d_map=key_attention_pool_2d_map)

        for key in rgb_keys:
            key_transform_map[key] = copy.deepcopy(transform)

        print('rgb keys:         ', rgb_keys)
        print('tactile rgb keys: ', tactile_keys)
        print('low_dim_keys keys:', low_dim_keys)
        print('rgb backbones:    ', key_model_name_map)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_to_model_key = key_to_model_key
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.tactile_keys = tactile_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.key_model_name_map = key_model_name_map
        self.key_feature_aggregation_map = key_feature_aggregation_map
        self.key_attention_map = key_attention_map
        self.key_spatial_embedding_map = key_spatial_embedding_map
        self.key_position_embedding_map = key_position_embedding_map
        self.key_aggregation_transformer_map = key_aggregation_transformer_map
        self.key_attention_pool_2d_map = key_attention_pool_2d_map
        self.use_tactile = bool(use_tactile)
        self.fusion_method = fusion_method
        self.fusion_output_dim = fusion_output_dim
        self.force_guide = bool(force_guide)
        self.force_keys = (
            sorted(force_keys) if force_keys is not None
            else sorted([key for key in low_dim_keys if 'force' in key])
        )
        self.force_range = tuple(float(x) for x in force_range)
        self.min_vision_weight = float(min_vision_weight)
        self.default_tactile_weight = float(default_tactile_weight)
        self.force_use_abs = bool(force_use_abs)

        if self.fusion_method in ['linear', 'film']:
            self.vision_fusion_proj = nn.LazyLinear(self.fusion_output_dim)
            self.tactile_fusion_proj = nn.LazyLinear(self.fusion_output_dim)
            if self.fusion_method == 'film':
                self.film_generator = nn.Sequential(
                    nn.LayerNorm(self.fusion_output_dim),
                    nn.Linear(self.fusion_output_dim, 2 * self.fusion_output_dim)
                )

        print('use_tactile:      ', self.use_tactile)
        print('fusion_method:    ', self.fusion_method)
        print('force_guide:      ', self.force_guide)
        print('force_keys:       ', self.force_keys)

        logger.info(
            "number of initialized parameters: %e",
            sum(
                p.numel() for p in self.parameters()
                if not isinstance(p, UninitializedParameter))
        )

    def _build_transform(self, transforms, image_shape, imagenet_norm):
        transform_list = []
        if transforms is not None:
            transforms = list(transforms)
            if len(transforms) > 0 and not isinstance(transforms[0], torch.nn.Module):
                assert transforms[0].type == 'RandomCrop'
                ratio = transforms[0].ratio
                transforms = [
                    torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                    torchvision.transforms.Resize(size=image_shape[0], antialias=True)
                ] + transforms[1:]
            transform_list.extend(transforms)

        if imagenet_norm:
            transform_list.append(torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))

        if len(transform_list) == 0:
            return nn.Identity()
        return nn.Sequential(*transform_list)

    def _build_model(
            self,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            checkpoint_path: str,
            downsample_ratio: int,
            image_shape: Tuple[int, int],
            use_group_norm: bool):
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0)

        if checkpoint_path:
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict, strict=False)
            except FileNotFoundError:
                logger.warning(
                    f"Pretrained checkpoint {checkpoint_path} not found. Skipping.")

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False

        feature_dim = getattr(model, 'num_features', None)
        if model_name.startswith('resnet'):
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 2048 if any(x in model_name for x in ['50', '101', '152']) else 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024 if any(x in model_name for x in ['50', '101', '152']) else 256
            else:
                raise NotImplementedError(
                    f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(
                    f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16)
                    if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )

        if feature_dim is None:
            raise RuntimeError(f"Cannot infer feature_dim for {model_name}")

        feature_map_shape = [x // downsample_ratio for x in image_shape]
        return model, feature_dim, feature_map_shape

    def _resolve_aggregation(self, model_name, aggregation, key):
        if model_name.startswith('vit'):
            if aggregation == 'all_tokens':
                return aggregation
            if aggregation is not None:
                logger.warning(
                    f"{key}: vit will use the CLS token. "
                    f"feature_aggregation ({aggregation}) is ignored.")
            return None
        if aggregation is None:
            logger.warning(
                f"{key}: non-ViT backbone {model_name} needs feature aggregation; "
                "using attention_pool_2d.")
            return 'attention_pool_2d'
        return aggregation

    def _build_aggregation_modules(
            self,
            key,
            model_name,
            aggregation,
            feature_dim,
            feature_map_shape,
            position_encording,
            key_attention_map,
            key_spatial_embedding_map,
            key_position_embedding_map,
            key_aggregation_transformer_map,
            key_attention_pool_2d_map):
        if model_name.startswith('vit'):
            return

        if aggregation == 'soft_attention':
            key_attention_map[key] = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif aggregation == 'spatial_embedding':
            key_spatial_embedding_map[key] = torch.nn.Parameter(
                torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif aggregation == 'transformer':
            num_features = feature_map_shape[0] * feature_map_shape[1] + 1
            if position_encording == 'learnable':
                key_position_embedding_map[key] = torch.nn.Parameter(
                    torch.randn(num_features, feature_dim))
            elif position_encording == 'sinusoidal':
                position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, feature_dim, 2).float()
                    * (-math.log(2 * num_features) / feature_dim))
                position_embedding[:, 0::2] = torch.sin(position * div_term)
                position_embedding[:, 1::2] = torch.cos(position * div_term)
                key_position_embedding_map[key] = torch.nn.Parameter(
                    position_embedding, requires_grad=False)
            else:
                raise ValueError(f"Unsupported position_encording: {position_encording}")
            key_aggregation_transformer_map[key] = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=feature_dim, nhead=4),
                num_layers=4)
        elif aggregation == 'attention_pool_2d':
            key_attention_pool_2d_map[key] = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=max(1, feature_dim // 64),
                output_dim=feature_dim
            )
        elif aggregation in ['avg', 'max']:
            return
        elif aggregation is None:
            return
        else:
            raise ValueError(f"Unsupported feature_aggregation: {aggregation}")

    def aggregate_feature(self, key, feature):
        model_name = self.key_model_name_map[key]
        aggregation = self.key_feature_aggregation_map[key]

        if model_name.startswith('vit'):
            if len(feature.shape) == 2:
                return feature
            if aggregation == 'all_tokens':
                return torch.flatten(feature, start_dim=1)
            assert aggregation is None
            return feature[:, 0, :]

        assert len(feature.shape) == 4
        if aggregation == 'attention_pool_2d':
            return self.key_attention_pool_2d_map[key](feature)

        feature = torch.flatten(feature, start_dim=-2)
        feature = torch.transpose(feature, 1, 2)

        if aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        if aggregation == 'max':
            return torch.amax(feature, dim=[1])
        if aggregation == 'soft_attention':
            weight = self.key_attention_map[key](feature)
            return torch.sum(feature * weight, dim=1)
        if aggregation == 'spatial_embedding':
            return torch.mean(feature * self.key_spatial_embedding_map[key], dim=1)
        if aggregation == 'transformer':
            zero_feature = torch.zeros(
                feature.shape[0], 1, feature.shape[-1], device=feature.device)
            position_embedding = self.key_position_embedding_map[key].to(feature.device)
            feature_with_pos_embedding = (
                torch.concat([zero_feature, feature], dim=1) + position_embedding)
            feature_output = self.key_aggregation_transformer_map[key](
                feature_with_pos_embedding)
            return feature_output[:, 0]

        raise RuntimeError(
            f"{key} uses {model_name}; set a non-null feature_aggregation "
            "for non-ViT backbones.")

    def _compute_modality_weights(
            self,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
            raw_obs_dict: Optional[Dict[str, torch.Tensor]] = None):
        if not self.force_guide:
            tactile_weight = torch.full(
                (batch_size, 1),
                self.default_tactile_weight,
                device=device,
                dtype=dtype)
            return 1.0 - tactile_weight, tactile_weight

        if raw_obs_dict is None:
            raw_obs_dict = {}

        force_values = list()
        for key in self.force_keys:
            if key not in raw_obs_dict:
                continue
            force = raw_obs_dict[key]
            if force.shape[0] != batch_size:
                continue
            if force.ndim >= 3:
                force = force[:, -1]
            else:
                force = force.reshape(batch_size, -1)
            force = force.to(device=device, dtype=dtype)
            if self.force_use_abs:
                force = force.abs()
            force_values.append(force.reshape(batch_size, -1))

        if len(force_values) == 0:
            force_scalar = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        else:
            force_scalar = torch.cat(force_values, dim=-1)
            force_scalar = torch.norm(force_scalar, p=2.0, dim=-1, keepdim=True)

        force_min, force_max = self.force_range
        denom = max(force_max - force_min, 1e-6)
        force_alpha = ((force_scalar - force_min) / denom).clamp(0.0, 1.0)

        tactile_weight = (1.0 - self.min_vision_weight) * force_alpha
        vision_weight = 1.0 - tactile_weight
        return vision_weight, tactile_weight

    def _fuse_features(
            self,
            vision_features,
            tactile_features,
            low_dim_features,
            batch_size: int,
            raw_obs_dict: Optional[Dict[str, torch.Tensor]] = None):
        if self.fusion_method == 'concat':
            return torch.cat(vision_features + tactile_features + low_dim_features, dim=-1)

        if len(vision_features) == 0:
            raise RuntimeError("linear/FiLM fusion requires at least one vision key.")

        vision_feature = torch.cat(vision_features, dim=-1)
        vision_feature = self.vision_fusion_proj(vision_feature)

        if len(tactile_features) == 0:
            fused_feature = vision_feature
        else:
            tactile_feature = torch.cat(tactile_features, dim=-1)
            tactile_feature = self.tactile_fusion_proj(tactile_feature)
            vision_weight, tactile_weight = self._compute_modality_weights(
                batch_size=batch_size,
                device=vision_feature.device,
                dtype=vision_feature.dtype,
                raw_obs_dict=raw_obs_dict)

            if self.fusion_method == 'linear':
                fused_feature = (
                    vision_weight * vision_feature
                    + tactile_weight * tactile_feature)
            elif self.fusion_method == 'film':
                gamma_beta = self.film_generator(tactile_feature)
                gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
                gamma = gamma * tactile_weight
                beta = beta * tactile_weight
                modulated_vision = vision_feature * (1.0 + gamma) + beta
                fused_feature = (
                    vision_weight * modulated_vision
                    + tactile_weight * tactile_feature)
            else:
                raise RuntimeError(f"Unsupported fusion_method: {self.fusion_method}")

        if len(low_dim_features) > 0:
            fused_feature = torch.cat([fused_feature] + low_dim_features, dim=-1)
        return fused_feature

    def forward(
            self,
            obs_dict: Dict[str, torch.Tensor],
            raw_obs_dict: Optional[Dict[str, torch.Tensor]] = None):
        rgb_features = list()
        vision_features = list()
        tactile_features = list()
        low_dim_features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            model_key = self.key_to_model_key[key]
            raw_feature = self.key_model_map[model_key](img)
            feature = self.aggregate_feature(key, raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            feature = feature.reshape(B, -1)
            rgb_features.append(feature)
            if key in self.tactile_keys:
                tactile_features.append(feature)
            else:
                vision_features.append(feature)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            low_dim_features.append(data.reshape(B, -1))

        if self.fusion_method == 'concat':
            return torch.cat(rgb_features + low_dim_features, dim=-1)

        return self._fuse_features(
            vision_features=vision_features,
            tactile_features=tactile_features,
            low_dim_features=low_dim_features,
            batch_size=batch_size,
            raw_obs_dict=raw_obs_dict)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        return example_output.shape
