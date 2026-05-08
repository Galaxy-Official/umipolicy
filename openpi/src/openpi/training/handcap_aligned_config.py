"""Isolated configs for force/time aligned Handcap multi-dataset training."""

from collections.abc import Sequence
import dataclasses
import pathlib
from typing import TypeAlias

import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.handcap_policy as handcap_policy
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AlignedHandcapRuntimeDataConfig(DataConfig):
    """Runtime DataConfig used only by aligned multi-health Handcap training."""

    use_aligned_multi_handcap: bool = True
    aligned_health_repo_ids: Sequence[str] = ()
    aligned_health_data_roots: Sequence[str] = ()
    aligned_health_labels: Sequence[str] = ("0", "50", "100")
    aligned_force_eps: float = 0.5
    aligned_max_progress_diff: float = 0.15
    aligned_time_weight: float = 0.25
    aligned_force_smoothing_window: int = 3
    aligned_anchor_dataset_index: int = 1
    aligned_max_alignments: int | None = None
    aligned_seed: int = 0
    aligned_cache_dir: str | None = None
    aligned_rebuild_cache: bool = False


@dataclasses.dataclass(frozen=True)
class LeRobotAlignedHandcapDataConfig(DataConfigFactory):
    """Data config for three health-conditioned Handcap datasets.

    The source datasets stay separate on disk. A runtime wrapper aligns frames by
    force first, then by normalized episode progress, and exposes grouped samples
    for contrastive training.
    """

    health_data_roots: Sequence[str] = tyro.MISSING
    health_repo_ids: Sequence[str] | None = None
    health_labels: Sequence[str] = ("0", "50", "100")
    action_sequence_keys: Sequence[str] = ("action",)
    force_eps: float = 0.5
    max_progress_diff: float = 0.15
    time_weight: float = 0.25
    force_smoothing_window: int = 3
    anchor_dataset_index: int = 1
    max_alignments: int | None = None
    alignment_seed: int = 0
    alignment_cache_dir: str | None = None
    rebuild_alignment_cache: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.health_data_roots is tyro.MISSING:
            raise ValueError("health_data_roots must contain the 0%, 50%, and 100% health dataset paths.")

        health_data_roots = tuple(self.health_data_roots)
        if len(health_data_roots) < 2:
            raise ValueError("At least two health_data_roots are required.")
        if len(self.health_labels) != len(health_data_roots):
            raise ValueError("health_labels must have the same length as health_data_roots.")

        if self.health_repo_ids is None:
            health_repo_ids = tuple(pathlib.Path(root).name for root in health_data_roots)
        else:
            health_repo_ids = tuple(self.health_repo_ids)
        if len(health_repo_ids) != len(health_data_roots):
            raise ValueError("health_repo_ids must have the same length as health_data_roots.")

        action_base_dim = getattr(model_config, "action_base_dim", 10)
        force_dim = getattr(model_config, "force_dim", 2)
        force_predict = getattr(model_config, "force_predict", False)
        force_guide = getattr(model_config, "force_guide", False)
        output_action_dim = action_base_dim + (force_dim if force_predict else 0)

        seq_keys = list(self.action_sequence_keys)
        if force_predict or force_guide:
            if "observation.forces.left" not in seq_keys:
                seq_keys.append("observation.forces.left")
            if "observation.forces.right" not in seq_keys:
                seq_keys.append("observation.forces.right")

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.HandcapRepackTransform(
                    {
                        "observation/wrist_image": "wrist_image",
                        "observation/left_tactile": "left_tactile",
                        "observation/right_tactile": "right_tactile",
                        "observation/state": "state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                handcap_policy.HandcapInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    force_predict=force_predict,
                    force_guide=force_guide,
                    action_base_dim=action_base_dim,
                    force_dim=force_dim,
                )
            ],
            outputs=[
                handcap_policy.HandcapOutputs(
                    output_action_dim=output_action_dim,
                    force_predict=force_predict,
                    action_base_dim=action_base_dim,
                    force_dim=force_dim,
                )
            ],
        )
        delta_action_mask = _transforms.make_bool_mask(action_base_dim, -(model_config.action_dim - action_base_dim))
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        base = self.create_base_config(assets_dirs, model_config)
        cache_dir = self.alignment_cache_dir or str(assets_dirs / "alignment_cache")
        return AlignedHandcapRuntimeDataConfig(
            repo_id=base.repo_id,
            asset_id=base.asset_id,
            use_handcap=True,
            data_root=base.data_root,
            norm_stats=base.norm_stats,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            use_quantile_norm=base.use_quantile_norm,
            action_sequence_keys=tuple(seq_keys),
            prompt_from_task=base.prompt_from_task,
            rlds_data_dir=base.rlds_data_dir,
            action_space=base.action_space,
            datasets=base.datasets,
            aligned_health_repo_ids=health_repo_ids,
            aligned_health_data_roots=health_data_roots,
            aligned_health_labels=tuple(self.health_labels),
            aligned_force_eps=self.force_eps,
            aligned_max_progress_diff=self.max_progress_diff,
            aligned_time_weight=self.time_weight,
            aligned_force_smoothing_window=self.force_smoothing_window,
            aligned_anchor_dataset_index=self.anchor_dataset_index,
            aligned_max_alignments=self.max_alignments,
            aligned_seed=self.alignment_seed,
            aligned_cache_dir=cache_dir,
            aligned_rebuild_cache=self.rebuild_alignment_cache,
        )


def get_aligned_handcap_configs():
    from openpi.training.config import TrainConfig

    ckpt_root = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt"
    pi05_base_params = f"{ckpt_root}/pi05_base/params"
    tactile_encoder_ckpt = f"{ckpt_root}/pretrained_tactile_encoder.pt"

    base_data_config = DataConfig(
        prompt_from_task=True,
        use_handcap=True,
    )

    return [
        TrainConfig(
            name="pi05_handcap_aligned_health_tactile_force_contrast",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt=tactile_encoder_ckpt,
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=True,
                camera_keys=("wrist_0_rgb",),
                contrastive_alignment=True,
                contrastive_weight=0.05,
                contrastive_temperature=0.07,
            ),
            data=LeRobotAlignedHandcapDataConfig(
                repo_id="handcap_aligned_health",
                health_data_roots=tyro.MISSING,
                base_config=base_data_config,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(pi05_base_params),
            num_train_steps=200_000,
            batch_size=384,
            num_workers=32,
            log_interval=100,
            save_interval=10000,
            keep_period=20_000,
        ),
    ]
