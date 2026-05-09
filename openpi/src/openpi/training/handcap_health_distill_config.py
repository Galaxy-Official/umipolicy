"""Isolated configs for multi-health wrist-force to tactile distillation."""

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
class HealthDistillRuntimeDataConfig(DataConfig):
    """Runtime DataConfig used only by multi-health distillation training."""

    use_health_distill_multi_handcap: bool = True
    health_repo_ids: Sequence[str] = ()
    health_data_roots: Sequence[str] = ()
    health_labels: Sequence[str] = ("0", "50", "100")
    vtla_tactile_history: int = 1
    vtla_tactile_keys: Sequence[str] = ()


@dataclasses.dataclass(frozen=True)
class LeRobotHealthDistillHandcapDataConfig(DataConfigFactory):
    """Data config for random cross-health wrist/tactile/force training."""

    health_data_roots: Sequence[str] = tyro.MISSING
    health_repo_ids: Sequence[str] | None = None
    health_labels: Sequence[str] = ("0", "50", "100")
    action_sequence_keys: Sequence[str] = ("action",)
    use_vtla_vgte: bool = False
    vtla_tactile_history: int = 4

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.health_data_roots is tyro.MISSING:
            raise ValueError("health_data_roots must contain the health-conditioned dataset paths.")

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
        if self.vtla_tactile_history <= 0:
            raise ValueError("vtla_tactile_history must be positive.")

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
                    force_observation_current_only=True,
                    tactile_temporal_grid=self.use_vtla_vgte and self.vtla_tactile_history > 1,
                    tactile_grid_shape=(2, 2),
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
        vtla_tactile_keys = ()
        if self.use_vtla_vgte and self.vtla_tactile_history > 1:
            vtla_tactile_keys = ("observation.tactiles.left", "observation.tactiles.right")
        return HealthDistillRuntimeDataConfig(
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
            health_repo_ids=health_repo_ids,
            health_data_roots=health_data_roots,
            health_labels=tuple(self.health_labels),
            vtla_tactile_history=self.vtla_tactile_history if self.use_vtla_vgte else 1,
            vtla_tactile_keys=vtla_tactile_keys,
        )


def get_health_distill_handcap_configs():
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
            name="pi05_handcap_health_distill_tactile_wrist_force",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt=tactile_encoder_ckpt,
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=True,
                camera_keys=("wrist_0_rgb",),
                health_distill=True,
                health_distill_weight=0.05,
                health_distill_gt_temperature=0.15,
                health_distill_tactile_temperature=0.07,
                health_distill_force_weight=0.5,
            ),
            data=LeRobotHealthDistillHandcapDataConfig(
                repo_id="handcap_health_distill",
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
        TrainConfig(
            name="pi05_handcap_health_vtla_vgte_baseline",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt=tactile_encoder_ckpt,
                tactile_variant="B/16",
                fusion_method="vtla_vgte",
                force_predict=True,
                force_guide=True,
                camera_keys=("wrist_0_rgb",),
            ),
            data=LeRobotHealthDistillHandcapDataConfig(
                repo_id="handcap_health_vtla_vgte",
                health_data_roots=tyro.MISSING,
                base_config=base_data_config,
                use_vtla_vgte=True,
                vtla_tactile_history=4,
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
