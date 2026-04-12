"""Handcap custom policies and data configs for OpenPI framework."""

import dataclasses
import pathlib
from typing import TypeAlias
from collections.abc import Sequence

import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.handcap_policy as handcap_policy
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

# In typical execution, config.py will import this file.
# To avoid circular imports, we import the necessary factories dynamically inside the class definition where needed,
# or just rely on openpi.training.config at the function level.
from openpi.training.config import DataConfigFactory, DataConfig, ModelTransformFactory

@dataclasses.dataclass(frozen=True)
class LeRobotHandcapDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    """
    # Exposing data_root here so the config instantiation knows about it
    data_root: str = tyro.MISSING

    # Action sequences are stored as 'action' in handcap datasets
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
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
            inputs=[handcap_policy.HandcapInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[handcap_policy.HandcapOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(10, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def get_handcap_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import TrainConfig

    return [
        TrainConfig(
            name="pi0_erase_board_and_write",
            model=pi0_config.Pi0Config(
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_and_write",
                data_root="Data/handcap/erase_board_and_write_handcap",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=30_000,
            batch_size=12,
        ),
        TrainConfig(
            name="pi0_erase_board_and_write_200",
            model=pi0_config.Pi0Config(
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_and_write",
                data_root="Data/handcap/erase_board_and_write_handcap_200/erase_board_and_write_handcap",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=30_000,
            batch_size=24,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi0_erase_board_and_write_tactile_200",
            model=pi0_config.Pi0Config(
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_and_write",
                data_root="Data/handcap/erase_board_and_write_handcap_200/erase_board_and_write_handcap",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=30_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi0_erase_board_and_write_tactile_200_debug",
            model=pi0_config.Pi0Config(
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_and_write",
                data_root="Data/handcap/erase_board_and_write_handcap_200/erase_board_and_write_handcap",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=30_000,
            batch_size=8,
            log_interval=100,
            save_interval=500,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi0_simple_sorting",
            model=pi0_config.Pi0Config(
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/block_to_pot_handcap",
                data_root="Data/handcap30",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi0_simple_sorting_tactile",
            model=pi0_config.Pi0Config(
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/block_to_pot_handcap",
                data_root="Data/handcap30",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_simple_sorting",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/block_to_pot_handcap",
                data_root="Data/handcap30",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_simple_sorting_tactile",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/block_to_pot_handcap",
                data_root="Data/handcap30",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/robopolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
    ]
