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

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=seq_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotHandcapWristDataConfig(DataConfigFactory):
    """Handcap LeRobot data config for datasets that only contain a wrist camera."""

    data_root: str = tyro.MISSING
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
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
                _transforms.RepackTransform(
                    {
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
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
                    include_tactile=False,
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

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=seq_keys,
        )


def get_handcap_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import TrainConfig

    pi05_base_params = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"
    tactile_encoder_ckpt = "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt"
    common_pi05_train_kwargs = {
        "num_train_steps": 200_000,
        "batch_size": 512,
        "num_workers": 76,
        "log_interval": 100,
        "save_interval": 10000,
        "keep_period": 20_000,
    }

    def make_pi05_505_configs(task_name: str, repo_id: str, data_root: str) -> list[TrainConfig]:
        def model_config(
            *,
            use_tactile: bool,
            force_predict: bool = False,
            force_guide: bool = False,
        ) -> pi0_config.Pi0Config:
            kwargs = {
                "pi05": True,
                "use_tactile": use_tactile,
                "camera_keys": ("wrist_0_rgb",),
            }
            if use_tactile:
                kwargs.update(
                    {
                        "tactile_pretrained_ckpt": tactile_encoder_ckpt,
                        "tactile_variant": "B/16",
                        "fusion_method": "linear",
                        "force_predict": force_predict,
                        "force_guide": force_guide,
                    }
                )
            else:
                kwargs["tactile_pretrained_ckpt"] = ""
            return pi0_config.Pi0Config(**kwargs)

        def base_data_config() -> DataConfig:
            return DataConfig(
                prompt_from_task=True,
                use_handcap=True,
            )

        return [
            TrainConfig(
                name=f"pi05_{task_name}",
                model=model_config(use_tactile=False),
                data=LeRobotHandcapWristDataConfig(
                    repo_id=repo_id,
                    data_root=data_root,
                    base_config=base_data_config(),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader(pi05_base_params),
                **common_pi05_train_kwargs,
            ),
            TrainConfig(
                name=f"pi05_{task_name}_tactile",
                model=model_config(use_tactile=True),
                data=LeRobotHandcapDataConfig(
                    repo_id=repo_id,
                    data_root=data_root,
                    base_config=base_data_config(),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader(pi05_base_params),
                **common_pi05_train_kwargs,
            ),
            TrainConfig(
                name=f"pi05_{task_name}_tactile_force_predict",
                model=model_config(use_tactile=True, force_predict=True),
                data=LeRobotHandcapDataConfig(
                    repo_id=repo_id,
                    data_root=data_root,
                    base_config=base_data_config(),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader(pi05_base_params),
                **common_pi05_train_kwargs,
            ),
            TrainConfig(
                name=f"pi05_{task_name}_tactile_force_guide",
                model=model_config(use_tactile=True, force_predict=True, force_guide=True),
                data=LeRobotHandcapDataConfig(
                    repo_id=repo_id,
                    data_root=data_root,
                    base_config=base_data_config(),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader(pi05_base_params),
                **common_pi05_train_kwargs,
            ),
        ]

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
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi0_base/params"),
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
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=30_000,
            batch_size=24,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_erase_board_wrist",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapWristDataConfig(
                repo_id="lihongcs/erase_board_wrist",
                data_root="Data/429_erase_board_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_erase_board_wrist_tactile",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=False,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_wrist",
                data_root="Data/429_erase_board_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_clamp_seal",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapWristDataConfig(
                repo_id="lihongcs/430_clamp_seal_lerobot",
                data_root="Data/430_clamp_seal_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_towel_hanging",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapWristDataConfig(
                repo_id="lihongcs/430_towel_hanging_lerobot",
                data_root="Data/430_towel_hanging_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_clamp_seal_tactile",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=False,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/430_clamp_seal_lerobot",
                data_root="Data/430_clamp_seal_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_towel_hanging_tactile",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=False,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/430_towel_hanging_lerobot",
                data_root="Data/430_towel_hanging_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi0_erase_board_and_write_tactile_200",
            model=pi0_config.Pi0Config(
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_and_write",
                data_root="Data/handcap/erase_board_and_write_handcap_200/erase_board_and_write_handcap",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi0_base/params"),
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
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_and_write",
                data_root="Data/handcap/erase_board_and_write_handcap_200/erase_board_and_write_handcap",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi0_base/params"),
            num_train_steps=30_000,
            batch_size=8,
            log_interval=100,
            save_interval=500,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_bread_moving",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=False,
                tactile_pretrained_ckpt="",
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapWristDataConfig(
                repo_id="lihongcs/501_bread_moving_lerobot",
                data_root="Data/501_bread_moving_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_bread_moving_tactile",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=False,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/501_bread_moving_lerobot",
                data_root="Data/501_bread_moving_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_bread_moving_tactile_force_predict",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/501_bread_moving_lerobot",
                data_root="Data/501_bread_moving_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_bread_moving_tactile_force_guide",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=True,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/501_bread_moving_lerobot",
                data_root="Data/501_bread_moving_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_clamp_seal_tactile_force_predict",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/430_clamp_seal_lerobot",
                data_root="Data/430_clamp_seal_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_clamp_seal_tactile_force_guide",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=True,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/430_clamp_seal_lerobot",
                data_root="Data/430_clamp_seal_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_erase_board_wrist_tactile_force_predict",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_wrist",
                data_root="Data/429_erase_board_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_erase_board_wrist_tactile_force_guide",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=True,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/erase_board_wrist",
                data_root="Data/429_erase_board_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_towel_hanging_tactile_force_predict",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=False,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/430_towel_hanging_lerobot",
                data_root="Data/430_towel_hanging_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        TrainConfig(
            name="pi05_430_towel_hanging_tactile_force_guide",
            model=pi0_config.Pi0Config(
                pi05=True,
                use_tactile=True,
                tactile_pretrained_ckpt="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pretrained_tactile_encoder.pt",
                tactile_variant="B/16",
                fusion_method="linear",
                force_predict=True,
                force_guide=True,
                camera_keys=("wrist_0_rgb",),),
            data=LeRobotHandcapDataConfig(
                repo_id="lihongcs/430_towel_hanging_lerobot",
                data_root="Data/430_towel_hanging_lerobot",
                base_config=DataConfig(
                    prompt_from_task=True,
                    use_handcap=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/openpi/ckpt/pi05_base/params"),
            num_train_steps=200_000,
            batch_size=8,
            log_interval=100,
            save_interval=5000,
            keep_period=20_000,
        ),
        *make_pi05_505_configs(
            task_name="505_screw",
            repo_id="lihongcs/505_screw_lerobot",
            data_root="Data/505_screw_lerobot",
        ),
        *make_pi05_505_configs(
            task_name="505_stiring",
            repo_id="lihongcs/505_stiring_lerobot",
            data_root="Data/505_stiring_lerobot",
        ),
        *make_pi05_505_configs(
            task_name="506_peg_flowers",
            repo_id="lihongcs/506_peg_flowers_lerobot",
            data_root="Data/506_peg_flowers_lerobot",
        ),
        *make_pi05_505_configs(
            task_name="506_open_bottle",
            repo_id="lihongcs/506_open_bottle_lerobot",
            data_root="Data/506_open_bottle_lerobot",
        ),
    ]
