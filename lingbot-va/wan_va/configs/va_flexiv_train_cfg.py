# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_flexiv_cfg import va_flexiv_cfg

va_flexiv_train_cfg = EasyDict(__name__='Config: VA flexiv train')
va_flexiv_train_cfg.update(va_flexiv_cfg)

va_flexiv_train_cfg.dataset_path = 'Data/flexiv_lerobot'
va_flexiv_train_cfg.empty_emb_path = os.path.join(
    va_flexiv_train_cfg.dataset_path, 'empty_emb.pt'
)

va_flexiv_train_cfg.use_tactile = False
va_flexiv_train_cfg.obs_cam_keys = ['observation.images.wrist']
va_flexiv_train_cfg.convert_action_to_relative_rot6d = True

va_flexiv_train_cfg.enable_wandb = True
va_flexiv_train_cfg.load_worker = 16
va_flexiv_train_cfg.save_interval = 1000
va_flexiv_train_cfg.gc_interval = 50
va_flexiv_train_cfg.cfg_prob = 0.1

va_flexiv_train_cfg.learning_rate = 1e-5
va_flexiv_train_cfg.beta1 = 0.9
va_flexiv_train_cfg.beta2 = 0.95
va_flexiv_train_cfg.weight_decay = 0.1
va_flexiv_train_cfg.warmup_steps = 10
va_flexiv_train_cfg.batch_size = 1
va_flexiv_train_cfg.gradient_accumulation_steps = 1
va_flexiv_train_cfg.num_steps = 50000
