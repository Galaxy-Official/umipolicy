# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_handcap_cfg import va_handcap_cfg

va_yellow_lerobot_428_train_cfg = EasyDict(__name__='Config: VA yellow_lerobot_428 train')
va_yellow_lerobot_428_train_cfg.update(va_handcap_cfg)

va_yellow_lerobot_428_train_cfg.dataset_path = 'Data/yellow_lerobot_428'
va_yellow_lerobot_428_train_cfg.empty_emb_path = os.path.join(
    va_yellow_lerobot_428_train_cfg.dataset_path, 'empty_emb.pt'
)

# This yellow_to_pink run is wrist-camera only for now.
va_yellow_lerobot_428_train_cfg.use_tactile = False
va_yellow_lerobot_428_train_cfg.obs_cam_keys = ['observation.images.wrist']

va_yellow_lerobot_428_train_cfg.enable_wandb = True
va_yellow_lerobot_428_train_cfg.load_worker = 16
va_yellow_lerobot_428_train_cfg.save_interval = 1000
va_yellow_lerobot_428_train_cfg.gc_interval = 50
va_yellow_lerobot_428_train_cfg.cfg_prob = 0.1

# Training parameters
va_yellow_lerobot_428_train_cfg.learning_rate = 1e-5
va_yellow_lerobot_428_train_cfg.beta1 = 0.9
va_yellow_lerobot_428_train_cfg.beta2 = 0.95
va_yellow_lerobot_428_train_cfg.weight_decay = 0.1
va_yellow_lerobot_428_train_cfg.warmup_steps = 10
va_yellow_lerobot_428_train_cfg.batch_size = 1
va_yellow_lerobot_428_train_cfg.gradient_accumulation_steps = 1
va_yellow_lerobot_428_train_cfg.num_steps = 50000
