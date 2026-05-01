# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_handcap_cfg import va_handcap_cfg
from .va_handcap_train_cfg import va_handcap_train_cfg
from .va_clamp_seal_430_train_cfg import va_clamp_seal_430_train_cfg
from .va_towel_hanging_430_train_cfg import va_towel_hanging_430_train_cfg
from .va_yellow_lerobot_428_train_cfg import va_yellow_lerobot_428_train_cfg
from .va_flexiv_cfg import va_flexiv_cfg
from .va_flexiv_train_cfg import va_flexiv_train_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
    'handcap': va_handcap_cfg,
    'handcap_train': va_handcap_train_cfg,
    'towel_hanging_430_train': va_towel_hanging_430_train_cfg,
    'clamp_seal_430_train': va_clamp_seal_430_train_cfg,
    'yellow_lerobot_428_train': va_yellow_lerobot_428_train_cfg,
    'flexiv': va_flexiv_cfg,
    'flexiv_train': va_flexiv_train_cfg,
}
