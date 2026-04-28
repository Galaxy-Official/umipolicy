# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_handcap_cfg = EasyDict(__name__='Config: VA handcap')
va_handcap_cfg.update(va_shared_cfg)
va_handcap_cfg.use_handcap = True
va_handcap_cfg.use_tactile = True

va_handcap_cfg.wan22_pretrained_model_name_or_path = "./ckpt/lingbot-va-base"

va_handcap_cfg.attn_window = 72
va_handcap_cfg.frame_chunk_size = 2
va_handcap_cfg.env_type = 'handcap'

va_handcap_cfg.height = 256
va_handcap_cfg.width = 320
va_handcap_cfg.action_dim = 10
va_handcap_cfg.action_per_frame = 16
va_handcap_cfg.convert_action_to_relative_rot6d = False
va_handcap_cfg.obs_cam_keys = [
    'observation.images.wrist', 'observation.tactiles.left',
    'observation.tactiles.right'
]
va_handcap_cfg.guidance_scale = 5
va_handcap_cfg.action_guidance_scale = 1

va_handcap_cfg.num_inference_steps = 25
va_handcap_cfg.video_exec_step = -1
va_handcap_cfg.action_num_inference_steps = 50

va_handcap_cfg.snr_shift = 5.0
va_handcap_cfg.action_snr_shift = 1.0

va_handcap_cfg.used_action_channel_ids = list(range(0, 10))
inverse_used_action_channel_ids = [
    len(va_handcap_cfg.used_action_channel_ids)
] * va_handcap_cfg.action_dim
for i, j in enumerate(va_handcap_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_handcap_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_handcap_cfg.action_norm_method = 'quantiles'
va_handcap_cfg.norm_stat = {
    "q01": [-1.0] * 10,
    "q99": [1.0] * 10,
}
