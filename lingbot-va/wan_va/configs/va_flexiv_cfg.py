# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_flexiv_cfg = EasyDict(__name__='Config: VA flexiv')
va_flexiv_cfg.update(va_shared_cfg)
va_flexiv_cfg.use_flexiv = True
va_flexiv_cfg.use_tactile = False

va_flexiv_cfg.wan22_pretrained_model_name_or_path = "./ckpt/lingbot-va-base"

va_flexiv_cfg.attn_window = 72
va_flexiv_cfg.frame_chunk_size = 2
va_flexiv_cfg.env_type = 'flexiv'

va_flexiv_cfg.height = 256
va_flexiv_cfg.width = 320
va_flexiv_cfg.action_dim = 10
va_flexiv_cfg.action_per_frame = 16
va_flexiv_cfg.convert_action_to_relative_rot6d = False
va_flexiv_cfg.obs_cam_keys = ['observation.images.wrist']
va_flexiv_cfg.guidance_scale = 5
va_flexiv_cfg.action_guidance_scale = 1

va_flexiv_cfg.num_inference_steps = 25
va_flexiv_cfg.video_exec_step = -1
va_flexiv_cfg.action_num_inference_steps = 50

va_flexiv_cfg.snr_shift = 5.0
va_flexiv_cfg.action_snr_shift = 1.0

va_flexiv_cfg.used_action_channel_ids = list(range(0, 10))
inverse_used_action_channel_ids = [
    len(va_flexiv_cfg.used_action_channel_ids)
] * va_flexiv_cfg.action_dim
for i, j in enumerate(va_flexiv_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_flexiv_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_flexiv_cfg.action_norm_method = 'quantiles'
va_flexiv_cfg.norm_stat = {
    "q01": [-1.0] * 10,
    "q99": [1.0] * 10,
}
