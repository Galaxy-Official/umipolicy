# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_handcap_cfg = EasyDict(__name__='Config: VA handcap')
va_handcap_cfg.update(va_shared_cfg)
va_handcap_cfg.use_handcap = True
va_handcap_cfg.use_tactile = True

va_handcap_cfg.wan22_pretrained_model_name_or_path = "./ckpt"

va_handcap_cfg.attn_window = 72
va_handcap_cfg.frame_chunk_size = 2
va_handcap_cfg.env_type = 'handcap'

va_handcap_cfg.height = 256
va_handcap_cfg.width = 320
va_handcap_cfg.action_dim = 30
va_handcap_cfg.action_per_frame = 16
va_handcap_cfg.obs_cam_keys = [
    'observation.images.wrist_0_rgb', 'observation.images.left_tactile',
    'observation.images.right_tactile'
]
va_handcap_cfg.guidance_scale = 5
va_handcap_cfg.action_guidance_scale = 1

va_handcap_cfg.num_inference_steps = 25
va_handcap_cfg.video_exec_step = -1
va_handcap_cfg.action_num_inference_steps = 50

va_handcap_cfg.snr_shift = 5.0
va_handcap_cfg.action_snr_shift = 1.0

va_handcap_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_handcap_cfg.used_action_channel_ids)
] * va_handcap_cfg.action_dim
for i, j in enumerate(va_handcap_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_handcap_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_handcap_cfg.action_norm_method = 'quantiles'
va_handcap_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
