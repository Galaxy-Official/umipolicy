from .base_model import BaseModel
from .base_clip_model import BaseClipModel
from .alpha_clip import AlphaClip
from .roi_siglip import Roi_Siglip
from .siglip import Siglip
from .siglip2 import Siglip2
from .vlm.base_vlm import BaseVLM

__all__ = [
    "BaseModel",
    "BaseClipModel",
    "Siglip",
    "AlphaClip",
    "Roi_Siglip",
    "Siglip2",
    "BaseVLM",
]

# def load_model(
#     model_name: str,
#     device: torch.device = None,
#     ckpt_type: str = None,
#     prompter_type: str = "VanillaClip",
# ) -> BaseModel:
#     if model_name == "Siglip":
#         return Siglip(device, ckpt_type, prompter_type)
#     elif model_name == "AlphaClip":
#         return AlphaClip(device, ckpt_type, prompter_type)
#     else:
#         raise KeyError(f"Invalid model name: {model_name}")
