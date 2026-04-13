
import numpy as np
import toml
import torch
from typing import List, Tuple
from PIL import Image
from torchvision import transforms
from .base_model import BaseModel
from .base_clip_model import BaseClipModel
from perception.utils.result import DetectionResult


@BaseModel.register_model("AlphaClip")
class AlphaClip(BaseClipModel):
    config_path = "ocl/configs/models/alpha_clip.toml"

    def __init__(
        self,
        device: torch.device = None,
        ckpt_type: str = None,
    ):
        super().__init__(device, ckpt_type)
        import alpha_clip

        with open(self.config_path, "r") as f:
            config = toml.load(f)
        self.model, self.preprocess = alpha_clip.load(
            name=config["path"]["model"],
            alpha_vision_ckpt_pth=config["path"]["checkpoint"][self.ckpt_type],
            device=device,
        )
        print(f"Loading model from {config['path']['model']}")
        print(
            f"Loading checkpoint from {config['path']['checkpoint'][self.ckpt_type]}"
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224)
                ),  # change to (336,336) when using ViT-L/14@336px
                transforms.Normalize(0.5, 0.26),
            ]
        )

    def generate_text_tokens(
        self,
        text_list,
        label_type: str = "attribute",
        prompter_type: str = "VanillaClip",
    ):
        all_text = super().generate_text_tokens(
            text_list, label_type, prompter_type
        )
        return alpha_clip.tokenize(all_text).to(self.device)

    def get_masked_image(
        self,
        rgb_image,
        binary_mask,
    ):
        # mask value should be 0 or 1
        alpha = self.mask_transform((binary_mask * 225).astype(np.uint8))
        alpha = alpha.half().to(self.device).unsqueeze(dim=0)
        image = (
            self.preprocess(rgb_image).unsqueeze(dim=0).half().to(self.device)
        )
        return image, alpha

    @torch.no_grad()
    def calculate_text_probabilities(
        self,
        image: Image.Image,
        detections: List[DetectionResult],
        attribute_list: List[str],
        affordance_list: List[str],
        obj_id_len: int,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        pass
