import logging
from typing import List

import toml
import torch
from loguru import logger
from PIL import Image
from .base_model import BaseModel
from .base_clip_model import BaseClipModel
from perception.utils.result import DetectionResult

 


@BaseModel.register_model("siglip")
class Siglip(BaseClipModel):
    config_path = "perception/configs/models/siglip.toml"

    def __init__(
        self,
        device: torch.device = None,
        ckpt_type: str = "default",
        prompter_type: str = "VanillaClip",
    ):
        super().__init__(device, ckpt_type, prompter_type)
        import open_clip
        with open(self.config_path, "r") as f:
            config = toml.load(f)
        clip_model_path = config["path"]["model"]
        self.clip_model, self.clip_processor = (
            open_clip.create_model_from_pretrained(
                model_name=f"local-dir:{clip_model_path}",
            )
        )
        self.clip_model.to(self.device)
        logger.info(f"Loading tokenizer from {clip_model_path}")

        self.clip_tokenizer = open_clip.get_tokenizer(
            model_name=f"local-dir:{clip_model_path}",
        )
        ckpt_path = config["path"]["checkpoint"].get(self.ckpt_type, None)
        if ckpt_path is not None:
            logger.info(f"Loading CLIP checkpoint from {ckpt_path}")
            clip_ckpt = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
            msg = self.clip_model.load_state_dict(clip_ckpt, strict=False)
            logger.info(msg)
        else:
            logger.info("No checkpoint loaded")

    def generate_text_tokens(
        self,
        text_list,
        label_type="attribute",
    ):
        all_text = super().generate_text_tokens(text_list, label_type)
        tokens = self.clip_tokenizer(
            texts=all_text,
            context_length=self.clip_model.context_length,
        )
        return tokens

    @torch.no_grad()
    def calculate_text_probabilities(
        self,
        image: Image.Image,
        detections: List[DetectionResult],
        labels: List[str] = None,
        attribute_list: List[str] = None,
        affordance_list: List[str] = None,
        obj_id_len: int = None,
    ):
        if not hasattr(self, "attr_text_features"):
            self.attr_text_features = self.encode_text(
                self.generate_text_tokens(attribute_list, "attribute").to(
                    self.device
                )
            )
        if not hasattr(self, "aff_text_features"):
            self.aff_text_features = self.encode_text(
                self.generate_text_tokens(affordance_list, "affordance").to(
                    self.device
                )
            )
        # Init Image Buffer
        image_buffer = []
        for i in range(obj_id_len):
            image_buffer.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        images = []
        label_list = []
        if labels is not None: 
            label2idx = {label: i for i, label in enumerate(labels)}
        else:
            label2idx = {detection.label: i for i, detection in enumerate(detections)}
        for detection in detections:
            label = detection.label
            label_list.append(label)
            box = detection.box
            detection.mask
            xmin, ymin, xmax, ymax = box.xyxy
            if xmin == -1:
                xmin = 400
                xmax = image.size[0]
                ymin = 280
                ymax = image.size[1]
            #     # TODO
            #     logger.info(f"{label}")
            #     try:
            #         last_image = image_buffer[label]
            #     except IndexError:
            #         logger.error(
            #             f"Index Error, the list length is {len(image_buffer)}, but the label is {label} "
            #         )
            #     images.append(self.clip_processor(last_image))
            # else:
                # for i in range(mask.shape[0]):
                #     for j in range(mask.shape[1]):
                #         if mask[i, j] > 0:
                #             mask[i, j] = 1
                #         else:
                #             mask[i, j] = 0`
                # array = np.array(image)
                # # array dim [W, H, C] -> [C, W, H]
                # array = np.transpose(array, [2, 0, 1])
                # # mask value should be 0 or 1
                # array = array * mask  # 点乘
                # # array dim [C, W, H] -> [W, H, C]
                # array = np.transpose(array, [1, 2, 0])
                # mask_image = Image.fromarray(array, mode="RGB")  # 打开图片
                # crop_image = mask_image.crop((xmin, ymin, xmax, ymax))
            crop_image = image.crop((xmin, ymin, xmax, ymax))
            image_buffer[label2idx[label]] = crop_image
            crop_image = self.clip_processor(crop_image)
            images.append(crop_image)
        images = torch.stack(images).cuda()
        with torch.no_grad():
            image_features = self.encode_image(images)
            if self.clip_model.logit_bias is not None:
                attr_probs = torch.sigmoid(
                    image_features
                    @ self.attr_text_features.T
                    * self.clip_model.logit_scale.exp()
                    + self.clip_model.logit_bias
                )
                aff_probs = torch.sigmoid(
                    image_features
                    @ self.aff_text_features.T
                    * self.clip_model.logit_scale.exp()
                    + self.clip_model.logit_bias
                )
            else:
                attr_probs = torch.sigmoid(
                    image_features
                    @ self.attr_text_features.T
                    * self.clip_model.logit_scale.exp()
                )
                aff_probs = torch.sigmoid(
                    image_features
                    @ self.aff_text_features.T
                    * self.clip_model.logit_scale.exp()
                )
        return label_list, attr_probs, aff_probs
