from abc import ABC, abstractmethod

import torch

from perception.utils.prompter import ClipPrompter, VanillaClipPrompter


class BaseClipModel(ABC):
    model_registry = {}

    def __init__(
        self,
        device: torch.device = None,
        ckpt_type: str = None,
        prompter_type: str = "VanillaClip",
    ):
        self.device = device
        self.ckpt_type = ckpt_type
        self.prompter_type = prompter_type

    @abstractmethod
    def calculate_text_probabilities(
        self,
        image,
        detections,
        labels,
        attribute_list,
        affordance_list,
        obj_id_len,
    ):
        pass

    def encode_image(self, image, normalize: bool = True):
        image_features = self.clip_model.encode_image(image, normalize)
        return image_features

    def encode_text(self, text, normalize: bool = True):
        text_features = self.clip_model.encode_text(text, normalize)
        return text_features

    def generate_text_tokens(
        self,
        text_list,
        label_type: str = "attribute",
    ):
        if self.prompter_type == "VanillaClip":
            prompter = VanillaClipPrompter(label_type=label_type)
        elif self.prompter_type == "Clip":
            prompter = ClipPrompter(label_type=label_type)
        else:
            raise KeyError(f"Invalid prompter type: {self.prompter_type}")
        all_text = [prompter.make_prompt(a) for a in text_list]
        return all_text
