import toml
import torch

from loguru import logger
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
from .base_model import BaseModel
from .base_clip_model import BaseClipModel
from perception.utils.result import DetectionResult

@BaseModel.register_model("siglip2")
class Siglip2(BaseClipModel):
    config_path = "perception/configs/models/siglip2.toml"

    def __init__(
        self,
        device: torch.device = None,
        ckpt_type: str = "default",
        prompter_type: str = "VanillaClip",
    ):
        super().__init__(device, ckpt_type, prompter_type)
        with open(self.config_path, "r") as f:
            config = toml.load(f)
        clip_model_path = config.get("path", {}).get("model", "google/siglip2-so400m-patch14-224")
        logger.info(f"Loading model from {clip_model_path}")
        self.clip_model = AutoModel.from_pretrained(clip_model_path, device_map="auto")
        self.clip_processor = AutoProcessor.from_pretrained(clip_model_path)
        self.clip_model.to(self.device)
        logger.info(f"Loading tokenizer from {clip_model_path}")
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path)

    def generate_text_tokens(
        self,
        text_list,
        label_type="attribute",
    ):
        all_text = super().generate_text_tokens(text_list, label_type)
        tokens = self.clip_tokenizer(
            text=all_text,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        return tokens

    def encode_image(self, image):
        image_features = self.clip_model.get_image_features(image)
        return image_features

    def encode_text(self, text):
        text_features = self.clip_model.get_text_features(**text)
        return text_features


    @torch.no_grad()
    def calculate_text_probabilities(
        self,
        image: Image.Image,
        detections: list[DetectionResult],
        attribute_list: list[str],
        affordance_list: list[str],
        obj_id_len: int,
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
        for detection in detections:
            label = detection.label
            label_list.append(label)
            box = detection.box
            detection.mask
            xmin, ymin, xmax, ymax = box.xyxy
            if xmin == -1:
                # TODO
                logger.info(f"{label}")
                try:
                    last_image = image_buffer[label]
                except IndexError:
                    logger.error(
                        f"Index Error, the list length is {len(image_buffer)}, but the label is {label} "
                    )
                images.append(self.clip_processor(
                    images=last_image,
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                    )['pixel_values']
                )
            else:
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
                image_buffer[label] = crop_image
                crop_image = self.clip_processor(
                    images=crop_image,
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                )
                images.append(crop_image['pixel_values'])
        images = torch.cat(images).to(self.device)
        # with torch.no_grad():
        #     outputs = self.clip_model(
        #         input_ids=self.attr_text_features
        #     )
        with torch.no_grad():
            image_embeds = self.encode_image(images)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            attr_text_embeds = self.attr_text_features / self.attr_text_features.norm(p=2, dim=-1, keepdim=True)
            aff_text_embeds = self.aff_text_features / self.aff_text_features.norm(p=2, dim=-1, keepdim=True)
            logits_per_attr_text =  torch.matmul(
                image_embeds,
                attr_text_embeds.t(),
            )
            logits_per_aff_text =  torch.matmul(
                image_embeds,
                aff_text_embeds.t(),
                
            )
            logit_scale, logit_bias = self.clip_model.logit_scale.to(attr_text_embeds.device), self.clip_model.logit_bias.to(attr_text_embeds.device)
            logits_per_attr_text = torch.sigmoid(logits_per_attr_text * logit_scale.exp() + logit_bias)
            logits_per_aff_text = torch.sigmoid(logits_per_aff_text * logit_scale.exp() + logit_bias)
            # if self.clip_model.logit_bias is not None:
            #     attr_probs = torch.sigmoid(
            #         image_features
            #         @ self.attr_text_features.T
            #         * self.clip_model.logit_scale.exp()
            #         + self.clip_model.logit_bias
            #     )
            #     aff_probs = torch.sigmoid(
            #         image_features
            #         @ self.aff_text_features.T
            #         * self.clip_model.logit_scale.exp()
            #         + self.clip_model.logit_bias
            #     )
            # else:
            #     attr_probs = torch.sigmoid(
            #         image_features
            #         @ self.attr_text_features.T
            #         * self.clip_model.logit_scale.exp()
            #     )
            #     aff_probs = torch.sigmoid(
            #         image_features
            #         @ self.aff_text_features.T
            #         * self.clip_model.logit_scale.exp()
            #     )
        return label_list, logits_per_attr_text, logits_per_aff_text
