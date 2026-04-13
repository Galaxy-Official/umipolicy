import logging
from math import ceil, floor
from typing import Optional
import torch
import torch.nn.functional as F
from PIL import Image
from .base_model import BaseModel
from .base_clip_model import BaseClipModel
from .siglip import Siglip

logger = logging.getLogger(__name__)


@BaseModel.register_model("roi_siglip")
class Roi_Siglip(Siglip):

    def __init__(
        self,
        device: torch.device = None,
        ckpt_type: str = "default",
        prompter_type: str = "VanillaClip",
        roi_pooling_layer: int = 13,
        **kwargs,
    ) -> None:
        super().__init__(device, ckpt_type, prompter_type)
        import open_clip
        self.roi_pooling_layer = roi_pooling_layer
        state_dict = self.clip_model.state_dict()
        self.grid_size = round(
            (state_dict["visual.trunk.pos_embed"].shape[1] - 1) ** 0.5
        )  # TODO why -1?
        assert (
            type(self.clip_model.visual) == open_clip.timm_model.TimmModel
        ), type(self.clip_model.visual)
        self.depth = len(self.clip_model.visual.trunk.blocks)
        assert self.roi_pooling_layer < self.depth

        self.n_heads = self.clip_model.visual.trunk.blocks[0].attn.num_heads

        self.logit_scale = self.clip_model.logit_scale
        self.logit_bias = self.clip_model.logit_bias

    def encode_image(
        self,
        images,
        bboxes: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ):
        if bboxes is None:
            return self.clip_model.encode_image(images, normalize)
        mask = self.box_to_mask(bboxes, self.grid_size)
        features = self.visual_forward_with_mask(images, mask)
        return F.normalize(features, dim=-1) if normalize else features

    def visual_forward_with_mask(
        self, images: torch.Tensor, mask: torch.FloatTensor = None
    ):
        model = self.clip_model
        if mask is None:
            return model.visual(images)
        else:
            x = model.visual.trunk.patch_embed(images)
            x = model.visual.trunk._pos_embed(x)
            x = model.visual.trunk.patch_drop(x)
            x = model.visual.trunk.norm_pre(x)

            for i in range(self.roi_pooling_layer):
                x = model.visual.trunk.blocks[i](x)

            # roi_pooling_layer, attention with mask
            blk = model.visual.trunk.blocks[self.roi_pooling_layer]
            x = blk.norm1(x)
            x = self.timm_attention_with_mask(blk.attn, x, mask)

            x = x + blk.drop_path1(blk.ls1(x))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

            for i in range(self.roi_pooling_layer + 1, self.depth):
                x = model.visual.trunk.blocks[i](x)

            x = model.visual.trunk.norm(x)
            x = model.visual.trunk.forward_head(x)
            x = model.visual.head(x)

            return x

    def timm_attention_with_mask(self, attn_layer, x, mask):
        B, N, C = x.shape
        qkv = (
            attn_layer.qkv(x)
            .reshape(B, N, 3, attn_layer.num_heads, attn_layer.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = attn_layer.q_norm(q), attn_layer.k_norm(k)

        mask = mask.to(q.device)
        assert mask.dtype == torch.bool, mask.dtype

        if attn_layer.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=(
                    attn_layer.attn_drop.p if attn_layer.training else 0.0
                ),
            )
            assert torch.isnan(x).sum() == 0, x
        else:
            q = q * attn_layer.scale
            attn = q @ k.transpose(-2, -1)
            mask = mask.masked_fill(mask, float("-inf"))
            attn = attn + mask
            attn = attn.softmax(dim=-1)
            attn = attn_layer.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn_layer.proj(x)
        x = attn_layer.proj_drop(x)

        return x

    def box_to_mask(self, bboxes, resolution):
        """Φ is a binary mask converted from the bounding boxes. Φi,j equals −∞ if i is the CLS token and j a patch outside the given bounding boxes, and 0 otherwise."""
        mask = torch.zeros(bboxes.size(0), resolution, resolution).to(
            torch.bool
        )

        for i in range(bboxes.size(0)):
            box = bboxes[i].to(torch.float32).cpu().numpy()
            x1, y1, x2, y2 = (box * resolution).tolist()
            x1, y1 = floor(x1), floor(y1)
            x2, y2 = ceil(x2), ceil(y2)
            # mask[i, y1:y2, x1:x2] = float("-inf")
            mask[i, y1:y2, x1:x2] = True
            assert x1 < x2, (box, x1, y1, x2, y2)
            assert y1 < y2, (box, x1, y1, x2, y2)
            assert mask[i].sum() > 0, (mask[i], box, x1, y1, x2, y2)

        # bz, width, width
        # mask = mask.reshape(mask.shape[0], -1)
        # # bz, width^2

        # n_patches = mask.size(1)  # = width*width

        # # mask = torch.cat([
        # #     torch.zeros(mask.size(0), 1),
        # #     mask,
        # # ], dim=1)  # bz, width^2+1

        # mask = mask.unsqueeze(1).expand(-1, 1, -1)
        # # bz, 1, width^2+1

        # mask = torch.cat([
        #     mask,
        #     torch.zeros(mask.size(0), n_patches, mask.size(2)),
        # ], dim=1) # bz, width^2+1, width^2+1

        mask = mask.reshape(mask.shape[0], -1)
        # mask [bz, d]
        n_patches = mask.shape[1]
        mask = mask.unsqueeze(1).expand(-1, n_patches, -1)
        # mask [bz, d, d]

        # mask = mask + mask.transpose(1, 2)

        mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        # bz, head, width^2+1, width^2+1
        # mask = mask.reshape(-1, mask.size(2), mask.size(3))
        # bz*head, width^2+1, width^2+1

        return mask

    @torch.no_grad()
    def calculate_text_probabilities(
        self,
        image: Image.Image,
        detections,
        attribute_list,
        affordance_list,
        obj_id_len,
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
        bboxes = []
        # TODO
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
                    normalized_box = [0, 0, 1, 1]  # Full image
                except IndexError:
                    logger.error(
                        f"Index Error, the list length is {len(image_buffer)}, but the label is {label} "
                    )
            else:
                image_buffer[label] = image
                normalized_box = self.normalize_bbox(box, image.size)
            images.append(self.clip_processor(image_buffer[label]))
            bboxes.append(normalized_box)
        images = torch.stack(images).to(self.device)
        bboxes = torch.tensor(bboxes, device=self.device)
        image_features = self.encode_image(
            images=images,
            bboxes=bboxes,
            normalize=True,
        )
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

    @staticmethod
    def normalize_bbox(box, image_size):
        """Convert box coordinates to normalized 0-1 range."""
        width, height = image_size
        xmin, ymin, xmax, ymax = box.xyxy
        return [
            max(0, xmin) / width,
            max(0, ymin) / height,
            min(xmax, width) / width,
            min(ymax, height) / height,
        ]
