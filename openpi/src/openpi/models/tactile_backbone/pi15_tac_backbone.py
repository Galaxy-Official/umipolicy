"""
Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Pi-0.5 Tactile Encoder
"""

import torch
from typing import *
import torch.nn as nn
import torch.nn.functional as F
from openpi.models.tactile_backbone.clip_model.vision_transformer import CLIPVisionModel

class ViFiCLIP(nn.Module):
    def __init__(self, clip_model, freeze_text_encoder, use_positional_embeds):
        super().__init__()
        self.clip_model = clip_model
        self.use_positional_embeds = use_positional_embeds

    def forward(self, frames,sensors):
        # video
        b, l, c, h, w = frames.shape # (b, l, c, h, w)
        frames = frames.reshape(b * l, c, h, w) # (b * l, c, h, w)
        frame_embeds = self.clip_model(frames)
        
        tactile_feature = frame_embeds.last_hidden_state # (b * l, num_tokens, patch_embed_size)
        _, num_tokens, patch_embed_size = tactile_feature.shape
        
        frame_features = frame_embeds.pooler_output # (b * l, patch_embed_size)
        frame_features = frame_features.reshape(b, l, patch_embed_size) # (b, l, patch_embed_size)
        tactile_feature = tactile_feature.reshape(b, l, num_tokens, patch_embed_size) # (b, l, num_tokens, patch_embed_size)
        video_features = frame_features.mean(dim=1, keepdim=False) # (b, patch_embed_size)
        tactile_feature = tactile_feature.mean(dim=1, keepdim=False) # (b, num_tokens, patch_embed_size)
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
        tactile_feature = tactile_feature / tactile_feature.norm(p=2, dim=-1, keepdim=True)
        return tactile_feature, video_features


class PI15_Tac_Encoder(nn.Module):
    def __init__(self, 
                 tactile_pretrained_ckpt):
        
        super(PI15_Tac_Encoder, self).__init__()

        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tactile_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tactile_vificlip = ViFiCLIP(self.tactile_model, freeze_text_encoder=True, use_positional_embeds=True).to(self.device)
        
        self.__load_ckpt__(tactile_pretrained_ckpt)
        
    def __load_ckpt__(self, tactile_pretrained_ckpt):
        """Load the pretrained tactile encoder and adapter weights.
        """
        state_dict = torch.load(tactile_pretrained_ckpt, map_location="cpu", weights_only=True)
        # for k, v in state_dict.items():
        #     print(k)
        # exit()
        new_state_dict = {}
        for k, v in state_dict.items():
            if "vision_model" in k and "VPT" not in k:
                new_state_dict[k] = v
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("clip_model.", ""): v for k, v in state_dict.items()}
        self.tactile_vificlip.load_state_dict(new_state_dict, strict=True)
        print("Loaded tactile ViFi-CLIP!")
    
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def contrastive_loss(self, visual_features, tactile_features, temperature=0.07, safe_mode=True):
        if safe_mode:
            if torch.any(torch.isnan(visual_features)) or torch.any(torch.isnan(tactile_features)):
                return torch.tensor(0.0, device=visual_features.device, dtype=visual_features.dtype)
            if torch.all(torch.abs(visual_features) < 1e-8) or torch.all(torch.abs(tactile_features) < 1e-8):
                return torch.tensor(0.0, device=visual_features.device, dtype=visual_features.dtype)
    
        eps = 1e-6
        visual_features = visual_features + eps
        tactile_features = tactile_features + eps
        
        visual_features = F.normalize(visual_features, dim=1)
        tactile_features = F.normalize(tactile_features, dim=1)
        
        logits = torch.matmul(visual_features, tactile_features.T) / temperature
        
        logits = logits + torch.eye(logits.shape[0], device=logits.device) * eps
        
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2.0
    
    def forward(self, 
                vision: torch.Tensor,
                tactile: torch.Tensor):
        vision_feature = self.vision_model(vision.to(self.device),output_hidden_states=True)
        vision_pooled_output = vision_feature.pooler_output
        
        tactile_feature, tactile_pooled_output = self.tactile_vificlip(tactile.to(self.device), None)
        
        
        contrastive_loss = self.contrastive_loss(vision_pooled_output, tactile_pooled_output)
        return contrastive_loss, tactile_feature
        
        
            
