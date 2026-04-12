
import torch
import utils3d
from typing import *
import torch.nn as nn
import torch.nn.functional as F
from openpi.models.tactile_backbone.clip_model.vision_transformer import CLIPVisionModel



class CLIPPretrainedTactileEncoderPi0(nn.Module):
    def __init__(self,
        tactile_pretrained_ckpt: Optional[str] = None, # Path to the pretrained checkpoint
        ):
        super(CLIPPretrainedTactileEncoderPi0, self).__init__()
        
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tactile_pretrained_ckpt = tactile_pretrained_ckpt
        
        
        self.mapping_dim = nn.Linear(1024, 2048)
        self.__init_ckpt__()
        
    def __init_ckpt__(self):
        ckpt_state_dict = torch.load(self.tactile_pretrained_ckpt, map_location='cpu')
        
        new_ckpt_state_dict = {}
        for key, value in ckpt_state_dict.items():
            if key.startswith("model."):
                new_key = key.replace("model.", "")
                new_ckpt_state_dict[new_key] = value
            else:
                new_ckpt_state_dict[key] = value
        ckpt_state_dict = new_ckpt_state_dict
        
        model_state_dict = self.model.state_dict()
        model_param_names = set(model_state_dict.keys())
        ckpt_param_names = set(ckpt_state_dict.keys())
        loaded_params = model_param_names.intersection(ckpt_param_names)
        print("\nSuccess load parameters:")
        for name in sorted(list(loaded_params))[:20]:
            print(f"  - {name}")
        if len(loaded_params) > 20:
            print(f"  ... and {len(loaded_params)-20} parameters")
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        del ckpt_state_dict, model_param_names, ckpt_param_names
    
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
                tactile: torch.Tensor):
        """
        forward pass of the model.
        Args:
            tactile: List of tensors, each tensor is a batch of tactile data.
                Each tensor should have shape [B, C, H, W] where B is the batch size,
                C is the number of channels, H is the height and W is the width.
        """ 
        
        tactile_feature = self.model(tactile.to(self.device),output_hidden_states=True)
        tactile_pooled_output = tactile_feature.pooler_output
        tactile_feature = tactile_feature.last_hidden_state


        tactile_feature = self.mapping_dim(tactile_feature)
        return tactile_feature[:, 1:, :]

class CLIPPretrainedTactileEncoderPooled(nn.Module):
    def __init__(self,
        tactile_pretrained_ckpt: Optional[str] = None, # Path to the pretrained checkpoint
        ):
        super(CLIPPretrainedTactileEncoderPooled, self).__init__()
        
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tactile_pretrained_ckpt = tactile_pretrained_ckpt
        
        
        self.mapping_dim = nn.Linear(1024, 64)
        self.__init_ckpt__()
        
    def __init_ckpt__(self):
        ckpt_state_dict = torch.load(self.tactile_pretrained_ckpt, map_location='cpu')
        
        new_ckpt_state_dict = {}
        for key, value in ckpt_state_dict.items():
            if key.startswith("model."):
                new_key = key.replace("model.", "")
                new_ckpt_state_dict[new_key] = value
            else:
                new_ckpt_state_dict[key] = value
        ckpt_state_dict = new_ckpt_state_dict
        
        model_state_dict = self.model.state_dict()
        model_param_names = set(model_state_dict.keys())
        ckpt_param_names = set(ckpt_state_dict.keys())
        loaded_params = model_param_names.intersection(ckpt_param_names)
        print("\nSuccess load parameters:")
        for name in sorted(list(loaded_params))[:20]:
            print(f"  - {name}")
        if len(loaded_params) > 20:
            print(f"  ... and {len(loaded_params)-20} parameters")
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        del ckpt_state_dict, model_param_names, ckpt_param_names
    
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
                tactile: torch.Tensor):
        """
        forward pass of the model.
        Args:
            tactile: List of tensors, each tensor is a batch of tactile data.
                Each tensor should have shape [B, C, H, W] where B is the batch size,
                C is the number of channels, H is the height and W is the width.
        """ 
        
        tactile_feature = self.model(tactile.to(self.device),output_hidden_states=True)
        tactile_pooled_output = tactile_feature.pooler_output
        # tactile_feature = tactile_feature.last_hidden_state


        tactile_feature = self.mapping_dim(tactile_pooled_output)
        # contrastive_loss = self.contrastive_loss(vision_pooled_output, tactile_pooled_output)
        return tactile_feature

class CLIPPretrainedTactileEncoder(nn.Module):
    def __init__(self,
        tactile_pretrained_ckpt: Optional[str] = None, # Path to the pretrained checkpoint
        ):
        super(CLIPPretrainedTactileEncoder, self).__init__()
        
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tactile_pretrained_ckpt = tactile_pretrained_ckpt
        
        
        self.mapping_dim = nn.Linear(1024, 300)
        self.__init_ckpt__()
        
        
        
    def __init_ckpt__(self):
        ckpt_state_dict = torch.load(self.tactile_pretrained_ckpt, map_location='cpu')
        
        new_ckpt_state_dict = {}
        for key, value in ckpt_state_dict.items():
            if key.startswith("model."):
                new_key = key.replace("model.", "")
                new_ckpt_state_dict[new_key] = value
            else:
                new_ckpt_state_dict[key] = value
        ckpt_state_dict = new_ckpt_state_dict
        
        model_state_dict = self.model.state_dict()
        model_param_names = set(model_state_dict.keys())
        ckpt_param_names = set(ckpt_state_dict.keys())
        loaded_params = model_param_names.intersection(ckpt_param_names)
        print("\nSuccess load parameters:")
        for name in sorted(list(loaded_params))[:20]:
            print(f"  - {name}")
        if len(loaded_params) > 20:
            print(f"  ... and {len(loaded_params)-20} parameters")
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        del ckpt_state_dict, model_param_names, ckpt_param_names
    
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
                tactile: torch.Tensor):
        """
        forward pass of the model.
        Args:
            tactile: List of tensors, each tensor is a batch of tactile data.
                Each tensor should have shape [B, C, H, W] where B is the batch size,
                C is the number of channels, H is the height and W is the width.
        """ 
        
        tactile_feature = self.model(tactile.to(self.device),output_hidden_states=True)
        tactile_pooled_output = tactile_feature.pooler_output
        tactile_feature = tactile_feature.last_hidden_state


        tactile_feature = self.mapping_dim(tactile_feature).view(tactile_feature.shape[0], tactile_feature.shape[1], 15, -1)
        # contrastive_loss = self.contrastive_loss(vision_pooled_output, tactile_pooled_output)
        return tactile_feature

import flax.linen as nn

class CLIPPretrainedTactileEncoderPi0Jax(nn.Module):
    """
    Flax implementation of a tactile encoder based on a pretrained Vision Transformer.
    This module is designed to be wrapped by `flax.nnx.bridge.ToNNX`.
    """
    # Define model parameters as class attributes for Flax
    output_dim: int = 2048
    vit_variant: str = "L/14"  # Corresponds to clip-vit-large-patch14
    dtype: str = "float32"
    scan: bool = True

    @nn.compact
    def __call__(self, tactile, train: bool = False):
        """
        Forward pass of the model.
        Args:
            tactile: An array with shape [B, H, W, C] representing the tactile image.
            train: A boolean indicating if the model is in training mode.
        """
        # Instantiate the SiglipViT model. `pool_type='tok'` ensures the [CLS] token is added.
        vit = CLIPPretrainedTactileEncoderPi0(tactile_pretrained_ckpt="/home/lixingyu/workspace_lxy2/workspace/robotpolicy/ckpt/pretrained_tactile_encoder.pt")

        # Get the token embeddings from the ViT.
        # The output `vit_features` has shape [B, num_tokens + 1, embed_dim]
        vit_features, _ = vit(tactile, train=train)

        # The original PyTorch code slices with `[:, 1:, :]` to exclude the [CLS] token.
        # We replicate that behavior here.
        # tactile_feature_tokens = vit_features[:, 1:, :]

        # Apply a linear projection to map features to the desired output dimension.
        # The original PyTorch code maps from 1024 to 2048. The SiglipViT 'L' variant has an embedding dim of 1024.
        # mapped_features = nn.Dense(
        #     features=self.output_dim,
        #     dtype=self.dtype,
        #     name="mapping_dim"
        # )(tactile_feature_tokens)

        return vit_features, _