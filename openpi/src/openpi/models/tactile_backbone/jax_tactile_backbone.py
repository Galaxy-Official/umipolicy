import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import FlaxCLIPVisionModel, CLIPVisionConfig
import torch
import numpy as np


class FlaxCLIPVisionModule(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 直接使用 FlaxCLIPVisionModel，而不是 FlaxCLIPVisionTransformer
        self.vision_model = FlaxCLIPVisionModel(self.config, dtype=self.dtype)

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # FlaxCLIPVisionModel 不接受 deterministic 参数，移除它
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class FlaxCLIPPretrainedTactileEncoderPi0(nn.Module):
    """
    CLIPPretrainedTactileEncoderPi0 的 JAX/Flax 实现。
    """
    output_dim: int = 2048
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        """
        在 Flax 中定义子模块。这在 __call__ 之前执行。
        """
        config = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )

        # 2. 使用这个显式配置来实例化模型
        self.vit_model = FlaxCLIPVisionModule(
            config=config,
            dtype=self.dtype
        )
        self.mapping_dim = nn.Dense(
            features=self.output_dim,
            dtype=self.dtype,
            name="mapping_dim"
        )
    
    
    def __call__(self, tactile: jnp.ndarray, train: bool = False):
        """
        模型的前向传播。
        Args:
            tactile: 输入的触觉图像张量，形状为 [B, H, W, C] (JAX 默认通道在后)。
            train: 是否为训练模式。
        """
        # 注意: Hugging Face Flax 模型期望输入形状为 [B, H, W, C]
        # 如果你的数据是 [B, C, H, W]，需要先进行转换
        # if tactile.ndim == 4 and tactile.shape[1] == 3:
        #      # 更稳健的检查：如果秩为4且第二个维度是3，则假定为B,C,H,W
        
        
        # tactile = jnp.transpose(tactile, (0, 2, 3, 1))
        print("input tactile shape:", tactile.shape)
        # 1. 从预训练的CLIP模型获取特征
        # `from_pretrained` 会下载并缓存模型权重，但我们稍后会用自己的权重覆盖它们
        outputs = self.vit_model(tactile, output_hidden_states=True)
        
        # last_hidden_state 形状: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.last_hidden_state

        # 2. 使用在 setup 中定义的 mapping_dim 应用线性映射层
        mapped_features = self.mapping_dim(last_hidden_state)

        # 3. 忽略 [CLS] token 的输出，与 PyTorch 版本 `[:, 1:, :]` 行为一致
        return mapped_features[:, 1:, :]

def load_pytorch_weights_in_flax(
    flax_model: nn.Module,
    pytorch_ckpt_path: str
) -> dict:
    """
    加载PyTorch检查点，将其转换为Flax参数格式，并返回Flax的状态。

    Args:
        flax_model: 要加载权重的Flax模型实例。
        pytorch_ckpt_path: PyTorch .pt 或 .pth 检查点文件的路径。

    Returns:
        一个包含 'params' 的字典，可用于Flax模型的 `apply` 方法。
    """
    # 1. 加载 PyTorch 权重
    pt_state_dict = torch.load(pytorch_ckpt_path, map_location='cpu')

    # 2. 初始化 Flax 模型以获取参数结构
    # 使用一个虚拟输入来初始化
    dummy_input = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
    flax_params = flax_model.init(jax.random.PRNGKey(0), dummy_input)['params']
    
    # 3. 转换并加载权重
    new_flax_params = flax_params.unfreeze() # 将 FrozenDict 转换为可变字典

    # a) 加载 CLIP Vision Transformer 权重
    vit_params = new_flax_params['FlaxCLIPVisionModule_0']
    for key, value in pt_state_dict.items():
        # 清理 PyTorch checkpoint 中的 'model.' 前缀
        if key.startswith("model."):
            key = key.replace("model.", "", 1)

        # 将 PyTorch 的层名转换为 Flax 的层名
        # 例如: 'vision_model.embeddings.patch_embedding.weight' -> 'vision_model/embeddings/patch_embedding/kernel'
        key_parts = key.split('.')
        
        # PyTorch: (out, in, k, k) -> Flax: (k, k, in, out)
        if 'patch_embedding.weight' in key:
            vit_params['vision_model']['embeddings']['patch_embedding']['kernel'] = jnp.array(value.permute(2, 3, 1, 0).numpy())
        # PyTorch: (hidden_size,) -> Flax: (hidden_size,)
        elif 'class_embedding' in key:
            vit_params['vision_model']['embeddings']['class_embedding'] = jnp.array(value.squeeze().numpy())
        # PyTorch: (num_pos, hidden_size) -> Flax: (1, num_pos, hidden_size)
        elif 'position_embedding.weight' in key:
            vit_params['vision_model']['embeddings']['position_embedding']['embedding'] = jnp.array(value.unsqueeze(0).numpy())
        # PyTorch Linear: (out, in) -> Flax Dense: (in, out)
        elif 'weight' in key_parts[-1] and 'layer_norm' not in key:
            # 递归地找到正确的字典位置
            current_level = vit_params
            for part in key_parts[:-1]:
                current_level = current_level[part]
            current_level['kernel'] = jnp.array(value.T.numpy())
        # LayerNorm/Bias: (size,) -> (size,)
        elif 'bias' in key_parts[-1] or 'layer_norm' in key:
            current_level = vit_params
            for part in key_parts[:-1]:
                current_level = current_level[part]
            current_level['bias' if 'bias' in key_parts[-1] else 'scale'] = jnp.array(value.numpy())

    # b) 加载自定义的 mapping_dim 层权重
    if 'mapping_dim.weight' in pt_state_dict:
        new_flax_params['mapping_dim']['kernel'] = jnp.array(pt_state_dict['mapping_dim.weight'].T.numpy())
    if 'mapping_dim.bias' in pt_state_dict:
        new_flax_params['mapping_dim']['bias'] = jnp.array(pt_state_dict['mapping_dim.bias'].numpy())

    print("✅ PyTorch weights successfully converted and loaded into Flax model.")
    
    # 将可变字典转换回 FrozenDict 并创建完整的状态
    return {'params': new_flax_params}


# --- 使用示例 ---
# if __name__ == '__main__':
#     # 1. 实例化Flax模型
#     flax_model = FlaxCLIPPretrainedTactileEncoderPi0()

#     # 2. 指定PyTorch检查点路径
#     ckpt_path = "/path/to/your/pretrained_tactile_encoder.pt"

#     # 3. 加载权重
#     # 这个函数会处理所有的转换逻辑
#     flax_state = load_pytorch_weights_in_flax(flax_model, ckpt_path)

#     # 4. 运行模型进行推理
#     # 创建一个随机的输入图像
#     # 注意JAX的输入格式是 (B, H, W, C)
#     dummy_tactile_image = jnp.ones((4, 224, 224, 3)) 

#     # 使用 `apply` 方法执行前向传播
#     output_features = flax_model.apply(flax_state, dummy_tactile_image)

#     print("Input shape:", dummy_tactile_image.shape)
#     print("Output features shape:", output_features.shape)
#     # 预期输出形状: (4, 256, 2048)
#     # (batch_size, num_patches, output_dim)
#     # num_patches = (224/14)^2 = 16^2 = 256