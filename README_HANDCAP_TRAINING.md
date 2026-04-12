# UmiPolicy Handcap 训练与使用全指南

本说明文档汇总了 `umipolicy` 生态下三套不同的训练架构：**LeRobot**, **OpenPI** 和 **LingBot-VA**。
这三套代码库各自有独立的特性与底层逻辑（PyTorch / JAX / 潜变量扩散）。以下是关于数据流、训练脚本入口、以及关键优化参数调节的详细指南。

---

## 📂 1. 数据流向与保存位置

所有的 Handcap 原始数据均在 `/Users/macbookpro/Desktop/workspace/handcap/Postprocess` 下通过脚本转换（带有断点续传功能的 `resume` 版本推荐用于防崩溃断点保护）。

- **LeRobot 与 OpenPI 数据层**
  - **脚本**: `_02_combine_and_transfer_data_into_lerobot_30.py` 或 `_30_resume.py`
  - **结构**: 纯正的 LeRobot v3.0 / v2.1 格式。
  - **默认输出/指向**: 可以在相应配置集的 `repo_id` 或者直接通过绝对路径传递，例如 OpenPI 配置中的 `dataset={"name": "lerobot", "repo_id": "/inspire/hdd/.../Data/handcap30"}`。数据统一保存在统一的存储节点（通常位于外挂挂载盘或集群共享目录）。
- **LingBot-VA 专属数据层**
  - **脚本**: `_02_combine_and_transfer_data_into_lerobot_lingbot.py` 或 `_lingbot_resume.py`
  - **结构**: 在 LeRobot v3.0 基础上，为 `meta/episodes.jsonl` **自动注入了 `action_config`** 语言控制特征。
  - **前置硬性要求**: 必须经过 Wan2.2 VAE 提取视觉特征。转换后请在您的数据根目录下生成对应的 `latents/` 文件夹（存有 `.pth` 后缀特征字典）后，再喂给网络。

*(💡 建议在代码中所有的 `dataset_path`、`repo_id` 参数里采用**绝对路径**指向集群中的同一份数据集副本，以节约磁盘！)*

---

## 🚀 2. 各架构启动脚本及说明

### 架构 A：LeRobot (纯 PyTorch)
常用于标准的 Actor-Critic 或简易 Diffusion/ACT 策略实验。
- **训练启动脚本**: 
  - `lerobot/src/scripts/handcap_training/simple_sorting_dp_0409.sh`
- **底层执行**: 调用标准的 `lerobot.scripts.lerobot_train` 并通过 DDP 展开。

### 架构 B：OpenPI (基于 JAX / XLA 分布式)
用于训练前沿大一统策略大模型（如 $\pi_0$, $\pi_{0.5}$）。因为底层是 JAX，编译极快，吞吐很高。
- **训练启动脚本**: 
  - `openpi/scripts/handcap_training/simple_sorting_pi0_0409.sh` 
  - `openpi/scripts/handcap_training/simple_sorting_pi05_0409.sh`
- **配置注册地**: `openpi/src/openpi/training/handcap_config.py`

### 架构 C：LingBot-VA (视觉-动作 潜变量扩散)
针对超长上下文的长序列任务有着极其恐怖的泛化性。
- **训练启动脚本**: 
  - `lingbot-va/script/run_va_posttrain_handcap.sh`
- **配置注册地**: `lingbot-va/wan_va/configs/va_handcap_train_cfg.py`

---

## ⚙️ 3. 核心优化参数与调节指南

三套架构底层和计算图均不同，遇到不同的泛化收敛或显存瓶颈时，调节的方向和入口也有所区别：

### 【LeRobot 优化指南】
- **修改位置**: YAML 配置文件或 bash 命令的重载传参。
- **关键参数**:
  1. `training.batch_size`: 决定显存占用和梯度稳定性。默认单卡多为 `4-16` 之间。如果在 LeRobot 中爆显存（OOM），首选下调此参数。
  2. `training.lr` (Learning Rate): 默认通常在 `1e-4` 到 `1e-5`。若收敛过慢可以上调，若 loss 震荡或发散则考虑调小。
  3. `training.grad_clip_norm`: 防止梯度爆炸的截断阈值，对于不稳定任务建议设定在 `1.0` 左右。

### 【OpenPI 优化指南 - JAX】
- **修改位置**: `openpi/src/openpi/training/handcap_config.py` 内注册的 `TrainConfig` 对象。
- **关键参数**:
  1. `batch_size`: 这里的 batch size 表示的是 **Global Batch Size** (全局)。因为您通常使用 2 卡训练，之前建议总数为 128 (单卡分担 64)。如果显存够大可拉高至 256 增加数据吞吐率。注意：JAX 是提前预分配并编译所有显存的 (XLA 原理)。
  2. `max_steps`: 训练步数。$\pi_0$ 收敛极快，由于它是 Fine-tune 预训练基础模型，如果数据集不大（只有几百个 Episodes），`max_steps` 设为 `20,000` - `50,000` 即可，太高会严重过拟合！
  3. `learning_rate` / `warmup_steps`: 遵循余弦退火。大模型微调（全参或 LoRA），学习率建议在 `1e-5` 量级以内，`warmup` 建议调到 `1000` 步让预训练分布平滑过渡衰减。

### 【LingBot-VA 优化指南 - Latent Diffusion】
- **修改位置**: `lingbot-va/wan_va/configs/va_handcap_train_cfg.py`
- **关键参数**:
  1. `batch_size` & `gradient_accumulation_steps` (累加步): 潜变量由于是全局整存，非常占显存。通常单卡 bs 只能设 `1` 到 `2`。为了保证大 Batch 训练效果（官方推荐 Global Batch 达到 32 或 64），你必须相应拉高 `gradient_accumulation_steps`（比如设定为 4 或 8）。
  2. `cfg_prob` (Classifier-Free Guidance Probability)：这个是 LingBot 的特色参数。默认是 `0.1` 左右。它代表在训练中抹除文本（Text Embedding 替换为空）的概率。如果您的动作文本指令只有简单的 "perform task"，不需要过于依赖语言泛化，这个可以维持或稍低。如果您指望模型强依赖语言区分不同动作，可调整。
  3. `attn_window` & `frame_chunk_size` (在 `va_handcap_cfg.py` 内): 极深层的底层结构参。如果长程序列长（例如视频很长），可以尝试改动，但对于资源占用极为敏感。普通任务建议保持 72 / 2 的原比例。
  4. `learning_rate`: 因涉及到复杂的扩散步降噪预测模型，初始推荐为极低的 `1e-5`。由于参数量极大，配合 `weight_decay = 0.1` 缓解过拟合。

---

> [!TIP] 
> **调试防爆显存最佳实践**
> 1. 先用测试网运行一次：启动只分配 1 块 GPU（设 CUDA_VISIBLE_DEVICES），并在命令里 override `batch_size=1`。
> 2. `LeRobot` 会因为视频序列直接过模型导致极高的显存峰值（可以调整帧数或者降低图像分辨率）。
> 3. `OpenPI` 因为是 JAX，它一启动会“锁”死整张卡的显存，别怕，只是预分配并不代表实际 OOM。如果真的中途触发 `ResourceExhausted`，才需要真正下调 XLA 的 batch 参数。
> 4. `LingBot` 则高度依赖您在转换代码阶段 `fps` 这个参数抽出来的视频 Latent 长度。视频潜序列越长，占用空间按几何倍数提升。

