# 训练记录与参数分析: `simple_sorting_dp_0409.sh`

这份分析报告总结了针对 `simple_sorting` 任务的 Diffusion Policy 训练脚本中的核心参数、硬件配置、模型架构及训练策略。

## 1. 硬件资源与分布式训练配置
- **使用显卡**: 2 张 (CUDA 0,1)
- **训练加速器**: Hugging Face `accelerate` (`--multi_gpu`, `--num_processes=2`)
- **精度与编译**: 使用了 `bf16` 混合精度和 PyTorch 2.0 的 `inductor` 编译后端（这能极大提升训练吞吐量和显存利用率）。
- **Dataloader 并发**: 10 个 Workers (`--num_workers=10`)，并指定使用 `pyav` 作为视频解码后端。

> [!TIP]
> **全局 Batch Size 说明**：LeRobot 默认采用单卡 Batch Size。脚本中设定了 `--batch_size=256`，配合 2 张卡，实际的**全局 Batch Size 为 512**。这是一个相当大的 Batch Size，需要确保学习率（Learning Rate）能与其匹配。

## 2. 数据集与任务设定
- **数据集根目录**: 本地路径 `Data/handcap_simple_sorting_409`
- **Hugging Face Repo ID**: `lihongcs/simple_sorting_handcap`
- **额外输入**: 禁用了力觉 (`use_force=false`) 和触觉 (`use_tactile=false`) 传感器，但开启了特定的 Handcap 标志 (`--use_handcap=true`)。

## 3. 策略模型与架构 (Diffusion Policy)
- **策略类型**: 扩散模型 (`diffusion`)
- **视觉主干网络 (Backbone)**: `resnet18`
- **预训练权重**: 未从互联网直接拉取 ImageNet 权重，而是使用了本地存放的权重 `ckpt/resnet18-f37072fd.pth`。
- **归一化层修改**: 明确禁用了 Group Norm (`--policy.use_group_norm=false`)。通常 Diffusion Policy 推荐用 Group Norm 来替代 Batch Norm 以稳定小 Batch 下的训练，这里禁用可能意味着主干网络保留了原始的 Batch Norm。

## 4. 优化器与训练周期
- **优化器**: AdamW (`--optimizer.type=adamw`)
- **总训练步数**: 200,000 步 (`--steps=200000`)
- **验证与保存频率**: 每 10,000 步评估一次并保存 Checkpoint (`--save_freq=10000`, `--eval_freq=10000`)。
- **日志记录频率**: 每 100 步输出一次日志 (`--log_freq=100`)。

## 5. 日志与存储管理
- **全局输出目录**: `outputs/train/simple_sorting_0409_1`
- **WandB**: 启用了离线模式 (`--wandb.mode="offline"`)，日志不会实时上传到云端，而是存在本地的 `wandb` 文件夹中。
- **防止覆盖保护与断点续训 (Resume)**: 
  - 脚本内置了**交互式检测逻辑**。若发现同名输出目录，会阻塞并在终端询问用户是否进行断点续训 (输入 `r`)。
  - 如果用户选择重新开始，脚本会**自动删除旧的输出文件夹**（清理“空壳残骸”），防止残留的历史 Checkpoint 干扰全新的训练过程。

## 结论与建议
1. **显存压力**: `ResNet18` 加上单卡 `256` 的 Batch Size 在通常的 24GB/40GB 显存上压力极大。如果遇到 OOM (Out of Memory)，请优先降低 `--batch_size`。
2. **训练时间估算**: 20w 步是一个非常完整的收敛周期。对于 `simple_sorting` 这类相对简单的任务，模型可能在 5w-10w 步之间就已经具备了不错的表现，建议在推理时多测几个中期的 Checkpoint（例如 100000 步时的权重）。
3. **离线日志上传**: 训练完成后，记得使用 `wandb sync` 命令将本地离线收集的 Loss 曲线同步到 WandB 网页端以供进一步分析。
