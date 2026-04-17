# OpenPI 训练脚本说明与路径分配机制

这份说明针对 `openpi` 架构的训练管理机制进行了梳理。相比起 LeRobot 或 LingBot 采用 `accelerate` 或者分布式框架的显式命令拉起多卡训练，OpenPI 原生利用了 JAX 框架特性 (或者专门为 PyTorch 打造的适配器)，因此启动与路径映射上有其独有的特征。

## 1. 训练脚本如何启动？

针对 Handcap 单独存放的启动 Shell 文件（位于 `scripts/handcap_training/` 目录中），如 `simple_sorting_pi0_0409.sh`。

**启动步骤：**
需要确保当前位于 `openpi` 根目录下执行：
```bash
cd /Users/macbookpro/Desktop/workspace/umipolicy/openpi

# 给定执行权限
chmod +x scripts/handcap_training/simple_sorting_pi0_0409.sh

# 一键启动训练
./scripts/handcap_training/simple_sorting_pi0_0409.sh
```

**脚本机制拆解：**
* **归一化统计数据 (Norm Stats) 计算**：在跑主训练前，脚本默认会运行一刀 `python scripts/compute_norm_stats.py --config-name pi0_simple_sorting`，自动为当前数据集提前计算并落盘所有的归一化边界值。
* **显式传入配置与实验名**：真正的训练程序由 `python scripts/train.py` 拉起（由于是 JAX，多卡 FSDP 会自动按照 `--fsdp-devices 1` 与环境变量自行扩展）。

## 2. 预训练的权重在什么路径？

预训练的 Base Model 位置在 OpenPI 配置中是以**组件参数**形式直接写合在代码里的！所有的具体路径可以通过查验注册配置文件得知。

以 `pi0_simple_sorting` 为例（其配置写在 `src/openpi/training/handcap_config.py`）：
* **主模型权重路径 (Base Params)**：预设的是绝对挂载地址 `/inspire/hdd/project/.../openpi/ckpt/pi0_base/params`（如果是 Pi0.5 则是 `pi05_base/params`）。
* **触觉模块预训练权重 (Tactile Encoder)**：如果用到了 tactile，还会自动外挂一个由 `tactile_pretrained_ckpt` 参数指定的附加权重 `.../pretrained_tactile_encoder.pt`。

如果项目有迁移（比如下载到了 Mac 本地 `openpi` 下），您需要去 `handcap_config.py` 这个文件里将硬编码写死的硬盘路径转换成属于您的**本地相对路径**或绝对路径。

## 3. 模型保存在什么地方？(及相关输出)

OpenPI 会整合日志（如未开启 wandb 会打印与记录本地）和模型文件，并完全遵照预制参数 `exp-name` 将它们聚合在一个统一目录底下。

由于我们在外层调用的参数是 `exp-name dp_simple_sorting_0409_handcap` 且 `--config-name pi0_simple_sorting`：
* **全局保存相对根目录**：所有的输出和持久化对象将被统一保存在 `openpi/checkpoints/` 
* **您的本次训练的落盘路径**：将会被串联在一起，存放在：
  `openpi/checkpoints/pi0_simple_sorting/dp_simple_sorting_0409_handcap/`

上述文件夹内会由 Checkpoint Manager 按规律截取步骤（如 `step_5000/`, `step_10000/`等），自动保存成符合 `nnx` 反序列化特征的模型参数块！
