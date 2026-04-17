# LeRobot Handcap 训练脚本使用指南

本目录 (`handcap_training/`) 包含了使用最新的 **LeRobot 3.0** 架构训练策略模型（主要是 Diffusion Policy）的 Shell 启动脚本。

通过解析这里的脚本文件（如 `tool_to_cups_0415.sh`），以下是关于如何启动、数据挂载以及权重保存位置的详细说明。

## 1. 训练脚本如何启动？

训练利用了 Hugging Face 的 `accelerate` 库进行多卡分布式训练。所有的启动都在这些 `.sh` 脚本中预设好了参数，您只需要在终端执行对应文件即可。

**启动步骤：**
```bash
# 1. 切换至带有训练脚本的根目录 (或者项目目录)
cd /Users/macbookpro/Desktop/workspace/umipolicy/lerobot

# 2. 赋予执行权限 (仅初次需要)
chmod +x src/scripts/handcap_training/tool_to_cups_0415.sh

# 3. 直接运行脚本启动多卡训练
./src/scripts/handcap_training/tool_to_cups_0415.sh
```

**脚本内部核心参数解析：**
* `CUDA_VISIBLE_DEVICES=0,1`：指定使用了前两张 GPU 显卡。
* `accelerate launch --multi_gpu --num_processes=2`：通知 PyTorch 启动 2 个并行进程。
* `--dataset.root=Data/tool_to_cups_0415`：指定本地使用的数据集路径。

## 2. 预训练权重在什么路径？

在目前的脚本中（比如 `tool_to_cups_0415.sh`），模型配置了 `--policy.type=diffusion` 并且是在使用数据集进行**从头训练 (Training from Scratch)**。

如果您需要**基于预训练权重微调 (Fine-tuning)** 或者加载已经完成预训练的开源基础权重（如 UMI 权重或者先前训练中断的 Checkpoint），您需要在脚本中添加 `--resume` 或是指向预训练权重的参数（遵循 LeRobot API）：
```bash
# 加入此参数以加载预训练权重目录
--pretrained_policy_name_or_path=outputs/train/tool_to_cups_0415_1/checkpoints/last/pretrained_model
```
LeRobot 在本地寻找预训练权重时，通常是指向一个包含 `config.json` 和 `model.safetensors` 等文件的 Hugging Face 标准本地目录。

## 3. 模型保存在什么地方？(输出与 Checkpoints)

模型的全局输出目录由脚本中的 `--output_dir` 参数全权决定。
以 `tool_to_cups_0415.sh` 为例，参数配置为：
`--output_dir=outputs/train/tool_to_cups_0415_1`

**存储结构解读：**
当训练开始后，所有的日志、配置文件和模型权重都会被保存在 `lerobot/outputs/train/tool_to_cups_0415_1/` 下面。

它的子目录结构通常如下：
* 📂 **`outputs/train/tool_to_cups_0415_1/checkpoints/`**：这里就是**模型保存的具体位置**。
  * 根据您的配置，里头会有多个类似 `005000/`, `010000/`, `last/` 这样按 Step 数命名的文件夹。
  * 每一个 Checkpoint 文件夹内包含了可以直接拿来 Inference 的完整策略权重文件（`model.safetensors`）和模型配置文件。
* 📂 **`outputs/train/tool_to_cups_0415_1/wandb/`**（或者本地 log）：训练的 Loss 曲线和其他评估日志，由于设置了 `--wandb.mode="offline"`，这些日志将在这里本地存储。

## 4. 全局 Batch Size 注意事项
在 LeRobot 的底层实现中，`--batch_size` 代表的是**单卡 (Per-GPU) Batch Size**。
如果您设置 `--batch_size=256` 且 `--num_processes=2`，那么每次全局迭代其实会吃掉总计 `256 * 2 = 512` 个样本。如果要维持原有的吞吐逻辑和学习率对齐，建议按总卡数除以目标全局 Batch size 来设置脚本里的单卡数字。
