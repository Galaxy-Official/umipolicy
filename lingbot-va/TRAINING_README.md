# LingBot-VA (Handcap) 训练脚本与路径配置解析

这份说明针对 `lingbot-va` 目录下的多模态模型 (Wan-VA架构) 训练进行了详细且清晰的梳理，主要聚焦于：如何启动训练、预训练权重位置，以及训练结果（日志和模型）的落盘位置。

## 1. 训练脚本如何启动？

LingBot 的训练使用了标准的多卡 `torch.distributed.run` (即 `torchrun`) 进行。为了方便配置，我们使用了一系列封装好的 Shell 脚本。针对专门采集的 `handcap` 任务（带有触觉、手腕多相机等），推荐的启动方法是调用 `script/run_va_posttrain_handcap.sh`。

**启动步骤：**
```bash
# 1. 切换至 lingbot-va 根目录
cd /Users/macbookpro/Desktop/workspace/umipolicy/lingbot-va

# 2. 赋予执行权限 (如需)
chmod +x script/run_va_posttrain_handcap.sh

# 3. 环境变量与一键启动
# 默认使用 8 卡训练 (NGPU=8)，您可以指定实际可用的卡数 (例如 NGPU=2)
# 默认使用 `handcap_train` 的配置文件
NGPU=2 bash script/run_va_posttrain_handcap.sh
```

**底层执行的 Python 脚本：**
该命令实质上拉起了：
`python -m torch.distributed.run ... -m wan_va.train_handcap --config-name handcap_train`

## 2. 预训练权重在什么路径？

在基于 Diffusion/VAE 的大架构训练时，极少会随机初始化重头训练，系统通常会去加载一个开源或已经完成第一阶段训练的基础大模型权重（Base Model）。

**预训练权重的默认路径：**
* **相对路径**：`./ckpt/lingbot-va-base`  (即 `lingbot-va/ckpt/lingbot-va-base`)

**如何修改？**
如果您更换了权重，或者想指到别的路径，可以在文件 `wan_va/configs/va_handcap_cfg.py` 中修改对应字段：
```python
# 第11行左右
va_handcap_cfg.wan22_pretrained_model_name_or_path = "./ckpt/lingbot-va-base"
```
这个路径内应包含标准 HuggingFace 格式的文件（如 `config.json`, 模型 `.safetensors` 文件等）。

## 3. 模型输出（日志等）保存在哪个目录？

训练产生的一般性输出文件（包括生成的实时 Profiler 图表如 `train_profile.md` 数据等）的根目录通过**共享配置 (`shared_config.py`)** 控制。

**默认输出路径：**
* **相对路径**：`./train_out` (即 `lingbot-va/train_out`)

您可以通过编辑 `wan_va/configs/shared_config.py` 文件中的第11行进行修改：
```python
va_shared_cfg.save_root = './train_out'
```

## 4. 最终的模型 (Checkpoints) 保存在什么地方？

真正能用来执行下游预测、被评估系统读取的权重文件 (Checkpoints) 会挂载于之前提及的 `save_root` 下的专属子文件夹中。

**Checkpoint 具体路径：**
* **相对路径**：`./train_out/checkpoints/`  (即 `lingbot-va/train_out/checkpoints/`)

**保存逻辑：**
* 在 `wan_va/train_handcap.py` 脚本内部运行逻辑里，系统会自动在这里创建文件夹。
* 按照配置，通常默认依据 `save_interval` (默认配置是 1000 steps 保存一次)，会在 `checkpoints/` 下方按照迭代 Step 数生成形如 `step_1000/`, `step_2000/` 等的子目录。
* 每完成一个阶段，就可以直接将这些子文件夹作为新模型的目录提供给推断脚本使用。
