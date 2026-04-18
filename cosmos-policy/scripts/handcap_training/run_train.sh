#!/bin/bash
# ==============================================================================
# Cosmos Policy: 通用训练启动脚本
# ==============================================================================

# 1. 设置系统所使用的 GPU 数量 (单机多卡)
NUM_GPUS=2

# 2. 设置您的预处理数据集所在的根目录
# (注: 下一级的目录应该是您转换好的类似于 ALOHA 或者 Handcap 的全量数据集)
export BASE_DATASETS_DIR="data/handcap_cosmos"

# 3. 指定您想要跑的实验配置名称 (来源于 cosmos_policy/config/experiment/)
# 我们创建了新的 cosmos_predict2_handcap 用于处理类似 ALOHA HDF5 结构的 handcapcosmos:
EXPERIMENT_NAME="cosmos_predict2_handcap"

# 4. 指定您的数据集的全局语言指令 (Prompt)
# 这句话将被嵌入并替代未知的文件夹名，传给文本编码大模型
TASK_PROMPT="put box and block into paper box"

# 5. 指定开放的内部主节点通讯端口 (防止与其他训练任务冲突)
MASTER_PORT=12341

echo "================================================================================"
echo "🚀 正在启动 Cosmos Policy 训练..."
echo "📂 数据集基准路径: $BASE_DATASETS_DIR"
echo "🔧 GPU 调度数量: $NUM_GPUS"
echo "🧪 加载的配置卡: $EXPERIMENT_NAME"
echo "🗣️  文本语言指令: $TASK_PROMPT"
echo "================================================================================"

# 5. 解决 Transformer Engine 在非 Docker 环境下的 ldconfig -p 报错bug
echo "正在搜索最佳的 libnvrtc.so 路径以修复 TransformerEngine..."
NVRTC_PATH=$(find $PWD/.venv/lib -name "libnvrtc.so*" | grep -v "stubs" | grep -v "builtins" | sort -r | head -n 1)
if [[ -z "$NVRTC_PATH" ]]; then
    NVRTC_PATH=$(find /usr/local/cuda*/lib64 -name "libnvrtc.so*" | grep -v "stubs" | grep -v "builtins" | sort -r | head -n 1)
fi
echo "✅ 最终选定的 NVRTC_PATH: $NVRTC_PATH"

if [[ -n "$NVRTC_PATH" ]]; then
    export LD_LIBRARY_PATH="$(dirname $NVRTC_PATH):$LD_LIBRARY_PATH"
    export LD_PRELOAD="$NVRTC_PATH"
    mkdir -p .tmp_bin
    cat << EOF > .tmp_bin/ldconfig
#!/bin/bash
if [[ "\$*" == *"-p"* ]]; then
    /sbin/ldconfig -p 2>/dev/null
    echo "	libnvrtc.so.11.2 (libc6,x86-64) => \$NVRTC_PATH"
    echo "	libnvrtc.so.12 (libc6,x86-64) => \$NVRTC_PATH"
    echo "	libnvrtc.so (libc6,x86-64) => \$NVRTC_PATH"
    exit 0
fi
/sbin/ldconfig "\$@"
EOF
    chmod +x .tmp_bin/ldconfig
    export PATH="$PWD/.tmp_bin:$PATH"
fi

# 6. 修复 Triton 编译时缺失 Python.h (No such file or directory) 的问题
echo "正在搜寻 Python.h 头文件路径用于 Triton 编译..."
PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
if [[ ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]]; then
    PYTHON_INCLUDE_DIR=$(find /root/miniforge3/envs /opt/conda -name "Python.h" 2>/dev/null | grep "python3" | head -n 1 | xargs dirname)
fi
if [[ -n "$PYTHON_INCLUDE_DIR" ]]; then
    export C_INCLUDE_PATH="$PYTHON_INCLUDE_DIR:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE_DIR:$CPLUS_INCLUDE_PATH"
    echo "✅ 已注入 C_INCLUDE_PATH=$PYTHON_INCLUDE_DIR"
fi

# 7. 自动修复由容器网络造成的 NCCL 死锁超时问题 (30分钟卡死)
echo "正在注入 NCCL 防死锁环境变量..."
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1  # 强制使死锁崩溃而不是干等30分钟
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 8. 彻底禁用 Wandb 联网登录 (防止 Rank 0 卡在输入 API Key 环节导致另一个卡等待 30 分钟)
export WANDB_MODE=offline
echo "✅ 已强制设定 WandB 离线模式"

# 9. 离线预处理文本指令的 T5 嵌入 (防止在训练主进程中多开导致极高的 CUDA OOM)
T5_CKPT_DIR="ckpt/google-t5/t5-11b"
echo "正在检查是否存在离线 T5 特征 ${BASE_DATASETS_DIR}/t5_embeddings.pkl ..."
if [[ ! -f "${BASE_DATASETS_DIR}/t5_embeddings.pkl" ]]; then
    echo "未检测到 T5 离线缓存，即将为您生成。请确保 ${T5_CKPT_DIR} 下存放了 t5-11b 权重!"
    PYTHONPATH=. python scripts/handcap_training/precompute_t5.py \
        --prompt "$TASK_PROMPT" \
        --data_dir "$BASE_DATASETS_DIR" \
        --model_name "$T5_CKPT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ T5 特征提取失败，停止训练。"
        exit 1
    fi
fi

# 9. 清理上一轮被 NaN 污染的特征状态文件，强制重新生成干净版本
echo "清理可能存在的僵尸 NaN 缓存..."
rm -vf "${BASE_DATASETS_DIR}/dataset_statistics_post_norm.json"

# 10. 直接拉起 torchrun (因为您已经在 cosmos-policy 的 conda 环境中了)
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="${EXPERIMENT_NAME}" \
  dataloader_train.dataset.default_command="${TASK_PROMPT}" \
  job.wandb_mode="offline" \
  dataloader_train.dataset.t5_text_embeddings_path="${BASE_DATASETS_DIR}/t5_embeddings.pkl"
