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

# 4. 指定开放的内部主节点通讯端口 (防止与其他训练任务冲突)
MASTER_PORT=12341

echo "================================================================================"
echo "🚀 正在启动 Cosmos Policy 训练..."
echo "📂 数据集基准路径: $BASE_DATASETS_DIR"
echo "🔧 GPU 调度数量: $NUM_GPUS"
echo "🧪 加载的配置卡: $EXPERIMENT_NAME"
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

# 7. 自动修复服务器缺失 GUI 库 (libGL.so.1) 导致的 OpenCV 导入错误
if ! python -c "import cv2" 2>/dev/null; then
    echo "❌ 检测到 cv2 导入失败 (通常是因为服务器缺失 libGL.so.1 基本库)"
    echo "正在自动将 opencv-python 替换为服务器专用的无头版本 opencv-python-headless..."
    python -m pip uninstall -y opencv-python opencv-contrib-python
    python -m pip install opencv-python-headless
fi

# 8. 直接拉起 torchrun (因为您已经在 cosmos-policy 的 conda 环境中了)
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="${EXPERIMENT_NAME}"
