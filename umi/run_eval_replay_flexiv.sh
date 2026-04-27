#!/bin/bash
# run_eval_replay_flexiv.sh
# 这个脚本用于读取 UMI 格式的 Zarr 数据集，并在 Flexiv 真实机械臂上进行开环回放。

# 获取当前工作目录的上级目录 (用于动态定位)
BASE_DIR=$(pwd)
ZARR_PATH="data/yellow_to_pink_block.zarr"

# 检查数据集是否存在
if [ ! -f "$ZARR_PATH" ] && [ ! -d "$ZARR_PATH" ]; then
    echo "错误：找不到 Zarr 数据集，请检查路径: $ZARR_PATH"
    exit 1
fi

echo "=========================================="
echo "🚀 准备启动 UMI Zarr 数据集回放"
echo "📂 数据集路径: $ZARR_PATH"
echo "=========================================="

# 默认回放第 0 个 Episode，如果传入参数则使用参数
EPISODE_IDX=${1:-0}
echo "▶️ 将回放 Episode Index: $EPISODE_IDX"

# --- 环境变量配置 (兼容 RDK 1.0+) ---
export FLEXIV_ROBOT_IP="192.168.2.100"
export FLEXIV_ROBOT_SN="Rizon4-062339"
export FLEXIV_GRIPPER_NAME="Flexiv-GN01"
export FLEXIV_LOCAL_IP="192.168.2.102"

# 设置真实的初始位姿参数 (7个关节角度，单位：弧度)，机械臂启动时会先自动运动到该安全位姿
export FLEXIV_INIT_POSE="[-0.0009,-0.1701,-0.0133,2.0214,-0.0058,0.6921,-0.0012]"
echo "▶️ 设置机械臂初始位姿: $FLEXIV_INIT_POSE"

# 启动脚本
python scripts_real/eval_replay_flexiv.py \
    --data_root "$ZARR_PATH" \
    --episode_index "$EPISODE_IDX" \
    --ctrl_freq 5 \
    --task_name "umi_replay"
