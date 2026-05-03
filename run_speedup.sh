#!/bin/bash

# 退出遇到错误时终止
set -e

# 检查输入参数
if [ -z "$1" ]; then
    echo "使用方法: bash run_speedup.sh <视频文件路径或文件夹> [加速倍数]"
    echo ""
    echo "示例 1 (处理单个视频, 5倍速):"
    echo "  bash run_speedup.sh my_video.mp4 5"
    echo ""
    echo "示例 2 (处理文件夹下的所有 mp4, 10倍速):"
    echo "  bash run_speedup.sh 'my_folder/*.mp4' 10"
    exit 1
fi

DATA_PATH="$1"
SPEED="${2:-5.0}" # 如果没提供第二个参数，默认 5 倍速

echo "===================================="
echo "🚀 开始视频加速处理"
echo "📂 输入路径: ${DATA_PATH}"
echo "⏩ 加速倍率: ${SPEED}x"
echo "===================================="

# 执行 Python 加速脚本 (允许通配符展开，所以不加双引号)
python speedup_video.py $DATA_PATH --speed ${SPEED}

echo "✅ 处理完成！加速后的视频已保存在原目录下。"
