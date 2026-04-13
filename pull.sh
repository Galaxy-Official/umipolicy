#!/bin/bash

echo "⬇️ [接收端] 开始从云端拉取最新代码..."

# 1. 安全保护：暂存服务器本地的临时修改
if [[ -n $(git status -s) ]]; then
    echo "⚠️ 检测到接收端本地有未提交的修改或新文件！"
    echo "🗂️ 正在将本地修改暂存 (git stash)..."
    # 💡 优化：加上 -u 参数，连同“未追踪的新文件”一起安全暂存
    git stash -u
    HAS_STASH=true
else
    HAS_STASH=false
fi

# 2. 拉取最新代码
echo "⬇️ 正在同步远端最新代码..."
# ⚠️ 关键修复：这里把 main 改成了 master ⚠️
git pull origin master

if [ $? -ne 0 ]; then
    echo "❌ 拉取失败！请检查网络或远端仓库状态。"
    exit 1
fi

# 3. 恢复服务器上的临时修改
if [ "$HAS_STASH" = true ]; then
    echo "📦 正在恢复您在接收端的临时修改 (git stash pop)..."
    git stash pop
    echo "⚠️ 提示：请留意是否有冲突警告。"
fi

echo "🎉 接收完成！当前机器代码已更新到最新版本。"