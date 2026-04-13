#!/bin/bash

echo "⬇️ [接收端] 开始从云端拉取最新代码..."

# 1. 安全检查：如果接收端本地有未提交的临时修改，先保护起来
if [[ -n $(git status -s) ]]; then
    echo "⚠️ 检测到接收端本地有未提交的修改！"
    echo "🗂️ 为了防止代码冲突，正在将本地修改暂存 (git stash)..."
    git stash
    HAS_STASH=true
else
    HAS_STASH=false
fi

# 2. 拉取最新代码
echo "⬇️ 正在同步远端最新代码..."
git pull origin main

if [ $? -ne 0 ]; then
    echo "❌ 拉取失败！请检查网络或远端仓库状态。"
    exit 1
fi

# 3. 如果刚才暂存了临时修改，现在恢复它们
if [ "$HAS_STASH" = true ]; then
    echo "📦 正在恢复您在接收端的临时修改 (git stash pop)..."
    git stash pop
    echo "⚠️ 提示：如果刚才拉取的新代码和你的临时修改在同一行，可能会提示冲突，请注意检查。"
fi

echo "🎉 接收完成！当前机器代码已更新到最新版本。"