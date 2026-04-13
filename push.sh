#!/bin/bash

echo "🚀 [发送端] 开始将本地代码同步到云端..."

# 1. 检查本地是否有修改
if [[ -n $(git status -s) ]]; then
    echo "📦 检测到本地有修改，正在记录并提交..."
    git add .
    git commit -m "Auto send: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "✅ 本地暂无新修改需要提交。"
fi

# 2. 无论有没有修改，推送前先拉取合并（防冲突黄金法则）
echo "⬇️ 正在拉取远端最新代码并合并..."
git pull --rebase origin main

if [ $? -ne 0 ]; then
    echo "❌ 拉取失败！可能产生了代码冲突，请手动解决！"
    exit 1
fi

# 3. 推送本地代码到远端
echo "⬆️ 正在推送到远端服务器..."
git push origin main

if [ $? -eq 0 ]; then
    echo "🎉 发送完成！云端代码已是最新。"
else
    echo "❌ 推送失败！请检查网络或 Gitee 权限配置。"
fi