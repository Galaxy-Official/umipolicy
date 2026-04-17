#!/bin/bash

echo "🚀 [发送端] 开始将本地代码同步到云端..."

# 1. 检查本地是否有修改
if [[ -n $(git status -s) ]]; then
    echo "📦 检测到本地有修改，正在检查大容量文件..."
    git add .
    
    # 查找暂存区刚加入的真实存在的文件 (排除仅做删除的文件)
    IFS=$'\n'
    staged_files=($(git diff --cached --name-only --diff-filter=d))
    unset IFS

    for file in "${staged_files[@]}"; do
        if [ -f "$file" ]; then
            # 兼容 Mac(-f%z) 和 Linux(-c%s) 的文件体积获取指令
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
            if [ -n "$size" ] && [ "$size" -gt 1048576 ]; then
                # 计算出 MB 为单位的值
                size_mb=$(awk "BEGIN {printf \"%.2f\", $size/1048576}")
                
                # < /dev/tty 保障在执行脚本时输入流不会断开
                read -p "⚠️ 警告: 准备拉取上传的文件 '$file' 很巨大 (${size_mb} MB)！是否确认将它上传到 Git 云端? (y/N): " choice < /dev/tty
                case "$choice" in
                    y|Y|yes|Yes ) 
                        echo "✅ 保留上传: $file"
                        ;;
                    * ) 
                        echo "🚫 已从本次提交流程中剔除该文件 (使其退回到未跟踪状态)。"
                        git restore --staged "$file"
                        ;;
                esac
            fi
        fi
    done

    # 检查踢出大文件后，是否还有剩余文件需要做 commit
    if [[ -n $(git diff --cached --name-only) ]]; then
        git commit -m "Auto send: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "✅ 大文件上传被取消完毕，目前已不存在其它修改变动的文件，跳过本次 Commit 动作。"
    fi
else
    echo "✅ 本地暂无新修改需要提交。"
fi

# 2. 无论有没有修改，推送前先拉取合并
echo "⬇️ 正在拉取远端最新代码并合并..."
# 这里改成了 master 👇
git pull --rebase origin master

if [ $? -ne 0 ]; then
    echo "❌ 拉取失败！可能产生了代码冲突，请手动解决！"
    exit 1
fi

# 3. 推送本地代码到远端
echo "⬆️ 正在推送到远端服务器..."
# 这里也改成了 master 👇
git push origin master

if [ $? -eq 0 ]; then
    echo "🎉 发送完成！云端代码已是最新。"
else
    echo "❌ 推送失败！请检查网络或配置。"
fi