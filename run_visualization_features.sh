#!/bin/bash

# 流体动力学模拟系统可视化功能运行脚本
echo "===== 流体动力学模拟系统高级可视化功能部署 ====="

# 确保在项目根目录下运行
if [ ! -f "src/main.py" ]; then
    echo "错误：请在项目根目录下运行此脚本"
    exit 1
fi

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 确保C++扩展已编译
echo -e "\n===== 检查并编译C++扩展 ====="
python scripts/build_extensions.py

# 添加高级知识库内容
echo -e "\n===== 添加高级知识库内容 ====="
python scripts/add_advanced_knowledge.py

# 添加涡环碰撞教程
echo -e "\n===== 添加涡环碰撞教程 ====="
python scripts/add_tutorials.py

# 启动应用服务器
echo -e "\n===== 启动应用服务器 ====="
echo "应用将在 http://localhost:8000 上运行"
echo "新功能包括:"
echo "  - 高级流体可视化（流线和粒子追踪）"
echo "  - 暗黑模式支持"
echo "  - 改进的预设场景选择界面"
echo "  - 涡环碰撞模拟场景"
echo -e "\n按 Ctrl+C 停止服务器"
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload 