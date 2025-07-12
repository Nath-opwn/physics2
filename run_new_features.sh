#!/bin/bash

# 流体动力学模拟系统新功能运行脚本
echo "===== 流体动力学模拟系统新功能部署 ====="

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

# 添加高级知识库内容
echo -e "\n===== 添加高级知识库内容 ====="
python scripts/add_advanced_knowledge.py

# 添加涡环碰撞教程
echo -e "\n===== 添加涡环碰撞教程 ====="
python scripts/add_tutorials.py

# 确保C++扩展已编译
echo -e "\n===== 检查并编译C++扩展 ====="
python scripts/build_extensions.py

# 运行性能测试
echo -e "\n===== 运行性能测试 ====="
python scripts/performance_test.py

# 启动应用服务器
echo -e "\n===== 启动应用服务器 ====="
echo "应用将在 http://localhost:8000 上运行"
echo "按 Ctrl+C 停止服务器"
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload 