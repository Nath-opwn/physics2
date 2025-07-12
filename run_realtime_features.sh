#!/bin/bash

# 流体动力学模拟系统 - 实时数据流与高级可视化功能启动脚本

# 确保虚拟环境已激活
if [ -d ".venv" ]; then
    echo "正在激活虚拟环境..."
    source .venv/bin/activate
fi

# 检查依赖库
echo "检查依赖库..."
pip install -r requirements.txt

# 确保必要的目录存在
echo "创建必要的目录..."
mkdir -p data
mkdir -p exports
mkdir -p static/js
mkdir -p static/css

# 初始化数据库
echo "初始化数据库..."
python -m src.init_db

# 填充知识库内容
echo "填充知识库内容..."
if [ -f "scripts/populate_knowledge.py" ]; then
    python scripts/populate_knowledge.py
fi

if [ -f "scripts/add_advanced_knowledge.py" ]; then
    python scripts/add_advanced_knowledge.py
fi

if [ -f "scripts/add_tutorials.py" ]; then
    python scripts/add_tutorials.py
fi

if [ -f "scripts/add_turbulence_knowledge.py" ]; then
    python scripts/add_turbulence_knowledge.py
fi

# 启动应用
echo "启动流体动力学模拟系统（实时数据流与高级可视化功能）..."
echo "访问 http://localhost:8000 使用系统"
echo "访问 http://localhost:8000/docs 查看API文档"
echo ""
echo "新增功能:"
echo "- WebSocket实时数据流"
echo "- 高级可视化API（流线、粒子追踪、等值面）"
echo "- 增强的数据导出功能（VTK、HDF5、批量导出）"
echo ""
echo "按Ctrl+C停止服务"

# 启动服务器
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload 