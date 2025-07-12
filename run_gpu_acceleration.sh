#!/bin/bash

# 流体动力学模拟系统 - GPU加速版本启动脚本

# 确保虚拟环境已激活
if [ -d ".venv" ]; then
    echo "正在激活虚拟环境..."
    source .venv/bin/activate
fi

# 检查依赖库
echo "检查依赖库..."
pip install -r requirements.txt

# 检查CUDA库是否可用
python -c "import cupy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: CuPy库不可用，无法使用GPU加速。"
    echo "请安装CuPy库: pip install cupy-cuda11x (根据您的CUDA版本选择合适的包)"
    echo "继续使用CPU版本运行..."
fi

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

# 运行性能测试
echo "运行性能测试..."
python scripts/performance_test.py --sizes "32,32,32,64,64,64" --steps 20 --output "gpu_performance.png"

# 设置环境变量以启用GPU加速
export USE_GPU=1

# 启动应用
echo "启动流体动力学模拟系统（GPU加速版本）..."
echo "访问 http://localhost:8000 使用系统"
echo "访问 http://localhost:8000/docs 查看API文档"
echo ""
echo "GPU加速功能:"
echo "- 使用CUDA加速的流体求解器"
echo "- 高性能计算模块"
echo "- 实时数据流与高级可视化"
echo ""
echo "按Ctrl+C停止服务"

# 启动服务器
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload 