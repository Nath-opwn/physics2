#!/bin/bash
# 构建和安装CUDA扩展

# 确保在脚本所在目录执行
cd "$(dirname "$0")"

echo "检查CUDA环境..."
if [ -z "$CUDA_HOME" ]; then
    # 尝试查找CUDA路径
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
        echo "设置CUDA_HOME为: $CUDA_HOME"
    else
        echo "未找到CUDA安装，请设置CUDA_HOME环境变量"
        exit 1
    fi
fi

# 检查pybind11是否安装
echo "检查依赖项..."
python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装pybind11..."
    pip install pybind11
fi

# 检查cupy是否安装
python3 -c "import cupy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装cupy..."
    pip install cupy
fi

echo "开始构建CUDA扩展..."
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "CUDA扩展构建成功!"
    echo "运行测试..."
    python3 test_cuda.py
else
    echo "CUDA扩展构建失败，请检查错误信息"
    exit 1
fi 