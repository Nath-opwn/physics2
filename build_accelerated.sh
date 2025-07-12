#!/bin/bash

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${GREEN}开始构建多相流加速模块...${NC}"

# 检查依赖项
echo -e "${YELLOW}检查依赖项...${NC}"

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}错误: 未找到CMake，请安装CMake后重试${NC}"
    exit 1
fi

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}警告: 未找到NVCC，将禁用CUDA支持${NC}"
    CUDA_AVAILABLE=0
else
    echo -e "${GREEN}找到CUDA: $(nvcc --version | head -n1)${NC}"
    CUDA_AVAILABLE=1
fi

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python3，请安装Python3后重试${NC}"
    exit 1
fi

# 检查pybind11
python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}未找到pybind11，正在安装...${NC}"
    pip install pybind11
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 安装pybind11失败${NC}"
        exit 1
    fi
fi

# 创建构建目录
echo -e "${GREEN}创建构建目录...${NC}"
mkdir -p build
cd build

# 配置CMake
echo -e "${GREEN}配置CMake...${NC}"
cmake ../src/core/cuda_accelerated

# 构建
echo -e "${GREEN}构建加速模块...${NC}"
cmake --build . -j $(nproc)

# 安装
echo -e "${GREEN}安装加速模块...${NC}"
cmake --install .

# 返回到项目根目录
cd ..

# 检查是否成功构建
if [ -f "multiphase_core.so" ] || [ -f "multiphase_core.pyd" ]; then
    echo -e "${GREEN}构建成功!${NC}"
    echo -e "${GREEN}可以使用以下命令运行基准测试:${NC}"
    echo -e "${YELLOW}python src/benchmark/benchmark_multiphase.py${NC}"
else
    echo -e "${RED}构建失败!${NC}"
    exit 1
fi 