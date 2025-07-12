#!/bin/bash

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
function print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
function check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 命令未找到，请安装后再试。"
        exit 1
    fi
}

# 检查必要的命令
check_command cmake
check_command python3

# 创建并激活虚拟环境
print_info "创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

print_info "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
print_info "安装Python依赖..."
pip install numpy matplotlib pybind11 scipy

# 创建构建目录
print_info "创建构建目录..."
mkdir -p build
cd build

# 检查CUDA是否可用
if command -v nvcc &> /dev/null; then
    print_info "检测到CUDA，启用CUDA支持..."
    CUDA_AVAILABLE=1
else
    print_warning "未检测到CUDA，禁用CUDA支持..."
    CUDA_AVAILABLE=0
fi

# 配置CMake
print_info "配置CMake..."
cmake ../src/core/cuda_accelerated

# 编译
print_info "编译中..."
cmake --build . -j$(nproc)

if [ $? -ne 0 ]; then
    print_error "编译失败！"
    deactivate
    exit 1
fi

print_success "编译成功！"

# 复制生成的库文件到项目根目录
print_info "复制库文件..."
find . -name "multiphase_core*.so" -exec cp {} ../ \;

# 运行测试
print_info "运行基本加速测试..."
cd ..
python test_acceleration.py

if [ $? -ne 0 ]; then
    print_error "基本加速测试失败！"
    deactivate
    exit 1
fi

print_success "基本加速测试通过！"

# 运行表面张力测试
print_info "运行表面张力测试..."
python test_surface_tension.py

if [ $? -ne 0 ]; then
    print_error "表面张力测试失败！"
    deactivate
    exit 1
fi

print_success "表面张力测试通过！"

# 运行接触角测试
print_info "运行接触角测试..."
python test_contact_angle.py

if [ $? -ne 0 ]; then
    print_error "接触角测试失败！"
    deactivate
    exit 1
fi

print_success "接触角测试通过！"

print_success "所有测试通过！"

# 生成性能报告
print_info "生成性能报告..."
echo "# 多相流模型性能报告" > performance_report.md
echo "## 测试环境" >> performance_report.md
echo "- 操作系统: $(uname -a)" >> performance_report.md
echo "- CPU: $(cat /proc/cpuinfo | grep 'model name' | uniq | cut -d':' -f2 | sed 's/^ //')" >> performance_report.md
echo "- 内存: $(free -h | grep Mem | awk '{print $2}')" >> performance_report.md

if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> performance_report.md
    echo "- CUDA版本: $(nvcc --version | grep release | awk '{print $6}')" >> performance_report.md
else
    echo "- GPU: 不可用" >> performance_report.md
    echo "- CUDA版本: 不可用" >> performance_report.md
fi

echo "" >> performance_report.md
echo "## 性能测试结果" >> performance_report.md
echo "### 基本加速测试" >> performance_report.md
echo "- Python vs C++/OpenMP: 见 benchmark_results.png" >> performance_report.md
echo "" >> performance_report.md
echo "### 表面张力测试" >> performance_report.md
echo "- 表面张力计算时间: 见测试输出" >> performance_report.md
echo "" >> performance_report.md
echo "### 接触角测试" >> performance_report.md
echo "- 不同接触角的收敛时间: 见测试输出" >> performance_report.md

print_success "性能报告已生成！"
print_info "查看 performance_report.md 获取详细信息。"

# 完成
print_success "构建和测试完成！"
print_info "可视化结果已保存为PNG文件。"

# 退出虚拟环境
deactivate 