# 加速多相流模型

本文档介绍如何使用C++/CUDA加速的多相流模型，以及如何构建和使用加速模块。

## 概述

多相流模型已经通过以下方式进行了优化：

1. 使用C++和OpenMP实现核心计算函数，提供多线程CPU加速
2. 使用CUDA实现GPU加速版本，适用于有NVIDIA GPU的系统
3. 通过pybind11创建Python绑定，实现与现有Python代码的无缝集成

优化后的模型支持两种多相流方法：
- 体积流体法（VOF）
- 水平集方法（Level Set）

## 性能提升

根据基准测试，优化后的模型相比纯Python实现可以获得显著的性能提升：

- C++/OpenMP版本：5-20倍加速（取决于网格大小和CPU核心数）
- CUDA GPU版本：20-100倍加速（取决于网格大小和GPU性能）

性能提升在大规模模拟中更为明显，例如在64x64x64网格上，CUDA版本比纯Python实现快约50倍。

## 依赖项

构建加速模块需要以下依赖项：

- C++14兼容编译器（如GCC 5+或MSVC 2017+）
- CMake 3.10+
- Python 3.6+
- pybind11
- 对于CUDA加速：CUDA工具包 10.0+

## 构建和安装

可以使用提供的构建脚本来编译和安装加速模块：

```bash
# 使脚本可执行
chmod +x build_accelerated.sh

# 运行构建脚本
./build_accelerated.sh
```

或者手动构建：

```bash
# 创建构建目录
mkdir -p build
cd build

# 配置CMake
cmake ../src/core/cuda_accelerated

# 构建
cmake --build .

# 安装
cmake --install .
```

成功构建后，将在项目根目录生成`multiphase_core.so`（Linux/macOS）或`multiphase_core.pyd`（Windows）文件。

## 使用方法

### 基本用法

```python
from src.models.multiphase_accelerated import AcceleratedMultiphaseModel

# 创建模型（自动检测并使用最快的可用实现）
model = AcceleratedMultiphaseModel(grid_size=(64, 64, 64), num_phases=2)

# 初始化球形界面
center = (32, 32, 32)
radius = 16
model.initialize_sphere(center, radius)

# 设置速度场
# ...（设置速度场的代码）

# 执行模拟步骤
dt = 0.1
model.step(dt, method='levelset')  # 或 'vof'

# 获取结果
volume_fractions = model.volume_fractions
phi = model.phi
interface_field = model.interface_field
```

### 强制使用特定实现

可以通过参数控制使用哪种实现：

```python
# 强制使用纯Python实现
model_py = AcceleratedMultiphaseModel(grid_size, use_acceleration=False)

# 使用C++/OpenMP实现（不使用CUDA）
model_cpp = AcceleratedMultiphaseModel(grid_size, use_acceleration=True, use_cuda=False)

# 使用CUDA实现（如果可用）
model_cuda = AcceleratedMultiphaseModel(grid_size, use_acceleration=True, use_cuda=True)
```

## 性能测试

可以使用提供的基准测试脚本来比较不同实现的性能：

```bash
python src/benchmark/benchmark_multiphase.py
```

这将在不同网格大小上测试Python、C++和CUDA实现（如果可用），并生成性能比较图表。

## 可视化

基准测试脚本也包含可视化功能，可以通过取消注释以下行来启用：

```python
# 在benchmark_multiphase.py的末尾
benchmark.run_and_visualize()
```

这将创建一个动画，显示模拟过程中的体积分数和水平集函数。

## 已知限制

1. CUDA实现需要NVIDIA GPU和CUDA工具包
2. 大型网格（>256³）可能需要大量GPU内存
3. 当前实现主要针对结构化网格优化

## 故障排除

如果遇到问题，请检查以下几点：

1. 确保已安装所有依赖项
2. 检查是否有编译错误
3. 对于CUDA问题，确保NVIDIA驱动程序和CUDA工具包版本兼容
4. 如果模块无法导入，检查Python路径是否包含项目根目录

## 性能优化建议

1. 对于大型模拟，优先使用CUDA实现
2. 如果没有GPU，确保启用OpenMP以获得多线程加速
3. 对于小型网格（<32³），纯Python实现可能已经足够快
4. 考虑减小时间步长以提高稳定性，特别是在使用VOF方法时 