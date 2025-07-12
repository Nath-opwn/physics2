# CUDA加速扩展模块

本模块提供了流体动力学模拟的CUDA加速实现，可显著提高大规模3D流体模拟的性能。

## 依赖项

- CUDA工具包 (建议10.0+)
- Python 3.6+
- pybind11
- cupy
- numpy

## 安装方法

1. 确保已安装CUDA工具包，并设置了`CUDA_HOME`环境变量
2. 运行构建脚本：

```bash
chmod +x build_cuda_ext.sh
./build_cuda_ext.sh
```

或者手动构建：

```bash
pip install pybind11 cupy
python setup.py build_ext --inplace
```

## 使用方法

```python
from src.core.cuda_ext import CudaFluidSolver, CUDA_AVAILABLE

# 检查CUDA是否可用
if CUDA_AVAILABLE:
    # 创建CUDA加速的流体求解器
    solver = CudaFluidSolver(width=64, height=64, depth=64, viscosity=0.1)
    
    # 添加力和障碍物
    solver.add_force(32, 32, 32, 0, 0, 10.0)
    solver.add_obstacle("sphere", {"center": [32, 32, 32], "radius": 8})
    
    # 运行模拟
    for i in range(100):
        solver.step(dt=0.1)
    
    # 获取结果
    velocity = solver.get_velocity_field()
    pressure = solver.get_pressure_field()
    vorticity = solver.get_vorticity_field()
else:
    print("CUDA不可用，将使用CPU版本")
```

## 性能测试

使用测试脚本比较CPU和CUDA版本的性能：

```bash
python test_cuda.py
```

这将生成性能比较结果和可视化图像。

## 实现细节

本扩展使用CUDA C++实现了流体动力学模拟的核心计算函数：

1. **平流（Advection）**：使用半拉格朗日方法和三线性插值
2. **扩散（Diffusion）**：使用隐式雅可比迭代求解
3. **投影（Projection）**：使用雅可比迭代求解泊松方程
4. **涡量计算（Vorticity）**：计算速度场的旋度

所有计算都在GPU上并行执行，显著提高了大规模模拟的性能。

## 自动降级机制

当CUDA不可用时，系统会自动降级使用CPU版本的求解器，确保代码在各种环境中都能正常运行。 