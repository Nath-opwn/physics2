# 流体求解器C++扩展

这个目录包含流体求解器的C++扩展模块，用于加速关键计算部分。

## 依赖项

- Python 3.6+
- NumPy
- 支持OpenMP的C++编译器（GCC 4.2+, Clang 3.8+, MSVC 2015+）

## 编译方法

在此目录下运行以下命令编译C++扩展：

```bash
python setup.py build_ext --inplace
```

这将在当前目录生成`fluid_solver_core.*.so`文件（Linux/Mac）或`fluid_solver_core.*.pyd`文件（Windows）。

## 使用方法

编译后，扩展模块会被自动导入到流体求解器中。要使用C++加速版本，只需在创建FluidSolver实例时设置`use_cpp_ext=True`（默认值）：

```python
from src.core.fluid_solver import FluidSolver

# 使用C++扩展
solver = FluidSolver(64, 64, 64, use_cpp_ext=True)

# 使用纯Python版本
solver = FluidSolver(64, 64, 64, use_cpp_ext=False)
```

## 性能测试

可以使用`scripts/performance_test.py`脚本来比较纯Python版本和C++扩展版本的性能差异：

```bash
python scripts/performance_test.py
```

这将生成一个性能比较图，显示不同网格尺寸下两个版本的运行时间和加速比。

## 实现的功能

C++扩展实现了以下关键计算函数：

1. `diffuse`: 扩散步骤
2. `advect`: 平流步骤
3. `project`: 投影步骤（确保速度场无散度）
4. `compute_vorticity`: 计算涡量场

这些函数使用OpenMP进行并行化，可以充分利用多核CPU提高性能。

## 注意事项

- 如果编译失败，请确保已安装适当的C++编译器和开发工具。
- 在Windows上，可能需要安装Visual C++ Build Tools。
- 在某些系统上，可能需要手动安装OpenMP库。
- 如果扩展模块无法加载，系统会自动回退到使用纯Python实现。 