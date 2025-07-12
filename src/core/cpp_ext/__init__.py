# 流体求解器C++扩展包
try:
    from .fluid_solver_core import diffuse, advect, project, compute_vorticity
    EXTENSION_LOADED = True
except ImportError:
    EXTENSION_LOADED = False
    print("警告：C++扩展模块未加载，将使用纯Python实现。")
    print("请运行 'cd src/core/cpp_ext && python setup.py build_ext --inplace' 编译扩展模块。") 