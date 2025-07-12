from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

# 检查CUDA是否可用
cuda_available = False
cuda_include = os.environ.get('CUDA_HOME', '/usr/local/cuda')
if os.path.exists(cuda_include):
    cuda_available = True
    print(f"找到CUDA路径: {cuda_include}")
else:
    print("未找到CUDA路径，将编译CPU版本")

# 定义CUDA扩展
cuda_ext = Extension(
    name="cuda_kernels",
    sources=[
        "cuda_kernels.cpp",
        "cuda_diffuse.cu",
        "cuda_advect.cu",
        "cuda_project.cu"
    ],
    include_dirs=[cuda_include + "/include"],
    library_dirs=[cuda_include + "/lib64"],
    libraries=["cudart"],
    language="c++",
    extra_compile_args={
        "nvcc": ["-O3", "--use_fast_math", "-Xcompiler", "-fPIC"],
        "cxx": ["-std=c++11", "-O3"]
    },
)

# 自定义构建扩展命令
class BuildExt(build_ext):
    def build_extensions(self):
        # 检测CUDA编译器
        if cuda_available:
            try:
                from torch.utils.cpp_extension import CUDAExtension, BuildExtension
                self.compiler = BuildExtension.compiler
            except ImportError:
                print("未找到PyTorch CUDA扩展工具，尝试使用直接方法")
                # 尝试直接设置CUDA编译器
                self.compiler.set_executable("compiler_so", "nvcc")
        
        build_ext.build_extensions(self)

# 设置
setup(
    name="fluid_sim_cuda",
    version="0.1",
    ext_modules=[cuda_ext] if cuda_available else [],
    cmdclass={"build_ext": BuildExt} if cuda_available else {},
) 