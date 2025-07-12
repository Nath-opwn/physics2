from setuptools import setup, Extension
import numpy as np

fluid_solver_module = Extension(
    'fluid_solver_core',
    sources=['fluid_solver_core.cpp'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp']
)

setup(
    name='fluid_solver_core',
    version='0.1',
    description='流体求解器的C++扩展',
    ext_modules=[fluid_solver_module]
) 