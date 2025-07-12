"""CUDA加速扩展模块"""
import logging
from ..fluid_solver import FluidSolver

try:
    from .cuda_solver import CudaFluidSolver, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False
    logging.warning("CUDA扩展模块导入失败，将使用CPU版本")
    
    # 创建一个包装类，在CUDA不可用时自动使用CPU版本
    class CudaFluidSolver:
        def __init__(self, width, height, depth, viscosity=0.1, density=1.0, boundary_type=0):
            logging.warning("CUDA库不可用，自动降级为CPU版本")
            self.solver = FluidSolver(width, height, depth, viscosity, density)
            self.width = width
            self.height = height
            self.depth = depth
            
        def step(self, dt):
            return self.solver.step(dt)
            
        def get_velocity_field(self):
            return self.solver.get_velocity_field()
            
        def get_pressure_field(self):
            return self.solver.get_pressure_field()
            
        def get_vorticity_field(self):
            return self.solver.get_vorticity_field()
            
        def add_force(self, x, y, z, fx, fy, fz):
            return self.solver.add_force(x, y, z, fx, fy, fz)
            
        def add_obstacle(self, shape, params):
            # 转换参数格式
            if shape == "sphere" and "center" in params:
                # CUDA版本使用[x,y,z]格式，CPU版本使用(x,y,z)格式
                params = params.copy()
                params["center"] = tuple(params["center"])
            return self.solver.add_obstacle(shape, params)
            
        def reset_performance_stats(self):
            return self.solver.reset_performance_stats() 