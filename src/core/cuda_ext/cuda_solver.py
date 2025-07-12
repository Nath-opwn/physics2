import numpy as np
import time
import logging

# 尝试导入CUDA相关库
try:
    import cupy as cp
    try:
        # 测试CUDA是否真正可用
        test_array = cp.array([1, 2, 3])
        test_result = cp.sum(test_array)
        CUDA_AVAILABLE = True
        logging.info("CUDA加速可用")
        
        # 尝试导入我们的CUDA扩展
        try:
            from . import cuda_kernels
            CUDA_KERNELS_AVAILABLE = True
            logging.info("CUDA核心函数扩展可用")
        except ImportError as e:
            CUDA_KERNELS_AVAILABLE = False
            logging.warning(f"CUDA核心函数扩展导入失败: {str(e)}，将使用CuPy实现")
    except Exception as e:
        CUDA_AVAILABLE = False
        CUDA_KERNELS_AVAILABLE = False
        logging.warning(f"CUDA初始化失败: {str(e)}，将使用CPU计算")
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_KERNELS_AVAILABLE = False
    logging.warning("CUDA库不可用，将使用CPU计算")

class CudaFluidSolver:
    """CUDA加速的流体求解器"""
    
    def __init__(self, width, height, depth, viscosity=0.1, density=1.0, boundary_type=0):
        """
        初始化CUDA加速的流体求解器
        
        参数:
        - width, height, depth: 模拟网格尺寸
        - viscosity: 流体粘度
        - density: 流体密度
        - boundary_type: 边界条件类型
        """
        if not CUDA_AVAILABLE:
            raise ImportError("CUDA库不可用或初始化失败，无法创建CUDA加速求解器")
        
        self.width = width
        self.height = height
        self.depth = depth
        self.viscosity = viscosity
        self.density = density
        self.boundary_type = boundary_type
        
        # 初始化场数据
        self.velocity = cp.zeros((width, height, depth, 3), dtype=cp.float32)
        self.velocity_prev = cp.zeros((width, height, depth, 3), dtype=cp.float32)
        self.pressure = cp.zeros((width, height, depth), dtype=cp.float32)
        self.pressure_prev = cp.zeros((width, height, depth), dtype=cp.float32)
        self.divergence = cp.zeros((width, height, depth), dtype=cp.float32)
        
        # 初始化外力场
        self.forces = cp.zeros((width, height, depth, 3), dtype=cp.float32)
        
        # 初始化障碍物标记
        self.obstacles = cp.zeros((width, height, depth), dtype=cp.bool_)
        
        # 性能统计
        self.step_time = 0.0
        self.step_count = 0
        
        # 记录是否使用CUDA核心函数
        self.using_cuda_kernels = CUDA_KERNELS_AVAILABLE
        if self.using_cuda_kernels:
            logging.info("使用CUDA核心函数扩展进行计算")
        else:
            logging.info("使用CuPy实现进行计算")
    
    def add_force(self, x, y, z, fx, fy, fz):
        """在指定位置添加力"""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.forces[x, y, z] = cp.array([fx, fy, fz], dtype=cp.float32)
    
    def add_obstacle(self, shape, params):
        """添加障碍物"""
        if shape == "sphere":
            center = params.get("center", [self.width//2, self.height//2, self.depth//2])
            radius = params.get("radius", min(self.width, self.height, self.depth) // 8)
            
            # 创建网格坐标
            x = cp.arange(self.width)
            y = cp.arange(self.height)
            z = cp.arange(self.depth)
            X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
            
            # 计算到中心的距离
            distance = cp.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
            
            # 标记障碍物
            self.obstacles = distance <= radius
        
        elif shape == "box":
            min_point = params.get("min_point", [0, 0, 0])
            max_point = params.get("max_point", [self.width//4, self.height//4, self.depth//4])
            
            # 标记障碍物
            self.obstacles[min_point[0]:max_point[0], min_point[1]:max_point[1], min_point[2]:max_point[2]] = True
        
        elif shape == "cylinder":
            center = params.get("center", [self.width//2, self.height//2])
            radius = params.get("radius", min(self.width, self.height) // 8)
            axis = params.get("axis", 2)  # 默认沿z轴
            
            # 创建网格坐标
            if axis == 0:  # 沿x轴
                y = cp.arange(self.height)
                z = cp.arange(self.depth)
                Y, Z = cp.meshgrid(y, z, indexing='ij')
                distance = cp.sqrt((Y - center[0])**2 + (Z - center[1])**2)
                
                for i in range(self.width):
                    self.obstacles[i] = distance <= radius
            
            elif axis == 1:  # 沿y轴
                x = cp.arange(self.width)
                z = cp.arange(self.depth)
                X, Z = cp.meshgrid(x, z, indexing='ij')
                distance = cp.sqrt((X - center[0])**2 + (Z - center[1])**2)
                
                for j in range(self.height):
                    self.obstacles[:, j] = distance <= radius
            
            else:  # 沿z轴
                x = cp.arange(self.width)
                y = cp.arange(self.height)
                X, Y = cp.meshgrid(x, y, indexing='ij')
                distance = cp.sqrt((X - center[0])**2 + (Y - center[1])**2)
                
                for k in range(self.depth):
                    self.obstacles[:, :, k] = distance <= radius
    
    def advect(self, field, field_prev, dt):
        """平流步骤 - 使用半拉格朗日方法"""
        if self.using_cuda_kernels:
            # 使用CUDA核心函数
            cuda_kernels.advect(field, field_prev, self.velocity, dt)
        else:
            # 使用CuPy实现
        # 创建网格坐标
        x = cp.arange(self.width)
        y = cp.arange(self.height)
        z = cp.arange(self.depth)
        X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
        
        # 计算回溯位置
        pos_x = X - dt * self.velocity[:, :, :, 0]
        pos_y = Y - dt * self.velocity[:, :, :, 1]
        pos_z = Z - dt * self.velocity[:, :, :, 2]
        
        # 边界处理
        pos_x = cp.clip(pos_x, 0, self.width - 1.01)
        pos_y = cp.clip(pos_y, 0, self.height - 1.01)
        pos_z = cp.clip(pos_z, 0, self.depth - 1.01)
        
        # 计算插值权重
        x0 = cp.floor(pos_x).astype(cp.int32)
        y0 = cp.floor(pos_y).astype(cp.int32)
        z0 = cp.floor(pos_z).astype(cp.int32)
        
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        # 确保索引在有效范围内
        x0 = cp.clip(x0, 0, self.width - 1)
        y0 = cp.clip(y0, 0, self.height - 1)
        z0 = cp.clip(z0, 0, self.depth - 1)
        x1 = cp.clip(x1, 0, self.width - 1)
        y1 = cp.clip(y1, 0, self.height - 1)
        z1 = cp.clip(z1, 0, self.depth - 1)
        
        # 计算插值系数
        sx = pos_x - x0
        sy = pos_y - y0
        sz = pos_z - z0
        
        # 三线性插值
        if len(field_prev.shape) == 3:  # 标量场
            c000 = field_prev[x0, y0, z0]
            c001 = field_prev[x0, y0, z1]
            c010 = field_prev[x0, y1, z0]
            c011 = field_prev[x0, y1, z1]
            c100 = field_prev[x1, y0, z0]
            c101 = field_prev[x1, y0, z1]
            c110 = field_prev[x1, y1, z0]
            c111 = field_prev[x1, y1, z1]
            
            c00 = c000 * (1 - sx) + c100 * sx
            c01 = c001 * (1 - sx) + c101 * sx
            c10 = c010 * (1 - sx) + c110 * sx
            c11 = c011 * (1 - sx) + c111 * sx
            
            c0 = c00 * (1 - sy) + c10 * sy
            c1 = c01 * (1 - sy) + c11 * sy
            
            field[:] = c0 * (1 - sz) + c1 * sz
        
        else:  # 矢量场
            for d in range(field_prev.shape[3]):
                c000 = field_prev[x0, y0, z0, d]
                c001 = field_prev[x0, y0, z1, d]
                c010 = field_prev[x0, y1, z0, d]
                c011 = field_prev[x0, y1, z1, d]
                c100 = field_prev[x1, y0, z0, d]
                c101 = field_prev[x1, y0, z1, d]
                c110 = field_prev[x1, y1, z0, d]
                c111 = field_prev[x1, y1, z1, d]
                
                c00 = c000 * (1 - sx) + c100 * sx
                c01 = c001 * (1 - sx) + c101 * sx
                c10 = c010 * (1 - sx) + c110 * sx
                c11 = c011 * (1 - sx) + c111 * sx
                
                c0 = c00 * (1 - sy) + c10 * sy
                c1 = c01 * (1 - sy) + c11 * sy
                
                field[:, :, :, d] = c0 * (1 - sz) + c1 * sz
    
    def diffuse(self, field, field_prev, dt, diffusion_rate):
        """扩散步骤 - 使用隐式求解"""
        alpha = dt * diffusion_rate
        beta = 1 / (1 + 6 * alpha)
        
        if self.using_cuda_kernels:
            # 使用CUDA核心函数
            cuda_kernels.diffuse(field, field_prev, alpha, beta, 20)
        else:
            # 使用CuPy实现
        # 迭代求解
        for _ in range(20):
            field[1:-1, 1:-1, 1:-1] = (
                field_prev[1:-1, 1:-1, 1:-1] + alpha * (
                    field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
                    field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
                    field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2]
                )
            ) * beta
            
            # 应用边界条件
            self.apply_boundary_conditions(field)
    
    def project(self, velocity, pressure, dt):
        """投影步骤 - 保证速度场无散度"""
        if self.using_cuda_kernels:
            # 使用CUDA核心函数
            cuda_kernels.project(velocity, pressure, self.divergence, dt, 50)
        else:
            # 使用CuPy实现
        # 计算速度场的散度
            self.divergence.fill(0)
        
            self.divergence[1:-1, 1:-1, 1:-1] = (
            (velocity[2:, 1:-1, 1:-1, 0] - velocity[:-2, 1:-1, 1:-1, 0]) / 2 +
            (velocity[1:-1, 2:, 1:-1, 1] - velocity[1:-1, :-2, 1:-1, 1]) / 2 +
            (velocity[1:-1, 1:-1, 2:, 2] - velocity[1:-1, 1:-1, :-2, 2]) / 2
        ) / dt
        
        # 求解泊松方程
        pressure.fill(0)
        for _ in range(50):
            pressure[1:-1, 1:-1, 1:-1] = (
                (pressure[2:, 1:-1, 1:-1] + pressure[:-2, 1:-1, 1:-1] +
                 pressure[1:-1, 2:, 1:-1] + pressure[1:-1, :-2, 1:-1] +
                 pressure[1:-1, 1:-1, 2:] + pressure[1:-1, 1:-1, :-2] -
                     self.divergence[1:-1, 1:-1, 1:-1]) / 6
            )
            
            # 应用边界条件
            self.apply_boundary_conditions(pressure)
        
        # 根据压力梯度更新速度场
        velocity[1:-1, 1:-1, 1:-1, 0] -= dt * (pressure[2:, 1:-1, 1:-1] - pressure[:-2, 1:-1, 1:-1]) / 2
        velocity[1:-1, 1:-1, 1:-1, 1] -= dt * (pressure[1:-1, 2:, 1:-1] - pressure[1:-1, :-2, 1:-1]) / 2
        velocity[1:-1, 1:-1, 1:-1, 2] -= dt * (pressure[1:-1, 1:-1, 2:] - pressure[1:-1, 1:-1, :-2]) / 2
        
        # 应用边界条件
        self.apply_boundary_conditions(velocity)
    
    def apply_boundary_conditions(self, field):
        """应用边界条件"""
        if self.boundary_type == 0:  # 固定边界
            if len(field.shape) == 3:  # 标量场
                field[0, :, :] = field[1, :, :]
                field[-1, :, :] = field[-2, :, :]
                field[:, 0, :] = field[:, 1, :]
                field[:, -1, :] = field[:, -2, :]
                field[:, :, 0] = field[:, :, 1]
                field[:, :, -1] = field[:, :, -2]
            else:  # 矢量场
                field[0, :, :, :] = field[1, :, :, :]
                field[-1, :, :, :] = field[-2, :, :, :]
                field[:, 0, :, :] = field[:, 1, :, :]
                field[:, -1, :, :] = field[:, -2, :, :]
                field[:, :, 0, :] = field[:, :, 1, :]
                field[:, :, -1, :] = field[:, :, -2, :]
        
        elif self.boundary_type == 1:  # 周期边界
            if len(field.shape) == 3:  # 标量场
                field[0, :, :] = field[-2, :, :]
                field[-1, :, :] = field[1, :, :]
                field[:, 0, :] = field[:, -2, :]
                field[:, -1, :] = field[:, 1, :]
                field[:, :, 0] = field[:, :, -2]
                field[:, :, -1] = field[:, :, 1]
            else:  # 矢量场
                field[0, :, :, :] = field[-2, :, :, :]
                field[-1, :, :, :] = field[1, :, :, :]
                field[:, 0, :, :] = field[:, -2, :, :]
                field[:, -1, :, :] = field[:, 1, :, :]
                field[:, :, 0, :] = field[:, :, -2, :]
                field[:, :, -1, :] = field[:, :, 1, :]
        
        # 应用障碍物边界条件
        if cp.any(self.obstacles):
            if len(field.shape) == 3:  # 标量场
                field[self.obstacles] = 0
            else:  # 矢量场
                field[self.obstacles, :] = 0
    
    def step(self, dt):
        """执行一步模拟"""
        start_time = time.time()
        
        # 保存当前场
        self.velocity_prev[:] = self.velocity
        self.pressure_prev[:] = self.pressure
        
        # 添加外力
        self.velocity += dt * self.forces
        
        # 平流步骤
        self.advect(self.velocity, self.velocity_prev, dt)
        
        # 扩散步骤
        self.diffuse(self.velocity, self.velocity_prev, dt, self.viscosity)
        
        # 投影步骤
        self.project(self.velocity, self.pressure, dt)
        
        # 清除力场
        self.forces.fill(0)
        
        # 记录步骤时间
        self.step_time += time.time() - start_time
        self.step_count += 1
    
    def get_velocity_field(self):
        """获取速度场"""
        return cp.asnumpy(self.velocity)
    
    def get_pressure_field(self):
        """获取压力场"""
        return cp.asnumpy(self.pressure)
    
    def get_vorticity_field(self):
        """计算涡量场"""
        if self.using_cuda_kernels:
            # 使用CUDA核心函数
            vorticity_np = cuda_kernels.compute_vorticity(self.velocity)
            return vorticity_np
        else:
            # 使用CuPy实现
        # 计算速度梯度
        vorticity = cp.zeros((self.width, self.height, self.depth, 3), dtype=cp.float32)
        
        # x分量: dw/dy - dv/dz
        vorticity[1:-1, 1:-1, 1:-1, 0] = (
            (self.velocity[1:-1, 2:, 1:-1, 2] - self.velocity[1:-1, :-2, 1:-1, 2]) / 2 -
            (self.velocity[1:-1, 1:-1, 2:, 1] - self.velocity[1:-1, 1:-1, :-2, 1]) / 2
        )
        
        # y分量: du/dz - dw/dx
        vorticity[1:-1, 1:-1, 1:-1, 1] = (
            (self.velocity[1:-1, 1:-1, 2:, 0] - self.velocity[1:-1, 1:-1, :-2, 0]) / 2 -
            (self.velocity[2:, 1:-1, 1:-1, 2] - self.velocity[:-2, 1:-1, 1:-1, 2]) / 2
        )
        
        # z分量: dv/dx - du/dy
        vorticity[1:-1, 1:-1, 1:-1, 2] = (
            (self.velocity[2:, 1:-1, 1:-1, 1] - self.velocity[:-2, 1:-1, 1:-1, 1]) / 2 -
            (self.velocity[1:-1, 2:, 1:-1, 0] - self.velocity[1:-1, :-2, 1:-1, 0]) / 2
        )
        
        # 应用边界条件
        self.apply_boundary_conditions(vorticity)
        
        return cp.asnumpy(vorticity)
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if self.step_count > 0:
            avg_step_time = self.step_time / self.step_count
        else:
            avg_step_time = 0
        
        return {
            "total_time": self.step_time,
            "step_count": self.step_count,
            "avg_step_time": avg_step_time,
            "using_cuda_kernels": self.using_cuda_kernels
        }
    
    def reset_performance_stats(self):
        """重置性能统计信息"""
        self.step_time = 0.0
        self.step_count = 0 