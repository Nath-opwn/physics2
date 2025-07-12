import numpy as np
import importlib.util
import os
import sys
import logging

# 设置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 尝试导入加速模块
try:
    import multiphase_core
    HAS_ACCELERATION = True
    logger.info("成功加载加速模块")
    CUDA_AVAILABLE = multiphase_core.is_cuda_available()
    logger.info(f"CUDA加速{'可用' if CUDA_AVAILABLE else '不可用'}")
except ImportError:
    HAS_ACCELERATION = False
    CUDA_AVAILABLE = False
    logger.warning("无法加载加速模块，将使用纯Python实现")

class AcceleratedMultiphaseModel:
    """
    使用C++/CUDA加速的多相流模型
    """
    
    def __init__(self, grid_size, num_phases=2, use_acceleration=True, use_cuda=True):
        """
        初始化加速的多相流模型
        
        参数:
            grid_size: 三维网格尺寸 (width, height, depth)
            num_phases: 相数量
            use_acceleration: 是否使用C++/CUDA加速
            use_cuda: 是否使用CUDA (如果可用)
        """
        self.width, self.height, self.depth = grid_size
        self.num_phases = num_phases
        self.use_acceleration = use_acceleration and HAS_ACCELERATION
        self.use_cuda = use_cuda and CUDA_AVAILABLE and self.use_acceleration
        
        # 初始化体积分数 (VOF方法)
        self.volume_fractions = np.zeros((num_phases, self.width, self.height, self.depth), dtype=np.float32)
        
        # 初始化水平集函数 (水平集方法)
        self.phi = np.ones((self.width, self.height, self.depth), dtype=np.float32) * 1.0
        
        # 界面场
        self.interface_field = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        
        # 初始化速度场
        self.velocity_field = np.zeros((self.width, self.height, self.depth, 3), dtype=np.float32)
        
        # 水平集方法参数
        self.epsilon = 1.5  # 界面厚度参数
        self.reinit_iterations = 10  # 重初始化迭代次数
        self.dt_reinit = 0.1  # 重初始化时间步长
        
        # 表面张力参数
        self.sigma = 0.07  # 表面张力系数 (N/m)
        self.curvature = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        self.surface_tension_force = np.zeros((self.width, self.height, self.depth, 3), dtype=np.float32)
        
        # 接触角参数
        self.contact_angle = 90.0  # 默认接触角 (度)
        self.has_solid_boundary = False  # 是否有固体边界
        self.solid_boundary = np.zeros((self.width, self.height, self.depth), dtype=np.int8)
        
        logger.info(f"初始化加速多相流模型: 网格尺寸={grid_size}, 相数量={num_phases}")
        logger.info(f"加速状态: C++/CUDA加速={'启用' if self.use_acceleration else '禁用'}, "
                   f"CUDA={'启用' if self.use_cuda else '禁用'}")
        
        # 性能统计
        self.performance_stats = {
            'advect_time': 0.0,
            'reinit_time': 0.0,
            'surface_tension_time': 0.0,
            'total_time': 0.0,
            'steps': 0
        }
    
    def initialize_sphere(self, center, radius, phase_idx=0, inside_value=1.0, outside_value=0.0):
        """
        初始化球形界面
        
        参数:
            center: 球心坐标 (x, y, z)
            radius: 球半径
            phase_idx: 相索引
            inside_value: 球内部值
            outside_value: 球外部值
        """
        cx, cy, cz = center
        
        # 创建网格坐标
        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        z = np.arange(0, self.depth)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算到中心的距离
        distance = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        
        # 设置体积分数
        self.volume_fractions[phase_idx] = np.where(distance <= radius, inside_value, outside_value)
        
        # 如果有多个相，确保体积分数和为1
        if self.num_phases > 1:
            other_phase = 1 if phase_idx == 0 else 0
            self.volume_fractions[other_phase] = 1.0 - self.volume_fractions[phase_idx]
        
        # 设置水平集函数 (有符号距离函数)
        self.phi = distance - radius
        
        # 更新界面场
        self.update_interface_field()
        
        logger.info(f"初始化球形界面: 中心={center}, 半径={radius}, 相索引={phase_idx}")
    
    def set_velocity_field(self, velocity_field):
        """
        设置速度场
        
        参数:
            velocity_field: 速度场数组，形状为 (width, height, depth, 3)
        """
        assert velocity_field.shape == (self.width, self.height, self.depth, 3), "速度场尺寸不匹配"
        self.velocity_field = velocity_field.astype(np.float32)
    
    def advect_vof(self, dt, phase_idx=0):
        """
        使用VOF方法进行平流
        
        参数:
            dt: 时间步长
            phase_idx: 要平流的相索引
        """
        if self.use_acceleration:
            # 使用加速模块
            self.volume_fractions = multiphase_core.advect_vof(
                self.volume_fractions, self.velocity_field, phase_idx, dt, self.use_cuda
            )
            
            # 归一化体积分数
            self.volume_fractions = multiphase_core.normalize_volume_fractions(
                self.volume_fractions, self.use_cuda
            )
        else:
            # 使用纯Python实现 (简化版本，实际应用中应使用更高级的方法)
            # 这里只是一个占位符，实际项目中应该有一个纯Python实现
            logger.warning("使用纯Python VOF平流 (简化实现)")
            
            # 创建临时数组存储结果
            vf_new = np.copy(self.volume_fractions)
            
            # 简单的半拉格朗日平流
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        idx = (x, y, z)
                        vel = self.velocity_field[idx]
                        
                        # 计算回溯位置
                        pos_x = x - dt * vel[0]
                        pos_y = y - dt * vel[1]
                        pos_z = z - dt * vel[2]
                        
                        # 边界检查
                        pos_x = max(0, min(self.width - 1.001, pos_x))
                        pos_y = max(0, min(self.height - 1.001, pos_y))
                        pos_z = max(0, min(self.depth - 1.001, pos_z))
                        
                        # 三线性插值 (简化实现)
                        x0, y0, z0 = int(pos_x), int(pos_y), int(pos_z)
                        x1, y1, z1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1), min(z0 + 1, self.depth - 1)
                        
                        sx, sy, sz = pos_x - x0, pos_y - y0, pos_z - z0
                        
                        # 插值
                        c000 = self.volume_fractions[phase_idx, x0, y0, z0]
                        c001 = self.volume_fractions[phase_idx, x0, y0, z1]
                        c010 = self.volume_fractions[phase_idx, x0, y1, z0]
                        c011 = self.volume_fractions[phase_idx, x0, y1, z1]
                        c100 = self.volume_fractions[phase_idx, x1, y0, z0]
                        c101 = self.volume_fractions[phase_idx, x1, y0, z1]
                        c110 = self.volume_fractions[phase_idx, x1, y1, z0]
                        c111 = self.volume_fractions[phase_idx, x1, y1, z1]
                        
                        c00 = c000 * (1 - sx) + c100 * sx
                        c01 = c001 * (1 - sx) + c101 * sx
                        c10 = c010 * (1 - sx) + c110 * sx
                        c11 = c011 * (1 - sx) + c111 * sx
                        
                        c0 = c00 * (1 - sy) + c10 * sy
                        c1 = c01 * (1 - sy) + c11 * sy
                        
                        vf_new[phase_idx, x, y, z] = c0 * (1 - sz) + c1 * sz
            
            self.volume_fractions = vf_new
            
            # 归一化体积分数
            total = np.sum(self.volume_fractions, axis=0)
            total = np.maximum(total, 1e-6)
            for p in range(self.num_phases):
                self.volume_fractions[p] /= total
    
    def advect_levelset(self, dt):
        """
        使用水平集方法进行平流
        
        参数:
            dt: 时间步长
        """
        if self.use_acceleration:
            # 使用加速模块
            self.phi = multiphase_core.advect_levelset(
                self.phi, self.velocity_field, dt, self.use_cuda
            )
            
            # 重初始化水平集函数
            self.phi = multiphase_core.reinitialize_levelset(
                self.phi, self.reinit_iterations, self.dt_reinit, self.use_cuda
            )
            
            # 更新相场
            self.volume_fractions = multiphase_core.update_phase_fields(
                self.phi, self.num_phases, self.epsilon, self.use_cuda
            )
        else:
            # 使用纯Python实现 (简化版本)
            logger.warning("使用纯Python水平集平流 (简化实现)")
            
            # 创建临时数组存储结果
            phi_new = np.zeros_like(self.phi)
            
            # 简单的半拉格朗日平流
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        idx = (x, y, z)
                        vel = self.velocity_field[idx]
                        
                        # 计算回溯位置
                        pos_x = x - dt * vel[0]
                        pos_y = y - dt * vel[1]
                        pos_z = z - dt * vel[2]
                        
                        # 边界检查
                        pos_x = max(0, min(self.width - 1.001, pos_x))
                        pos_y = max(0, min(self.height - 1.001, pos_y))
                        pos_z = max(0, min(self.depth - 1.001, pos_z))
                        
                        # 三线性插值 (简化实现)
                        x0, y0, z0 = int(pos_x), int(pos_y), int(pos_z)
                        x1, y1, z1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1), min(z0 + 1, self.depth - 1)
                        
                        sx, sy, sz = pos_x - x0, pos_y - y0, pos_z - z0
                        
                        # 插值
                        c000 = self.phi[x0, y0, z0]
                        c001 = self.phi[x0, y0, z1]
                        c010 = self.phi[x0, y1, z0]
                        c011 = self.phi[x0, y1, z1]
                        c100 = self.phi[x1, y0, z0]
                        c101 = self.phi[x1, y0, z1]
                        c110 = self.phi[x1, y1, z0]
                        c111 = self.phi[x1, y1, z1]
                        
                        c00 = c000 * (1 - sx) + c100 * sx
                        c01 = c001 * (1 - sx) + c101 * sx
                        c10 = c010 * (1 - sx) + c110 * sx
                        c11 = c011 * (1 - sx) + c111 * sx
                        
                        c0 = c00 * (1 - sy) + c10 * sy
                        c1 = c01 * (1 - sy) + c11 * sy
                        
                        phi_new[x, y, z] = c0 * (1 - sz) + c1 * sz
            
            self.phi = phi_new
            
            # 简单的重初始化 (实际应用中应使用更高级的方法)
            # 这里只是一个占位符
            logger.warning("跳过Python实现的重初始化")
            
            # 更新相场
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        # 使用平滑Heaviside函数
                        self.volume_fractions[0, x, y, z] = 0.5 * (1.0 - np.tanh(self.phi[x, y, z] / self.epsilon))
                        
                        # 更新其他相
                        if self.num_phases > 1:
                            self.volume_fractions[1, x, y, z] = 1.0 - self.volume_fractions[0, x, y, z]
    
    def update_interface_field(self):
        """
        更新界面场
        """
        if self.use_acceleration:
            # 使用加速模块
            self.interface_field = multiphase_core.compute_interface_field(
                self.phi, self.epsilon, self.use_cuda
            )
        else:
            # 使用纯Python实现 (简化版本)
            logger.warning("使用纯Python界面场计算 (简化实现)")
            
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        phi_val = self.phi[x, y, z]
                        if abs(phi_val) < self.epsilon:
                            self.interface_field[x, y, z] = 0.5 * (1.0 + np.cos(np.pi * phi_val / self.epsilon)) / self.epsilon
                        else:
                            self.interface_field[x, y, z] = 0.0
    
    def compute_curvature(self):
        """
        计算界面曲率
        """
        if self.use_acceleration:
            # 使用加速模块
            self.curvature = multiphase_core.compute_curvature(
                self.phi, self.use_cuda
            )
        else:
            # 使用纯Python实现 (简化版本)
            logger.warning("使用纯Python曲率计算 (简化实现)")
            
            # 创建临时梯度场
            grad_x = np.zeros_like(self.phi)
            grad_y = np.zeros_like(self.phi)
            grad_z = np.zeros_like(self.phi)
            
            # 计算梯度 (中心差分)
            for x in range(1, self.width - 1):
                for y in range(1, self.height - 1):
                    for z in range(1, self.depth - 1):
                        grad_x[x, y, z] = (self.phi[x+1, y, z] - self.phi[x-1, y, z]) * 0.5
                        grad_y[x, y, z] = (self.phi[x, y+1, z] - self.phi[x, y-1, z]) * 0.5
                        grad_z[x, y, z] = (self.phi[x, y, z+1] - self.phi[x, y, z-1]) * 0.5
            
            # 计算曲率
            for x in range(1, self.width - 1):
                for y in range(1, self.height - 1):
                    for z in range(1, self.depth - 1):
                        # 计算梯度大小
                        grad_mag = np.sqrt(grad_x[x, y, z]**2 + grad_y[x, y, z]**2 + grad_z[x, y, z]**2)
                        
                        # 避免除以零
                        if grad_mag < 1e-6:
                            self.curvature[x, y, z] = 0.0
                            continue
                        
                        # 归一化梯度
                        nx = grad_x[x, y, z] / grad_mag
                        ny = grad_y[x, y, z] / grad_mag
                        nz = grad_z[x, y, z] / grad_mag
                        
                        # 计算梯度的散度 (简化实现)
                        dnx_dx = (grad_x[x+1, y, z] / max(1e-6, np.sqrt(grad_x[x+1, y, z]**2 + grad_y[x+1, y, z]**2 + grad_z[x+1, y, z]**2)) - 
                                 grad_x[x-1, y, z] / max(1e-6, np.sqrt(grad_x[x-1, y, z]**2 + grad_y[x-1, y, z]**2 + grad_z[x-1, y, z]**2))) * 0.5
                        
                        dny_dy = (grad_y[x, y+1, z] / max(1e-6, np.sqrt(grad_x[x, y+1, z]**2 + grad_y[x, y+1, z]**2 + grad_z[x, y+1, z]**2)) - 
                                 grad_y[x, y-1, z] / max(1e-6, np.sqrt(grad_x[x, y-1, z]**2 + grad_y[x, y-1, z]**2 + grad_z[x, y-1, z]**2))) * 0.5
                        
                        dnz_dz = (grad_z[x, y, z+1] / max(1e-6, np.sqrt(grad_x[x, y, z+1]**2 + grad_y[x, y, z+1]**2 + grad_z[x, y, z+1]**2)) - 
                                 grad_z[x, y, z-1] / max(1e-6, np.sqrt(grad_x[x, y, z-1]**2 + grad_y[x, y, z-1]**2 + grad_z[x, y, z-1]**2))) * 0.5
                        
                        # 曲率是梯度场的散度
                        self.curvature[x, y, z] = dnx_dx + dny_dy + dnz_dz
            
            # 应用边界条件
            self.curvature[0, :, :] = self.curvature[1, :, :]
            self.curvature[-1, :, :] = self.curvature[-2, :, :]
            self.curvature[:, 0, :] = self.curvature[:, 1, :]
            self.curvature[:, -1, :] = self.curvature[:, -2, :]
            self.curvature[:, :, 0] = self.curvature[:, :, 1]
            self.curvature[:, :, -1] = self.curvature[:, :, -2]
    
    def compute_surface_tension(self):
        """
        计算表面张力力
        """
        # 首先计算曲率
        self.compute_curvature()
        
        if self.use_acceleration:
            # 使用加速模块
            self.surface_tension_force = multiphase_core.compute_surface_tension(
                self.phi, self.curvature, self.sigma, self.epsilon, self.use_cuda
            )
        else:
            # 使用纯Python实现 (简化版本)
            logger.warning("使用纯Python表面张力计算 (简化实现)")
            
            # 更新界面场
            self.update_interface_field()
            
            # 创建临时梯度场
            grad_x = np.zeros_like(self.phi)
            grad_y = np.zeros_like(self.phi)
            grad_z = np.zeros_like(self.phi)
            
            # 计算梯度 (中心差分)
            for x in range(1, self.width - 1):
                for y in range(1, self.height - 1):
                    for z in range(1, self.depth - 1):
                        grad_x[x, y, z] = (self.phi[x+1, y, z] - self.phi[x-1, y, z]) * 0.5
                        grad_y[x, y, z] = (self.phi[x, y+1, z] - self.phi[x, y-1, z]) * 0.5
                        grad_z[x, y, z] = (self.phi[x, y, z+1] - self.phi[x, y, z-1]) * 0.5
            
            # 计算表面张力力
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        # 计算梯度大小
                        grad_mag = np.sqrt(grad_x[x, y, z]**2 + grad_y[x, y, z]**2 + grad_z[x, y, z]**2)
                        
                        # 避免除以零
                        grad_mag = max(1e-6, grad_mag)
                        
                        # 计算法向量
                        nx = grad_x[x, y, z] / grad_mag
                        ny = grad_y[x, y, z] / grad_mag
                        nz = grad_z[x, y, z] / grad_mag
                        
                        # 计算表面张力力
                        force_magnitude = self.sigma * self.curvature[x, y, z] * self.interface_field[x, y, z]
                        
                        self.surface_tension_force[x, y, z, 0] = force_magnitude * nx
                        self.surface_tension_force[x, y, z, 1] = force_magnitude * ny
                        self.surface_tension_force[x, y, z, 2] = force_magnitude * nz
    
    def set_solid_boundary(self, solid_boundary):
        """
        设置固体边界
        
        参数:
            solid_boundary: 固体边界数组，形状为 (width, height, depth)，值为0表示流体，1表示固体
        """
        assert solid_boundary.shape == (self.width, self.height, self.depth), "固体边界尺寸不匹配"
        self.solid_boundary = solid_boundary.astype(np.int8)
        self.has_solid_boundary = True
    
    def set_contact_angle(self, contact_angle):
        """
        设置接触角
        
        参数:
            contact_angle: 接触角 (度)
        """
        self.contact_angle = contact_angle
    
    def apply_contact_angle_boundary_condition(self):
        """
        应用接触角边界条件
        """
        if not self.has_solid_boundary:
            return
        
        # 接触角转换为弧度
        theta = self.contact_angle * np.pi / 180.0
        
        # 计算目标接触角的法向量
        n_target = np.cos(theta)
        
        # 创建临时梯度场
        grad_x = np.zeros_like(self.phi)
        grad_y = np.zeros_like(self.phi)
        grad_z = np.zeros_like(self.phi)
        
        # 计算梯度 (中心差分)
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                for z in range(1, self.depth - 1):
                    grad_x[x, y, z] = (self.phi[x+1, y, z] - self.phi[x-1, y, z]) * 0.5
                    grad_y[x, y, z] = (self.phi[x, y+1, z] - self.phi[x, y-1, z]) * 0.5
                    grad_z[x, y, z] = (self.phi[x, y, z+1] - self.phi[x, y, z-1]) * 0.5
        
        # 应用接触角边界条件
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                for z in range(1, self.depth - 1):
                    # 检查是否在固体边界附近
                    if self.solid_boundary[x, y, z] == 0:
                        # 检查周围是否有固体
                        has_solid_neighbor = False
                        solid_normal_x, solid_normal_y, solid_normal_z = 0.0, 0.0, 0.0
                        
                        if x > 0 and self.solid_boundary[x-1, y, z] == 1:
                            has_solid_neighbor = True
                            solid_normal_x += 1.0
                        if x < self.width - 1 and self.solid_boundary[x+1, y, z] == 1:
                            has_solid_neighbor = True
                            solid_normal_x -= 1.0
                        
                        if y > 0 and self.solid_boundary[x, y-1, z] == 1:
                            has_solid_neighbor = True
                            solid_normal_y += 1.0
                        if y < self.height - 1 and self.solid_boundary[x, y+1, z] == 1:
                            has_solid_neighbor = True
                            solid_normal_y -= 1.0
                        
                        if z > 0 and self.solid_boundary[x, y, z-1] == 1:
                            has_solid_neighbor = True
                            solid_normal_z += 1.0
                        if z < self.depth - 1 and self.solid_boundary[x, y, z+1] == 1:
                            has_solid_neighbor = True
                            solid_normal_z -= 1.0
                        
                        if has_solid_neighbor:
                            # 归一化固体法向量
                            solid_normal_mag = np.sqrt(solid_normal_x**2 + solid_normal_y**2 + solid_normal_z**2)
                            if solid_normal_mag > 1e-6:
                                solid_normal_x /= solid_normal_mag
                                solid_normal_y /= solid_normal_mag
                                solid_normal_z /= solid_normal_mag
                                
                                # 计算流体法向量
                                fluid_normal_mag = np.sqrt(grad_x[x, y, z]**2 + grad_y[x, y, z]**2 + grad_z[x, y, z]**2)
                                if fluid_normal_mag > 1e-6:
                                    fluid_normal_x = grad_x[x, y, z] / fluid_normal_mag
                                    fluid_normal_y = grad_y[x, y, z] / fluid_normal_mag
                                    fluid_normal_z = grad_z[x, y, z] / fluid_normal_mag
                                    
                                    # 计算当前接触角
                                    dot_product = fluid_normal_x * solid_normal_x + fluid_normal_y * solid_normal_y + fluid_normal_z * solid_normal_z
                                    current_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                                    
                                    # 调整水平集函数以匹配目标接触角
                                    target_dot = np.cos(theta)
                                    if abs(dot_product - target_dot) > 1e-3:
                                        # 计算旋转轴 (叉积)
                                        axis_x = fluid_normal_y * solid_normal_z - fluid_normal_z * solid_normal_y
                                        axis_y = fluid_normal_z * solid_normal_x - fluid_normal_x * solid_normal_z
                                        axis_z = fluid_normal_x * solid_normal_y - fluid_normal_y * solid_normal_x
                                        
                                        # 归一化旋转轴
                                        axis_mag = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
                                        if axis_mag > 1e-6:
                                            axis_x /= axis_mag
                                            axis_y /= axis_mag
                                            axis_z /= axis_mag
                                            
                                            # 计算旋转角度
                                            angle_diff = current_angle - theta
                                            
                                            # 使用罗德里格旋转公式计算新的法向量
                                            cos_angle = np.cos(angle_diff)
                                            sin_angle = np.sin(angle_diff)
                                            
                                            new_normal_x = fluid_normal_x * cos_angle + (axis_y * fluid_normal_z - axis_z * fluid_normal_y) * sin_angle + axis_x * (1 - cos_angle) * (axis_x * fluid_normal_x + axis_y * fluid_normal_y + axis_z * fluid_normal_z)
                                            new_normal_y = fluid_normal_y * cos_angle + (axis_z * fluid_normal_x - axis_x * fluid_normal_z) * sin_angle + axis_y * (1 - cos_angle) * (axis_x * fluid_normal_x + axis_y * fluid_normal_y + axis_z * fluid_normal_z)
                                            new_normal_z = fluid_normal_z * cos_angle + (axis_x * fluid_normal_y - axis_y * fluid_normal_x) * sin_angle + axis_z * (1 - cos_angle) * (axis_x * fluid_normal_x + axis_y * fluid_normal_y + axis_z * fluid_normal_z)
                                            
                                            # 更新梯度场
                                            grad_x[x, y, z] = new_normal_x * fluid_normal_mag
                                            grad_y[x, y, z] = new_normal_y * fluid_normal_mag
                                            grad_z[x, y, z] = new_normal_z * fluid_normal_mag
        
        # 使用调整后的梯度更新水平集函数 (简化实现)
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                for z in range(1, self.depth - 1):
                    if self.interface_field[x, y, z] > 0.0:
                        # 在界面附近应用调整
                        self.phi[x, y, z] = self.phi[x, y, z] - 0.1 * (
                            grad_x[x, y, z] * (self.phi[x+1, y, z] - self.phi[x-1, y, z]) * 0.5 +
                            grad_y[x, y, z] * (self.phi[x, y+1, z] - self.phi[x, y-1, z]) * 0.5 +
                            grad_z[x, y, z] * (self.phi[x, y, z+1] - self.phi[x, y, z-1]) * 0.5
                        )

    def step(self, dt, method='levelset'):
        """
        执行一个时间步
        
        参数:
            dt: 时间步长
            method: 使用的方法 ('levelset' 或 'vof')
        """
        import time
        start_time = time.time()
        
        if method == 'levelset':
            # 使用水平集方法
            
            # 平流水平集函数
            advect_start = time.time()
            self.advect_levelset(dt)
            advect_time = time.time() - advect_start
            
            # 应用接触角边界条件
            if self.has_solid_boundary:
                self.apply_contact_angle_boundary_condition()
            
            # 计算表面张力力
            st_start = time.time()
            self.compute_surface_tension()
            st_time = time.time() - st_start
            
        elif method == 'vof':
            # 使用VOF方法
            
            # 平流每个相
            advect_start = time.time()
            for phase_idx in range(self.num_phases):
                self.advect_vof(dt, phase_idx)
            advect_time = time.time() - advect_start
            
            # 更新水平集函数 (简化实现)
            # 在实际应用中，应该从VOF重建水平集函数
            # 这里我们只是简单地设置一个阈值
            threshold = 0.5
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        if self.volume_fractions[0, x, y, z] > threshold:
                            self.phi[x, y, z] = -1.0
                        else:
                            self.phi[x, y, z] = 1.0
            
            # 重初始化水平集函数
            reinit_start = time.time()
            self.phi = multiphase_core.reinitialize_levelset(
                self.phi, self.reinit_iterations, self.dt_reinit, self.use_cuda
            )
            reinit_time = time.time() - reinit_start
            
            # 应用接触角边界条件
            if self.has_solid_boundary:
                self.apply_contact_angle_boundary_condition()
            
            # 计算表面张力力
            st_start = time.time()
            self.compute_surface_tension()
            st_time = time.time() - st_start
        
        else:
            raise ValueError(f"未知方法: {method}")
        
        # 更新性能统计
        total_time = time.time() - start_time
        self.performance_stats['advect_time'] += advect_time
        self.performance_stats['surface_tension_time'] += st_time
        self.performance_stats['total_time'] += total_time
        self.performance_stats['steps'] += 1
        
        return total_time
    
    def get_volume(self, phase_idx=0):
        """
        计算指定相的体积
        
        参数:
            phase_idx: 相索引
        
        返回:
            体积
        """
        return np.sum(self.volume_fractions[phase_idx])
    
    def get_surface_area(self):
        """
        计算界面面积
        
        返回:
            界面面积
        """
        return np.sum(self.interface_field)
    
    def get_curvature(self):
        """
        获取曲率场
        """
        return self.curvature.copy()
    
    def get_surface_tension_force(self):
        """
        获取表面张力力场
        """
        return self.surface_tension_force.copy()
    
    def reset_performance_stats(self):
        """
        重置性能统计
        """
        self.performance_stats = {
            'advect_time': 0.0,
            'reinit_time': 0.0,
            'surface_tension_time': 0.0,
            'total_time': 0.0,
            'steps': 0
        }
    
    def get_performance_stats(self):
        """
        获取性能统计
        """
        if self.performance_stats['steps'] > 0:
            stats = {
                'advect_time_avg': self.performance_stats['advect_time'] / self.performance_stats['steps'],
                'reinit_time_avg': self.performance_stats['reinit_time'] / self.performance_stats['steps'],
                'surface_tension_time_avg': self.performance_stats['surface_tension_time'] / self.performance_stats['steps'],
                'total_time_avg': self.performance_stats['total_time'] / self.performance_stats['steps'],
                'steps': self.performance_stats['steps']
            }
        else:
            stats = {
                'advect_time_avg': 0.0,
                'reinit_time_avg': 0.0,
                'surface_tension_time_avg': 0.0,
                'total_time_avg': 0.0,
                'steps': 0
            }
        
        return stats


# 测试加速模块
if __name__ == "__main__":
    # 创建一个小网格进行测试
    grid_size = (32, 32, 32)
    model = AcceleratedMultiphaseModel(grid_size, num_phases=2)
    
    # 初始化球形界面
    center = (16, 16, 16)
    radius = 8
    model.initialize_sphere(center, radius)
    
    # 设置旋转速度场
    velocity_field = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3), dtype=np.float32)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                # 创建旋转速度场
                dx, dy, dz = x - center[0], y - center[1], z - center[2]
                velocity_field[x, y, z, 0] = -dy * 0.1
                velocity_field[x, y, z, 1] = dx * 0.1
                velocity_field[x, y, z, 2] = 0.0
    
    model.set_velocity_field(velocity_field)
    
    # 记录初始体积
    initial_volume = model.get_volume()
    print(f"初始体积: {initial_volume}")
    
    # 执行100个时间步
    dt = 0.1
    steps = 100
    
    for i in range(steps):
        model.step(dt, method='levelset')
        
        if i % 10 == 0:
            current_volume = model.get_volume()
            volume_change = (current_volume - initial_volume) / initial_volume * 100
            print(f"步骤 {i}: 体积 = {current_volume}, 变化 = {volume_change:.2f}%")
    
    # 最终体积变化
    final_volume = model.get_volume()
    volume_change = (final_volume - initial_volume) / initial_volume * 100
    print(f"最终体积: {final_volume}, 变化: {volume_change:.2f}%") 