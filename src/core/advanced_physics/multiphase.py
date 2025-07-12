"""多相流模型

实现多相流体的模拟，包括VOF（体积流体法）和水平集方法。
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List

class MultiphaseModel:
    """多相流模型基类
    
    为多相流模型提供通用接口。
    """
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 depth: int,
                 num_phases: int = 2):
        """初始化多相流模型
        
        参数:
            width: 网格宽度
            height: 网格高度
            depth: 网格深度
            num_phases: 相数
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.num_phases = num_phases
        
        # 初始化相场
        self.phase_fields = np.zeros((num_phases, width, height, depth), dtype=np.float32)
        
        # 初始化密度和粘度
        self.densities = np.ones(num_phases, dtype=np.float32)
        self.viscosities = np.ones(num_phases, dtype=np.float32)
    
    def set_phase_properties(self, phase_idx: int, density: float, viscosity: float) -> None:
        """设置相的物理属性
        
        参数:
            phase_idx: 相索引
            density: 密度
            viscosity: 粘度
        """
        if 0 <= phase_idx < self.num_phases:
            self.densities[phase_idx] = density
            self.viscosities[phase_idx] = viscosity
    
    def initialize_phase(self, phase_idx: int, region_func) -> None:
        """初始化相分布
        
        参数:
            phase_idx: 相索引
            region_func: 判断点是否在相区域内的函数
        """
        if 0 <= phase_idx < self.num_phases:
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        if region_func(x, y, z):
                            self.phase_fields[phase_idx, x, y, z] = 1.0
    
    def step(self, dt: float, velocity_field: np.ndarray) -> None:
        """执行一步多相流计算
        
        参数:
            dt: 时间步长
            velocity_field: 速度场
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_density_field(self) -> np.ndarray:
        """计算密度场
        
        返回:
            密度场数组
        """
        density_field = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        
        for i in range(self.num_phases):
            density_field += self.phase_fields[i] * self.densities[i]
        
        return density_field
    
    def get_viscosity_field(self) -> np.ndarray:
        """计算粘度场
        
        返回:
            粘度场数组
        """
        viscosity_field = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        
        for i in range(self.num_phases):
            viscosity_field += self.phase_fields[i] * self.viscosities[i]
        
        return viscosity_field
    
    def get_phase_field(self, phase_idx: int) -> np.ndarray:
        """获取指定相的相场
        
        参数:
            phase_idx: 相索引
        
        返回:
            相场数组
        """
        if 0 <= phase_idx < self.num_phases:
            return self.phase_fields[phase_idx]
        else:
            raise ValueError(f"相索引 {phase_idx} 超出范围 [0, {self.num_phases-1}]")


class VOFModel(MultiphaseModel):
    """体积流体法(VOF)模型
    
    使用体积分数追踪界面的多相流模型。
    """
    
    def __init__(self, width: int, height: int, depth: int, num_phases: int = 2):
        """初始化VOF模型
        
        参数:
            width: 网格宽度
            height: 网格高度
            depth: 网格深度
            num_phases: 相数
        """
        super().__init__(width, height, depth, num_phases)
        
        # 初始化体积分数场
        self.volume_fractions = np.zeros((num_phases, width, height, depth), dtype=np.float32)
        
        # 用于存储临时计算结果
        self.volume_fractions_prev = np.zeros_like(self.volume_fractions)
    
    def initialize_phase(self, phase_idx: int, region_func) -> None:
        """初始化相分布
        
        参数:
            phase_idx: 相索引
            region_func: 判断点是否在相区域内的函数
        """
        super().initialize_phase(phase_idx, region_func)
        
        # 初始化体积分数
        self.volume_fractions[phase_idx] = self.phase_fields[phase_idx].copy()
        
        # 确保体积分数和为1
        self._normalize_volume_fractions()
    
    def _normalize_volume_fractions(self) -> None:
        """归一化体积分数，确保每个单元的体积分数和为1"""
        # 计算体积分数和
        total = np.sum(self.volume_fractions, axis=0)
        
        # 避免除以零
        total = np.maximum(total, 1e-6)
        
        # 归一化
        for i in range(self.num_phases):
            self.volume_fractions[i] /= total
    
    def step(self, dt: float, velocity_field: np.ndarray) -> None:
        """执行一步VOF计算
        
        参数:
            dt: 时间步长
            velocity_field: 速度场
        """
        # 保存当前体积分数
        self.volume_fractions_prev[:] = self.volume_fractions
        
        # 对每个相进行平流计算
        for phase_idx in range(self.num_phases):
            self._advect_phase(phase_idx, dt, velocity_field)
        
        # 归一化体积分数
        self._normalize_volume_fractions()
        
        # 更新相场
        for phase_idx in range(self.num_phases):
            self.phase_fields[phase_idx] = self.volume_fractions[phase_idx].copy()
    
    def _advect_phase(self, phase_idx: int, dt: float, velocity_field: np.ndarray) -> None:
        """计算相的平流
        
        参数:
            phase_idx: 相索引
            dt: 时间步长
            velocity_field: 速度场
        """
        # 创建网格坐标
        x = np.arange(self.width)
        y = np.arange(self.height)
        z = np.arange(self.depth)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算回溯位置
        pos_x = X - dt * velocity_field[:, :, :, 0]
        pos_y = Y - dt * velocity_field[:, :, :, 1]
        pos_z = Z - dt * velocity_field[:, :, :, 2]
        
        # 边界处理
        pos_x = np.clip(pos_x, 0, self.width - 1.001)
        pos_y = np.clip(pos_y, 0, self.height - 1.001)
        pos_z = np.clip(pos_z, 0, self.depth - 1.001)
        
        # 计算插值权重
        x0 = np.floor(pos_x).astype(np.int32)
        y0 = np.floor(pos_y).astype(np.int32)
        z0 = np.floor(pos_z).astype(np.int32)
        
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        # 确保索引在有效范围内
        x0 = np.clip(x0, 0, self.width - 1)
        y0 = np.clip(y0, 0, self.height - 1)
        z0 = np.clip(z0, 0, self.depth - 1)
        x1 = np.clip(x1, 0, self.width - 1)
        y1 = np.clip(y1, 0, self.height - 1)
        z1 = np.clip(z1, 0, self.depth - 1)
        
        # 计算插值系数
        sx = pos_x - x0
        sy = pos_y - y0
        sz = pos_z - z0
        
        # 三线性插值
        c000 = self.volume_fractions_prev[phase_idx, x0, y0, z0]
        c001 = self.volume_fractions_prev[phase_idx, x0, y0, z1]
        c010 = self.volume_fractions_prev[phase_idx, x0, y1, z0]
        c011 = self.volume_fractions_prev[phase_idx, x0, y1, z1]
        c100 = self.volume_fractions_prev[phase_idx, x1, y0, z0]
        c101 = self.volume_fractions_prev[phase_idx, x1, y0, z1]
        c110 = self.volume_fractions_prev[phase_idx, x1, y1, z0]
        c111 = self.volume_fractions_prev[phase_idx, x1, y1, z1]
        
        c00 = c000 * (1 - sx) + c100 * sx
        c01 = c001 * (1 - sx) + c101 * sx
        c10 = c010 * (1 - sx) + c110 * sx
        c11 = c011 * (1 - sx) + c111 * sx
        
        c0 = c00 * (1 - sy) + c10 * sy
        c1 = c01 * (1 - sy) + c11 * sy
        
        # 更新体积分数
        self.volume_fractions[phase_idx] = c0 * (1 - sz) + c1 * sz
        
        # 应用边界条件
        self._apply_boundary_conditions(phase_idx)
    
    def _apply_boundary_conditions(self, phase_idx: int) -> None:
        """应用边界条件
        
        参数:
            phase_idx: 相索引
        """
        # 零梯度边界条件
        self.volume_fractions[phase_idx, 0, :, :] = self.volume_fractions[phase_idx, 1, :, :]
        self.volume_fractions[phase_idx, -1, :, :] = self.volume_fractions[phase_idx, -2, :, :]
        self.volume_fractions[phase_idx, :, 0, :] = self.volume_fractions[phase_idx, :, 1, :]
        self.volume_fractions[phase_idx, :, -1, :] = self.volume_fractions[phase_idx, :, -2, :]
        self.volume_fractions[phase_idx, :, :, 0] = self.volume_fractions[phase_idx, :, :, 1]
        self.volume_fractions[phase_idx, :, :, -1] = self.volume_fractions[phase_idx, :, :, -2]
    
    def get_interface_field(self) -> np.ndarray:
        """获取界面场
        
        返回:
            界面场数组，值在[0,1]之间，表示界面位置
        """
        # 计算界面场 (简化为第一相的体积分数)
        return self.volume_fractions[0]


class LevelSetModel(MultiphaseModel):
    """水平集方法模型
    
    使用水平集函数追踪界面的多相流模型。
    """
    
    def __init__(self, width: int, height: int, depth: int, num_phases: int = 2):
        """初始化水平集模型
        
        参数:
            width: 网格宽度
            height: 网格高度
            depth: 网格深度
            num_phases: 相数
        """
        super().__init__(width, height, depth, num_phases)
        
        # 初始化水平集函数
        self.phi = np.ones((width, height, depth), dtype=np.float32)
        self.phi_prev = np.ones_like(self.phi)
        
        # 重初始化参数
        self.reinit_steps = 5  # 重初始化迭代次数
        self.dt_reinit = 0.1   # 重初始化时间步长
    
    def initialize_phase(self, phase_idx: int, region_func) -> None:
        """初始化相分布
        
        参数:
            phase_idx: 相索引
            region_func: 判断点是否在相区域内的函数
        """
        super().initialize_phase(phase_idx, region_func)
        
        # 初始化水平集函数
        if phase_idx == 0:  # 只对第一个相初始化水平集
            # 简单初始化为有符号距离函数：内部为负，外部为正
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        if self.phase_fields[phase_idx, x, y, z] > 0.5:
                            self.phi[x, y, z] = -1.0
                        else:
                            self.phi[x, y, z] = 1.0
            
            # 执行重初始化，使phi成为真正的有符号距离函数
            self._reinitialize()
    
    def step(self, dt: float, velocity_field: np.ndarray) -> None:
        """执行一步水平集计算
        
        参数:
            dt: 时间步长
            velocity_field: 速度场
        """
        # 保存当前水平集函数
        self.phi_prev[:] = self.phi
        
        # 计算水平集函数的平流
        self._advect_levelset(dt, velocity_field)
        
        # 重初始化水平集函数
        self._reinitialize()
        
        # 更新相场
        self._update_phase_fields()
    
    def _advect_levelset(self, dt: float, velocity_field: np.ndarray) -> None:
        """计算水平集函数的平流
        
        参数:
            dt: 时间步长
            velocity_field: 速度场
        """
        # 创建网格坐标
        x = np.arange(self.width)
        y = np.arange(self.height)
        z = np.arange(self.depth)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 计算回溯位置
        pos_x = X - dt * velocity_field[:, :, :, 0]
        pos_y = Y - dt * velocity_field[:, :, :, 1]
        pos_z = Z - dt * velocity_field[:, :, :, 2]
        
        # 边界处理
        pos_x = np.clip(pos_x, 0, self.width - 1.001)
        pos_y = np.clip(pos_y, 0, self.height - 1.001)
        pos_z = np.clip(pos_z, 0, self.depth - 1.001)
        
        # 计算插值权重
        x0 = np.floor(pos_x).astype(np.int32)
        y0 = np.floor(pos_y).astype(np.int32)
        z0 = np.floor(pos_z).astype(np.int32)
        
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        # 确保索引在有效范围内
        x0 = np.clip(x0, 0, self.width - 1)
        y0 = np.clip(y0, 0, self.height - 1)
        z0 = np.clip(z0, 0, self.depth - 1)
        x1 = np.clip(x1, 0, self.width - 1)
        y1 = np.clip(y1, 0, self.height - 1)
        z1 = np.clip(z1, 0, self.depth - 1)
        
        # 计算插值系数
        sx = pos_x - x0
        sy = pos_y - y0
        sz = pos_z - z0
        
        # 三线性插值
        c000 = self.phi_prev[x0, y0, z0]
        c001 = self.phi_prev[x0, y0, z1]
        c010 = self.phi_prev[x0, y1, z0]
        c011 = self.phi_prev[x0, y1, z1]
        c100 = self.phi_prev[x1, y0, z0]
        c101 = self.phi_prev[x1, y0, z1]
        c110 = self.phi_prev[x1, y1, z0]
        c111 = self.phi_prev[x1, y1, z1]
        
        c00 = c000 * (1 - sx) + c100 * sx
        c01 = c001 * (1 - sx) + c101 * sx
        c10 = c010 * (1 - sx) + c110 * sx
        c11 = c011 * (1 - sx) + c111 * sx
        
        c0 = c00 * (1 - sy) + c10 * sy
        c1 = c01 * (1 - sy) + c11 * sy
        
        # 更新水平集函数
        self.phi = c0 * (1 - sz) + c1 * sz
        
        # 应用边界条件
        self._apply_boundary_conditions()
    
    def _reinitialize(self) -> None:
        """重初始化水平集函数，使其成为有符号距离函数"""
        # 临时存储
        phi_temp = np.zeros_like(self.phi)
        
        # 计算符号函数
        sign_phi = self.phi / np.sqrt(self.phi**2 + 1e-6)
        
        # 迭代重初始化
        for _ in range(self.reinit_steps):
            # 计算梯度
            grad_x = np.zeros_like(self.phi)
            grad_y = np.zeros_like(self.phi)
            grad_z = np.zeros_like(self.phi)
            
            # 中心差分计算梯度
            grad_x[1:-1, :, :] = (self.phi[2:, :, :] - self.phi[:-2, :, :]) / 2.0
            grad_y[:, 1:-1, :] = (self.phi[:, 2:, :] - self.phi[:, :-2, :]) / 2.0
            grad_z[:, :, 1:-1] = (self.phi[:, :, 2:] - self.phi[:, :, :-2]) / 2.0
            
            # 计算梯度大小
            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            
            # 更新水平集函数
            phi_temp = self.phi - self.dt_reinit * sign_phi * (grad_mag - 1.0)
            
            # 更新
            self.phi = phi_temp.copy()
            
            # 应用边界条件
            self._apply_boundary_conditions()
    
    def _apply_boundary_conditions(self) -> None:
        """应用边界条件"""
        # 零梯度边界条件
        self.phi[0, :, :] = self.phi[1, :, :]
        self.phi[-1, :, :] = self.phi[-2, :, :]
        self.phi[:, 0, :] = self.phi[:, 1, :]
        self.phi[:, -1, :] = self.phi[:, -2, :]
        self.phi[:, :, 0] = self.phi[:, :, 1]
        self.phi[:, :, -1] = self.phi[:, :, -2]
    
    def _update_phase_fields(self) -> None:
        """根据水平集函数更新相场"""
        # 使用平滑Heaviside函数
        epsilon = 1.5  # 界面厚度参数
        
        # 更新第一相
        self.phase_fields[0] = 0.5 * (1.0 - np.tanh(self.phi / epsilon))
        
        # 更新其他相 (简化为二相情况)
        if self.num_phases > 1:
            self.phase_fields[1] = 1.0 - self.phase_fields[0]
    
    def get_interface_field(self) -> np.ndarray:
        """获取界面场
        
        返回:
            界面场数组，值在[0,1]之间，表示界面位置
        """
        # 计算界面场 (使用平滑Delta函数)
        epsilon = 1.5  # 界面厚度参数
        return 0.5 * (1.0 + np.cos(np.pi * self.phi / epsilon)) * (np.abs(self.phi) < epsilon) 