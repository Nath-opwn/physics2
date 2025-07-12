"""热传导模型

实现流体热传导和对流的计算，支持温度场求解和浮力计算。
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

class ThermalModel:
    """热传导模型类
    
    实现基于能量方程的热传导和对流计算。
    """
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 depth: int, 
                 thermal_diffusivity: float = 0.1,
                 specific_heat: float = 4200.0,
                 thermal_expansion_coeff: float = 2.1e-4,
                 reference_temperature: float = 293.15,  # 20°C
                 gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)):
        """初始化热传导模型
        
        参数:
            width: 网格宽度
            height: 网格高度
            depth: 网格深度
            thermal_diffusivity: 热扩散率 (m^2/s)
            specific_heat: 比热容 (J/(kg·K))
            thermal_expansion_coeff: 热膨胀系数 (1/K)
            reference_temperature: 参考温度 (K)
            gravity: 重力向量 (m/s^2)
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.thermal_diffusivity = thermal_diffusivity
        self.specific_heat = specific_heat
        self.thermal_expansion_coeff = thermal_expansion_coeff
        self.reference_temperature = reference_temperature
        self.gravity = np.array(gravity, dtype=np.float32)
        
        # 初始化温度场
        self.temperature = np.ones((width, height, depth), dtype=np.float32) * reference_temperature
        self.temperature_prev = np.ones((width, height, depth), dtype=np.float32) * reference_temperature
        
        # 初始化热源
        self.heat_sources = np.zeros((width, height, depth), dtype=np.float32)
    
    def add_heat_source(self, x: int, y: int, z: int, power: float) -> None:
        """添加热源
        
        参数:
            x, y, z: 热源位置
            power: 热源功率 (W)
        """
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.heat_sources[x, y, z] = power
    
    def set_temperature(self, x: int, y: int, z: int, temperature: float) -> None:
        """设置指定位置的温度
        
        参数:
            x, y, z: 位置
            temperature: 温度值 (K)
        """
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.temperature[x, y, z] = temperature
            self.temperature_prev[x, y, z] = temperature
    
    def set_temperature_field(self, temperature_field: np.ndarray) -> None:
        """设置整个温度场
        
        参数:
            temperature_field: 温度场数组
        """
        if temperature_field.shape == self.temperature.shape:
            self.temperature = temperature_field.copy()
            self.temperature_prev = temperature_field.copy()
    
    def step(self, dt: float, velocity_field: np.ndarray, density_field: Optional[np.ndarray] = None) -> None:
        """执行一步热传导和对流计算
        
        参数:
            dt: 时间步长
            velocity_field: 速度场，形状为(width, height, depth, 3)
            density_field: 密度场，形状为(width, height, depth)
        """
        # 保存当前温度场
        self.temperature_prev[:] = self.temperature
        
        # 如果没有提供密度场，使用均匀密度
        if density_field is None:
            density_field = np.ones((self.width, self.height, self.depth), dtype=np.float32)
        
        # 计算热扩散
        self._diffuse(dt)
        
        # 计算热对流
        self._advect(dt, velocity_field)
        
        # 添加热源
        self._add_sources(dt, density_field)
    
    def _diffuse(self, dt: float) -> None:
        """计算热扩散
        
        参数:
            dt: 时间步长
        """
        # 热扩散系数
        alpha = dt * self.thermal_diffusivity
        
        # 雅可比迭代系数
        beta = 1.0 / (1.0 + 6.0 * alpha)
        
        # 雅可比迭代求解扩散方程
        for _ in range(20):  # 20次迭代通常足够
            self.temperature[1:-1, 1:-1, 1:-1] = (
                self.temperature_prev[1:-1, 1:-1, 1:-1] + alpha * (
                    self.temperature[2:, 1:-1, 1:-1] + self.temperature[:-2, 1:-1, 1:-1] +
                    self.temperature[1:-1, 2:, 1:-1] + self.temperature[1:-1, :-2, 1:-1] +
                    self.temperature[1:-1, 1:-1, 2:] + self.temperature[1:-1, 1:-1, :-2]
                )
            ) * beta
            
            # 应用边界条件
            self._apply_boundary_conditions()
    
    def _advect(self, dt: float, velocity_field: np.ndarray) -> None:
        """计算热对流
        
        参数:
            dt: 时间步长
            velocity_field: 速度场
        """
        # 创建临时数组存储结果
        temp = np.zeros_like(self.temperature)
        
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
        c000 = self.temperature_prev[x0, y0, z0]
        c001 = self.temperature_prev[x0, y0, z1]
        c010 = self.temperature_prev[x0, y1, z0]
        c011 = self.temperature_prev[x0, y1, z1]
        c100 = self.temperature_prev[x1, y0, z0]
        c101 = self.temperature_prev[x1, y0, z1]
        c110 = self.temperature_prev[x1, y1, z0]
        c111 = self.temperature_prev[x1, y1, z1]
        
        c00 = c000 * (1 - sx) + c100 * sx
        c01 = c001 * (1 - sx) + c101 * sx
        c10 = c010 * (1 - sx) + c110 * sx
        c11 = c011 * (1 - sx) + c111 * sx
        
        c0 = c00 * (1 - sy) + c10 * sy
        c1 = c01 * (1 - sy) + c11 * sy
        
        temp = c0 * (1 - sz) + c1 * sz
        
        # 更新温度场
        self.temperature = temp
        
        # 应用边界条件
        self._apply_boundary_conditions()
    
    def _add_sources(self, dt: float, density_field: np.ndarray) -> None:
        """添加热源
        
        参数:
            dt: 时间步长
            density_field: 密度场
        """
        # 计算温度增量 ΔT = Q * dt / (ρ * cp)
        # 其中Q是热源功率，ρ是密度，cp是比热容
        delta_T = self.heat_sources * dt / (density_field * self.specific_heat)
        
        # 更新温度场
        self.temperature += delta_T
    
    def _apply_boundary_conditions(self) -> None:
        """应用边界条件"""
        # 绝热边界条件 (零梯度)
        self.temperature[0, :, :] = self.temperature[1, :, :]
        self.temperature[-1, :, :] = self.temperature[-2, :, :]
        self.temperature[:, 0, :] = self.temperature[:, 1, :]
        self.temperature[:, -1, :] = self.temperature[:, -2, :]
        self.temperature[:, :, 0] = self.temperature[:, :, 1]
        self.temperature[:, :, -1] = self.temperature[:, :, -2]
    
    def compute_buoyancy_force(self, density_field: Optional[np.ndarray] = None) -> np.ndarray:
        """计算浮力
        
        参数:
            density_field: 密度场
        
        返回:
            浮力场，形状为(width, height, depth, 3)
        """
        # 初始化力场
        force = np.zeros((self.width, self.height, self.depth, 3), dtype=np.float32)
        
        # 如果没有提供密度场，使用均匀密度
        if density_field is None:
            density_field = np.ones((self.width, self.height, self.depth), dtype=np.float32)
        
        # 计算温度差
        delta_T = self.temperature - self.reference_temperature
        
        # 计算浮力 F_b = ρ * β * ΔT * g
        # 其中ρ是密度，β是热膨胀系数，ΔT是温度差，g是重力加速度
        for i in range(3):
            force[:, :, :, i] = density_field * self.thermal_expansion_coeff * delta_T * self.gravity[i]
        
        return force
    
    def get_temperature_field(self) -> np.ndarray:
        """获取温度场
        
        返回:
            温度场数组
        """
        return self.temperature
    
    def get_heat_source_field(self) -> np.ndarray:
        """获取热源场
        
        返回:
            热源场数组
        """
        return self.heat_sources 