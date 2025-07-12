"""表面张力模型

实现流体表面张力的计算和应用，支持界面追踪和表面张力力的计算。
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

class SurfaceTensionModel:
    """表面张力模型类
    
    实现基于连续表面力(CSF)方法的表面张力计算。
    """
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 depth: int, 
                 surface_tension_coefficient: float = 0.07,
                 contact_angle: float = 90.0):
        """初始化表面张力模型
        
        参数:
            width: 网格宽度
            height: 网格高度
            depth: 网格深度
            surface_tension_coefficient: 表面张力系数 (N/m)
            contact_angle: 接触角 (度)
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.surface_tension_coefficient = surface_tension_coefficient
        self.contact_angle_rad = np.radians(contact_angle)
        
        # 初始化界面追踪场
        self.interface_field = np.zeros((width, height, depth), dtype=np.float32)
        
        # 初始化曲率场
        self.curvature = np.zeros((width, height, depth), dtype=np.float32)
        
        # 初始化表面法线场
        self.normals = np.zeros((width, height, depth, 3), dtype=np.float32)
        
        # 缓存梯度计算的临时数组
        self._tmp_grad = np.zeros((width, height, depth, 3), dtype=np.float32)
    
    def update_interface_field(self, phase_field: np.ndarray) -> None:
        """更新界面追踪场
        
        参数:
            phase_field: 相场数组，值在[0,1]之间，表示流体相分数
        """
        self.interface_field = phase_field.copy()
        
        # 计算界面场的梯度（法线方向）
        self._compute_normals()
        
        # 计算界面曲率
        self._compute_curvature()
    
    def _compute_normals(self) -> None:
        """计算界面法线方向"""
        # 计算梯度
        grad_x = np.zeros_like(self.interface_field)
        grad_y = np.zeros_like(self.interface_field)
        grad_z = np.zeros_like(self.interface_field)
        
        # 中心差分计算梯度
        grad_x[1:-1, :, :] = (self.interface_field[2:, :, :] - self.interface_field[:-2, :, :]) / 2.0
        grad_y[:, 1:-1, :] = (self.interface_field[:, 2:, :] - self.interface_field[:, :-2, :]) / 2.0
        grad_z[:, :, 1:-1] = (self.interface_field[:, :, 2:] - self.interface_field[:, :, :-2]) / 2.0
        
        # 计算梯度大小
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-10
        
        # 计算单位法线向量
        self.normals[:, :, :, 0] = grad_x / grad_magnitude
        self.normals[:, :, :, 1] = grad_y / grad_magnitude
        self.normals[:, :, :, 2] = grad_z / grad_magnitude
        
        # 存储梯度大小（用于界面检测）
        self._tmp_grad[:, :, :, 0] = grad_x
        self._tmp_grad[:, :, :, 1] = grad_y
        self._tmp_grad[:, :, :, 2] = grad_z
    
    def _compute_curvature(self) -> None:
        """计算界面曲率"""
        # 曲率计算 (散度法)
        div_x = np.zeros_like(self.interface_field)
        div_y = np.zeros_like(self.interface_field)
        div_z = np.zeros_like(self.interface_field)
        
        # 计算法线向量的散度
        div_x[1:-1, :, :] = (self.normals[2:, :, :, 0] - self.normals[:-2, :, :, 0]) / 2.0
        div_y[:, 1:-1, :] = (self.normals[:, 2:, :, 1] - self.normals[:, :-2, :, 1]) / 2.0
        div_z[:, :, 1:-1] = (self.normals[:, :, 2:, 2] - self.normals[:, :, :-2, 2]) / 2.0
        
        # 曲率 = -散度(法线)
        self.curvature = -(div_x + div_y + div_z)
        
        # 仅在界面附近计算曲率
        interface_mask = np.abs(self.interface_field - 0.5) < 0.25
        self.curvature = self.curvature * interface_mask
    
    def compute_surface_tension_force(self) -> np.ndarray:
        """计算表面张力力
        
        返回:
            表面张力力场，形状为(width, height, depth, 3)
        """
        # 初始化力场
        force = np.zeros((self.width, self.height, self.depth, 3), dtype=np.float32)
        
        # 计算界面梯度大小
        grad_magnitude = np.sqrt(np.sum(self._tmp_grad**2, axis=3))
        
        # 仅在界面附近应用表面张力
        interface_mask = (grad_magnitude > 0.01)
        
        # 计算表面张力力 F_st = sigma * kappa * grad(c) * delta_s
        # 其中sigma是表面张力系数，kappa是曲率，grad(c)是界面法线，delta_s是界面Dirac函数
        for i in range(3):
            force[:, :, :, i] = self.surface_tension_coefficient * self.curvature * self.normals[:, :, :, i] * grad_magnitude
        
        # 应用界面掩码
        for i in range(3):
            force[:, :, :, i] = force[:, :, :, i] * interface_mask
        
        return force
    
    def apply_boundary_conditions(self, solid_mask: Optional[np.ndarray] = None) -> None:
        """应用边界条件
        
        参数:
            solid_mask: 固体边界掩码，True表示固体单元
        """
        if solid_mask is None:
            return
        
        # 在固体边界处应用接触角边界条件
        # 这里简化处理，实际应用中需要更复杂的接触角处理
        boundary_cells = np.zeros_like(solid_mask, dtype=bool)
        
        # 检测固体边界附近的流体单元
        boundary_cells[1:-1, 1:-1, 1:-1] = (
            (~solid_mask[1:-1, 1:-1, 1:-1]) & 
            ((solid_mask[2:, 1:-1, 1:-1]) | (solid_mask[:-2, 1:-1, 1:-1]) |
             (solid_mask[1:-1, 2:, 1:-1]) | (solid_mask[1:-1, :-2, 1:-1]) |
             (solid_mask[1:-1, 1:-1, 2:]) | (solid_mask[1:-1, 1:-1, :-2]))
        )
        
        # 在边界单元处应用接触角
        if np.any(boundary_cells):
            # 计算到固体的方向
            solid_direction = np.zeros((self.width, self.height, self.depth, 3), dtype=np.float32)
            
            # 简化：假设固体方向是从边界单元到最近固体单元的方向
            for x in range(1, self.width-1):
                for y in range(1, self.height-1):
                    for z in range(1, self.depth-1):
                        if boundary_cells[x, y, z]:
                            # 检查六个方向
                            dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
                            for dx, dy, dz in dirs:
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                                    if solid_mask[nx, ny, nz]:
                                        solid_direction[x, y, z, 0] = -dx
                                        solid_direction[x, y, z, 1] = -dy
                                        solid_direction[x, y, z, 2] = -dz
                                        break
            
            # 归一化固体方向
            solid_dir_mag = np.sqrt(np.sum(solid_direction**2, axis=3)) + 1e-10
            for i in range(3):
                solid_direction[:, :, :, i] = solid_direction[:, :, :, i] / solid_dir_mag
            
            # 应用接触角边界条件
            # n_wall = n_interface * cos(theta) + t * sin(theta)
            # 其中n_wall是壁面法线，n_interface是界面法线，t是切向量
            cos_theta = np.cos(self.contact_angle_rad)
            sin_theta = np.sin(self.contact_angle_rad)
            
            # 计算切向量 (简化为界面法线和固体方向的叉积的单位向量)
            tangent = np.zeros((self.width, self.height, self.depth, 3), dtype=np.float32)
            tangent[:, :, :, 0] = self.normals[:, :, :, 1] * solid_direction[:, :, :, 2] - self.normals[:, :, :, 2] * solid_direction[:, :, :, 1]
            tangent[:, :, :, 1] = self.normals[:, :, :, 2] * solid_direction[:, :, :, 0] - self.normals[:, :, :, 0] * solid_direction[:, :, :, 2]
            tangent[:, :, :, 2] = self.normals[:, :, :, 0] * solid_direction[:, :, :, 1] - self.normals[:, :, :, 1] * solid_direction[:, :, :, 0]
            
            # 归一化切向量
            tangent_mag = np.sqrt(np.sum(tangent**2, axis=3)) + 1e-10
            for i in range(3):
                tangent[:, :, :, i] = tangent[:, :, :, i] / tangent_mag
            
            # 修改边界单元的法线
            for i in range(3):
                self.normals[boundary_cells, i] = (
                    self.normals[boundary_cells, i] * cos_theta + 
                    tangent[boundary_cells, i] * sin_theta
                )
    
    def get_interface_field(self) -> np.ndarray:
        """获取界面场
        
        返回:
            界面场数组
        """
        return self.interface_field
    
    def get_curvature_field(self) -> np.ndarray:
        """获取曲率场
        
        返回:
            曲率场数组
        """
        return self.curvature
    
    def get_normal_field(self) -> np.ndarray:
        """获取法线场
        
        返回:
            法线场数组
        """
        return self.normals 