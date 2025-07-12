"""多相流与物理耦合示例

展示如何将多相流模型与表面张力和热传导模型结合使用。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.advanced_physics.multiphase import VOFModel, LevelSetModel
from src.core.advanced_physics.surface_tension import SurfaceTensionModel
from src.core.advanced_physics.thermal import ThermalModel


def create_velocity_field(width, height, depth, center, gravity=np.array([0, -9.8, 0])):
    """创建初始速度场
    
    参数:
        width, height, depth: 网格尺寸
        center: 中心点坐标
        gravity: 重力向量
        
    返回:
        速度场数组
    """
    velocity_field = np.zeros((width, height, depth, 3), dtype=np.float32)
    
    # 添加小的随机扰动
    np.random.seed(42)
    velocity_field += np.random.normal(0, 0.01, velocity_field.shape)
    
    return velocity_field


def run_multiphase_with_surface_tension():
    """运行多相流与表面张力耦合的示例"""
    print("运行多相流与表面张力耦合示例...")
    
    # 创建模型实例
    width, height, depth = 64, 64, 64
    
    # 使用水平集方法
    ls = LevelSetModel(width, height, depth, num_phases=2)
    
    # 设置相的物理属性
    ls.set_phase_properties(0, density=1000.0, viscosity=1.0)  # 水
    ls.set_phase_properties(1, density=1.0, viscosity=0.01)    # 空气
    
    # 初始化相分布（变形液滴）
    center = (width // 2, height // 2, depth // 2)
    radius = min(width, height, depth) // 5
    
    def ellipsoid_region(x, y, z):
        return ((x - center[0])**2 / (1.5*radius)**2 + 
                (y - center[1])**2 / radius**2 + 
                (z - center[2])**2 / (0.8*radius)**2) < 1.0
    
    ls.initialize_phase(0, ellipsoid_region)
    
    # 创建表面张力模型，直接在构造函数中设置表面张力系数
    surface_tension = SurfaceTensionModel(width, height, depth, surface_tension_coefficient=0.07)
    
    # 创建初始速度场
    velocity_field = create_velocity_field(width, height, depth, center)
    
    # 运行模拟
    num_steps = 50
    dt = 0.05
    
    # 存储结果
    phase_history = []
    interface_history = []
    
    for step in range(num_steps):
        # 获取界面法向和曲率
        interface_field = ls.get_interface_field()
        
        # 计算表面张力力
        surface_tension.update_interface_field(ls.get_phase_field(0))
        surface_tension_force = surface_tension.compute_surface_tension_force()
        
        # 更新速度场（简化，实际应用中需要求解Navier-Stokes方程）
        # 这里我们只添加表面张力的影响
        for i in range(3):
            velocity_field[:, :, :, i] += dt * surface_tension_force[:, :, :, i] / ls.get_density_field()
        
        # 执行水平集计算步骤
        ls.step(dt, velocity_field)
        
        # 存储结果
        if step % 5 == 0:
            phase_history.append(ls.get_phase_field(0).copy())
            interface_history.append(ls.get_interface_field().copy())
            print(f"步骤 {step+1}/{num_steps} 完成")
    
    # 可视化结果
    mid_z = depth // 2
    
    # 创建动画帧
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (phase, interface) in enumerate(zip(phase_history, interface_history)):
        axes[0].clear()
        axes[1].clear()
        
        im1 = axes[0].imshow(phase[:, :, mid_z].T, origin='lower', cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title(f'相场 (步骤 {i*5})')
        
        im2 = axes[1].imshow(interface[:, :, mid_z].T, origin='lower', cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'界面场 (步骤 {i*5})')
        
        plt.tight_layout()
        plt.savefig(f'surface_tension_step_{i*5:03d}.png')
    
    print("表面张力示例结果已保存")


def run_multiphase_with_thermal():
    """运行多相流与热传导耦合的示例"""
    print("\n运行多相流与热传导耦合示例...")
    
    # 创建模型实例
    width, height, depth = 64, 64, 64
    
    # 使用VOF方法
    vof = VOFModel(width, height, depth, num_phases=2)
    
    # 设置相的物理属性
    vof.set_phase_properties(0, density=1000.0, viscosity=1.0)  # 水
    vof.set_phase_properties(1, density=1.0, viscosity=0.01)    # 空气
    
    # 初始化相分布（分层）
    def stratified_region(x, y, z):
        return y < height // 2
    
    vof.initialize_phase(0, stratified_region)
    
    # 创建热传导模型
    thermal = ThermalModel(width, height, depth)
    
    # 设置热物理属性 - 直接在构造函数中设置
    # 初始化温度场（底部高温，顶部低温）
    temperature = np.zeros((width, height, depth), dtype=np.float32)
    for y in range(height):
        temperature[:, y, :] = 20 + 80 * (1 - y / height)  # 20-100°C
    
    thermal.set_temperature_field(temperature)
    
    # 创建初始速度场
    velocity_field = create_velocity_field(width, height, depth, (width//2, height//2, depth//2))
    
    # 运行模拟
    num_steps = 50
    dt = 0.05
    
    # 存储结果
    phase_history = []
    temp_history = []
    
    for step in range(num_steps):
        # 获取当前密度场
        density_field = vof.get_density_field()
        
        # 计算浮力
        buoyancy_force = thermal.compute_buoyancy_force(density_field)
        
        # 更新速度场（简化，实际应用中需要求解Navier-Stokes方程）
        for i in range(3):
            velocity_field[:, :, :, i] += dt * buoyancy_force[:, :, :, i]
        
        # 执行VOF计算步骤
        vof.step(dt, velocity_field)
        
        # 更新温度场
        thermal.step(dt, velocity_field, density_field)
        
        # 存储结果
        if step % 5 == 0:
            phase_history.append(vof.get_phase_field(0).copy())
            temp_history.append(thermal.get_temperature_field().copy())
            print(f"步骤 {step+1}/{num_steps} 完成")
    
    # 可视化结果
    mid_z = depth // 2
    
    # 创建动画帧
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (phase, temp) in enumerate(zip(phase_history, temp_history)):
        axes[0].clear()
        axes[1].clear()
        
        im1 = axes[0].imshow(phase[:, :, mid_z].T, origin='lower', cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title(f'相场 (步骤 {i*5})')
        
        im2 = axes[1].imshow(temp[:, :, mid_z].T, origin='lower', cmap='hot', vmin=20, vmax=100)
        axes[1].set_title(f'温度场 (步骤 {i*5})')
        plt.colorbar(im2, ax=axes[1], label='温度 (°C)')
        
        plt.tight_layout()
        plt.savefig(f'thermal_step_{i*5:03d}.png')
    
    print("热传导示例结果已保存")


def run_combined_physics():
    """运行结合多相流、表面张力和热传导的综合示例"""
    print("\n运行综合物理模型示例...")
    
    # 创建模型实例
    width, height, depth = 64, 64, 64
    
    # 使用水平集方法
    ls = LevelSetModel(width, height, depth, num_phases=2)
    
    # 设置相的物理属性
    ls.set_phase_properties(0, density=1000.0, viscosity=1.0)  # 水
    ls.set_phase_properties(1, density=1.0, viscosity=0.01)    # 空气
    
    # 初始化相分布（两个液滴）
    center1 = (width // 3, height // 2, depth // 2)
    center2 = (2 * width // 3, height // 2, depth // 2)
    radius = min(width, height, depth) // 8
    
    def two_drops_region(x, y, z):
        dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2)
        dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2)
        return dist1 < radius or dist2 < radius
    
    ls.initialize_phase(0, two_drops_region)
    
    # 创建表面张力模型
    surface_tension = SurfaceTensionModel(width, height, depth, surface_tension_coefficient=0.07)
    
    # 创建热传导模型
    thermal = ThermalModel(width, height, depth)
    
    # 初始化温度场（一个液滴热，一个液滴冷）
    temperature = np.ones((width, height, depth), dtype=np.float32) * 20  # 基础温度20°C
    
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2)
                if dist1 < radius:
                    temperature[x, y, z] = 80  # 热液滴
                
                dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2)
                if dist2 < radius:
                    temperature[x, y, z] = 10  # 冷液滴
    
    thermal.set_temperature_field(temperature)
    
    # 创建初始速度场（两个液滴相向移动）
    velocity_field = np.zeros((width, height, depth, 3), dtype=np.float32)
    
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2)
                dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2)
                
                if dist1 < radius * 1.5:
                    velocity_field[x, y, z, 0] = 0.5  # 右移
                
                if dist2 < radius * 1.5:
                    velocity_field[x, y, z, 0] = -0.5  # 左移
    
    # 运行模拟
    num_steps = 100
    dt = 0.05
    
    # 存储结果
    phase_history = []
    temp_history = []
    
    for step in range(num_steps):
        # 获取当前密度场
        density_field = ls.get_density_field()
        
        # 计算表面张力力
        surface_tension.update_interface_field(ls.get_phase_field(0))
        surface_tension_force = surface_tension.compute_surface_tension_force()
        
        # 计算浮力
        buoyancy_force = thermal.compute_buoyancy_force(density_field)
        
        # 更新速度场（简化，实际应用中需要求解Navier-Stokes方程）
        for i in range(3):
            # 添加表面张力和浮力
            velocity_field[:, :, :, i] += dt * (
                surface_tension_force[:, :, :, i] / density_field +
                buoyancy_force[:, :, :, i]
            )
        
        # 执行水平集计算步骤
        ls.step(dt, velocity_field)
        
        # 更新温度场
        thermal.step(dt, velocity_field, density_field)
        
        # 存储结果
        if step % 10 == 0:
            phase_history.append(ls.get_phase_field(0).copy())
            temp_history.append(thermal.get_temperature_field().copy())
            print(f"步骤 {step+1}/{num_steps} 完成")
    
    # 可视化结果
    mid_z = depth // 2
    
    # 创建动画帧
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (phase, temp) in enumerate(zip(phase_history, temp_history)):
        axes[0].clear()
        axes[1].clear()
        
        im1 = axes[0].imshow(phase[:, :, mid_z].T, origin='lower', cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title(f'相场 (步骤 {i*10})')
        
        im2 = axes[1].imshow(temp[:, :, mid_z].T, origin='lower', cmap='hot', vmin=10, vmax=80)
        axes[1].set_title(f'温度场 (步骤 {i*10})')
        plt.colorbar(im2, ax=axes[1], label='温度 (°C)')
        
        plt.tight_layout()
        plt.savefig(f'combined_step_{i*10:03d}.png')
    
    print("综合物理模型示例结果已保存")


if __name__ == "__main__":
    run_multiphase_with_surface_tension()
    run_multiphase_with_thermal()
    run_combined_physics() 