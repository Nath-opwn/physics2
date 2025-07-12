"""多相流模型测试

测试VOF模型和水平集方法的基本功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.advanced_physics.multiphase import VOFModel, LevelSetModel


def test_vof_model():
    """测试VOF模型的基本功能"""
    print("测试VOF模型...")
    
    # 创建VOF模型实例
    width, height, depth = 50, 50, 50
    vof = VOFModel(width, height, depth, num_phases=2)
    
    # 设置相的物理属性
    vof.set_phase_properties(0, density=1000.0, viscosity=1.0)  # 水
    vof.set_phase_properties(1, density=1.0, viscosity=0.01)    # 空气
    
    # 初始化相分布（球形液滴）
    center = (width // 2, height // 2, depth // 2)
    radius = min(width, height, depth) // 4
    
    def sphere_region(x, y, z):
        return ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < radius**2
    
    vof.initialize_phase(0, sphere_region)
    
    # 创建速度场（旋转场）
    velocity_field = np.zeros((width, height, depth, 3), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                dx, dy, dz = x - center[0], y - center[1], z - center[2]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if dist > 0:
                    # 创建旋转速度场
                    velocity_field[x, y, z, 0] = -0.1 * dy / dist
                    velocity_field[x, y, z, 1] = 0.1 * dx / dist
                    velocity_field[x, y, z, 2] = 0.05 * np.sin(dist / 10)
    
    # 运行模拟
    num_steps = 10
    dt = 0.1
    
    start_time = time.time()
    for step in range(num_steps):
        vof.step(dt, velocity_field)
        print(f"步骤 {step+1}/{num_steps} 完成")
    
    end_time = time.time()
    print(f"VOF模拟完成，用时: {end_time - start_time:.2f} 秒")
    
    # 验证结果
    phase0 = vof.get_phase_field(0)
    total_volume = np.sum(phase0)
    expected_volume = 4/3 * np.pi * radius**3
    
    print(f"初始体积: {expected_volume:.2f}")
    print(f"最终体积: {total_volume:.2f}")
    print(f"体积变化: {(total_volume - expected_volume) / expected_volume * 100:.2f}%")
    
    # 可视化中间切片
    mid_z = depth // 2
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(phase0[:, :, mid_z].T, origin='lower', cmap='Blues')
    plt.title('VOF相场 (z中间切片)')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(vof.get_density_field()[:, :, mid_z].T, origin='lower', cmap='viridis')
    plt.title('密度场 (z中间切片)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('vof_test_result.png')
    print("结果已保存到 'vof_test_result.png'")


def test_levelset_model():
    """测试水平集方法的基本功能"""
    print("\n测试水平集方法...")
    
    # 创建水平集模型实例
    width, height, depth = 50, 50, 50
    ls = LevelSetModel(width, height, depth, num_phases=2)
    
    # 设置相的物理属性
    ls.set_phase_properties(0, density=1000.0, viscosity=1.0)  # 水
    ls.set_phase_properties(1, density=1.0, viscosity=0.01)    # 空气
    
    # 初始化相分布（球形液滴）
    center = (width // 2, height // 2, depth // 2)
    radius = min(width, height, depth) // 4
    
    def sphere_region(x, y, z):
        return ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < radius**2
    
    ls.initialize_phase(0, sphere_region)
    
    # 创建速度场（旋转场）
    velocity_field = np.zeros((width, height, depth, 3), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                dx, dy, dz = x - center[0], y - center[1], z - center[2]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if dist > 0:
                    # 创建旋转速度场
                    velocity_field[x, y, z, 0] = -0.1 * dy / dist
                    velocity_field[x, y, z, 1] = 0.1 * dx / dist
                    velocity_field[x, y, z, 2] = 0.05 * np.sin(dist / 10)
    
    # 运行模拟
    num_steps = 10
    dt = 0.1
    
    start_time = time.time()
    for step in range(num_steps):
        ls.step(dt, velocity_field)
        print(f"步骤 {step+1}/{num_steps} 完成")
    
    end_time = time.time()
    print(f"水平集模拟完成，用时: {end_time - start_time:.2f} 秒")
    
    # 验证结果
    phase0 = ls.get_phase_field(0)
    total_volume = np.sum(phase0)
    expected_volume = 4/3 * np.pi * radius**3
    
    print(f"初始体积: {expected_volume:.2f}")
    print(f"最终体积: {total_volume:.2f}")
    print(f"体积变化: {(total_volume - expected_volume) / expected_volume * 100:.2f}%")
    
    # 可视化中间切片
    mid_z = depth // 2
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(phase0[:, :, mid_z].T, origin='lower', cmap='Blues')
    plt.title('水平集相场 (z中间切片)')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(ls.get_interface_field()[:, :, mid_z].T, origin='lower', cmap='hot')
    plt.title('界面场 (z中间切片)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('levelset_test_result.png')
    print("结果已保存到 'levelset_test_result.png'")


def compare_methods():
    """比较VOF和水平集方法的性能和精度"""
    print("\n比较VOF和水平集方法...")
    
    # 创建模型实例
    width, height, depth = 50, 50, 50
    vof = VOFModel(width, height, depth)
    ls = LevelSetModel(width, height, depth)
    
    # 设置相同的初始条件
    center = (width // 2, height // 2, depth // 2)
    radius = min(width, height, depth) // 4
    
    def sphere_region(x, y, z):
        return ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < radius**2
    
    vof.initialize_phase(0, sphere_region)
    ls.initialize_phase(0, sphere_region)
    
    # 创建速度场（旋转场）
    velocity_field = np.zeros((width, height, depth, 3), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                dx, dy, dz = x - center[0], y - center[1], z - center[2]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                if dist > 0:
                    # 创建旋转速度场
                    velocity_field[x, y, z, 0] = -0.1 * dy / dist
                    velocity_field[x, y, z, 1] = 0.1 * dx / dist
                    velocity_field[x, y, z, 2] = 0.05 * np.sin(dist / 10)
    
    # 运行模拟
    num_steps = 20
    dt = 0.1
    
    vof_times = []
    ls_times = []
    vof_volumes = []
    ls_volumes = []
    
    # VOF模拟
    for step in range(num_steps):
        start_time = time.time()
        vof.step(dt, velocity_field)
        end_time = time.time()
        vof_times.append(end_time - start_time)
        vof_volumes.append(np.sum(vof.get_phase_field(0)))
    
    # 水平集模拟
    for step in range(num_steps):
        start_time = time.time()
        ls.step(dt, velocity_field)
        end_time = time.time()
        ls_times.append(end_time - start_time)
        ls_volumes.append(np.sum(ls.get_phase_field(0)))
    
    # 比较结果
    print(f"VOF平均计算时间: {np.mean(vof_times):.4f} 秒/步")
    print(f"水平集平均计算时间: {np.mean(ls_times):.4f} 秒/步")
    
    expected_volume = 4/3 * np.pi * radius**3
    vof_vol_error = abs(vof_volumes[-1] - expected_volume) / expected_volume * 100
    ls_vol_error = abs(ls_volumes[-1] - expected_volume) / expected_volume * 100
    
    print(f"VOF体积误差: {vof_vol_error:.2f}%")
    print(f"水平集体积误差: {ls_vol_error:.2f}%")
    
    # 可视化比较
    plt.figure(figsize=(15, 10))
    
    # 相场比较
    plt.subplot(2, 2, 1)
    mid_z = depth // 2
    plt.imshow(vof.get_phase_field(0)[:, :, mid_z].T, origin='lower', cmap='Blues')
    plt.title('VOF相场 (z中间切片)')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(ls.get_phase_field(0)[:, :, mid_z].T, origin='lower', cmap='Blues')
    plt.title('水平集相场 (z中间切片)')
    plt.colorbar()
    
    # 体积保持比较
    plt.subplot(2, 2, 3)
    plt.plot(range(num_steps), vof_volumes, 'b-', label='VOF')
    plt.plot(range(num_steps), ls_volumes, 'r-', label='水平集')
    plt.axhline(y=expected_volume, color='k', linestyle='--', label='理论体积')
    plt.xlabel('时间步')
    plt.ylabel('体积')
    plt.legend()
    plt.title('体积保持比较')
    
    # 计算时间比较
    plt.subplot(2, 2, 4)
    plt.plot(range(num_steps), vof_times, 'b-', label='VOF')
    plt.plot(range(num_steps), ls_times, 'r-', label='水平集')
    plt.xlabel('时间步')
    plt.ylabel('计算时间 (秒)')
    plt.legend()
    plt.title('计算性能比较')
    
    plt.tight_layout()
    plt.savefig('multiphase_comparison.png')
    print("比较结果已保存到 'multiphase_comparison.png'")


if __name__ == "__main__":
    test_vof_model()
    test_levelset_model()
    compare_methods() 