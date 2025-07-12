#!/usr/bin/env python3
"""
CUDA扩展测试脚本
"""
import numpy as np
import time
import logging
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入求解器
from src.core.fluid_solver import FluidSolver
from src.core.cuda_ext import CudaFluidSolver, CUDA_AVAILABLE

def test_performance_comparison(size=64, steps=100):
    """
    比较CPU和CUDA版本的性能
    """
    if not CUDA_AVAILABLE:
        logging.warning("CUDA不可用，无法进行性能比较")
        return
    
    # 初始化求解器
    cpu_solver = FluidSolver(size, size, size, viscosity=0.1)
    cuda_solver = CudaFluidSolver(size, size, size, viscosity=0.1)
    
    # 添加相同的初始条件
    # 在中心添加力
    center = size // 2
    for solver in [cpu_solver, cuda_solver]:
        solver.add_force(center, center, center, 0, 0, 10.0)
        
        # 添加障碍物
        solver.add_obstacle("sphere", {"center": [center, center, center], "radius": size // 8})
    
    # 测试CPU版本
    logging.info("开始测试CPU版本...")
    cpu_start = time.time()
    for i in range(steps):
        if i % 10 == 0:
            logging.info(f"CPU步骤: {i}/{steps}")
        cpu_solver.step(0.1)
    cpu_time = time.time() - cpu_start
    
    # 测试CUDA版本
    logging.info("开始测试CUDA版本...")
    cuda_start = time.time()
    for i in range(steps):
        if i % 10 == 0:
            logging.info(f"CUDA步骤: {i}/{steps}")
        cuda_solver.step(0.1)
    cuda_time = time.time() - cuda_start
    
    # 获取性能统计
    cpu_stats = cpu_solver.get_performance_stats()
    cuda_stats = cuda_solver.get_performance_stats()
    
    # 输出结果
    logging.info(f"网格大小: {size}x{size}x{size}, 步数: {steps}")
    logging.info(f"CPU总时间: {cpu_time:.4f}秒, 每步平均: {cpu_stats['avg_step_time']:.4f}秒")
    logging.info(f"CUDA总时间: {cuda_time:.4f}秒, 每步平均: {cuda_stats['avg_step_time']:.4f}秒")
    logging.info(f"加速比: {cpu_time / cuda_time:.2f}x")
    
    # 验证结果是否一致
    cpu_vel = cpu_solver.get_velocity_field()
    cuda_vel = cuda_solver.get_velocity_field()
    
    # 计算差异
    diff = np.abs(cpu_vel - cuda_vel).mean()
    max_diff = np.abs(cpu_vel - cuda_vel).max()
    logging.info(f"速度场平均差异: {diff:.6f}, 最大差异: {max_diff:.6f}")
    
    return {
        "cpu_time": cpu_time,
        "cuda_time": cuda_time,
        "speedup": cpu_time / cuda_time,
        "avg_diff": diff,
        "max_diff": max_diff
    }

def visualize_flow(solver, slice_pos=None):
    """
    可视化流场
    """
    # 获取速度场
    velocity = solver.get_velocity_field()
    vorticity = solver.get_vorticity_field()
    
    # 计算速度大小和涡量大小
    vel_mag = np.sqrt(np.sum(velocity**2, axis=3))
    vort_mag = np.sqrt(np.sum(vorticity**2, axis=3))
    
    # 如果没有指定切片位置，使用中心
    if slice_pos is None:
        slice_pos = velocity.shape[0] // 2
    
    # 创建图形
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制速度场
    im1 = axs[0].imshow(vel_mag[:, :, slice_pos].T, origin='lower', cmap='viridis')
    axs[0].set_title('速度场大小')
    fig.colorbar(im1, ax=axs[0])
    
    # 绘制涡量场
    im2 = axs[1].imshow(vort_mag[:, :, slice_pos].T, origin='lower', cmap='plasma')
    axs[1].set_title('涡量场大小')
    fig.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig('flow_visualization.png')
    logging.info("可视化结果已保存为 flow_visualization.png")

def main():
    """
    主函数
    """
    # 测试不同大小的网格
    sizes = [32, 64, 128]
    results = []
    
    for size in sizes:
        logging.info(f"测试网格大小: {size}x{size}x{size}")
        result = test_performance_comparison(size=size, steps=50)
        if result:
            results.append((size, result))
    
    # 如果有结果，创建加速比图表
    if results:
        sizes = [r[0] for r in results]
        speedups = [r[1]["speedup"] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, speedups, 'o-', linewidth=2)
        plt.xlabel('网格大小')
        plt.ylabel('加速比 (CPU时间/CUDA时间)')
        plt.title('CUDA加速比随网格大小的变化')
        plt.grid(True)
        plt.savefig('cuda_speedup.png')
        logging.info("加速比图表已保存为 cuda_speedup.png")
    
    # 创建一个更大的模拟并可视化
    if CUDA_AVAILABLE:
        logging.info("创建可视化示例...")
        size = 64
        solver = CudaFluidSolver(size, size, size, viscosity=0.05)
        
        # 添加一些有趣的初始条件
        center = size // 2
        solver.add_force(center, center, center, 0, 0, 20.0)
        solver.add_force(center + 10, center, center, 5.0, 0, 0)
        solver.add_force(center - 10, center, center, -5.0, 0, 0)
        
        # 添加障碍物
        solver.add_obstacle("sphere", {"center": [center, center, center + 15], "radius": size // 10})
        
        # 运行模拟
        for i in range(100):
            if i % 10 == 0:
                logging.info(f"可视化步骤: {i}/100")
            solver.step(0.1)
        
        # 可视化结果
        visualize_flow(solver)

if __name__ == "__main__":
    main() 