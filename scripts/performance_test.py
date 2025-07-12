#!/usr/bin/env python3
"""
流体动力学模拟系统性能测试脚本

比较CPU和GPU版本的性能差异
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import argparse
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fluid_solver import FluidSolver
try:
    from src.core.cuda_ext import CudaFluidSolver, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_performance_test(sizes, num_steps=100, use_gpu=True):
    """
    运行性能测试
    
    参数:
    - sizes: 网格尺寸列表 [(width, height, depth), ...]
    - num_steps: 每个尺寸运行的步数
    - use_gpu: 是否使用GPU
    
    返回:
    - 性能结果字典
    """
    results = {
        'grid_sizes': [],
        'total_cells': [],
        'cpu_times': [],
        'gpu_times': [],
        'speedup': []
    }
    
    for size in sizes:
        width, height, depth = size
        total_cells = width * height * depth
        
        logger.info(f"测试网格尺寸: {width}x{height}x{depth} (总计 {total_cells} 个单元)")
        results['grid_sizes'].append(f"{width}x{height}x{depth}")
        results['total_cells'].append(total_cells)
        
        # CPU测试
        logger.info("运行CPU版本...")
        cpu_solver = FluidSolver(width, height, depth, viscosity=0.1, density=1.0)
        
        # 预热
        for _ in range(5):
            cpu_solver.step(0.01)
        
        # 重置性能统计
        cpu_solver.reset_performance_stats()
        
        # 计时测试
        start_time = time.time()
        for _ in range(num_steps):
            cpu_solver.step(0.01)
        cpu_time = time.time() - start_time
        
        logger.info(f"CPU版本完成 {num_steps} 步，耗时 {cpu_time:.2f} 秒")
        results['cpu_times'].append(cpu_time)
        
        # GPU测试（如果可用）
        if use_gpu and CUDA_AVAILABLE:
            logger.info("运行GPU版本...")
            try:
                gpu_solver = CudaFluidSolver(width, height, depth, viscosity=0.1, density=1.0)
                
                # 预热
                for _ in range(5):
                    gpu_solver.step(0.01)
                
                # 重置性能统计
                gpu_solver.reset_performance_stats()
                
                # 计时测试
                start_time = time.time()
                for _ in range(num_steps):
                    gpu_solver.step(0.01)
                gpu_time = time.time() - start_time
                
                logger.info(f"GPU版本完成 {num_steps} 步，耗时 {gpu_time:.2f} 秒")
                results['gpu_times'].append(gpu_time)
                
                # 计算加速比
                speedup = cpu_time / gpu_time
                results['speedup'].append(speedup)
                logger.info(f"GPU加速比: {speedup:.2f}x")
            
            except Exception as e:
                logger.error(f"GPU测试失败: {str(e)}")
                results['gpu_times'].append(None)
                results['speedup'].append(None)
        else:
            results['gpu_times'].append(None)
            results['speedup'].append(None)
            if not CUDA_AVAILABLE:
                logger.warning("CUDA不可用，跳过GPU测试")
            
        logger.info("-" * 50)
    
    return results

def plot_results(results, output_file=None):
    """绘制性能测试结果"""
    plt.figure(figsize=(14, 10))
    
    # 创建两个子图
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    # 准备数据
    grid_sizes = results['grid_sizes']
    total_cells = results['total_cells']
    cpu_times = results['cpu_times']
    gpu_times = results['gpu_times']
    speedup = results['speedup']
    
    # 绘制运行时间对比
    x = np.arange(len(grid_sizes))
    width = 0.35
    
    cpu_bars = ax1.bar(x - width/2, cpu_times, width, label='CPU')
    
    # 只有在GPU数据可用时才绘制
    if any(t is not None for t in gpu_times):
        gpu_bars = ax1.bar(x + width/2, [t if t is not None else 0 for t in gpu_times], width, label='GPU')
    
    ax1.set_ylabel('运行时间 (秒)')
    ax1.set_title('CPU vs GPU 运行时间对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grid_sizes)
    ax1.legend()
    
    # 添加数值标签
    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            if value is not None:
                height = bar.get_height()
                ax1.annotate(f'{value:.2f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    add_labels(cpu_bars, cpu_times)
    if any(t is not None for t in gpu_times):
        add_labels(gpu_bars, gpu_times)
    
    # 绘制加速比
    if any(s is not None for s in speedup):
        ax2.plot(x, [s if s is not None else 0 for s in speedup], 'o-', linewidth=2, markersize=8)
        ax2.set_ylabel('加速比 (CPU时间 / GPU时间)')
        ax2.set_title('GPU加速比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(grid_sizes)
        ax2.grid(True)
        
        # 添加数值标签
        for i, s in enumerate(speedup):
            if s is not None:
                ax2.annotate(f'{s:.2f}x',
                            xy=(i, s),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center')
        
        # 添加辅助线
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    # 设置x轴标签
    plt.xlabel('网格尺寸')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"结果图表已保存到: {output_file}")
    
    # 显示图表
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='流体动力学模拟系统性能测试')
    parser.add_argument('--sizes', type=str, default="32,32,32,64,64,64,96,96,96,128,128,128",
                        help='测试网格尺寸，格式为"width1,height1,depth1,width2,height2,depth2,..."')
    parser.add_argument('--steps', type=int, default=50,
                        help='每个尺寸运行的步数')
    parser.add_argument('--output', type=str, default="performance_comparison.png",
                        help='输出图表文件名')
    parser.add_argument('--no-gpu', action='store_true',
                        help='不使用GPU测试')
    
    args = parser.parse_args()
    
    # 解析网格尺寸
    try:
        size_values = list(map(int, args.sizes.split(',')))
        if len(size_values) % 3 != 0:
            raise ValueError("网格尺寸参数数量必须是3的倍数")
        
        sizes = []
        for i in range(0, len(size_values), 3):
            sizes.append((size_values[i], size_values[i+1], size_values[i+2]))
    except Exception as e:
        logger.error(f"解析网格尺寸参数失败: {str(e)}")
        return
    
    # 运行测试
    results = run_performance_test(sizes, args.steps, not args.no_gpu)
    
    # 绘制结果
    plot_results(results, args.output)

if __name__ == "__main__":
    main() 