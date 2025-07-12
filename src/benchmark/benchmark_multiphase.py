import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入模型
try:
    from src.models.multiphase_accelerated import AcceleratedMultiphaseModel
    HAS_ACCELERATED = True
    # 检查是否有CUDA可用
    import multiphase_core
    CUDA_AVAILABLE = multiphase_core.is_cuda_available()
except ImportError:
    HAS_ACCELERATED = False
    CUDA_AVAILABLE = False
    print("警告: 无法导入加速模型，将只测试纯Python实现")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkMultiphase:
    """多相流模型性能测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.grid_sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
        self.num_steps = 10  # 减少步数以加快测试
        self.dt = 0.1
        
        # 测试结果
        self.results = {
            'python': {},
            'cpp': {},
            'cuda': {}
        }
    
    def create_velocity_field(self, grid_size, center=None):
        """创建旋转速度场"""
        if center is None:
            center = (grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2)
            
        velocity_field = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3), dtype=np.float32)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    # 创建旋转速度场
                    dx, dy, dz = x - center[0], y - center[1], z - center[2]
                    velocity_field[x, y, z, 0] = -dy * 0.1
                    velocity_field[x, y, z, 1] = dx * 0.1
                    velocity_field[x, y, z, 2] = 0.0
        
        return velocity_field
    
    def run_test(self, grid_size, implementation='python', method='levelset'):
        """运行单个测试"""
        logger.info(f"运行测试: 网格尺寸={grid_size}, 实现={implementation}, 方法={method}")
        
        # 创建模型
        if implementation == 'python':
            model = AcceleratedMultiphaseModel(grid_size, use_acceleration=False)
        elif implementation == 'cpp':
            model = AcceleratedMultiphaseModel(grid_size, use_acceleration=True, use_cuda=False)
        elif implementation == 'cuda':
            model = AcceleratedMultiphaseModel(grid_size, use_acceleration=True, use_cuda=True)
        else:
            raise ValueError(f"未知实现: {implementation}")
        
        # 初始化球形界面
        center = (grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2)
        radius = min(grid_size) // 4
        model.initialize_sphere(center, radius)
        
        # 设置速度场
        velocity_field = self.create_velocity_field(grid_size, center)
        model.set_velocity_field(velocity_field)
        
        # 记录初始体积
        initial_volume = model.get_volume()
        
        # 计时
        start_time = time.time()
        
        # 执行时间步
        for i in range(self.num_steps):
            model.step(self.dt, method=method)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 计算最终体积变化
        final_volume = model.get_volume()
        volume_change = (final_volume - initial_volume) / initial_volume * 100
        
        # 记录结果
        result = {
            'elapsed_time': elapsed_time,
            'steps_per_second': self.num_steps / elapsed_time,
            'initial_volume': initial_volume,
            'final_volume': final_volume,
            'volume_change': volume_change
        }
        
        logger.info(f"测试完成: 耗时={elapsed_time:.2f}秒, 每秒步数={result['steps_per_second']:.2f}, "
                   f"体积变化={volume_change:.2f}%")
        
        return result
    
    def run_benchmark(self):
        """运行所有测试"""
        for grid_size in self.grid_sizes:
            grid_key = f"{grid_size[0]}x{grid_size[1]}x{grid_size[2]}"
            
            # Python实现
            self.results['python'][grid_key] = self.run_test(grid_size, implementation='python')
            
            # C++实现 (如果可用)
            if HAS_ACCELERATED:
                self.results['cpp'][grid_key] = self.run_test(grid_size, implementation='cpp')
            
            # CUDA实现 (如果可用)
            if HAS_ACCELERATED and CUDA_AVAILABLE:
                self.results['cuda'][grid_key] = self.run_test(grid_size, implementation='cuda')
    
    def plot_results(self):
        """绘制测试结果"""
        # 准备数据
        grid_labels = list(self.results['python'].keys())
        python_times = [self.results['python'][grid]['steps_per_second'] for grid in grid_labels]
        
        cpp_times = []
        cuda_times = []
        
        if HAS_ACCELERATED:
            cpp_times = [self.results['cpp'][grid]['steps_per_second'] for grid in grid_labels]
            
            if CUDA_AVAILABLE:
                cuda_times = [self.results['cuda'][grid]['steps_per_second'] for grid in grid_labels]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(grid_labels))
        width = 0.3
        
        # 绘制柱状图
        rects1 = ax.bar(x - width/2, python_times, width, label='Python')
        
        if HAS_ACCELERATED:
            rects2 = ax.bar(x + width/2, cpp_times, width, label='C++/OpenMP')
            
            if CUDA_AVAILABLE:
                rects3 = ax.bar(x + width*1.5, cuda_times, width, label='CUDA')
        
        # 添加标签和标题
        ax.set_xlabel('网格尺寸')
        ax.set_ylabel('每秒步数')
        ax.set_title('多相流模型性能比较')
        ax.set_xticks(x)
        ax.set_xticklabels(grid_labels)
        ax.legend()
        
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(rects1)
        
        if HAS_ACCELERATED:
            autolabel(rects2)
            
            if CUDA_AVAILABLE:
                autolabel(rects3)
        
        # 计算加速比
        if HAS_ACCELERATED:
            print("\n加速比:")
            for grid in grid_labels:
                python_time = self.results['python'][grid]['elapsed_time']
                cpp_time = self.results['cpp'][grid]['elapsed_time']
                cpp_speedup = python_time / cpp_time
                
                print(f"{grid}: C++/OpenMP vs Python: {cpp_speedup:.2f}x")
                
                if CUDA_AVAILABLE:
                    cuda_time = self.results['cuda'][grid]['elapsed_time']
                    cuda_speedup = python_time / cuda_time
                    cuda_vs_cpp = cpp_time / cuda_time
                    
                    print(f"{grid}: CUDA vs Python: {cuda_speedup:.2f}x")
                    print(f"{grid}: CUDA vs C++/OpenMP: {cuda_vs_cpp:.2f}x")
        
        # 保存图表
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        plt.show()
    
    def run_and_visualize(self, grid_size=(32, 32, 32), implementation='cpp'):
        """运行并可视化模拟"""
        # 创建模型
        if implementation == 'python':
            model = AcceleratedMultiphaseModel(grid_size, use_acceleration=False)
        elif implementation == 'cpp':
            model = AcceleratedMultiphaseModel(grid_size, use_acceleration=True, use_cuda=False)
        elif implementation == 'cuda':
            model = AcceleratedMultiphaseModel(grid_size, use_acceleration=True, use_cuda=True)
        else:
            raise ValueError(f"未知实现: {implementation}")
        
        # 初始化球形界面
        center = (grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2)
        radius = min(grid_size) // 4
        model.initialize_sphere(center, radius)
        
        # 设置速度场
        velocity_field = self.create_velocity_field(grid_size, center)
        model.set_velocity_field(velocity_field)
        
        # 创建图表
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # 初始数据
        slice_idx = grid_size[2] // 2
        im1 = ax1.imshow(model.volume_fractions[0, :, :, slice_idx], cmap='viridis', vmin=0, vmax=1)
        im2 = ax2.imshow(model.phi[:, :, slice_idx], cmap='RdBu', vmin=-radius, vmax=radius)
        
        ax1.set_title('体积分数')
        ax2.set_title('水平集函数')
        
        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        
        # 记录体积
        volumes = [model.get_volume()]
        
        # 动画更新函数
        def update(frame):
            model.step(0.1, method='levelset')
            
            # 更新图像
            im1.set_array(model.volume_fractions[0, :, :, slice_idx])
            im2.set_array(model.phi[:, :, slice_idx])
            
            # 记录体积
            volumes.append(model.get_volume())
            
            # 更新标题
            volume_change = (volumes[-1] - volumes[0]) / volumes[0] * 100
            ax1.set_title(f'体积分数 (步骤: {frame+1}, 变化: {volume_change:.2f}%)')
            
            return im1, im2
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=20, interval=100, blit=True)
        
        # 显示动画
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 运行基准测试
    benchmark = BenchmarkMultiphase()
    
    # 运行所有测试
    benchmark.run_benchmark()
    
    # 绘制结果
    benchmark.plot_results()
    
    # 可视化模拟 (可选)
    # benchmark.run_and_visualize() 