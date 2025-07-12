import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from matplotlib.colors import LinearSegmentedColormap

# 导入加速模块
try:
    import multiphase_core
    print("成功导入加速模块")
    print(f"CUDA可用: {multiphase_core.is_cuda_available()}")
except ImportError:
    print("无法导入加速模块")
    sys.exit(1)

# 导入Python接口
try:
    from src.models.multiphase_accelerated import AcceleratedMultiphaseModel
    print("成功导入Python接口")
except ImportError:
    print("无法导入Python接口")
    sys.exit(1)

# 创建自定义颜色映射
def create_custom_colormap():
    # 相1: 蓝色
    # 相2: 红色
    # 界面: 黄色
    colors = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]
    return LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

# 可视化多相流
def visualize_multiphase(model, step, save_path=None):
    # 获取中心切片
    x_mid = model.width // 2
    y_mid = model.height // 2
    z_mid = model.depth // 2
    
    # 获取数据
    phi_slice_xz = model.phi[x_mid, :, :]
    phi_slice_xy = model.phi[:, :, z_mid]
    
    # 创建颜色映射
    cmap = create_custom_colormap()
    
    # 创建图形
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制XZ切片
    im1 = axs[0].imshow(phi_slice_xz, cmap='coolwarm', vmin=-2, vmax=2, origin='lower')
    axs[0].set_title(f'水平集函数 (XZ切片, 步骤={step})')
    axs[0].set_xlabel('Z')
    axs[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axs[0])
    
    # 绘制XY切片
    im2 = axs[1].imshow(phi_slice_xy, cmap='coolwarm', vmin=-2, vmax=2, origin='lower')
    axs[1].set_title(f'水平集函数 (XY切片, 步骤={step})')
    axs[1].set_xlabel('Y')
    axs[1].set_ylabel('X')
    plt.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 可视化表面张力力
def visualize_surface_tension(model, step, save_path=None):
    # 获取中心切片
    x_mid = model.width // 2
    y_mid = model.height // 2
    z_mid = model.depth // 2
    
    # 获取数据
    interface_slice = model.interface_field[:, :, z_mid]
    curvature_slice = model.curvature[:, :, z_mid]
    force_x_slice = model.surface_tension_force[:, :, z_mid, 0]
    force_y_slice = model.surface_tension_force[:, :, z_mid, 1]
    
    # 创建图形
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 绘制界面场
    im1 = axs[0, 0].imshow(interface_slice, cmap='viridis', origin='lower')
    axs[0, 0].set_title(f'界面场 (XY切片, 步骤={step})')
    axs[0, 0].set_xlabel('Y')
    axs[0, 0].set_ylabel('X')
    plt.colorbar(im1, ax=axs[0, 0])
    
    # 绘制曲率
    im2 = axs[0, 1].imshow(curvature_slice, cmap='coolwarm', origin='lower')
    axs[0, 1].set_title(f'曲率 (XY切片, 步骤={step})')
    axs[0, 1].set_xlabel('Y')
    axs[0, 1].set_ylabel('X')
    plt.colorbar(im2, ax=axs[0, 1])
    
    # 绘制表面张力力 (X分量)
    im3 = axs[1, 0].imshow(force_x_slice, cmap='coolwarm', origin='lower')
    axs[1, 0].set_title(f'表面张力力 X分量 (XY切片, 步骤={step})')
    axs[1, 0].set_xlabel('Y')
    axs[1, 0].set_ylabel('X')
    plt.colorbar(im3, ax=axs[1, 0])
    
    # 绘制表面张力力 (Y分量)
    im4 = axs[1, 1].imshow(force_y_slice, cmap='coolwarm', origin='lower')
    axs[1, 1].set_title(f'表面张力力 Y分量 (XY切片, 步骤={step})')
    axs[1, 1].set_xlabel('Y')
    axs[1, 1].set_ylabel('X')
    plt.colorbar(im4, ax=axs[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 可视化表面张力力向量场
def visualize_force_field(model, step, save_path=None):
    # 获取中心切片
    z_mid = model.depth // 2
    
    # 获取数据
    interface_slice = model.interface_field[:, :, z_mid]
    force_x_slice = model.surface_tension_force[:, :, z_mid, 0]
    force_y_slice = model.surface_tension_force[:, :, z_mid, 1]
    
    # 创建网格
    Y, X = np.meshgrid(np.arange(model.height), np.arange(model.width))
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制界面场
    plt.imshow(interface_slice, cmap='viridis', origin='lower', alpha=0.7)
    plt.colorbar(label='界面场')
    
    # 绘制力向量场 (降采样以提高可视化效果)
    skip = 2
    plt.quiver(Y[::skip, ::skip], X[::skip, ::skip], 
               force_y_slice[::skip, ::skip], force_x_slice[::skip, ::skip], 
               color='r', scale=0.5)
    
    plt.title(f'表面张力力向量场 (步骤={step})')
    plt.xlabel('Y')
    plt.ylabel('X')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 运行表面张力测试
def run_surface_tension_test():
    print("\n=== 运行表面张力测试 ===")
    
    # 创建网格
    grid_size = (64, 64, 64)
    
    # 创建模型
    model = AcceleratedMultiphaseModel(grid_size, num_phases=2, use_acceleration=True)
    
    # 设置表面张力系数
    model.sigma = 0.05  # 表面张力系数
    
    # 初始化非球形界面 (椭球体)
    center = (32, 32, 32)
    a, b, c = 20, 15, 15  # 椭球体半轴长度
    
    # 创建椭球体水平集函数
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    z = np.arange(grid_size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    phi = np.sqrt(((X - center[0])/a)**2 + ((Y - center[1])/b)**2 + ((Z - center[2])/c)**2) - 1.0
    model.phi = phi.astype(np.float32)
    
    # 更新体积分数
    model.volume_fractions = model.update_phase_fields(model.phi, model.num_phases, model.epsilon, model.use_cuda)
    
    # 更新界面场
    model.update_interface_field()
    
    # 可视化初始状态
    visualize_multiphase(model, 0, save_path="surface_tension_step_000.png")
    
    # 计算表面张力
    model.compute_surface_tension()
    
    # 可视化表面张力
    visualize_surface_tension(model, 0, save_path="surface_tension_forces_000.png")
    visualize_force_field(model, 0, save_path="surface_tension_vectors_000.png")
    
    # 设置速度场 (初始为零)
    velocity_field = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3), dtype=np.float32)
    model.set_velocity_field(velocity_field)
    
    # 执行模拟
    dt = 0.1
    steps = 50
    
    for i in range(steps):
        # 更新速度场 (添加表面张力力)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    velocity_field[x, y, z, 0] += dt * model.surface_tension_force[x, y, z, 0]
                    velocity_field[x, y, z, 1] += dt * model.surface_tension_force[x, y, z, 1]
                    velocity_field[x, y, z, 2] += dt * model.surface_tension_force[x, y, z, 2]
        
        model.set_velocity_field(velocity_field)
        
        # 执行一个时间步
        model.step(dt, method='levelset')
        
        # 每10步可视化一次
        if (i + 1) % 10 == 0:
            step = i + 1
            print(f"完成步骤 {step}/{steps}")
            visualize_multiphase(model, step, save_path=f"surface_tension_step_{step:03d}.png")
            visualize_surface_tension(model, step, save_path=f"surface_tension_forces_{step:03d}.png")
            visualize_force_field(model, step, save_path=f"surface_tension_vectors_{step:03d}.png")
    
    # 打印性能统计
    stats = model.get_performance_stats()
    print("\n性能统计:")
    print(f"平均平流时间: {stats['advect_time_avg']:.4f}秒")
    print(f"平均表面张力计算时间: {stats['surface_tension_time_avg']:.4f}秒")
    print(f"平均总时间: {stats['total_time_avg']:.4f}秒")
    print(f"总步数: {stats['steps']}")

if __name__ == "__main__":
    run_surface_tension_test() 