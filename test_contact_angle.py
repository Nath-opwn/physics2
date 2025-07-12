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

# 可视化多相流和固体边界
def visualize_multiphase_with_solid(model, step, save_path=None):
    # 获取中心切片
    x_mid = model.width // 2
    y_mid = model.height // 2
    z_mid = model.depth // 2
    
    # 获取数据
    phi_slice_xz = model.phi[x_mid, :, :]
    phi_slice_xy = model.phi[:, :, z_mid]
    solid_slice_xz = model.solid_boundary[x_mid, :, :]
    solid_slice_xy = model.solid_boundary[:, :, z_mid]
    
    # 创建图形
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制XZ切片
    im1 = axs[0].imshow(phi_slice_xz, cmap='coolwarm', vmin=-2, vmax=2, origin='lower')
    axs[0].contour(solid_slice_xz, levels=[0.5], colors='k', linewidths=2)
    axs[0].set_title(f'水平集函数 (XZ切片, 步骤={step})')
    axs[0].set_xlabel('Z')
    axs[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axs[0])
    
    # 绘制XY切片
    im2 = axs[1].imshow(phi_slice_xy, cmap='coolwarm', vmin=-2, vmax=2, origin='lower')
    axs[1].contour(solid_slice_xy, levels=[0.5], colors='k', linewidths=2)
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

# 可视化接触角
def visualize_contact_angle(model, step, save_path=None):
    # 获取中心切片
    z_mid = model.depth // 2
    
    # 获取数据
    phi_slice = model.phi[:, :, z_mid]
    solid_slice = model.solid_boundary[:, :, z_mid]
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制水平集函数
    plt.imshow(phi_slice, cmap='coolwarm', vmin=-2, vmax=2, origin='lower', alpha=0.7)
    plt.colorbar(label='水平集函数')
    
    # 绘制固体边界
    plt.contour(solid_slice, levels=[0.5], colors='k', linewidths=2)
    
    # 绘制零水平集 (界面)
    plt.contour(phi_slice, levels=[0], colors='r', linewidths=2)
    
    plt.title(f'接触角可视化 (步骤={step}, 角度={model.contact_angle}°)')
    plt.xlabel('Y')
    plt.ylabel('X')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 创建平板固体边界
def create_plate_boundary(width, height, depth, position, normal_direction='x'):
    solid_boundary = np.zeros((width, height, depth), dtype=np.int8)
    
    if normal_direction == 'x':
        solid_boundary[:position, :, :] = 1
    elif normal_direction == 'y':
        solid_boundary[:, :position, :] = 1
    elif normal_direction == 'z':
        solid_boundary[:, :, :position] = 1
    
    return solid_boundary

# 创建液滴初始条件
def create_droplet(width, height, depth, center, radius):
    x = np.arange(width)
    y = np.arange(height)
    z = np.arange(depth)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    phi = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2) - radius
    
    return phi.astype(np.float32)

# 运行接触角测试
def run_contact_angle_test():
    print("\n=== 运行接触角测试 ===")
    
    # 创建网格
    grid_size = (64, 64, 64)
    
    # 测试不同的接触角
    contact_angles = [30, 60, 90, 120, 150]
    
    for angle in contact_angles:
        print(f"\n--- 测试接触角: {angle}° ---")
        
        # 创建模型
        model = AcceleratedMultiphaseModel(grid_size, num_phases=2, use_acceleration=True)
        
        # 设置表面张力系数
        model.sigma = 0.05  # 表面张力系数
        
        # 设置接触角
        model.set_contact_angle(angle)
        
        # 创建固体边界 (底部平板)
        solid_boundary = create_plate_boundary(grid_size[0], grid_size[1], grid_size[2], 10, normal_direction='y')
        model.set_solid_boundary(solid_boundary)
        
        # 创建液滴 (在固体边界上方)
        center = (32, 15, 32)
        radius = 10
        phi = create_droplet(grid_size[0], grid_size[1], grid_size[2], center, radius)
        model.phi = phi
        
        # 更新体积分数
        model.volume_fractions = model.update_phase_fields(model.phi, model.num_phases, model.epsilon, model.use_cuda)
        
        # 更新界面场
        model.update_interface_field()
        
        # 可视化初始状态
        visualize_multiphase_with_solid(model, 0, save_path=f"contact_angle_{angle}_step_000.png")
        visualize_contact_angle(model, 0, save_path=f"contact_angle_{angle}_interface_000.png")
        
        # 应用接触角边界条件
        model.apply_contact_angle_boundary_condition()
        
        # 计算表面张力
        model.compute_surface_tension()
        
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
                        if model.solid_boundary[x, y, z] == 0:  # 只在流体区域更新速度
                            velocity_field[x, y, z, 0] += dt * model.surface_tension_force[x, y, z, 0]
                            velocity_field[x, y, z, 1] += dt * model.surface_tension_force[x, y, z, 1]
                            velocity_field[x, y, z, 2] += dt * model.surface_tension_force[x, y, z, 2]
            
            # 在固体区域设置速度为零
            velocity_field[solid_boundary == 1] = 0.0
            
            model.set_velocity_field(velocity_field)
            
            # 执行一个时间步
            model.step(dt, method='levelset')
            
            # 每10步可视化一次
            if (i + 1) % 10 == 0:
                step = i + 1
                print(f"完成步骤 {step}/{steps}")
                visualize_multiphase_with_solid(model, step, save_path=f"contact_angle_{angle}_step_{step:03d}.png")
                visualize_contact_angle(model, step, save_path=f"contact_angle_{angle}_interface_{step:03d}.png")
        
        # 打印性能统计
        stats = model.get_performance_stats()
        print(f"\n接触角 {angle}° 性能统计:")
        print(f"平均平流时间: {stats['advect_time_avg']:.4f}秒")
        print(f"平均表面张力计算时间: {stats['surface_tension_time_avg']:.4f}秒")
        print(f"平均总时间: {stats['total_time_avg']:.4f}秒")

# 运行测试
if __name__ == "__main__":
    run_contact_angle_test() 