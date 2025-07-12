import numpy as np
import time
import sys

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

# 创建一个小型测试
def run_simple_test():
    print("\n=== 运行简单测试 ===")
    
    # 创建一个小网格
    grid_size = (32, 32, 32)
    
    # 创建Python实现的模型
    print("\n--- 测试Python实现 ---")
    model_py = AcceleratedMultiphaseModel(grid_size, use_acceleration=False)
    
    # 初始化球形界面
    center = (16, 16, 16)
    radius = 8
    model_py.initialize_sphere(center, radius)
    
    # 设置旋转速度场
    velocity_field = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3), dtype=np.float32)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                # 创建旋转速度场
                dx, dy, dz = x - center[0], y - center[1], z - center[2]
                velocity_field[x, y, z, 0] = -dy * 0.1
                velocity_field[x, y, z, 1] = dx * 0.1
                velocity_field[x, y, z, 2] = 0.0
    
    model_py.set_velocity_field(velocity_field)
    
    # 记录初始体积
    initial_volume = model_py.get_volume()
    print(f"初始体积: {initial_volume}")
    
    # 执行10个时间步并计时
    dt = 0.1
    steps = 10
    
    start_time = time.time()
    for i in range(steps):
        model_py.step(dt, method='levelset')
    py_time = time.time() - start_time
    
    # 最终体积变化
    final_volume = model_py.get_volume()
    volume_change = (final_volume - initial_volume) / initial_volume * 100
    print(f"Python实现: 耗时 = {py_time:.4f}秒, 最终体积 = {final_volume}, 变化 = {volume_change:.2f}%")
    
    # 创建C++实现的模型
    print("\n--- 测试C++实现 ---")
    model_cpp = AcceleratedMultiphaseModel(grid_size, use_acceleration=True, use_cuda=False)
    
    # 初始化球形界面
    model_cpp.initialize_sphere(center, radius)
    model_cpp.set_velocity_field(velocity_field)
    
    # 记录初始体积
    initial_volume = model_cpp.get_volume()
    print(f"初始体积: {initial_volume}")
    
    # 执行10个时间步并计时
    start_time = time.time()
    for i in range(steps):
        model_cpp.step(dt, method='levelset')
    cpp_time = time.time() - start_time
    
    # 最终体积变化
    final_volume = model_cpp.get_volume()
    volume_change = (final_volume - initial_volume) / initial_volume * 100
    print(f"C++实现: 耗时 = {cpp_time:.4f}秒, 最终体积 = {final_volume}, 变化 = {volume_change:.2f}%")
    
    # 计算加速比
    speedup = py_time / cpp_time
    print(f"\n加速比: C++/OpenMP vs Python = {speedup:.2f}x")

if __name__ == "__main__":
    run_simple_test() 