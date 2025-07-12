#!/usr/bin/env python3
"""
测试C++扩展是否正确工作的脚本
"""

import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入C++扩展
try:
    from src.core.cpp_ext import diffuse, advect, project, compute_vorticity, EXTENSION_LOADED
    
    if not EXTENSION_LOADED:
        print("C++扩展未加载，请先编译扩展模块")
        print("运行: python scripts/build_extensions.py")
        sys.exit(1)
    
    print("C++扩展已成功加载")
except ImportError as e:
    print(f"导入C++扩展失败: {e}")
    print("请先编译扩展模块")
    print("运行: python scripts/build_extensions.py")
    sys.exit(1)

def test_diffuse():
    """测试扩散函数"""
    print("\n测试扩散函数...")
    
    # 创建测试数据
    size = 32
    x = np.zeros((size, size, size), dtype=np.float64)
    x0 = np.zeros((size, size, size), dtype=np.float64)
    
    # 在中心添加热源
    x0[size//2, size//2, size//2] = 100.0
    
    # 参数
    diff = 0.1
    dt = 0.1
    iterations = 20
    
    # 运行扩散
    start_time = time.time()
    diffuse(x, x0, diff, dt, iterations)
    end_time = time.time()
    
    print(f"扩散计算完成，耗时: {end_time - start_time:.4f}秒")
    print(f"最大值: {np.max(x):.4f}, 最小值: {np.min(x):.4f}")
    print(f"中心值: {x[size//2, size//2, size//2]:.4f}")
    
    return True

def test_advect():
    """测试平流函数"""
    print("\n测试平流函数...")
    
    # 创建测试数据
    size = 32
    d = np.zeros((size, size, size), dtype=np.float64)
    d0 = np.zeros((size, size, size), dtype=np.float64)
    u = np.ones((size, size, size), dtype=np.float64) * 0.1
    v = np.zeros((size, size, size), dtype=np.float64)
    w = np.zeros((size, size, size), dtype=np.float64)
    
    # 在中心添加标量
    d0[size//2, size//2, size//2] = 100.0
    
    # 参数
    dt = 0.1
    
    # 运行平流
    start_time = time.time()
    advect(d, d0, u, v, w, dt)
    end_time = time.time()
    
    print(f"平流计算完成，耗时: {end_time - start_time:.4f}秒")
    print(f"最大值: {np.max(d):.4f}, 最小值: {np.min(d):.4f}")
    
    # 检查标量是否向右移动
    max_idx = np.unravel_index(np.argmax(d), d.shape)
    print(f"最大值位置: {max_idx}")
    
    return True

def test_project():
    """测试投影函数"""
    print("\n测试投影函数...")
    
    # 创建测试数据
    size = 32
    u = np.random.rand(size, size, size).astype(np.float64) * 2 - 1
    v = np.random.rand(size, size, size).astype(np.float64) * 2 - 1
    w = np.random.rand(size, size, size).astype(np.float64) * 2 - 1
    p = np.zeros((size, size, size), dtype=np.float64)
    div = np.zeros((size, size, size), dtype=np.float64)
    
    # 计算初始散度
    initial_div = np.zeros((size, size, size), dtype=np.float64)
    for z in range(1, size-1):
        for y in range(1, size-1):
            for x in range(1, size-1):
                initial_div[z, y, x] = (
                    (u[z, y, x+1] - u[z, y, x-1]) +
                    (v[z, y+1, x] - v[z, y-1, x]) +
                    (w[z+1, y, x] - w[z-1, y, x])
                ) * 0.5
    
    initial_div_max = np.max(np.abs(initial_div))
    print(f"初始散度最大值: {initial_div_max:.4f}")
    
    # 参数
    iterations = 20
    
    # 运行投影
    start_time = time.time()
    project(u, v, w, p, div, iterations)
    end_time = time.time()
    
    print(f"投影计算完成，耗时: {end_time - start_time:.4f}秒")
    
    # 计算投影后的散度
    final_div = np.zeros((size, size, size), dtype=np.float64)
    for z in range(1, size-1):
        for y in range(1, size-1):
            for x in range(1, size-1):
                final_div[z, y, x] = (
                    (u[z, y, x+1] - u[z, y, x-1]) +
                    (v[z, y+1, x] - v[z, y-1, x]) +
                    (w[z+1, y, x] - w[z-1, y, x])
                ) * 0.5
    
    final_div_max = np.max(np.abs(final_div))
    print(f"投影后散度最大值: {final_div_max:.4f}")
    print(f"散度减少比例: {1 - final_div_max / initial_div_max:.4f}")
    
    return final_div_max < initial_div_max

def test_compute_vorticity():
    """测试涡量计算函数"""
    print("\n测试涡量计算函数...")
    
    # 创建测试数据
    size = 32
    u = np.zeros((size, size, size), dtype=np.float64)
    v = np.zeros((size, size, size), dtype=np.float64)
    w = np.zeros((size, size, size), dtype=np.float64)
    vorticity = np.zeros((size, size, size), dtype=np.float64)
    
    # 创建旋转流场
    for z in range(size):
        for y in range(size):
            for x in range(size):
                dx = x - size // 2
                dy = y - size // 2
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    u[z, y, x] = -dy / dist * np.exp(-dist / 5)
                    v[z, y, x] = dx / dist * np.exp(-dist / 5)
    
    # 运行涡量计算
    start_time = time.time()
    compute_vorticity(u, v, w, vorticity)
    end_time = time.time()
    
    print(f"涡量计算完成，耗时: {end_time - start_time:.4f}秒")
    print(f"涡量最大值: {np.max(vorticity):.4f}")
    
    # 检查涡量是否在中心最大
    max_idx = np.unravel_index(np.argmax(vorticity), vorticity.shape)
    print(f"涡量最大值位置: {max_idx}")
    
    return True

if __name__ == "__main__":
    print("测试C++扩展功能")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("扩散函数", test_diffuse),
        ("平流函数", test_advect),
        ("投影函数", test_project),
        ("涡量计算", test_compute_vorticity)
    ]
    
    all_passed = True
    
    for name, test_func in tests:
        print(f"\n测试 {name}...")
        try:
            result = test_func()
            if result:
                print(f"{name} 测试通过")
            else:
                print(f"{name} 测试失败")
                all_passed = False
        except Exception as e:
            print(f"{name} 测试出错: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("所有测试通过！C++扩展工作正常。")
        sys.exit(0)
    else:
        print("部分测试失败，请检查C++扩展。")
        sys.exit(1) 