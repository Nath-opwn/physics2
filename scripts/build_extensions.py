#!/usr/bin/env python3
"""
编译C++扩展的脚本
"""

import os
import sys
import subprocess
import platform

def build_cpp_extension():
    """编译流体求解器的C++扩展"""
    print("开始编译流体求解器C++扩展...")
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # C++扩展目录
    cpp_ext_dir = os.path.join(root_dir, "src", "core", "cpp_ext")
    
    # 检查目录是否存在
    if not os.path.exists(cpp_ext_dir):
        print(f"错误：C++扩展目录不存在: {cpp_ext_dir}")
        return False
    
    # 切换到C++扩展目录
    os.chdir(cpp_ext_dir)
    
    # 构建命令
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    
    try:
        # 执行构建命令
        subprocess.run(cmd, check=True)
        print("C++扩展编译成功!")
        
        # 检查是否生成了扩展文件
        extension_found = False
        for file in os.listdir("."):
            if file.startswith("fluid_solver_core") and (file.endswith(".so") or file.endswith(".pyd")):
                extension_found = True
                print(f"生成的扩展文件: {file}")
                break
        
        if not extension_found:
            print("警告：未找到编译后的扩展文件，编译可能失败。")
            return False
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"编译失败，错误码: {e.returncode}")
        print(f"错误信息: {e}")
        return False
    
    except Exception as e:
        print(f"编译过程中发生错误: {e}")
        return False

def check_dependencies():
    """检查编译依赖项"""
    print("检查编译依赖项...")
    
    # 检查Python版本
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")
    if tuple(map(int, python_version.split("."))) < (3, 6):
        print("警告: 需要Python 3.6或更高版本")
    
    # 检查NumPy
    try:
        import numpy
        print(f"NumPy版本: {numpy.__version__}")
    except ImportError:
        print("错误: 未安装NumPy，请先安装NumPy")
        return False
    
    # 检查C++编译器
    if platform.system() == "Windows":
        # 在Windows上检查MSVC
        try:
            result = subprocess.run(["cl"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("找到MSVC编译器")
        except FileNotFoundError:
            print("警告: 未找到MSVC编译器，可能需要安装Visual C++ Build Tools")
    else:
        # 在Linux/Mac上检查GCC/Clang
        try:
            result = subprocess.run(["g++", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("找到GCC编译器")
        except FileNotFoundError:
            try:
                result = subprocess.run(["clang++", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("找到Clang编译器")
            except FileNotFoundError:
                print("警告: 未找到GCC或Clang编译器，请安装C++编译器")
                return False
    
    return True

if __name__ == "__main__":
    print("流体求解器C++扩展构建工具")
    print("=" * 50)
    
    # 检查依赖项
    if not check_dependencies():
        print("\n依赖项检查失败，请解决上述问题后重试。")
        sys.exit(1)
    
    print("\n依赖项检查通过，开始编译...")
    
    # 编译扩展
    if build_cpp_extension():
        print("\n编译成功完成！现在可以使用C++加速版本的流体求解器了。")
        sys.exit(0)
    else:
        print("\n编译失败，将使用纯Python版本的流体求解器。")
        sys.exit(1) 