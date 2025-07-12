#!/usr/bin/env python3
"""
系统初始化脚本
用于设置流体动力学模拟系统并添加所有知识库内容和教程
"""

import sys
import os
import subprocess
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.database import engine
from src.models.models import Base

def initialize_system():
    """初始化整个系统"""
    print("开始初始化流体动力学模拟系统...")
    
    # 1. 确保数据库表存在
    print("创建数据库表...")
    Base.metadata.create_all(bind=engine)
    print("数据库表创建完成")
    
    # 2. 编译C++扩展
    print("\n编译C++扩展模块...")
    try:
        result = subprocess.run(
            ["python", "scripts/build_extensions.py"], 
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print("C++扩展编译成功")
    except subprocess.CalledProcessError as e:
        print(f"C++扩展编译失败: {e}")
        print(f"错误输出: {e.stderr}")
    
    # 3. 清空知识库（可选）
    print("\n清空现有知识库...")
    try:
        result = subprocess.run(
            ["python", "scripts/clear_knowledge.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"清空知识库失败: {e}")
    
    # 4. 添加基础知识条目
    print("\n添加基础知识条目...")
    try:
        result = subprocess.run(
            ["python", "scripts/populate_knowledge.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"添加基础知识条目失败: {e}")
    
    # 5. 添加高级知识条目
    print("\n添加高级知识条目...")
    try:
        result = subprocess.run(
            ["python", "scripts/add_advanced_knowledge.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"添加高级知识条目失败: {e}")
    
    # 6. 添加教程
    print("\n添加教程内容...")
    try:
        result = subprocess.run(
            ["python", "scripts/add_tutorials.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"添加教程内容失败: {e}")
    
    # 7. 运行性能测试（可选）
    print("\n运行性能测试...")
    try:
        result = subprocess.run(
            ["python", "scripts/performance_test.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"性能测试失败: {e}")
    
    print("\n系统初始化完成！")
    print("您现在可以启动应用服务器: python src/main.py")

if __name__ == "__main__":
    initialize_system() 