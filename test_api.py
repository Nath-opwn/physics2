#!/usr/bin/env python3
"""
流体动力学模拟系统API测试脚本
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查端点"""
    print("测试健康检查端点...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("健康检查测试通过！\n")

def test_auth():
    """测试认证功能"""
    print("测试用户注册...")
    # 注册新用户
    register_data = {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "testpassword"
    }
    response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        print("用户注册成功！")
    else:
        print(f"响应: {response.text}")
        print("用户可能已存在，继续测试登录...")
    
    print("\n测试用户登录...")
    # 登录
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = requests.post(
        f"{BASE_URL}/api/auth/login", 
        data={"username": "testuser", "password": "testpassword"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        token_data = response.json()
        print(f"登录成功！获取到令牌: {token_data['access_token'][:20]}...")
        return token_data["access_token"]
    else:
        print(f"响应: {response.text}")
        print("使用默认测试用户登录...")
        # 使用初始化数据库时创建的测试用户
        response = requests.post(
            f"{BASE_URL}/api/auth/login", 
            data={"username": "test", "password": "test123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            token_data = response.json()
            print(f"登录成功！获取到令牌: {token_data['access_token'][:20]}...")
            return token_data["access_token"]
        else:
            print(f"响应: {response.text}")
            print("认证测试失败！")
            sys.exit(1)

def test_simulation(token):
    """测试模拟功能"""
    print("\n测试初始化模拟...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 初始化模拟
    sim_data = {
        "name": "测试模拟",
        "width": 50,
        "height": 50,
        "depth": 50,
        "viscosity": 0.1,
        "density": 1.0,
        "boundary_type": 0
    }
    response = requests.post(f"{BASE_URL}/api/simulation/initialize", json=sim_data, headers=headers)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        sim_result = response.json()
        session_id = sim_result["session_id"]
        print(f"模拟初始化成功！会话ID: {session_id}")
        
        # 执行单步模拟
        print("\n测试执行单步模拟...")
        step_data = {
            "session_id": session_id,
            "dt": 0.01
        }
        response = requests.post(f"{BASE_URL}/api/simulation/step", json=step_data, headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            step_result = response.json()
            print(f"单步模拟成功！当前步数: {step_result['step']}")
            
            # 测试数据探针
            print("\n测试数据探针...")
            probe_data = {
                "session_id": session_id,
                "x": 25,
                "y": 25,
                "z": 25
            }
            response = requests.post(f"{BASE_URL}/api/simulation/probe", json=probe_data, headers=headers)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                probe_result = response.json()
                print(f"数据探针成功！位置: {probe_result['position']}")
                print(f"速度: {probe_result['velocity']}")
                print(f"压力: {probe_result['pressure']}")
            else:
                print(f"响应: {response.text}")
                print("数据探针测试失败！")
            
            # 测试控制模拟
            print("\n测试控制模拟...")
            control_data = {
                "session_id": session_id,
                "action": "pause"
            }
            response = requests.post(f"{BASE_URL}/api/simulation/control", json=control_data, headers=headers)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                control_result = response.json()
                print(f"控制模拟成功！状态: {control_result['status']}")
            else:
                print(f"响应: {response.text}")
                print("控制模拟测试失败！")
            
            return session_id
        else:
            print(f"响应: {response.text}")
            print("单步模拟测试失败！")
    else:
        print(f"响应: {response.text}")
        print("模拟初始化测试失败！")
        return None

def test_parameters(token, session_id):
    """测试参数管理功能"""
    if not session_id:
        print("\n跳过参数管理测试，因为没有有效的会话ID")
        return
    
    print("\n测试获取参数预设...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/api/parameters/presets", headers=headers)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        presets = response.json()
        print(f"获取参数预设成功！预设数量: {len(presets)}")
        
        # 测试更新参数
        print("\n测试更新参数...")
        update_data = {
            "session_id": session_id,
            "parameters": {
                "viscosity": 0.05,
                "density": 1.2
            }
        }
        response = requests.post(f"{BASE_URL}/api/parameters/update", json=update_data, headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            update_result = response.json()
            print(f"更新参数成功！状态: {update_result['status']}")
        else:
            print(f"响应: {response.text}")
            print("更新参数测试失败！")
    else:
        print(f"响应: {response.text}")
        print("获取参数预设测试失败！")

def test_analysis(token, session_id):
    """测试数据分析功能"""
    if not session_id:
        print("\n跳过数据分析测试，因为没有有效的会话ID")
        return
    
    print("\n测试获取统计数据...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/api/analysis/statistics?session_id={session_id}", headers=headers)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print("获取统计数据成功！")
        print(f"速度统计: {stats['velocity']}")
        print(f"压力统计: {stats['pressure']}")
        
        # 测试获取分布数据
        print("\n测试获取分布数据...")
        response = requests.get(f"{BASE_URL}/api/analysis/distribution?session_id={session_id}&field=velocity", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            dist = response.json()
            print("获取分布数据成功！")
            print(f"分布数据点数: {len(dist['counts'])}")
        else:
            print(f"响应: {response.text}")
            print("获取分布数据测试失败！")
    else:
        print(f"响应: {response.text}")
        print("获取统计数据测试失败！")

def test_vortex_analysis(token, session_id):
    """测试涡度分析功能"""
    if not session_id:
        print("\n跳过涡度分析测试，因为没有有效的会话ID")
        return
    
    print("\n测试涡度分析...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 执行涡度分析
    analysis_data = {
        "session_id": session_id,
        "threshold": 0.05,
        "method": "vorticity_magnitude"
    }
    response = requests.post(f"{BASE_URL}/api/analysis/vortex-analysis", json=analysis_data, headers=headers)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        analysis_result = response.json()
        print(f"涡度分析成功！识别到 {len(analysis_result['vortex_structures'])} 个涡结构")
        
        if analysis_result['vortex_structures']:
            # 显示第一个涡结构的详细信息
            vortex = analysis_result['vortex_structures'][0]
            print(f"涡结构示例:")
            print(f"  - 位置: ({vortex['position'][0]:.2f}, {vortex['position'][1]:.2f}, {vortex['position'][2]:.2f})")
            print(f"  - 强度: {vortex['strength']:.4f}")
            print(f"  - 大小: {vortex['size']:.4f}")
            print(f"  - 方向: [{vortex['orientation'][0]:.2f}, {vortex['orientation'][1]:.2f}, {vortex['orientation'][2]:.2f}]")
    else:
        print(f"响应: {response.text}")
        print("涡度分析测试失败！")

def test_turbulence_analysis(token, session_id):
    """测试湍流分析功能"""
    if not session_id:
        print("\n跳过湍流分析测试，因为没有有效的会话ID")
        return
    
    print("\n测试湍流分析...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 执行湍流分析
    analysis_data = {
        "session_id": session_id,
        "region": {
            "x_min": 10,
            "y_min": 10,
            "z_min": 10,
            "x_max": 40,
            "y_max": 40,
            "z_max": 40
        }
    }
    response = requests.post(f"{BASE_URL}/api/analysis/turbulence-analysis", json=analysis_data, headers=headers)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        analysis_result = response.json()
        print(f"湍流分析成功！")
        print(f"湍流强度: {analysis_result['turbulence_intensity']:.4f}")
        print(f"雷诺应力张量:")
        for i, row in enumerate(analysis_result['reynolds_stresses']):
            print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
        
        # 显示能量谱信息
        energy_spectrum = analysis_result['energy_spectrum']
        if energy_spectrum['frequencies'] and energy_spectrum['energy_density']:
            print(f"能量谱: {len(energy_spectrum['frequencies'])} 个频率点")
            print(f"  频率范围: {min(energy_spectrum['frequencies']):.4f} - {max(energy_spectrum['frequencies']):.4f}")
            print(f"  最大能量密度: {max(energy_spectrum['energy_density']):.4f}")
        else:
            print("能量谱: 无数据")
    else:
        print(f"响应: {response.text}")
        print("湍流分析测试失败！")

def test_knowledge(token):
    """测试知识库功能"""
    print("\n测试获取知识分类...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/api/knowledge/categories", headers=headers)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        categories = response.json()
        print(f"获取知识分类成功！分类数量: {len(categories)}")
        
        # 测试获取知识内容
        print("\n测试获取知识内容...")
        response = requests.get(f"{BASE_URL}/api/knowledge/content", headers=headers)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            content = response.json()
            print(f"获取知识内容成功！内容数量: {len(content)}")
            
            # 测试搜索知识库
            if len(content) > 0:
                search_term = content[0]["title"].split()[0]
                print(f"\n测试搜索知识库 (搜索词: {search_term})...")
                response = requests.get(f"{BASE_URL}/api/knowledge/search?query={search_term}", headers=headers)
                print(f"状态码: {response.status_code}")
                
                if response.status_code == 200:
                    search_results = response.json()
                    print(f"搜索知识库成功！结果数量: {len(search_results)}")
                else:
                    print(f"响应: {response.text}")
                    print("搜索知识库测试失败！")
        else:
            print(f"响应: {response.text}")
            print("获取知识内容测试失败！")
    else:
        print(f"响应: {response.text}")
        print("获取知识分类测试失败！")

def test_export(token, session_id):
    """测试数据导出功能"""
    if not session_id:
        print("\n跳过数据导出测试，因为没有有效的会话ID")
        return
    
    print("\n测试获取导出格式...")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/api/export/formats", headers=headers)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        formats = response.json()
        print(f"获取导出格式成功！格式数量: {len(formats)}")
        
        # 测试导出数据
        print("\n测试导出数据...")
        response = requests.post(
            f"{BASE_URL}/api/export/data?session_id={session_id}&format=json", 
            headers=headers
        )
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            export_result = response.json()
            print("导出数据成功！")
            if "task_id" in export_result:
                task_id = export_result["task_id"]
                print(f"导出任务ID: {task_id}")
                
                # 测试获取导出任务状态
                print("\n测试获取导出任务状态...")
                time.sleep(2)  # 等待任务处理
                response = requests.get(f"{BASE_URL}/api/export/status/{task_id}", headers=headers)
                print(f"状态码: {response.status_code}")
                
                if response.status_code == 200:
                    status = response.json()
                    print(f"获取导出任务状态成功！状态: {status['status']}")
                else:
                    print(f"响应: {response.text}")
                    print("获取导出任务状态测试失败！")
            else:
                print("导出数据直接返回结果")
        else:
            print(f"响应: {response.text}")
            print("导出数据测试失败！")
    else:
        print(f"响应: {response.text}")
        print("获取导出格式测试失败！")

def main():
    """主函数"""
    print("开始测试流体动力学模拟系统API...\n")
    
    try:
        # 测试健康检查
        test_health()
        
        # 测试认证
        token = test_auth()
        
        # 测试模拟功能
        session_id = test_simulation(token)
        
        # 测试参数管理
        test_parameters(token, session_id)
        
        # 测试数据分析
        test_analysis(token, session_id)
        
        # 测试涡度分析
        test_vortex_analysis(token, session_id)
        
        # 测试湍流分析
        test_turbulence_analysis(token, session_id)
        
        # 测试知识库
        test_knowledge(token)
        
        # 测试数据导出
        test_export(token, session_id)
        
        print("\n所有测试完成！")
    
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 