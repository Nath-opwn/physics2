from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
import asyncio
import json
import logging
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

from src.api.simulation import active_simulations
from src.api.auth import get_current_active_user
from src.models.models import User
from src.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ws",
    tags=["websocket"],
    responses={404: {"description": "未找到"}},
)

# 存储活跃的WebSocket连接
active_connections: Dict[str, List[WebSocket]] = {}

async def get_user_from_token(websocket: WebSocket):
    """从WebSocket连接中获取用户信息"""
    try:
        # 从查询参数中获取token
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=1008, reason="缺少认证令牌")
            return None
        
        # 验证token并获取用户
        # 注意：这里简化了认证过程，实际应用中应该复用auth.py中的认证逻辑
        db = next(get_db())
        user = get_current_active_user(token=token, db=db)
        return user
    except Exception as e:
        logger.error(f"WebSocket认证错误: {str(e)}")
        await websocket.close(code=1008, reason="认证失败")
        return None

@router.websocket("/simulation/{session_id}")
async def websocket_simulation_endpoint(websocket: WebSocket, session_id: str):
    """模拟数据WebSocket端点"""
    # 接受WebSocket连接
    await websocket.accept()
    
    # 验证用户身份
    user = await get_user_from_token(websocket)
    if not user:
        return
    
    # 检查会话是否存在
    if session_id not in active_simulations:
        await websocket.close(code=1008, reason="模拟会话不存在")
        return
    
    # 将连接添加到活跃连接列表
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)
    
    try:
        # 发送初始数据
        sim = active_simulations[session_id]
        solver = sim["solver"]
        
        # 发送初始状态
        await send_simulation_data(websocket, sim, solver)
        
        # 持续接收客户端消息
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理客户端命令
            if message.get("type") == "command":
                command = message.get("command")
                if command == "get_data":
                    await send_simulation_data(websocket, sim, solver)
                elif command == "set_region":
                    # 设置感兴趣区域
                    region = message.get("region", {})
                    # 存储区域设置到连接上下文中
                    websocket.state.region = region
    except WebSocketDisconnect:
        # 客户端断开连接
        if session_id in active_connections:
            active_connections[session_id].remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
        await websocket.close(code=1011, reason=f"服务器内部错误: {str(e)}")

async def send_simulation_data(websocket: WebSocket, sim: Dict[str, Any], solver):
    """发送模拟数据到WebSocket客户端"""
    # 获取区域设置（如果有）
    region = getattr(websocket.state, "region", None)
    
    # 获取场数据
    if region:
        # 获取区域数据
        x_min, y_min, z_min = region.get("x_min", 0), region.get("y_min", 0), region.get("z_min", 0)
        x_max, y_max, z_max = region.get("x_max", solver.width), region.get("y_max", solver.height), region.get("z_max", solver.depth)
        
        velocity = solver.get_velocity_field()[x_min:x_max, y_min:y_max, z_min:z_max, :]
        pressure = solver.get_pressure_field()[x_min:x_max, y_min:y_max, z_min:z_max]
    else:
        # 获取下采样数据（避免发送过大数据）
        velocity = downsample_field(solver.get_velocity_field())
        pressure = downsample_field(solver.get_pressure_field())
    
    # 计算统计数据
    velocity_magnitude = np.sqrt(np.sum(velocity**2, axis=3))
    velocity_mean = float(np.mean(velocity_magnitude))
    velocity_max = float(np.max(velocity_magnitude))
    pressure_mean = float(np.mean(pressure))
    pressure_min = float(np.min(pressure))
    pressure_max = float(np.max(pressure))
    
    # 构建响应数据
    response = {
        "type": "simulation_data",
        "timestamp": datetime.now().isoformat(),
        "step": sim["step"],
        "statistics": {
            "velocity_mean": velocity_mean,
            "velocity_max": velocity_max,
            "pressure_mean": pressure_mean,
            "pressure_min": pressure_min,
            "pressure_max": pressure_max
        }
    }
    
    # 发送数据
    await websocket.send_json(response)

def downsample_field(field, target_size=50):
    """下采样场数据以减少传输数据量"""
    shape = field.shape
    
    if len(shape) == 3:  # 标量场
        width, height, depth = shape
        if max(width, height, depth) <= target_size:
            return field
        
        # 计算下采样步长
        step_x = max(1, width // target_size)
        step_y = max(1, height // target_size)
        step_z = max(1, depth // target_size)
        
        return field[::step_x, ::step_y, ::step_z]
    
    elif len(shape) == 4:  # 矢量场
        width, height, depth, components = shape
        if max(width, height, depth) <= target_size:
            return field
        
        # 计算下采样步长
        step_x = max(1, width // target_size)
        step_y = max(1, height // target_size)
        step_z = max(1, depth // target_size)
        
        return field[::step_x, ::step_y, ::step_z, :]
    
    return field

@router.websocket("/analysis/{session_id}")
async def websocket_analysis_endpoint(websocket: WebSocket, session_id: str):
    """分析数据WebSocket端点"""
    # 接受WebSocket连接
    await websocket.accept()
    
    # 验证用户身份
    user = await get_user_from_token(websocket)
    if not user:
        return
    
    # 检查会话是否存在
    if session_id not in active_simulations:
        await websocket.close(code=1008, reason="模拟会话不存在")
        return
    
    # 将连接添加到活跃连接列表
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)
    
    try:
        # 发送初始数据
        sim = active_simulations[session_id]
        solver = sim["solver"]
        
        # 持续接收客户端消息
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理客户端命令
            if message.get("type") == "command":
                command = message.get("command")
                if command == "get_vortex_analysis":
                    # 执行涡结构分析并发送结果
                    await send_vortex_analysis(websocket, sim, solver, message.get("params", {}))
                elif command == "get_turbulence_analysis":
                    # 执行湍流分析并发送结果
                    await send_turbulence_analysis(websocket, sim, solver, message.get("params", {}))
    except WebSocketDisconnect:
        # 客户端断开连接
        if session_id in active_connections:
            active_connections[session_id].remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
        await websocket.close(code=1011, reason=f"服务器内部错误: {str(e)}")

async def send_vortex_analysis(websocket: WebSocket, sim: Dict[str, Any], solver, params: Dict[str, Any]):
    """发送涡结构分析结果"""
    try:
        # 获取参数
        threshold = params.get("threshold", 0.1)
        method = params.get("method", "q_criterion")
        
        # 获取速度场
        velocity = solver.get_velocity_field()
        
        # 计算涡量场
        vorticity = solver.get_vorticity_field()
        
        # 根据方法识别涡结构
        if method == "q_criterion":
            # 计算Q准则
            # 简化实现，实际应计算速度梯度张量的第二不变量
            structures = identify_vortex_structures_by_vorticity(vorticity, threshold)
        elif method == "lambda2":
            # Lambda2准则
            # 简化实现
            structures = identify_vortex_structures_by_vorticity(vorticity, threshold)
        else:
            # 默认使用涡量大小法
            structures = identify_vortex_structures_by_vorticity(vorticity, threshold)
        
        # 构建响应
        response = {
            "type": "vortex_analysis",
            "timestamp": datetime.now().isoformat(),
            "step": sim["step"],
            "vortex_structures": structures
        }
        
        # 发送数据
        await websocket.send_json(response)
    except Exception as e:
        logger.error(f"涡结构分析错误: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"涡结构分析错误: {str(e)}"
        })

def identify_vortex_structures_by_vorticity(vorticity, threshold):
    """使用涡量大小识别涡结构"""
    # 计算涡量大小
    vorticity_magnitude = np.sqrt(np.sum(vorticity**2, axis=3))
    
    # 识别超过阈值的区域
    vortex_regions = vorticity_magnitude > threshold
    
    # 简化实现：返回前10个最强的涡结构
    structures = []
    count = 0
    
    # 找到涡量最大的点
    indices = np.argsort(vorticity_magnitude.flatten())[::-1]
    coordinates = np.unravel_index(indices, vorticity_magnitude.shape)
    
    for i in range(min(10, len(indices))):
        if vorticity_magnitude[coordinates[0][i], coordinates[1][i], coordinates[2][i]] <= threshold:
            break
            
        x, y, z = coordinates[0][i], coordinates[1][i], coordinates[2][i]
        strength = float(vorticity_magnitude[x, y, z])
        
        # 估计涡结构大小（简化）
        size = 1.0
        
        # 获取涡轴方向（使用涡量向量）
        direction = vorticity[x, y, z, :].tolist()
        
        structures.append({
            "position": [float(x), float(y), float(z)],
            "strength": strength,
            "size": size,
            "orientation": direction
        })
        
        count += 1
        if count >= 10:
            break
    
    return structures

async def send_turbulence_analysis(websocket: WebSocket, sim: Dict[str, Any], solver, params: Dict[str, Any]):
    """发送湍流分析结果"""
    try:
        # 获取区域参数
        region = params.get("region", None)
        
        # 获取速度场
        velocity = solver.get_velocity_field()
        
        # 如果指定了区域，则提取区域数据
        if region:
            x_min, y_min, z_min = region.get("x_min", 0), region.get("y_min", 0), region.get("z_min", 0)
            x_max, y_max, z_max = region.get("x_max", solver.width), region.get("y_max", solver.height), region.get("z_max", solver.depth)
            velocity = velocity[x_min:x_max, y_min:y_max, z_min:z_max, :]
        
        # 计算平均速度
        mean_velocity = np.mean(velocity, axis=(0, 1, 2))
        
        # 计算湍流脉动
        fluctuation = velocity - mean_velocity
        
        # 计算湍流强度
        fluctuation_magnitude = np.sqrt(np.sum(fluctuation**2, axis=3))
        mean_velocity_magnitude = np.sqrt(np.sum(mean_velocity**2))
        turbulence_intensity = float(np.mean(fluctuation_magnitude) / mean_velocity_magnitude if mean_velocity_magnitude > 0 else 0)
        
        # 计算雷诺应力张量
        reynolds_stresses = []
        for i in range(3):
            row = []
            for j in range(3):
                stress = float(np.mean(fluctuation[:,:,:,i] * fluctuation[:,:,:,j]))
                row.append(stress)
            reynolds_stresses.append(row)
        
        # 简化的能量谱计算
        energy_spectrum = calculate_simplified_energy_spectrum(fluctuation)
        
        # 构建响应
        response = {
            "type": "turbulence_analysis",
            "timestamp": datetime.now().isoformat(),
            "step": sim["step"],
            "turbulence_intensity": turbulence_intensity,
            "reynolds_stresses": reynolds_stresses,
            "energy_spectrum": energy_spectrum
        }
        
        # 发送数据
        await websocket.send_json(response)
    except Exception as e:
        logger.error(f"湍流分析错误: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"湍流分析错误: {str(e)}"
        })

def calculate_simplified_energy_spectrum(fluctuation):
    """计算简化的能量谱"""
    try:
        # 获取中心切片
        center_x = fluctuation.shape[0] // 2
        slice_data = fluctuation[center_x, :, :, :]
        
        # 计算2D FFT
        fft_data = np.fft.fft2(slice_data[:,:,0])
        fft_data = np.fft.fftshift(fft_data)
        
        # 计算功率谱
        power_spectrum = np.abs(fft_data)**2
        
        # 计算径向平均
        ny, nx = power_spectrum.shape
        y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]
        r = np.hypot(x, y)
        r = r.astype(np.int32)
        
        # 创建径向箱
        max_radius = min(nx, ny) // 2
        bins = np.arange(0, max_radius + 1)
        
        # 计算每个半径的平均值
        radial_mean = np.zeros(max_radius)
        for i in range(max_radius):
            mask = (r >= i) & (r < i+1)
            if mask.sum() > 0:
                radial_mean[i] = power_spectrum[mask].mean()
        
        # 创建波数数组
        wavenumbers = np.arange(max_radius)
        
        return {
            "wavenumbers": wavenumbers.tolist(),
            "energy": radial_mean.tolist()
        }
    except Exception as e:
        logger.error(f"能量谱计算错误: {str(e)}")
        return {"wavenumbers": [], "energy": []}

# 广播消息到所有连接的客户端
async def broadcast_simulation_update(session_id: str):
    """广播模拟更新到所有连接的客户端"""
    if session_id not in active_connections or session_id not in active_simulations:
        return
    
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 构建更新消息
    update = {
        "type": "simulation_update",
        "timestamp": datetime.now().isoformat(),
        "step": sim["step"]
    }
    
    # 广播给所有连接的客户端
    for connection in active_connections[session_id]:
        try:
            await connection.send_json(update)
        except Exception as e:
            logger.error(f"广播错误: {str(e)}") 