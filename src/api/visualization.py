from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import numpy as np
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.database.database import get_db
from src.models.models import User, SimulationSession
from src.api.auth import get_current_active_user
from src.api.simulation import active_simulations
import src.models.models as models

router = APIRouter(
    prefix="/api/visualization",
    tags=["visualization"],
    responses={404: {"description": "未找到"}},
)

@router.get("/streamlines")
async def generate_streamlines(
    session_id: str,
    num_lines: int = Query(10, ge=1, le=100),
    max_steps: int = Query(100, ge=10, le=1000),
    step_size: float = Query(0.5, gt=0, le=5.0),
    seed_method: str = Query("uniform", regex="^(uniform|random|plane|line)$"),
    seed_params: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    生成流线数据
    
    - **session_id**: 模拟会话ID
    - **num_lines**: 生成的流线数量
    - **max_steps**: 每条流线的最大步数
    - **step_size**: 积分步长
    - **seed_method**: 种子点生成方法 (uniform, random, plane, line)
    - **seed_params**: 种子点生成参数
    """
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 获取速度场
    velocity_field = solver.get_velocity_field()
    
    # 生成种子点
    seed_points = generate_seed_points(
        solver.width, 
        solver.height, 
        solver.depth, 
        num_lines, 
        seed_method, 
        seed_params
    )
    
    # 计算流线
    streamlines = []
    for seed in seed_points:
        line = integrate_streamline(
            velocity_field, 
            seed, 
            max_steps, 
            step_size, 
            solver.width, 
            solver.height, 
            solver.depth
        )
        if len(line) > 1:  # 只保留有效的流线
            streamlines.append(line)
    
    return {
        "session_id": session_id,
        "step": sim["step"],
        "streamlines": streamlines,
        "timestamp": datetime.now().isoformat()
    }

def generate_seed_points(width, height, depth, num_points, method, params=None):
    """生成种子点"""
    if params is None:
        params = {}
    
    if method == "uniform":
        # 均匀分布的种子点
        x_points = np.linspace(width * 0.1, width * 0.9, int(np.cbrt(num_points)))
        y_points = np.linspace(height * 0.1, height * 0.9, int(np.cbrt(num_points)))
        z_points = np.linspace(depth * 0.1, depth * 0.9, int(np.cbrt(num_points)))
        
        seeds = []
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    seeds.append([float(x), float(y), float(z)])
                    if len(seeds) >= num_points:
                        return seeds
        return seeds
    
    elif method == "random":
        # 随机分布的种子点
        seeds = []
        for _ in range(num_points):
            x = np.random.uniform(width * 0.1, width * 0.9)
            y = np.random.uniform(height * 0.1, height * 0.9)
            z = np.random.uniform(depth * 0.1, depth * 0.9)
            seeds.append([float(x), float(y), float(z)])
        return seeds
    
    elif method == "plane":
        # 平面上的种子点
        plane = params.get("plane", "xy")
        position = params.get("position", 0.5)
        
        if plane == "xy":
            z_pos = depth * position
            x_points = np.linspace(width * 0.1, width * 0.9, int(np.sqrt(num_points)))
            y_points = np.linspace(height * 0.1, height * 0.9, int(np.sqrt(num_points)))
            
            seeds = []
            for x in x_points:
                for y in y_points:
                    seeds.append([float(x), float(y), float(z_pos)])
                    if len(seeds) >= num_points:
                        return seeds
        
        elif plane == "xz":
            y_pos = height * position
            x_points = np.linspace(width * 0.1, width * 0.9, int(np.sqrt(num_points)))
            z_points = np.linspace(depth * 0.1, depth * 0.9, int(np.sqrt(num_points)))
            
            seeds = []
            for x in x_points:
                for z in z_points:
                    seeds.append([float(x), float(y_pos), float(z)])
                    if len(seeds) >= num_points:
                        return seeds
        
        elif plane == "yz":
            x_pos = width * position
            y_points = np.linspace(height * 0.1, height * 0.9, int(np.sqrt(num_points)))
            z_points = np.linspace(depth * 0.1, depth * 0.9, int(np.sqrt(num_points)))
            
            seeds = []
            for y in y_points:
                for z in z_points:
                    seeds.append([float(x_pos), float(y), float(z)])
                    if len(seeds) >= num_points:
                        return seeds
        
        return seeds
    
    elif method == "line":
        # 直线上的种子点
        start = params.get("start", [0.1, 0.1, 0.1])
        end = params.get("end", [0.9, 0.9, 0.9])
        
        # 将相对坐标转换为绝对坐标
        start_abs = [start[0] * width, start[1] * height, start[2] * depth]
        end_abs = [end[0] * width, end[1] * height, end[2] * depth]
        
        # 生成直线上的点
        t = np.linspace(0, 1, num_points)
        seeds = []
        for ti in t:
            x = start_abs[0] + ti * (end_abs[0] - start_abs[0])
            y = start_abs[1] + ti * (end_abs[1] - start_abs[1])
            z = start_abs[2] + ti * (end_abs[2] - start_abs[2])
            seeds.append([float(x), float(y), float(z)])
        
        return seeds
    
    # 默认返回随机种子点
    return generate_seed_points(width, height, depth, num_points, "random")

def integrate_streamline(velocity_field, seed_point, max_steps, step_size, width, height, depth):
    """使用RK4方法积分流线"""
    line = [seed_point]
    pos = np.array(seed_point, dtype=float)
    
    for _ in range(max_steps):
        # 检查是否超出边界
        if (pos[0] < 0 or pos[0] >= width - 1 or
            pos[1] < 0 or pos[1] >= height - 1 or
            pos[2] < 0 or pos[2] >= depth - 1):
            break
        
        # RK4积分
        k1 = interpolate_velocity(velocity_field, pos)
        k2 = interpolate_velocity(velocity_field, pos + 0.5 * step_size * k1)
        k3 = interpolate_velocity(velocity_field, pos + 0.5 * step_size * k2)
        k4 = interpolate_velocity(velocity_field, pos + step_size * k3)
        
        # 更新位置
        pos = pos + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 添加到流线
        line.append(pos.tolist())
        
        # 检查速度是否太小
        velocity = interpolate_velocity(velocity_field, pos)
        if np.linalg.norm(velocity) < 1e-6:
            break
    
    return line

def interpolate_velocity(velocity_field, pos):
    """三线性插值获取速度"""
    x, y, z = pos
    
    # 获取整数索引
    i0 = int(np.floor(x))
    j0 = int(np.floor(y))
    k0 = int(np.floor(z))
    
    # 确保索引在有效范围内
    width, height, depth, _ = velocity_field.shape
    i0 = max(0, min(i0, width - 2))
    j0 = max(0, min(j0, height - 2))
    k0 = max(0, min(k0, depth - 2))
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    
    # 计算插值权重
    s = x - i0
    t = y - j0
    u = z - k0
    
    # 三线性插值
    velocity = (
        velocity_field[i0, j0, k0] * (1-s) * (1-t) * (1-u) +
        velocity_field[i1, j0, k0] * s * (1-t) * (1-u) +
        velocity_field[i0, j1, k0] * (1-s) * t * (1-u) +
        velocity_field[i0, j0, k1] * (1-s) * (1-t) * u +
        velocity_field[i1, j0, k1] * s * (1-t) * u +
        velocity_field[i0, j1, k1] * (1-s) * t * u +
        velocity_field[i1, j1, k0] * s * t * (1-u) +
        velocity_field[i1, j1, k1] * s * t * u
    )
    
    return velocity

@router.get("/pathlines")
async def generate_pathlines(
    session_id: str,
    num_particles: int = Query(10, ge=1, le=100),
    num_steps: int = Query(20, ge=5, le=100),
    seed_method: str = Query("uniform", regex="^(uniform|random|plane|line)$"),
    seed_params: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    生成粒子路径线
    
    - **session_id**: 模拟会话ID
    - **num_particles**: 粒子数量
    - **num_steps**: 时间步数
    - **seed_method**: 种子点生成方法
    - **seed_params**: 种子点生成参数
    """
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 生成种子点
    particles = generate_seed_points(
        solver.width, 
        solver.height, 
        solver.depth, 
        num_particles, 
        seed_method, 
        seed_params
    )
    
    # 模拟粒子运动
    pathlines = []
    for particle in particles:
        path = [particle]
        pos = np.array(particle, dtype=float)
        
        for _ in range(num_steps):
            # 获取当前位置的速度
            if (pos[0] < 0 or pos[0] >= solver.width - 1 or
                pos[1] < 0 or pos[1] >= solver.height - 1 or
                pos[2] < 0 or pos[2] >= solver.depth - 1):
                break
            
            # 获取速度并更新位置
            velocity = interpolate_velocity(solver.get_velocity_field(), pos)
            pos = pos + velocity * solver.dt
            path.append(pos.tolist())
            
            # 执行一步模拟
            solver.step(solver.dt)
        
        pathlines.append(path)
    
    return {
        "session_id": session_id,
        "pathlines": pathlines,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/isosurface")
async def generate_isosurface(
    session_id: str,
    field: str = Query(..., regex="^(velocity|pressure|vorticity)$"),
    value: float = Query(...),
    simplify: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    生成等值面数据
    
    - **session_id**: 模拟会话ID
    - **field**: 场类型 (velocity, pressure, vorticity)
    - **value**: 等值面值
    - **simplify**: 是否简化网格
    """
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 获取场数据
    if field == "velocity":
        data = solver.get_velocity_field()
        # 计算速度大小
        scalar_field = np.sqrt(np.sum(data**2, axis=3))
    elif field == "pressure":
        scalar_field = solver.get_pressure_field()
    elif field == "vorticity":
        data = solver.get_vorticity_field()
        scalar_field = np.sqrt(np.sum(data**2, axis=3))
    else:
        raise HTTPException(status_code=400, detail="不支持的场类型")
    
    # 生成等值面（简化实现，返回顶点和三角形）
    vertices, triangles = marching_cubes_simplified(scalar_field, value)
    
    # 如果需要简化网格
    if simplify and len(vertices) > 1000:
        vertices, triangles = simplify_mesh(vertices, triangles, target_percent=0.5)
    
    return {
        "session_id": session_id,
        "field": field,
        "value": value,
        "vertices": vertices,
        "triangles": triangles,
        "timestamp": datetime.now().isoformat()
    }

def marching_cubes_simplified(scalar_field, isovalue):
    """简化版的移动立方体算法，用于生成等值面"""
    # 注意：这是一个简化实现，实际应用中应使用专业库如scikit-image或VTK
    
    # 获取场的尺寸
    width, height, depth = scalar_field.shape
    
    # 存储顶点和三角形
    vertices = []
    triangles = []
    
    # 遍历体素
    for i in range(width - 1):
        for j in range(height - 1):
            for k in range(depth - 1):
                # 获取体素的8个顶点值
                v0 = scalar_field[i, j, k]
                v1 = scalar_field[i+1, j, k]
                v2 = scalar_field[i+1, j+1, k]
                v3 = scalar_field[i, j+1, k]
                v4 = scalar_field[i, j, k+1]
                v5 = scalar_field[i+1, j, k+1]
                v6 = scalar_field[i+1, j+1, k+1]
                v7 = scalar_field[i, j+1, k+1]
                
                # 检查是否与等值面相交
                if ((v0 > isovalue) != (v1 > isovalue) or
                    (v1 > isovalue) != (v2 > isovalue) or
                    (v2 > isovalue) != (v3 > isovalue) or
                    (v3 > isovalue) != (v0 > isovalue) or
                    (v4 > isovalue) != (v5 > isovalue) or
                    (v5 > isovalue) != (v6 > isovalue) or
                    (v6 > isovalue) != (v7 > isovalue) or
                    (v7 > isovalue) != (v4 > isovalue) or
                    (v0 > isovalue) != (v4 > isovalue) or
                    (v1 > isovalue) != (v5 > isovalue) or
                    (v2 > isovalue) != (v6 > isovalue) or
                    (v3 > isovalue) != (v7 > isovalue)):
                    
                    # 简化处理：添加体素中心作为顶点
                    vertex_index = len(vertices)
                    vertices.append([i + 0.5, j + 0.5, k + 0.5])
                    
                    # 添加与等值面相交的面作为三角形
                    if (v0 > isovalue) != (v1 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v1 > isovalue) != (v2 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v2 > isovalue) != (v3 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v3 > isovalue) != (v0 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v4 > isovalue) != (v5 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v5 > isovalue) != (v6 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v6 > isovalue) != (v7 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v7 > isovalue) != (v4 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v0 > isovalue) != (v4 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v1 > isovalue) != (v5 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v2 > isovalue) != (v6 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
                    if (v3 > isovalue) != (v7 > isovalue):
                        triangles.append([vertex_index, vertex_index, vertex_index])
    
    return vertices, triangles

def simplify_mesh(vertices, triangles, target_percent=0.5):
    """简化网格，减少顶点和三角形数量"""
    # 简化实现：随机采样顶点和三角形
    target_vertices = int(len(vertices) * target_percent)
    target_triangles = int(len(triangles) * target_percent)
    
    if target_vertices >= len(vertices):
        return vertices, triangles
    
    # 随机选择顶点
    selected_indices = np.random.choice(len(vertices), target_vertices, replace=False)
    new_vertices = [vertices[i] for i in selected_indices]
    
    # 创建索引映射
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
    
    # 更新三角形
    new_triangles = []
    for triangle in triangles[:target_triangles]:
        # 检查三角形的所有顶点是否都在新的顶点集中
        if all(v in index_map for v in triangle):
            new_triangles.append([index_map[v] for v in triangle])
    
    return new_vertices, new_triangles

@router.get("/vector-field")
async def generate_vector_field(
    session_id: str,
    density: int = Query(5, ge=1, le=20),
    scale: float = Query(1.0, gt=0, le=10.0),
    region: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    生成矢量场可视化数据
    
    - **session_id**: 模拟会话ID
    - **density**: 矢量密度
    - **scale**: 矢量缩放因子
    - **region**: 区域参数 (可选)
    """
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 获取速度场
    velocity_field = solver.get_velocity_field()
    
    # 如果指定了区域，则提取区域数据
    if region:
        x_min = max(0, region.get("x_min", 0))
        y_min = max(0, region.get("y_min", 0))
        z_min = max(0, region.get("z_min", 0))
        x_max = min(solver.width, region.get("x_max", solver.width))
        y_max = min(solver.height, region.get("y_max", solver.height))
        z_max = min(solver.depth, region.get("z_max", solver.depth))
    else:
        x_min, y_min, z_min = 0, 0, 0
        x_max, y_max, z_max = solver.width, solver.height, solver.depth
    
    # 根据密度采样点
    step_x = max(1, (x_max - x_min) // density)
    step_y = max(1, (y_max - y_min) // density)
    step_z = max(1, (z_max - z_min) // density)
    
    # 生成矢量场数据
    vectors = []
    for i in range(x_min, x_max, step_x):
        for j in range(y_min, y_max, step_y):
            for k in range(z_min, z_max, step_z):
                if i < solver.width and j < solver.height and k < solver.depth:
                    # 获取位置和速度
                    position = [float(i), float(j), float(k)]
                    velocity = velocity_field[i, j, k].tolist()
                    
                    # 计算速度大小
                    magnitude = np.sqrt(sum(v**2 for v in velocity))
                    
                    # 添加到结果
                    vectors.append({
                        "position": position,
                        "direction": velocity,
                        "magnitude": float(magnitude)
                    })
    
    return {
        "session_id": session_id,
        "step": sim["step"],
        "vectors": vectors,
        "scale": scale,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/slice")
async def generate_slice(
    session_id: str,
    field: str = Query(..., regex="^(velocity|pressure|vorticity)$"),
    plane: str = Query(..., regex="^(xy|yz|xz)$"),
    position: float = Query(..., ge=0.0, le=1.0),
    resolution: int = Query(100, ge=10, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    生成场的切片数据
    
    - **session_id**: 模拟会话ID
    - **field**: 场类型 (velocity, pressure, vorticity)
    - **plane**: 切片平面 (xy, yz, xz)
    - **position**: 切片位置 (0.0-1.0)
    - **resolution**: 切片分辨率
    """
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 获取场数据
    if field == "velocity":
        data = solver.get_velocity_field()
        # 计算速度大小
        scalar_data = np.sqrt(np.sum(data**2, axis=3))
    elif field == "pressure":
        scalar_data = solver.get_pressure_field()
    elif field == "vorticity":
        data = solver.get_vorticity_field()
        scalar_data = np.sqrt(np.sum(data**2, axis=3))
    else:
        raise HTTPException(status_code=400, detail="不支持的场类型")
    
    # 计算切片
    if plane == "xy":
        k = int(position * (solver.depth - 1))
        slice_data = scalar_data[:, :, k]
        x_dim, y_dim = solver.width, solver.height
    elif plane == "yz":
        i = int(position * (solver.width - 1))
        slice_data = scalar_data[i, :, :]
        x_dim, y_dim = solver.height, solver.depth
    elif plane == "xz":
        j = int(position * (solver.height - 1))
        slice_data = scalar_data[:, j, :]
        x_dim, y_dim = solver.width, solver.depth
    else:
        raise HTTPException(status_code=400, detail="不支持的切片平面")
    
    # 调整分辨率
    if resolution != slice_data.shape[0] or resolution != slice_data.shape[1]:
        from scipy.ndimage import zoom
        zoom_factor = (resolution / slice_data.shape[0], resolution / slice_data.shape[1])
        slice_data = zoom(slice_data, zoom_factor)
    
    # 构建响应
    return {
        "session_id": session_id,
        "field": field,
        "plane": plane,
        "position": position,
        "dimensions": [x_dim, y_dim],
        "data": slice_data.tolist(),
        "min_value": float(np.min(slice_data)),
        "max_value": float(np.max(slice_data)),
        "timestamp": datetime.now().isoformat()
    } 