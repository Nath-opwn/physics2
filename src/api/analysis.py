from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from src.database.database import get_db
from src.models.models import User, SimulationSession, SimulationResult, AnalysisResult
from src.api.auth import get_current_active_user
from src.api.simulation import active_simulations
from src.core import schemas
import src.models.models as models

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analysis",
    tags=["analysis"],
    responses={404: {"description": "未找到"}},
)

@router.get("/timeseries")
async def get_time_series(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取时间序列数据"""
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 从数据库获取历史结果
    results = db.query(SimulationResult).filter(
        SimulationResult.session_id == session_id
    ).order_by(SimulationResult.step).all()
    
    if not results:
        # 如果没有历史数据，返回当前模拟的数据
        sim = active_simulations[session_id]
        solver = sim["solver"]
        
        # 计算当前能量
        velocity = solver.get_velocity_field()
        pressure = solver.get_pressure_field()
        
        # 计算动能
        kinetic_energy = np.mean(velocity[:,:,:,0]**2 + velocity[:,:,:,1]**2 + velocity[:,:,:,2]**2) * solver.density / 2
        
        # 计算势能（简化版，仅使用压力场）
        potential_energy = np.mean(np.abs(pressure))
        
        # 总能量
        total_energy = kinetic_energy + potential_energy
        
        return {
            "energy_series": [total_energy],
            "kinetic_energy_series": [kinetic_energy],
            "potential_energy_series": [potential_energy],
            "steps": [sim["step"]]
        }
    
    # 提取能量数据
    energy_series = []
    kinetic_energy_series = []
    potential_energy_series = []
    steps = []
    
    for result in results:
        if result.statistics and "kinetic_energy" in result.statistics and "potential_energy" in result.statistics:
            kinetic_energy = result.statistics["kinetic_energy"]
            potential_energy = result.statistics["potential_energy"]
            total_energy = kinetic_energy + potential_energy
            
            energy_series.append(total_energy)
            kinetic_energy_series.append(kinetic_energy)
            potential_energy_series.append(potential_energy)
            steps.append(result.step)
    
    return {
        "energy_series": energy_series,
        "kinetic_energy_series": kinetic_energy_series,
        "potential_energy_series": potential_energy_series,
        "steps": steps
    }

@router.get("/statistics")
async def get_statistics(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取统计数据"""
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 检查会话所有权
    session = db.query(SimulationSession).filter(
        SimulationSession.id == session_id,
        SimulationSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 获取场数据
    velocity = solver.get_velocity_field()
    pressure = solver.get_pressure_field()
    vorticity = solver.get_vorticity_field()
    
    # 计算速度大小
    velocity_magnitude = np.sqrt(np.sum(velocity**2, axis=3))
    vorticity_magnitude = np.sqrt(np.sum(vorticity**2, axis=3))
    
    # 计算统计信息
    velocity_mean = float(np.mean(velocity_magnitude))
    velocity_max = float(np.max(velocity_magnitude))
    pressure_mean = float(np.mean(pressure))
    pressure_min = float(np.min(pressure))
    pressure_max = float(np.max(pressure))
    vorticity_mean = float(np.mean(vorticity_magnitude))
    
    # 计算动能和势能
    kinetic_energy = float(np.mean(velocity_magnitude**2) * solver.density / 2)
    potential_energy = float(np.mean(np.abs(pressure)))
    
    # 计算直方图数据
    velocity_hist, _ = np.histogram(velocity_magnitude.flatten(), bins=20)
    pressure_hist, _ = np.histogram(pressure.flatten(), bins=20)
    vorticity_hist, _ = np.histogram(vorticity_magnitude.flatten(), bins=20)
    
    # 转换为列表
    velocity_histogram = velocity_hist.tolist()
    pressure_histogram = pressure_hist.tolist()
    vorticity_histogram = vorticity_hist.tolist()
    
    # 保存结果到数据库
    if sim["step"] % 10 == 0:  # 每10步保存一次
        db_result = SimulationResult(
            session_id=session_id,
            step=sim["step"],
            statistics={
                "velocity_mean": velocity_mean,
                "velocity_max": velocity_max,
                "pressure_mean": pressure_mean,
                "pressure_min": pressure_min,
                "pressure_max": pressure_max,
                "vorticity_mean": vorticity_mean,
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy
            }
        )
        db.add(db_result)
        db.commit()
    
    return {
        "velocity_mean": velocity_mean,
        "velocity_max": velocity_max,
        "pressure_mean": pressure_mean,
        "pressure_min": pressure_min,
        "pressure_max": pressure_max,
        "vorticity_mean": vorticity_mean,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "velocity_histogram": velocity_histogram,
        "pressure_histogram": pressure_histogram,
        "vorticity_histogram": vorticity_histogram
    }

@router.get("/distribution")
async def get_distribution(
    session_id: str,
    field: str,
    bins: int = 20,
    current_user: User = Depends(get_current_active_user)
):
    """获取分布数据"""
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
        data = np.sqrt(np.sum(data**2, axis=3)).flatten()
    elif field == "pressure":
        data = solver.get_pressure_field().flatten()
    elif field == "vorticity":
        data = solver.get_vorticity_field()
        # 计算涡量大小
        data = np.sqrt(np.sum(data**2, axis=3)).flatten()
    else:
        raise HTTPException(status_code=400, detail="无效的场类型")
    
    # 计算分布
    hist, bin_edges = np.histogram(data, bins=bins)
    
    # 转换为可序列化格式
    distribution = {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    }
    
    return distribution

@router.post("/region/statistics")
async def get_region_statistics(
    session_id: str,
    x_min: int,
    y_min: int,
    z_min: int,
    x_max: int,
    y_max: int,
    z_max: int,
    current_user: User = Depends(get_current_active_user)
):
    """获取区域统计数据"""
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 获取场数据
    velocity = solver.get_velocity_field()
    pressure = solver.get_pressure_field()
    vorticity = solver.get_vorticity_field()
    
    # 确保坐标在有效范围内
    x_min = max(0, min(solver.width - 1, x_min))
    y_min = max(0, min(solver.height - 1, y_min))
    z_min = max(0, min(solver.depth - 1, z_min))
    x_max = max(0, min(solver.width, x_max))
    y_max = max(0, min(solver.height, y_max))
    z_max = max(0, min(solver.depth, z_max))
    
    # 提取子区域
    v_region = velocity[z_min:z_max, y_min:y_max, x_min:x_max]
    p_region = pressure[z_min:z_max, y_min:y_max, x_min:x_max]
    w_region = vorticity[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # 计算速度大小
    v_magnitude = np.sqrt(np.sum(v_region**2, axis=3))
    
    # 计算统计信息
    statistics = {
        "region": {
            "x_min": x_min, "y_min": y_min, "z_min": z_min,
            "x_max": x_max, "y_max": y_max, "z_max": z_max,
            "volume": (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        },
        "velocity": {
            "mean": float(np.mean(v_magnitude)),
            "min": float(np.min(v_magnitude)),
            "max": float(np.max(v_magnitude)),
            "std": float(np.std(v_magnitude)),
            "component_means": np.mean(v_region, axis=(0,1,2)).tolist()
        },
        "pressure": {
            "mean": float(np.mean(p_region)),
            "min": float(np.min(p_region)),
            "max": float(np.max(p_region)),
            "std": float(np.std(p_region))
        },
        "vorticity": {
            "magnitude_mean": float(np.mean(np.sqrt(np.sum(w_region**2, axis=3))))
        }
    }
    
    return statistics

@router.post("/export/data")
async def export_data(
    session_id: str,
    format: str = "json",
    include_velocity: bool = True,
    include_pressure: bool = True,
    include_vorticity: bool = False,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_active_user)
):
    """导出数据"""
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 准备导出数据
    export_data = {
        "metadata": {
            "session_id": session_id,
            "step": sim["step"],
            "parameters": sim["params"]
        }
    }
    
    # 根据请求添加数据
    if include_velocity:
        velocity = solver.get_velocity_field()
        # 降采样以减少数据量
        export_data["velocity"] = velocity[::2, ::2, ::2].tolist()
    
    if include_pressure:
        pressure = solver.get_pressure_field()
        # 降采样以减少数据量
        export_data["pressure"] = pressure[::2, ::2, ::2].tolist()
    
    if include_vorticity:
        vorticity = solver.get_vorticity_field()
        # 降采样以减少数据量
        export_data["vorticity"] = vorticity[::2, ::2, ::2].tolist()
    
    # 根据格式返回数据
    if format == "json":
        return export_data
    elif format == "csv":
        # 实现CSV转换逻辑
        raise HTTPException(status_code=501, detail="CSV格式暂未实现")
    elif format == "vtk":
        # 实现VTK转换逻辑
        raise HTTPException(status_code=501, detail="VTK格式暂未实现")
    else:
        raise HTTPException(status_code=400, detail="不支持的格式") 

@router.post("/vortex-analysis", response_model=schemas.VortexAnalysisResponse)
async def analyze_vortex_structures(
    request: schemas.VortexAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    分析流体模拟中的涡结构
    
    此端点识别并分析流体中的关键涡结构，返回它们的位置、强度和其他特性
    """
    # 验证会话存在并属于当前用户
    session = db.query(models.SimulationSession).filter(
        models.SimulationSession.id == request.session_id,
        models.SimulationSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="模拟会话不存在或不属于当前用户")
    
    # 获取最新的模拟结果
    latest_result = db.query(models.SimulationResult).filter(
        models.SimulationResult.session_id == request.session_id
    ).order_by(models.SimulationResult.step.desc()).first()
    
    if not latest_result:
        raise HTTPException(status_code=404, detail="未找到模拟结果")
    
    try:
        # 调用核心计算模块进行涡结构分析
        from src.core.fluid_solver import analyze_vortex_structures
        
        # 从数据库中获取涡量场数据
        vorticity_data = json.loads(latest_result.vorticity_data) if latest_result.vorticity_data else None
        
        if not vorticity_data:
            # 如果没有直接存储涡量数据，则需要从速度场计算
            velocity_data = json.loads(latest_result.velocity_data) if latest_result.velocity_data else None
            if not velocity_data:
                raise HTTPException(status_code=400, detail="无法获取速度场或涡量场数据")
            
            # 计算涡量场
            vorticity_data = compute_vorticity_from_velocity(velocity_data, session.width, session.height, session.depth)
        
        # 分析涡结构
        vortex_structures = analyze_vortex_structures(
            vorticity_data, 
            session.width, 
            session.height, 
            session.depth,
            threshold=request.threshold,
            method=request.method
        )
        
        # 保存分析结果到数据库
        analysis_result = models.AnalysisResult(
            session_id=request.session_id,
            result_type="vortex_analysis",
            result_data=json.dumps({
                "vortex_structures": vortex_structures,
                "parameters": {
                    "threshold": request.threshold,
                    "method": request.method
                }
            }),
            created_at=datetime.now()
        )
        db.add(analysis_result)
        db.commit()
        
        return {
            "session_id": request.session_id,
            "step": latest_result.step,
            "vortex_structures": vortex_structures,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"涡结构分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"涡结构分析失败: {str(e)}")

@router.post("/turbulence-analysis", response_model=schemas.TurbulenceAnalysisResponse)
async def analyze_turbulence(
    request: schemas.TurbulenceAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    分析流体模拟中的湍流特性
    
    此端点计算湍流强度、雷诺应力和能量谱等湍流统计特性
    """
    # 验证会话存在并属于当前用户
    session = db.query(models.SimulationSession).filter(
        models.SimulationSession.id == request.session_id,
        models.SimulationSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="模拟会话不存在或不属于当前用户")
    
    # 获取最新的模拟结果
    latest_result = db.query(models.SimulationResult).filter(
        models.SimulationResult.session_id == request.session_id
    ).order_by(models.SimulationResult.step.desc()).first()
    
    if not latest_result:
        raise HTTPException(status_code=404, detail="未找到模拟结果")
    
    try:
        # 从数据库中获取速度场数据
        velocity_data = json.loads(latest_result.velocity_data) if latest_result.velocity_data else None
        
        if not velocity_data:
            raise HTTPException(status_code=400, detail="无法获取速度场数据")
        
        # 定义分析区域
        region = request.region if request.region else {
            "x_min": 0, "y_min": 0, "z_min": 0,
            "x_max": session.width, "y_max": session.height, "z_max": session.depth
        }
        
        # 调用湍流分析函数
        from src.core.fluid_solver import analyze_turbulence
        
        turbulence_results = analyze_turbulence(
            velocity_data,
            session.width,
            session.height,
            session.depth,
            region
        )
        
        # 保存分析结果到数据库
        analysis_result = models.AnalysisResult(
            session_id=request.session_id,
            result_type="turbulence_analysis",
            result_data=json.dumps(turbulence_results),
            created_at=datetime.now()
        )
        db.add(analysis_result)
        db.commit()
        
        return {
            "session_id": request.session_id,
            "step": latest_result.step,
            "turbulence_intensity": turbulence_results["turbulence_intensity"],
            "reynolds_stresses": turbulence_results["reynolds_stresses"],
            "energy_spectrum": turbulence_results["energy_spectrum"],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"湍流分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"湍流分析失败: {str(e)}")

def compute_vorticity_from_velocity(velocity_data, width, height, depth):
    """从速度场计算涡量场"""
    import numpy as np
    
    # 将速度场数据转换为NumPy数组
    u = np.zeros((width, height, depth))
    v = np.zeros((width, height, depth))
    w = np.zeros((width, height, depth))
    
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                idx = x + y * width + z * width * height
                if idx < len(velocity_data):
                    u[x, y, z] = velocity_data[idx][0]
                    v[x, y, z] = velocity_data[idx][1]
                    w[x, y, z] = velocity_data[idx][2]
    
    # 计算涡量场 (curl of velocity)
    dx = 1.0
    dy = 1.0
    dz = 1.0
    
    # 初始化涡量场
    vorticity = []
    
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                # 计算x方向导数
                x_next = min(x + 1, width - 1)
                x_prev = max(x - 1, 0)
                dv_dx = (v[x_next, y, z] - v[x_prev, y, z]) / (2 * dx)
                dw_dx = (w[x_next, y, z] - w[x_prev, y, z]) / (2 * dx)
                
                # 计算y方向导数
                y_next = min(y + 1, height - 1)
                y_prev = max(y - 1, 0)
                du_dy = (u[x, y_next, z] - u[x, y_prev, z]) / (2 * dy)
                dw_dy = (w[x, y_next, z] - w[x, y_prev, z]) / (2 * dy)
                
                # 计算z方向导数
                z_next = min(z + 1, depth - 1)
                z_prev = max(z - 1, 0)
                du_dz = (u[x, y, z_next] - u[x, y, z_prev]) / (2 * dz)
                dv_dz = (v[x, y, z_next] - v[x, y, z_prev]) / (2 * dz)
                
                # 计算涡量 (curl = ∇ × v)
                curl_x = dw_dy - dv_dz
                curl_y = du_dz - dw_dx
                curl_z = dv_dx - du_dy
                
                vorticity.append([curl_x, curl_y, curl_z])
    
    return vorticity 