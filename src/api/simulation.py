from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json

from src.database.database import get_db
from src.services.data_service import DataService
from src.api.auth import get_current_active_user
from src.models.models import User

router = APIRouter(
    prefix="/api/simulation",
    tags=["simulation"],
    responses={404: {"description": "Not found"}},
)

@router.post("/session")
async def create_simulation_session(
    name: str,
    description: str,
    params: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """创建新的模拟会话"""
    data_service = DataService(db)
    session = data_service.create_simulation_session(
        user_id=current_user.id,
        name=name,
        description=description,
        params=params
    )
    
    return {"session_id": session.id, "message": "模拟会话创建成功"}

@router.post("/result/{session_id}")
async def save_simulation_result(
    session_id: str,
    step: int,
    velocity_data: Optional[str] = None,
    pressure_data: Optional[str] = None,
    vorticity_data: Optional[str] = None,
    statistics: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """保存模拟结果"""
    data_service = DataService(db)
    result = data_service.save_simulation_result(
        session_id=session_id,
        step=step,
        velocity_data=velocity_data,
        pressure_data=pressure_data,
        vorticity_data=vorticity_data,
        statistics=statistics
    )
    
    return {"result_id": result.id, "message": "模拟结果保存成功"}

@router.post("/data/{session_id}")
async def save_simulation_data(
    session_id: str,
    step: int,
    grid_size: Dict[str, int],
    time_step: float,
    viscosity: float,
    density: float,
    surface_tension_coefficient: Optional[float] = None,
    interface_curvature_data: Optional[str] = None,
    contact_angle: Optional[float] = None,
    contact_line_data: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """保存详细模拟数据，包括表面张力和接触角数据"""
    data_service = DataService(db)
    sim_data = data_service.save_simulation_data(
        session_id=session_id,
        step=step,
        grid_size=grid_size,
        time_step=time_step,
        viscosity=viscosity,
        density=density,
        surface_tension_coefficient=surface_tension_coefficient,
        interface_curvature_data=interface_curvature_data,
        contact_angle=contact_angle,
        contact_line_data=contact_line_data
    )
    
    return {"data_id": sim_data.id, "message": "模拟数据保存成功"}

@router.post("/surface-tension/{session_id}")
async def save_surface_tension_config(
    session_id: str,
    method: str,
    coefficient: float,
    parameters: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """保存表面张力配置"""
    data_service = DataService(db)
    config = data_service.save_surface_tension_config(
        session_id=session_id,
        method=method,
        coefficient=coefficient,
        parameters=parameters
    )
    
    return {"config_id": config.id, "message": "表面张力配置保存成功"}

@router.post("/contact-angle/{session_id}")
async def save_contact_angle_config(
    session_id: str,
    angle: float,
    model: str,
    parameters: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """保存接触角配置"""
    data_service = DataService(db)
    config = data_service.save_contact_angle_config(
        session_id=session_id,
        angle=angle,
        model=model,
        parameters=parameters
    )
    
    return {"config_id": config.id, "message": "接触角配置保存成功"}

@router.post("/performance/{session_id}")
async def save_performance_metric(
    session_id: str,
    implementation: str,
    step_count: int,
    total_time: float,
    steps_per_second: float,
    memory_usage: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """保存性能指标数据"""
    data_service = DataService(db)
    metric = data_service.save_performance_metric(
        session_id=session_id,
        implementation=implementation,
        step_count=step_count,
        total_time=total_time,
        steps_per_second=steps_per_second,
        memory_usage=memory_usage,
        details=details
    )
    
    return {"metric_id": metric.id, "message": "性能指标数据保存成功"}

@router.get("/sessions")
async def get_simulation_sessions(
    user_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取模拟会话列表"""
    data_service = DataService(db)
    
    # 如果没有指定用户ID，则使用当前用户的ID
    if user_id is None:
        user_id = current_user.id
    
    sessions = data_service.get_simulation_sessions(
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    
    # 转换为可序列化的字典列表
    result = []
    for session in sessions:
        result.append({
            "id": session.id,
            "name": session.name,
            "description": session.description,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "width": session.width,
            "height": session.height,
            "depth": session.depth,
            "viscosity": session.viscosity,
            "density": session.density,
            "boundary_type": session.boundary_type
        })
    
    return result

@router.get("/results/{session_id}")
async def get_simulation_results(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取模拟结果列表"""
    data_service = DataService(db)
    results = data_service.get_simulation_results(
        session_id=session_id,
        limit=limit,
        offset=offset
    )
    
    # 转换为可序列化的字典列表
    result = []
    for res in results:
        result.append({
            "id": res.id,
            "session_id": res.session_id,
            "step": res.step,
            "timestamp": res.timestamp,
            "has_velocity_data": res.velocity_data is not None,
            "has_pressure_data": res.pressure_data is not None,
            "has_vorticity_data": res.vorticity_data is not None,
            "statistics": res.statistics
        })
    
    return result

@router.get("/data/{session_id}")
async def get_simulation_data(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取模拟详细数据列表"""
    data_service = DataService(db)
    data_list = data_service.get_simulation_data(
        session_id=session_id,
        limit=limit,
        offset=offset
    )
    
    # 转换为可序列化的字典列表
    result = []
    for data in data_list:
        result.append({
            "id": data.id,
            "session_id": data.session_id,
            "step": data.step,
            "timestamp": data.timestamp,
            "grid_size": data.grid_size,
            "time_step": data.time_step,
            "viscosity": data.viscosity,
            "density": data.density,
            "surface_tension_coefficient": data.surface_tension_coefficient,
            "has_interface_curvature_data": data.interface_curvature_data is not None,
            "contact_angle": data.contact_angle,
            "has_contact_line_data": data.contact_line_data is not None
        })
    
    return result

@router.get("/performance")
async def get_performance_metrics(
    session_id: Optional[str] = None,
    implementation: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取性能指标数据"""
    data_service = DataService(db)
    metrics = data_service.get_performance_metrics(
        session_id=session_id,
        implementation=implementation
    )
    
    # 转换为可序列化的字典列表
    result = []
    for metric in metrics:
        result.append({
            "id": metric.id,
            "session_id": metric.session_id,
            "implementation": metric.implementation,
            "step_count": metric.step_count,
            "total_time": metric.total_time,
            "steps_per_second": metric.steps_per_second,
            "memory_usage": metric.memory_usage,
            "timestamp": metric.timestamp,
            "details": metric.details
        })
    
    return result

@router.get("/surface-tension/{session_id}")
async def get_surface_tension_config(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取表面张力配置"""
    data_service = DataService(db)
    config = data_service.get_surface_tension_config(session_id)
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到表面张力配置"
        )
    
    return {
        "id": config.id,
        "session_id": config.session_id,
        "method": config.method,
        "coefficient": config.coefficient,
        "created_at": config.created_at,
        "parameters": config.parameters
    }

@router.get("/contact-angle/{session_id}")
async def get_contact_angle_config(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """获取接触角配置"""
    data_service = DataService(db)
    config = data_service.get_contact_angle_config(session_id)
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到接触角配置"
        )
    
    return {
        "id": config.id,
        "session_id": config.session_id,
        "angle": config.angle,
        "model": config.model,
        "created_at": config.created_at,
        "parameters": config.parameters
    } 