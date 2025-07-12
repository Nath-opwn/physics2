from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any

from src.database.database import get_db
from src.models.models import User, SimulationSession, Parameter
from src.schemas.schemas import ParameterUpdate
from src.api.auth import get_current_active_user
from src.api.simulation import active_simulations

router = APIRouter(
    prefix="/api/parameters",
    tags=["parameters"],
    responses={404: {"description": "未找到"}},
)

@router.post("/update")
async def update_parameters(
    params: ParameterUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """更新模拟参数"""
    # 检查会话是否存在
    if params.session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 检查会话所有权
    session = db.query(SimulationSession).filter(
        SimulationSession.id == params.session_id,
        SimulationSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    
    # 获取模拟实例
    sim = active_simulations[params.session_id]
    solver = sim["solver"]
    
    # 更新参数
    for key, value in params.parameters.items():
        # 根据参数类型更新求解器
        if key == "viscosity":
            solver.viscosity = float(value)
        elif key == "density":
            solver.density = float(value)
        elif key == "boundary_type":
            solver.set_boundary_condition(int(value))
        
        # 记录参数更新到数据库
        db_param = Parameter(
            session_id=params.session_id,
            name=key,
            value=str(value)
        )
        db.add(db_param)
    
    # 更新会话参数
    for key, value in params.parameters.items():
        if hasattr(session, key):
            setattr(session, key, value)
    
    db.commit()
    
    return {"status": "parameters updated"}

@router.get("/presets", response_model=List[Dict[str, Any]])
async def get_parameter_presets():
    """获取参数预设"""
    # 返回一些常见流体模拟场景的预设参数
    return [
        {
            "name": "低粘度水流",
            "description": "模拟低粘度水流动",
            "parameters": {
                "viscosity": 0.01,
                "density": 1.0,
                "boundary_type": 0
            }
        },
        {
            "name": "高粘度油流",
            "description": "模拟高粘度油流动",
            "parameters": {
                "viscosity": 0.5,
                "density": 0.9,
                "boundary_type": 0
            }
        },
        {
            "name": "卡门涡街",
            "description": "模拟卡门涡街现象",
            "parameters": {
                "viscosity": 0.1,
                "density": 1.0,
                "boundary_type": 1
            }
        },
        {
            "name": "隧道气流",
            "description": "模拟隧道内气流",
            "parameters": {
                "viscosity": 0.05,
                "density": 0.2,
                "boundary_type": 2
            }
        }
    ]

@router.post("/validate")
async def validate_parameters(params: Dict[str, Any]):
    """验证参数有效性"""
    validation_results = {}
    
    # 验证粘度
    if "viscosity" in params:
        viscosity = params["viscosity"]
        if not isinstance(viscosity, (int, float)):
            validation_results["viscosity"] = "粘度必须是数值类型"
        elif viscosity < 0:
            validation_results["viscosity"] = "粘度不能为负值"
        elif viscosity > 1.0:
            validation_results["viscosity"] = "粘度超出正常范围"
    
    # 验证密度
    if "density" in params:
        density = params["density"]
        if not isinstance(density, (int, float)):
            validation_results["density"] = "密度必须是数值类型"
        elif density <= 0:
            validation_results["density"] = "密度必须大于0"
    
    # 验证边界条件
    if "boundary_type" in params:
        boundary_type = params["boundary_type"]
        if not isinstance(boundary_type, int):
            validation_results["boundary_type"] = "边界条件类型必须是整数"
        elif boundary_type not in [0, 1, 2, 3]:
            validation_results["boundary_type"] = "边界条件类型无效"
    
    # 验证网格尺寸
    for dim in ["width", "height", "depth"]:
        if dim in params:
            size = params[dim]
            if not isinstance(size, int):
                validation_results[dim] = f"{dim}必须是整数"
            elif size < 10:
                validation_results[dim] = f"{dim}太小，至少需要10"
            elif size > 500:
                validation_results[dim] = f"{dim}太大，不能超过500"
    
    if validation_results:
        return {"valid": False, "errors": validation_results}
    else:
        return {"valid": True} 