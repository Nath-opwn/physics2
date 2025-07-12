from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from src.utils.monitoring import (
    get_latest_metrics, get_metrics_history, get_average_metrics
)
from src.utils.error_handler import (
    get_error_history, get_error_stats, clear_error_history
)
from src.api.auth import get_current_active_user
from src.models.models import User

router = APIRouter(
    prefix="/api/monitoring",
    tags=["monitoring"],
    responses={404: {"description": "Not found"}},
)

# 响应模型
class MetricsResponse(BaseModel):
    timestamp: str
    system: Dict[str, Any]
    process: Dict[str, Any]

class ErrorResponse(BaseModel):
    timestamp: str
    error_type: str
    message: str
    traceback: Optional[str] = None
    context: Dict[str, Any] = {}

class ErrorStatsResponse(BaseModel):
    stats: Dict[str, int]
    total: int

@router.get("/metrics/current", response_model=MetricsResponse)
async def get_current_metrics(current_user: User = Depends(get_current_active_user)):
    """获取当前系统指标"""
    metrics = get_latest_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics available")
    return metrics

@router.get("/metrics/history", response_model=List[MetricsResponse])
async def get_metrics_history_api(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user)
):
    """获取历史系统指标"""
    return get_metrics_history(limit)

@router.get("/metrics/average", response_model=Dict[str, Any])
async def get_average_metrics_api(
    minutes: int = Query(5, ge=1, le=60),
    current_user: User = Depends(get_current_active_user)
):
    """获取平均系统指标"""
    return get_average_metrics(minutes)

@router.get("/errors/recent", response_model=List[ErrorResponse])
async def get_recent_errors(
    limit: int = Query(20, ge=1, le=100),
    error_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """获取最近错误"""
    return get_error_history(limit, error_type)

@router.get("/errors/stats", response_model=ErrorStatsResponse)
async def get_error_statistics(current_user: User = Depends(get_current_active_user)):
    """获取错误统计"""
    stats = get_error_stats()
    return {
        "stats": stats,
        "total": sum(stats.values())
    }

@router.post("/errors/clear")
async def clear_errors(current_user: User = Depends(get_current_active_user)):
    """清除错误历史"""
    clear_error_history()
    return {"status": "success", "message": "Error history cleared"}

@router.get("/health")
async def health_check():
    """健康检查端点，不需要认证"""
    metrics = get_latest_metrics()
    
    # 简单的健康状态判断
    health_status = "healthy"
    if metrics:
        if metrics["system"]["cpu_percent"] > 90 or metrics["system"]["memory_percent"] > 90:
            health_status = "warning"
    
    return {
        "status": health_status,
        "timestamp": metrics["timestamp"] if metrics else None,
    } 