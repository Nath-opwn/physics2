from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class VortexAnalysisRequest(BaseModel):
    """涡结构分析请求"""
    session_id: str
    threshold: float = Field(default=0.1, description="涡识别阈值")
    method: str = Field(default="q_criterion", description="涡识别方法，支持'q_criterion', 'lambda2', 'vorticity'")
    region: Optional[Dict[str, int]] = Field(default=None, description="分析区域，格式为{'x_min': 0, 'y_min': 0, 'z_min': 0, 'x_max': 10, 'y_max': 10, 'z_max': 10}")

class VortexStructure(BaseModel):
    """涡结构信息"""
    id: int
    center: List[float]
    size: float
    intensity: float
    rotation: str

class VortexAnalysisResponse(BaseModel):
    """涡结构分析响应"""
    session_id: str
    vortex_count: int
    structures: List[VortexStructure]
    average_intensity: float
    max_intensity: float
    statistics: Dict[str, Any]

class TurbulenceAnalysisRequest(BaseModel):
    """湍流分析请求"""
    session_id: str
    region: Optional[Dict[str, int]] = Field(default=None, description="分析区域，格式为{'x_min': 0, 'y_min': 0, 'z_min': 0, 'x_max': 10, 'y_max': 10, 'z_max': 10}")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="分析参数")

class TurbulenceAnalysisResponse(BaseModel):
    """湍流分析响应"""
    session_id: str
    reynolds_number: float
    energy_spectrum: Dict[str, List[float]]
    dissipation_rate: float
    kolmogorov_scale: float
    taylor_scale: float
    integral_scale: float
    statistics: Dict[str, Any] 