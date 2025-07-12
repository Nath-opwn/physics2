from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class SimulationInit(BaseModel):
    """模拟初始化参数"""
    width: int = Field(..., description="宽度", ge=8, le=512)
    height: int = Field(..., description="高度", ge=8, le=512)
    depth: int = Field(..., description="深度", ge=8, le=512)
    viscosity: float = Field(0.1, description="粘度", ge=0.0, le=1.0)
    density: float = Field(1.0, description="密度", gt=0.0)
    name: str = Field("未命名模拟", description="模拟名称")
    boundary_type: int = Field(0, description="边界条件类型: 0=固定, 1=周期性")
    template: Optional[str] = Field(None, description="模板名称")
    use_gpu: bool = Field(False, description="是否使用GPU加速")

class SimulationStep(BaseModel):
    session_id: str
    dt: float = Field(gt=0.0, le=0.1, default=0.01, description="时间步长")
    
class SimulationControl(BaseModel):
    session_id: str
    action: str = Field(description="控制动作: start, pause, stop")
    
class ParameterUpdate(BaseModel):
    session_id: str
    parameters: Dict[str, Any]
    
class SessionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    
class SessionInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class DataProbeRequest(BaseModel):
    session_id: str
    x: float
    y: float
    z: float

class DataProbeResponse(BaseModel):
    position: Dict[str, float]
    velocity: List[float]
    pressure: float
    vorticity: List[float]

class RegionDataRequest(BaseModel):
    session_id: str
    x_min: int
    y_min: int
    z_min: int
    x_max: int
    y_max: int
    z_max: int

class RegionDataResponse(BaseModel):
    velocity: List
    pressure: List
    vorticity: List
    statistics: Dict[str, float]

class ForceRequest(BaseModel):
    session_id: str
    x: int
    y: int
    z: int
    fx: float
    fy: float
    fz: float

class ObstacleRequest(BaseModel):
    session_id: str
    shape: str = Field(description="障碍物形状: sphere, cylinder, box")
    params: Dict[str, Any] = Field(description="障碍物参数")

class PresetSimulationRequest(BaseModel):
    session_id: str
    preset_type: str = Field(description="预设模拟类型: cylinder_flow, karman_vortex, channel_flow, lid_driven_cavity")
    params: Optional[Dict[str, Any]] = Field(default=None, description="预设模拟参数")

class ExportDataRequest(BaseModel):
    session_id: str
    format: str = "json"
    include_velocity: bool = True
    include_pressure: bool = True
    include_vorticity: bool = False

class KnowledgeItemCreate(BaseModel):
    title: str
    category: str
    content: str
    tags: str

class KnowledgeItemResponse(BaseModel):
    id: int
    title: str
    category: str
    content: str
    tags: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class TutorialCreate(BaseModel):
    title: str
    description: str
    difficulty: str
    
class TutorialResponse(BaseModel):
    id: int
    title: str
    description: str
    difficulty: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class TutorialStepCreate(BaseModel):
    tutorial_id: int
    step_number: int
    title: str
    content: str
    
class TutorialStepResponse(BaseModel):
    id: int
    tutorial_id: int
    step_number: int
    title: str
    content: str
    
    class Config:
        orm_mode = True

class ExperimentCreate(BaseModel):
    title: str
    description: str
    parameters: Dict[str, Any]
    
class ExperimentResponse(BaseModel):
    id: int
    title: str
    description: str
    parameters: Dict[str, Any]
    created_at: datetime
    
    class Config:
        orm_mode = True 

class VortexAnalysisRequest(BaseModel):
    session_id: str
    threshold: float = Field(ge=0.0, default=0.1, description="涡结构识别阈值")
    method: str = Field(default="q_criterion", description="涡结构识别方法: q_criterion, lambda2, vorticity_magnitude")

class VortexStructure(BaseModel):
    position: List[float]  # [x, y, z]
    strength: float  # 涡结构强度
    size: float  # 涡结构大小估计
    orientation: List[float]  # 涡轴方向 [dx, dy, dz]

class VortexAnalysisResponse(BaseModel):
    session_id: str
    step: int
    vortex_structures: List[VortexStructure]
    timestamp: datetime

class TurbulenceAnalysisRequest(BaseModel):
    session_id: str
    region: Optional[Dict[str, int]] = Field(default=None, description="分析区域: {x_min, y_min, z_min, x_max, y_max, z_max}")

class TurbulenceAnalysisResponse(BaseModel):
    session_id: str
    step: int
    turbulence_intensity: float
    reynolds_stresses: List[List[float]]  # 雷诺应力张量
    energy_spectrum: Dict[str, List[float]]  # 能量谱
    timestamp: datetime 

class GPUAccelerationRequest(BaseModel):
    """GPU加速请求模型"""
    session_id: str
    use_gpu: bool = True 