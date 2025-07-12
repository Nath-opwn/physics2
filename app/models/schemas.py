from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# 参数定义模型
class ParameterDefinition(BaseModel):
    """参数定义"""
    name: str
    type: str  # 'number', 'select', 'boolean'等
    default: Any
    range: Optional[Tuple[float, float]] = None
    options: Optional[List[str]] = None
    unit: Optional[str] = None
    description: Optional[str] = None

# 实验类型模型
class ExperimentTypeBase(BaseModel):
    """实验类型基础模型"""
    type: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None

class ExperimentTypeCreate(ExperimentTypeBase):
    """创建实验类型请求"""
    parameters: List[ParameterDefinition]

class ExperimentTypeResponse(ExperimentTypeBase):
    """实验类型响应"""
    id: str
    parameters: List[ParameterDefinition]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

# 实验会话模型
class ExperimentSessionBase(BaseModel):
    """实验会话基础模型"""
    name: str
    description: Optional[str] = None

class ExperimentSessionCreate(ExperimentSessionBase):
    """创建实验会话请求"""
    experiment_type: str
    parameters: Dict[str, Any]

class ExperimentSessionUpdate(BaseModel):
    """更新实验会话请求"""
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ExperimentSessionResponse(ExperimentSessionBase):
    """实验会话响应"""
    id: str
    experiment_type: str
    user_id: str
    parameters: Dict[str, Any]
    status: str
    progress: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

# 实验结果模型
class ExperimentResultBase(BaseModel):
    """实验结果基础模型"""
    computation_time: Optional[float] = None
    
class ExperimentResultCreate(ExperimentResultBase):
    """创建实验结果请求"""
    session_id: str
    result_data: Optional[Dict[str, Any]] = None
    storage_path: Optional[str] = None
    error_message: Optional[str] = None

class ExperimentResultResponse(ExperimentResultBase):
    """实验结果响应"""
    id: str
    session_id: str
    result_data: Optional[Dict[str, Any]] = None
    storage_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# 可视化模型
class VisualizationBase(BaseModel):
    """可视化基础模型"""
    type: str
    meta_info: Optional[Dict[str, Any]] = None

class VisualizationCreate(VisualizationBase):
    """创建可视化请求"""
    result_id: str
    storage_path: str

class VisualizationResponse(VisualizationBase):
    """可视化响应"""
    id: str
    result_id: str
    storage_path: str
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# 运行实验响应
class RunExperimentResponse(BaseModel):
    """运行实验响应"""
    job_id: str
    session_id: str
    status: str = "pending"

# 实验状态响应
class ExperimentStatusResponse(BaseModel):
    """实验状态响应"""
    session_id: str
    status: str
    progress: int
    error_message: Optional[str] = None 