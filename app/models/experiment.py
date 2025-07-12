from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, JSON, Text, Enum
from sqlalchemy.orm import relationship
import enum

from app.models.base import BaseModel

class ExperimentStatusEnum(str, enum.Enum):
    """实验状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExperimentType(BaseModel):
    """实验类型定义模型"""
    
    __tablename__ = "experiment_types"
    
    id = Column(String(36), primary_key=True, default=BaseModel.generate_id)
    type = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True)
    parameters_schema = Column(JSON, nullable=False)  # 存储参数定义的JSON Schema
    
    # 关系
    sessions = relationship("ExperimentSession", back_populates="experiment_type", cascade="all, delete-orphan")

class ExperimentSession(BaseModel):
    """实验会话模型"""
    
    __tablename__ = "experiment_sessions"
    
    id = Column(String(36), primary_key=True, default=BaseModel.generate_id)
    user_id = Column(String(36), nullable=False, index=True)
    experiment_type_id = Column(String(36), ForeignKey("experiment_types.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=False)  # 存储实际参数值
    status = Column(Enum(ExperimentStatusEnum), default=ExperimentStatusEnum.PENDING, nullable=False)
    progress = Column(Integer, default=0, nullable=False)
    job_id = Column(String(100), nullable=True)  # Celery任务ID
    
    # 关系
    experiment_type = relationship("ExperimentType", back_populates="sessions")
    results = relationship("ExperimentResult", back_populates="session", uselist=False, cascade="all, delete-orphan")

class ExperimentResult(BaseModel):
    """实验结果模型"""
    
    __tablename__ = "experiment_results"
    
    id = Column(String(36), primary_key=True, default=BaseModel.generate_id)
    session_id = Column(String(36), ForeignKey("experiment_sessions.id"), nullable=False, unique=True)
    result_data = Column(JSON, nullable=True)  # 存储基本结果数据
    storage_path = Column(String(255), nullable=True)  # 存储大型结果数据的路径
    computation_time = Column(Float, nullable=True)  # 计算耗时(秒)
    error_message = Column(Text, nullable=True)  # 如果失败，存储错误信息
    
    # 关系
    session = relationship("ExperimentSession", back_populates="results")
    visualizations = relationship("Visualization", back_populates="result", cascade="all, delete-orphan")

class Visualization(BaseModel):
    """可视化数据模型"""
    
    __tablename__ = "visualizations"
    
    id = Column(String(36), primary_key=True, default=BaseModel.generate_id)
    result_id = Column(String(36), ForeignKey("experiment_results.id"), nullable=False)
    type = Column(String(50), nullable=False)  # velocity, pressure, streamlines等
    storage_path = Column(String(255), nullable=False)  # 存储可视化数据的路径
    meta_info = Column(JSON, nullable=True)  # 可视化元数据
    
    # 关系
    result = relationship("ExperimentResult", back_populates="visualizations") 