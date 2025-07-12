from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, DateTime, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from src.database.database import Base

class SimulationData(Base):
    __tablename__ = "simulation_data"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"))
    step = Column(Integer)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # 基础数据
    grid_size = Column(JSON)  # {"width": w, "height": h, "depth": d}
    time_step = Column(Float)
    
    # 物理参数
    viscosity = Column(Float)
    density = Column(Float)
    
    # 表面张力参数
    surface_tension_coefficient = Column(Float, nullable=True)
    interface_curvature_data = Column(Text, nullable=True)  # 存储路径或压缩JSON
    
    # 接触角参数
    contact_angle = Column(Float, nullable=True)
    contact_line_data = Column(Text, nullable=True)  # 存储路径或压缩JSON
    
    # 关联
    session = relationship("SimulationSession", back_populates="simulation_data")

# 更新SimulationSession模型中的关系
from src.models.models import SimulationSession
SimulationSession.simulation_data = relationship("SimulationData", back_populates="session")

class SurfaceTensionConfig(Base):
    __tablename__ = "surface_tension_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"))
    method = Column(String)  # "CSF", "SSF", "PF" 等表面张力计算方法
    coefficient = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 额外配置参数
    parameters = Column(JSON, nullable=True)
    
    # 关联
    session = relationship("SimulationSession")

class ContactAngleConfig(Base):
    __tablename__ = "contact_angle_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"))
    angle = Column(Float)  # 接触角（度）
    model = Column(String)  # "static", "dynamic", "hysteresis" 等接触角模型
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 额外配置参数（如接触角滞后参数）
    parameters = Column(JSON, nullable=True)
    
    # 关联
    session = relationship("SimulationSession")

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"))
    implementation = Column(String)  # "python", "cpp", "cuda"
    step_count = Column(Integer)
    total_time = Column(Float)  # 总执行时间（秒）
    steps_per_second = Column(Float)  # 每秒执行步数
    memory_usage = Column(Float, nullable=True)  # 内存使用（MB）
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # 详细性能数据
    details = Column(JSON, nullable=True)
    
    # 关联
    session = relationship("SimulationSession") 