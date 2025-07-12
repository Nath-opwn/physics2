from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, DateTime, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from src.database.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    sessions = relationship("SimulationSession", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    key = Column(String, index=True)
    value = Column(String)
    
    user = relationship("User", back_populates="preferences")

class SimulationSession(Base):
    __tablename__ = "simulation_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 模拟参数
    width = Column(Integer)
    height = Column(Integer)
    depth = Column(Integer)
    viscosity = Column(Float)
    density = Column(Float)
    boundary_type = Column(Integer)
    
    # 关联
    user = relationship("User", back_populates="sessions")
    results = relationship("SimulationResult", back_populates="session")
    parameters = relationship("Parameter", back_populates="session")

class SimulationResult(Base):
    __tablename__ = "simulation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"))
    step = Column(Integer)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # 结果数据 - 存储路径或压缩数据
    velocity_data = Column(Text, nullable=True)  # 存储路径或压缩JSON
    pressure_data = Column(Text, nullable=True)
    vorticity_data = Column(Text, nullable=True)
    
    # 统计数据
    statistics = Column(JSON, nullable=True)
    
    session = relationship("SimulationSession", back_populates="results")

class Parameter(Base):
    __tablename__ = "parameters"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"))
    name = Column(String, index=True)
    value = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    session = relationship("SimulationSession", back_populates="parameters")

class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    category = Column(String, index=True)
    content = Column(Text)
    tags = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Tutorial(Base):
    __tablename__ = "tutorials"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    difficulty = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    steps = relationship("TutorialStep", back_populates="tutorial")

class TutorialStep(Base):
    __tablename__ = "tutorial_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    tutorial_id = Column(Integer, ForeignKey("tutorials.id"))
    step_number = Column(Integer)
    title = Column(String)
    content = Column(Text)
    
    tutorial = relationship("Tutorial", back_populates="steps")

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    parameters = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now()) 

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("simulation_sessions.id"))
    result_type = Column(String, index=True)  # 例如: "vortex_analysis", "turbulence_analysis"
    result_data = Column(Text)  # JSON格式存储的分析结果
    created_at = Column(DateTime(timezone=True))
    
    session = relationship("SimulationSession") 