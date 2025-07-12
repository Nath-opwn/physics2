from sqlalchemy.orm import Session
import json
import numpy as np
import uuid
from datetime import datetime

from src.models.models import SimulationSession, SimulationResult, Parameter
from src.models.simulation_data import SimulationData, SurfaceTensionConfig, ContactAngleConfig, PerformanceMetric

class DataService:
    """数据服务类，处理与数据库的交互"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_simulation_session(self, user_id: int, name: str, description: str, params: dict) -> SimulationSession:
        """创建新的模拟会话"""
        session_id = str(uuid.uuid4())
        
        # 创建会话记录
        session = SimulationSession(
            id=session_id,
            user_id=user_id,
            name=name,
            description=description,
            width=params.get("width", 100),
            height=params.get("height", 100),
            depth=params.get("depth", 100),
            viscosity=params.get("viscosity", 0.01),
            density=params.get("density", 1.0),
            boundary_type=params.get("boundary_type", 1)
        )
        
        self.db.add(session)
        
        # 添加额外参数
        for key, value in params.items():
            if key not in ["width", "height", "depth", "viscosity", "density", "boundary_type"]:
                param = Parameter(
                    session_id=session_id,
                    name=key,
                    value=str(value) if not isinstance(value, (dict, list)) else json.dumps(value)
                )
                self.db.add(param)
        
        self.db.commit()
        return session
    
    def save_simulation_result(self, session_id: str, step: int, 
                               velocity_data=None, pressure_data=None, vorticity_data=None,
                               statistics=None) -> SimulationResult:
        """保存模拟结果"""
        result = SimulationResult(
            session_id=session_id,
            step=step,
            velocity_data=velocity_data,
            pressure_data=pressure_data,
            vorticity_data=vorticity_data,
            statistics=statistics
        )
        
        self.db.add(result)
        self.db.commit()
        return result
    
    def save_simulation_data(self, session_id: str, step: int, 
                             grid_size: dict, time_step: float, 
                             viscosity: float, density: float,
                             surface_tension_coefficient: float = None,
                             interface_curvature_data: str = None,
                             contact_angle: float = None,
                             contact_line_data: str = None) -> SimulationData:
        """保存详细模拟数据，包括表面张力和接触角数据"""
        sim_data = SimulationData(
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
        
        self.db.add(sim_data)
        self.db.commit()
        return sim_data
    
    def save_surface_tension_config(self, session_id: str, method: str, 
                                   coefficient: float, parameters: dict = None) -> SurfaceTensionConfig:
        """保存表面张力配置"""
        config = SurfaceTensionConfig(
            session_id=session_id,
            method=method,
            coefficient=coefficient,
            parameters=parameters
        )
        
        self.db.add(config)
        self.db.commit()
        return config
    
    def save_contact_angle_config(self, session_id: str, angle: float, 
                                 model: str, parameters: dict = None) -> ContactAngleConfig:
        """保存接触角配置"""
        config = ContactAngleConfig(
            session_id=session_id,
            angle=angle,
            model=model,
            parameters=parameters
        )
        
        self.db.add(config)
        self.db.commit()
        return config
    
    def save_performance_metric(self, session_id: str, implementation: str,
                               step_count: int, total_time: float, 
                               steps_per_second: float, memory_usage: float = None,
                               details: dict = None) -> PerformanceMetric:
        """保存性能指标数据"""
        metric = PerformanceMetric(
            session_id=session_id,
            implementation=implementation,
            step_count=step_count,
            total_time=total_time,
            steps_per_second=steps_per_second,
            memory_usage=memory_usage,
            details=details
        )
        
        self.db.add(metric)
        self.db.commit()
        return metric
    
    def get_simulation_sessions(self, user_id: int = None, limit: int = 100, offset: int = 0):
        """获取模拟会话列表"""
        query = self.db.query(SimulationSession)
        
        if user_id:
            query = query.filter(SimulationSession.user_id == user_id)
        
        return query.order_by(SimulationSession.created_at.desc()).offset(offset).limit(limit).all()
    
    def get_simulation_results(self, session_id: str, limit: int = 100, offset: int = 0):
        """获取模拟结果列表"""
        return self.db.query(SimulationResult)\
            .filter(SimulationResult.session_id == session_id)\
            .order_by(SimulationResult.step)\
            .offset(offset).limit(limit).all()
    
    def get_simulation_data(self, session_id: str, limit: int = 100, offset: int = 0):
        """获取模拟详细数据列表"""
        return self.db.query(SimulationData)\
            .filter(SimulationData.session_id == session_id)\
            .order_by(SimulationData.step)\
            .offset(offset).limit(limit).all()
    
    def get_performance_metrics(self, session_id: str = None, implementation: str = None):
        """获取性能指标数据"""
        query = self.db.query(PerformanceMetric)
        
        if session_id:
            query = query.filter(PerformanceMetric.session_id == session_id)
        
        if implementation:
            query = query.filter(PerformanceMetric.implementation == implementation)
        
        return query.order_by(PerformanceMetric.timestamp.desc()).all()
    
    def get_surface_tension_config(self, session_id: str):
        """获取表面张力配置"""
        return self.db.query(SurfaceTensionConfig)\
            .filter(SurfaceTensionConfig.session_id == session_id)\
            .order_by(SurfaceTensionConfig.created_at.desc())\
            .first()
    
    def get_contact_angle_config(self, session_id: str):
        """获取接触角配置"""
        return self.db.query(ContactAngleConfig)\
            .filter(ContactAngleConfig.session_id == session_id)\
            .order_by(ContactAngleConfig.created_at.desc())\
            .first() 