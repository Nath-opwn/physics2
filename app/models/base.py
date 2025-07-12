from sqlalchemy import Column, DateTime, func
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
import uuid

# 创建基础模型类
Base: DeclarativeMeta = declarative_base()

class BaseModel(Base):
    """所有模型的基类，提供共用字段和功能"""
    
    __abstract__ = True
    
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    
    @classmethod
    def generate_id(cls):
        """生成唯一ID"""
        return str(uuid.uuid4()) 