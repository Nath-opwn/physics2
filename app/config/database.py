from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config.settings import settings

# 创建数据库引擎
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG)

# 创建会话类
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 初始化数据库
def init_db():
    """初始化数据库表"""
    from app.models.base import Base
    # 导入所有模型确保表被创建
    from app.models import experiment
    
    # 创建所有表
    Base.metadata.create_all(bind=engine) 