from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# 创建数据目录
os.makedirs("data", exist_ok=True)

# PostgreSQL数据库连接URL
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:t662ghf5@ps-0-postgresql.ns-dt9r90wb.svc:5432/fluiddb"

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 依赖项，用于获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 