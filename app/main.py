from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from app.api import experiment
from app.config.settings import settings
from app.config.database import init_db
from app.services.storage import StorageService

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

# 创建应用
app = FastAPI(
    title="流体力学仿真系统 API",
    description="高性能流体力学模拟系统的后端API，处理前端提交的模拟请求，执行计算密集型任务，并提供结果数据。",
    version="0.1.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定确切的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量用于跟踪数据库可用状态
db_available = False

# 初始化服务
@app.on_event("startup")
async def startup_services():
    global db_available
    
    # 初始化存储服务
    try:
        app.state.storage_service = StorageService()
        logger.info("存储服务初始化完成")
    except Exception as e:
        logger.error(f"存储服务初始化失败: {str(e)}")
        app.state.storage_service = StorageService()  # 重试，将使用本地存储
    
    # 初始化数据库
    try:
        init_db()
        db_available = True
        logger.info("数据库初始化完成")
    except Exception as e:
        db_available = False
        logger.error(f"数据库初始化失败: {str(e)}")
        logger.warning("API将以有限功能模式启动，某些需要数据库的功能可能不可用")

# 注册API路由
app.include_router(
    experiment.router,
    prefix="/api/lab",
    tags=["实验室"]
)

# 健康检查端点
@app.get("/health", tags=["系统"])
async def health_check():
    """系统健康检查端点"""
    return {
        "status": "healthy", 
        "version": "0.1.0",
        "database_available": db_available,
        "storage_mode": "minio" if app.state.storage_service.use_minio else "local"
    }

# 根路径
@app.get("/", tags=["系统"])
async def root():
    """API根端点，提供API信息"""
    return {
        "name": "流体力学仿真系统 API",
        "version": "0.1.0",
        "documentation": "/docs",
        "status": "运行中",
        "database_available": db_available
    } 