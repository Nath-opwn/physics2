from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
from dotenv import load_dotenv

# 尝试加载环境变量
load_dotenv()

class Settings(BaseSettings):
    """应用程序设置配置类"""
    
    # API配置
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # 数据库配置
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "fluiddb")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    @property
    def DATABASE_URL(self) -> str:
        """构建数据库连接URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # Celery配置
    @property
    def CELERY_BROKER_URL(self) -> str:
        """构建Celery代理URL"""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        """构建Celery结果后端URL"""
        return self.CELERY_BROKER_URL
    
    # JWT配置
    SECRET_KEY: str = os.getenv("SECRET_KEY", "default_secret_key_for_development_only")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # MinIO配置
    MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "minio")
    MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
    MINIO_HOST: str = os.getenv("MINIO_HOST", "localhost")
    MINIO_PORT: str = os.getenv("MINIO_PORT", "9000")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "fluidresults")
    
    @property
    def MINIO_ENDPOINT(self) -> str:
        """构建MinIO端点URL"""
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"
    
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "fluidlab")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    # 实验配置
    MAX_CONCURRENT_TASKS: int = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
    MAX_SIMULATION_TIME: int = int(os.getenv("MAX_SIMULATION_TIME", "3600"))  # 秒
    DATA_STORAGE_PATH: str = os.getenv("DATA_STORAGE_PATH", "/app/data")
    TEMP_STORAGE_PATH: str = os.getenv("TEMP_STORAGE_PATH", "/app/temp")
    
    # GPU配置
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore"
    )

# 创建全局设置实例
settings = Settings() 