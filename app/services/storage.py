from minio import Minio
from minio.error import S3Error
import io
import os
import json
from typing import BinaryIO, Dict, Any, Optional
import logging
import shutil
import pathlib

from app.config.settings import settings

logger = logging.getLogger(__name__)

class StorageService:
    """存储服务，负责处理大型文件的存储和检索，支持MinIO或本地文件系统"""
    
    def __init__(self):
        """初始化存储服务，尝试连接MinIO，失败则使用本地文件系统"""
        self.use_minio = True
        self.local_storage_path = os.path.join(os.getcwd(), "data", "storage")
        
        # 确保本地存储目录存在
        if not os.path.exists(self.local_storage_path):
            os.makedirs(self.local_storage_path, exist_ok=True)
            logger.info(f"已创建本地存储目录: {self.local_storage_path}")
            
        try:
            self.client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )
            
            # 确保存储桶存在
            if not self.client.bucket_exists(settings.MINIO_BUCKET):
                self.client.make_bucket(settings.MINIO_BUCKET)
                logger.info(f"已创建存储桶: {settings.MINIO_BUCKET}")
        except Exception as e:
            logger.warning(f"初始化MinIO客户端失败: {str(e)}")
            logger.info("将使用本地文件系统作为存储后端")
            self.use_minio = False
    
    def store_file(self, object_name: str, data: BinaryIO, content_type: str = "application/octet-stream") -> str:
        """
        存储文件到MinIO或本地文件系统
        
        参数:
            object_name: 对象名称/路径
            data: 文件数据
            content_type: 内容类型
            
        返回:
            对象的完整路径
        """
        if self.use_minio:
            try:
                # 获取文件大小
                data.seek(0, os.SEEK_END)
                size = data.tell()
                data.seek(0)
                
                # 上传到MinIO
                self.client.put_object(
                    settings.MINIO_BUCKET,
                    object_name,
                    data,
                    size,
                    content_type=content_type
                )
                
                return f"{settings.MINIO_BUCKET}/{object_name}"
            except S3Error as e:
                logger.error(f"存储文件失败 {object_name}: {str(e)}")
                logger.info("尝试使用本地文件系统作为后备")
                self.use_minio = False
                return self.store_file(object_name, data, content_type)
        else:
            # 使用本地文件系统
            try:
                file_path = os.path.join(self.local_storage_path, object_name)
                directory = os.path.dirname(file_path)
                
                # 确保目录存在
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                
                # 写入文件
                with open(file_path, 'wb') as f:
                    f.write(data.read())
                
                return f"local:{object_name}"
            except Exception as e:
                logger.error(f"存储本地文件失败 {object_name}: {str(e)}")
                raise
    
    def store_json(self, object_name: str, data: Dict[str, Any]) -> str:
        """
        存储JSON数据
        
        参数:
            object_name: 对象名称/路径
            data: 要存储的JSON数据
            
        返回:
            对象的完整路径
        """
        try:
            # 将字典转换为JSON字符串
            json_data = json.dumps(data).encode('utf-8')
            
            # 创建内存文件对象
            data_stream = io.BytesIO(json_data)
            
            # 上传文件
            return self.store_file(object_name, data_stream, "application/json")
        except Exception as e:
            logger.error(f"存储JSON数据失败 {object_name}: {str(e)}")
            raise
    
    def get_file(self, object_name: str) -> BinaryIO:
        """
        获取文件
        
        参数:
            object_name: 对象名称/路径
            
        返回:
            文件数据的字节流
        """
        if object_name.startswith("local:"):
            # 从本地文件系统获取
            local_path = object_name.replace("local:", "")
            file_path = os.path.join(self.local_storage_path, local_path)
            try:
                with open(file_path, 'rb') as f:
                    return io.BytesIO(f.read())
            except Exception as e:
                logger.error(f"获取本地文件失败 {file_path}: {str(e)}")
                raise
        elif self.use_minio:
            try:
                # 从完整路径中提取对象名称（如果需要）
                if object_name.startswith(f"{settings.MINIO_BUCKET}/"):
                    object_name = object_name[len(f"{settings.MINIO_BUCKET}/"):]
                
                # 获取对象
                response = self.client.get_object(settings.MINIO_BUCKET, object_name)
                
                # 读取所有数据并返回
                return io.BytesIO(response.read())
            except S3Error as e:
                logger.error(f"获取文件失败 {object_name}: {str(e)}")
                raise
        else:
            # 从本地文件系统获取
            file_path = os.path.join(self.local_storage_path, object_name)
            try:
                with open(file_path, 'rb') as f:
                    return io.BytesIO(f.read())
            except Exception as e:
                logger.error(f"获取本地文件失败 {file_path}: {str(e)}")
                raise
    
    def get_json(self, object_name: str) -> Dict[str, Any]:
        """
        获取JSON数据
        
        参数:
            object_name: 对象名称/路径
            
        返回:
            JSON数据的字典
        """
        try:
            # 获取文件
            data_stream = self.get_file(object_name)
            
            # 解析JSON
            return json.loads(data_stream.getvalue().decode('utf-8'))
        except Exception as e:
            logger.error(f"获取JSON数据失败 {object_name}: {str(e)}")
            raise
    
    def generate_presigned_url(self, object_name: str, expires: int = 3600) -> str:
        """
        生成预签名URL或本地文件路径
        
        参数:
            object_name: 对象名称/路径
            expires: 过期时间（秒）
            
        返回:
            预签名URL或本地文件路径
        """
        if object_name.startswith("local:"):
            # 本地文件路径
            local_path = object_name.replace("local:", "")
            file_path = os.path.join(self.local_storage_path, local_path)
            return f"file://{file_path}"
        elif self.use_minio:
            try:
                # 从完整路径中提取对象名称（如果需要）
                if object_name.startswith(f"{settings.MINIO_BUCKET}/"):
                    object_name = object_name[len(f"{settings.MINIO_BUCKET}/"):]
                
                # 生成URL
                return self.client.presigned_get_object(
                    settings.MINIO_BUCKET,
                    object_name,
                    expires=expires
                )
            except S3Error as e:
                logger.error(f"生成预签名URL失败 {object_name}: {str(e)}")
                raise
        else:
            # 本地文件路径
            file_path = os.path.join(self.local_storage_path, object_name)
            return f"file://{file_path}"
    
    def delete_file(self, object_name: str) -> bool:
        """
        删除文件
        
        参数:
            object_name: 对象名称/路径
            
        返回:
            是否成功删除
        """
        if object_name.startswith("local:"):
            # 从本地文件系统删除
            local_path = object_name.replace("local:", "")
            file_path = os.path.join(self.local_storage_path, local_path)
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                logger.error(f"删除本地文件失败 {file_path}: {str(e)}")
                return False
        elif self.use_minio:
            try:
                # 从完整路径中提取对象名称（如果需要）
                if object_name.startswith(f"{settings.MINIO_BUCKET}/"):
                    object_name = object_name[len(f"{settings.MINIO_BUCKET}/"):]
                
                # 删除对象
                self.client.remove_object(settings.MINIO_BUCKET, object_name)
                return True
            except S3Error as e:
                logger.error(f"删除文件失败 {object_name}: {str(e)}")
                return False
        else:
            # 从本地文件系统删除
            file_path = os.path.join(self.local_storage_path, object_name)
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                logger.error(f"删除本地文件失败 {file_path}: {str(e)}")
                return False 