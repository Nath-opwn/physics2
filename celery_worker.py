"""
Celery工作进程启动脚本

使用方法:
    celery -A celery_worker.celery_app worker --loglevel=info
"""
from app.tasks.worker import celery_app

if __name__ == "__main__":
    celery_app.start() 