from celery import Celery
import logging
import os
import time
import io
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import json

from app.config.settings import settings
from app.config.database import SessionLocal
from app.models.experiment import ExperimentSession, ExperimentResult, Visualization, ExperimentStatusEnum
from app.experiments import get_experiment_calculator
from app.services.storage import StorageService

logger = logging.getLogger(__name__)

# 创建Celery应用
celery_app = Celery(
    "fluid_lab",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# 配置Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    worker_max_tasks_per_child=10,
    task_time_limit=settings.MAX_SIMULATION_TIME,  # 最大任务运行时间
)

# 创建本地存储服务实例
storage_service = StorageService()

def update_session_status(session_id: str, status: str, progress: int):
    """
    更新会话状态
    
    参数:
        session_id: 会话ID
        status: 状态 ("running", "completed", "failed", "cancelled")
        progress: 进度 (0-100)
    """
    try:
        db = SessionLocal()
        session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
        
        if session:
            # 更新状态
            if status == "running":
                session.status = ExperimentStatusEnum.RUNNING
            elif status == "completed":
                session.status = ExperimentStatusEnum.COMPLETED
            elif status == "failed":
                session.status = ExperimentStatusEnum.FAILED
            elif status == "cancelled":
                session.status = ExperimentStatusEnum.CANCELLED
            
            # 更新进度
            session.progress = progress
            
            db.commit()
    except Exception as e:
        logger.error(f"更新会话状态失败: {str(e)}")
    finally:
        db.close()

def log_experiment_error(session_id: str, error_message: str):
    """
    记录实验错误
    
    参数:
        session_id: 会话ID
        error_message: 错误信息
    """
    try:
        db = SessionLocal()
        session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
        
        if session:
            # 查找或创建结果记录
            result = session.results
            if not result:
                result = ExperimentResult(session_id=session_id, error_message=error_message)
                db.add(result)
            else:
                result.error_message = error_message
            
            db.commit()
    except Exception as e:
        logger.error(f"记录实验错误失败: {str(e)}")
    finally:
        db.close()

def store_experiment_results(
    session_id: str, 
    results: Dict[str, Any], 
    time_steps_data: List[Dict[str, Any]], 
    visualization_data: Dict[str, Any]
):
    """
    存储实验结果
    
    参数:
        session_id: 会话ID
        results: 基本结果数据
        time_steps_data: 时间步数据
        visualization_data: 可视化数据
    """
    try:
        db = SessionLocal()
        session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
        
        if not session:
            logger.error(f"找不到会话: {session_id}")
            return
        
        # 创建基本结果对象
        result = session.results
        if not result:
            result = ExperimentResult(
                session_id=session_id,
                result_data=results,
                computation_time=results.get("computation_time", 0.0)
            )
            db.add(result)
        else:
            result.result_data = results
            result.computation_time = results.get("computation_time", 0.0)
        
        # 将详细结果存储到MinIO
        detailed_results = {
            "results": results,
            "time_steps_data": time_steps_data
        }
        
        # 创建存储路径
        results_path = f"results/{session_id}/detailed_results.json"
        
        # 存储详细结果
        try:
            storage_path = storage_service.store_json(results_path, detailed_results)
            result.storage_path = storage_path
        except Exception as e:
            logger.error(f"存储详细结果失败: {str(e)}")
        
        # 保存基本结果到数据库
        db.commit()
        db.refresh(result)
        
        # 处理可视化数据
        for step_key, step_data in visualization_data.items():
            # 对于每种可视化类型创建记录
            if "plots" in step_data:
                for vis_type, plot_data in step_data["plots"].items():
                    # 存储图像
                    vis_path = f"results/{session_id}/visualizations/{step_key}_{vis_type}.png"
                    
                    try:
                        # 存储图像文件
                        img_storage_path = storage_service.store_file(
                            vis_path, 
                            plot_data, 
                            "image/png"
                        )
                        
                        # 创建可视化记录
                        vis = Visualization(
                            result_id=result.id,
                            type=vis_type,
                            storage_path=img_storage_path,
                            meta_info={
                                "step": step_data.get("step"),
                                "time": step_data.get("time")
                            }
                        )
                        db.add(vis)
                    except Exception as e:
                        logger.error(f"存储可视化数据失败: {str(e)}")
            
            # 移除大型二进制数据以避免过大的JSON对象
            if "plots" in step_data:
                del step_data["plots"]
        
        # 存储可视化数据的元数据
        vis_meta_path = f"results/{session_id}/visualization_metadata.json"
        try:
            storage_service.store_json(vis_meta_path, visualization_data)
        except Exception as e:
            logger.error(f"存储可视化元数据失败: {str(e)}")
        
        # 提交所有更改
        db.commit()
        
    except Exception as e:
        logger.error(f"存储实验结果失败: {str(e)}")
    finally:
        db.close()

@celery_app.task
def run_experiment_task(session_id: str, experiment_type: str, parameters: Dict[str, Any]):
    """
    后台运行实验计算的Celery任务
    
    参数:
        session_id: 会话ID
        experiment_type: 实验类型
        parameters: 实验参数
        
    返回:
        任务状态信息
    """
    try:
        logger.info(f"开始实验计算: {experiment_type}, 会话ID: {session_id}")
        
        # 更新状态为运行中
        update_session_status(session_id, "running", 0)
        
        # 创建进度回调
        def progress_callback(progress):
            update_session_status(session_id, "running", progress)
        
        # 获取计算模块
        try:
            calculator = get_experiment_calculator(experiment_type)
        except KeyError:
            error_msg = f"未知实验类型: {experiment_type}"
            logger.error(error_msg)
            update_session_status(session_id, "failed", 0)
            log_experiment_error(session_id, error_msg)
            return {"status": "error", "error": error_msg}
        
        # 执行计算
        try:
            results, time_steps_data, visualization_data = calculator(
                parameters,
                progress_callback=progress_callback
            )
        except Exception as e:
            error_msg = f"计算过程发生错误: {str(e)}"
            logger.error(error_msg)
            update_session_status(session_id, "failed", 0)
            log_experiment_error(session_id, error_msg)
            return {"status": "error", "error": error_msg}
        
        # 存储结果
        try:
            store_experiment_results(session_id, results, time_steps_data, visualization_data)
        except Exception as e:
            error_msg = f"存储结果失败: {str(e)}"
            logger.error(error_msg)
            update_session_status(session_id, "failed", 0)
            log_experiment_error(session_id, error_msg)
            return {"status": "error", "error": error_msg}
        
        # 更新状态为完成
        update_session_status(session_id, "completed", 100)
        
        logger.info(f"实验计算完成: {experiment_type}, 会话ID: {session_id}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "experiment_type": experiment_type
        }
        
    except Exception as e:
        error_msg = f"任务执行失败: {str(e)}"
        logger.error(error_msg)
        update_session_status(session_id, "failed", 0)
        log_experiment_error(session_id, error_msg)
        return {
            "status": "error",
            "session_id": session_id,
            "error": error_msg
        } 