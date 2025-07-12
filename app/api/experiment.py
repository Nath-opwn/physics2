from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import uuid

from app.models.schemas import (
    ExperimentTypeResponse, 
    ExperimentSessionCreate, 
    ExperimentSessionUpdate,
    ExperimentSessionResponse,
    RunExperimentResponse,
    ExperimentStatusResponse
)
from app.models.experiment import ExperimentType, ExperimentSession, ExperimentStatusEnum
from app.config.database import get_db
from app.tasks.worker import run_experiment_task
# 不再直接导入storage_service
# from app.services.storage import storage_service
from app.experiments.registry import get_experiment_types, get_experiment_config

router = APIRouter()

# 获取可用的实验类型
@router.get("/experiments", response_model=List[ExperimentTypeResponse])
async def get_available_experiments(db: Session = Depends(get_db)):
    """
    返回所有可用的流体力学实验列表，包括实验类型、名称和描述。
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        experiments = db.query(ExperimentType).all()
        logger.debug(f"从数据库获取到 {len(experiments)} 个实验类型")
        
        # 如果数据库中没有实验类型，从注册表中加载
        if not experiments:
            logger.info("数据库中没有实验类型，从注册表加载")
            experiment_types = get_experiment_types()
            for exp_type in experiment_types:
                try:
                    parameters = [param.model_dump() for param in exp_type.parameters]
                    db_experiment = ExperimentType(
                        type=exp_type.type,
                        name=exp_type.name,
                        description=exp_type.description,
                        category=exp_type.category,
                        parameters_schema={"parameters": parameters}
                    )
                    db.add(db_experiment)
                except Exception as e:
                    logger.error(f"添加实验类型时出错: {str(e)}")
                    raise
            
            db.commit()
            experiments = db.query(ExperimentType).all()
            logger.debug(f"加载后，从数据库获取到 {len(experiments)} 个实验类型")
        
        # 将实验类型转换为响应模型
        return [
            ExperimentTypeResponse(
                id=exp.id,
                type=exp.type,
                name=exp.name,
                description=exp.description,
                category=exp.category,
                parameters=exp.parameters_schema.get("parameters", []),
                created_at=exp.created_at,
                updated_at=exp.updated_at
            )
            for exp in experiments
        ]
    except Exception as e:
        logger.error(f"获取实验类型时出错: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实验类型时出错: {str(e)}"
        )

# 获取特定实验类型的配置
@router.get("/experiments/{experiment_type}/configuration")
async def get_experiment_configuration(experiment_type: str, db: Session = Depends(get_db)):
    """
    获取特定实验类型的配置参数列表，包括参数名称、类型、默认值、范围等。
    """
    # 从数据库中查找实验类型
    db_experiment = db.query(ExperimentType).filter(ExperimentType.type == experiment_type).first()
    
    if not db_experiment:
        # 如果数据库中没有，尝试从注册表中获取
        try:
            experiment_config = get_experiment_config(experiment_type)
            return experiment_config
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"实验类型 '{experiment_type}' 不存在"
            )
    
    return db_experiment.parameters_schema

# 创建实验会话
@router.post("/sessions", response_model=ExperimentSessionResponse)
async def create_session(
    session_data: ExperimentSessionCreate, 
    db: Session = Depends(get_db)
):
    """
    创建新的实验会话，保存用户选择的参数。
    """
    # 验证实验类型是否存在
    experiment_type = db.query(ExperimentType).filter(ExperimentType.type == session_data.experiment_type).first()
    if not experiment_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"实验类型 '{session_data.experiment_type}' 不存在"
        )
    
    # TODO: 在实际应用中，从认证系统获取用户ID
    user_id = str(uuid.uuid4())  # 模拟用户ID
    
    # 创建实验会话
    db_session = ExperimentSession(
        user_id=user_id,
        experiment_type_id=experiment_type.id,
        name=session_data.name,
        description=session_data.description,
        parameters=session_data.parameters,
        status=ExperimentStatusEnum.PENDING
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    # 构造响应
    response = ExperimentSessionResponse(
        id=db_session.id,
        experiment_type=experiment_type.type,
        user_id=db_session.user_id,
        name=db_session.name,
        description=db_session.description,
        parameters=db_session.parameters,
        status=db_session.status.value,
        progress=db_session.progress,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at
    )
    
    return response

# 获取用户的所有会话
@router.get("/sessions/user", response_model=List[ExperimentSessionResponse])
async def get_user_sessions(db: Session = Depends(get_db)):
    """
    获取当前用户创建的所有实验会话。
    """
    # TODO: 在实际应用中，从认证系统获取用户ID
    user_id = str(uuid.uuid4())  # 模拟用户ID
    
    sessions = db.query(ExperimentSession).filter(ExperimentSession.user_id == user_id).all()
    
    # 构造响应
    response = []
    for session in sessions:
        experiment_type = db.query(ExperimentType).filter(ExperimentType.id == session.experiment_type_id).first()
        response.append(
            ExperimentSessionResponse(
                id=session.id,
                experiment_type=experiment_type.type,
                user_id=session.user_id,
                name=session.name,
                description=session.description,
                parameters=session.parameters,
                status=session.status.value,
                progress=session.progress,
                created_at=session.created_at,
                updated_at=session.updated_at
            )
        )
    
    return response

# 获取特定会话
@router.get("/sessions/{session_id}", response_model=ExperimentSessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """
    获取特定实验会话的详情。
    """
    session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 不存在"
        )
    
    # TODO: 在实际应用中，验证用户是否有权限访问此会话
    
    # 获取实验类型
    experiment_type = db.query(ExperimentType).filter(ExperimentType.id == session.experiment_type_id).first()
    
    # 构造响应
    response = ExperimentSessionResponse(
        id=session.id,
        experiment_type=experiment_type.type,
        user_id=session.user_id,
        name=session.name,
        description=session.description,
        parameters=session.parameters,
        status=session.status.value,
        progress=session.progress,
        created_at=session.created_at,
        updated_at=session.updated_at
    )
    
    return response

# 更新会话参数
@router.put("/sessions/{session_id}/parameters", response_model=ExperimentSessionResponse)
async def update_session_parameters(
    session_id: str, 
    update_data: ExperimentSessionUpdate, 
    db: Session = Depends(get_db)
):
    """
    更新实验会话的参数。
    """
    session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 不存在"
        )
    
    # TODO: 在实际应用中，验证用户是否有权限更新此会话
    
    # 检查会话是否可以更新
    if session.status != ExperimentStatusEnum.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法更新非待处理状态的会话参数，当前状态: {session.status.value}"
        )
    
    # 更新字段
    if update_data.name is not None:
        session.name = update_data.name
    
    if update_data.description is not None:
        session.description = update_data.description
    
    if update_data.parameters is not None:
        session.parameters = update_data.parameters
    
    db.commit()
    db.refresh(session)
    
    # 获取实验类型
    experiment_type = db.query(ExperimentType).filter(ExperimentType.id == session.experiment_type_id).first()
    
    # 构造响应
    response = ExperimentSessionResponse(
        id=session.id,
        experiment_type=experiment_type.type,
        user_id=session.user_id,
        name=session.name,
        description=session.description,
        parameters=session.parameters,
        status=session.status.value,
        progress=session.progress,
        created_at=session.created_at,
        updated_at=session.updated_at
    )
    
    return response

# 运行实验
@router.post("/sessions/{session_id}/run", response_model=RunExperimentResponse)
async def run_experiment(session_id: str, db: Session = Depends(get_db)):
    """
    启动实验计算，将任务提交到异步队列。
    返回job_id用于追踪计算状态。
    """
    session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 不存在"
        )
    
    # TODO: 在实际应用中，验证用户是否有权限运行此会话
    
    # 检查会话是否可以运行
    if session.status in [ExperimentStatusEnum.RUNNING, ExperimentStatusEnum.COMPLETED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法运行{session.status.value}状态的会话"
        )
    
    # 获取实验类型
    experiment_type = db.query(ExperimentType).filter(ExperimentType.id == session.experiment_type_id).first()
    
    # 提交异步任务
    task = run_experiment_task.apply_async(
        args=[session.id, experiment_type.type, session.parameters]
    )
    
    # 更新会话状态和任务ID
    session.status = ExperimentStatusEnum.RUNNING
    session.progress = 0
    session.job_id = task.id
    db.commit()
    
    # 构造响应
    response = RunExperimentResponse(
        job_id=task.id,
        session_id=session.id,
        status="running"
    )
    
    return response

# 获取实验状态
@router.get("/sessions/{session_id}/status", response_model=ExperimentStatusResponse)
async def get_experiment_status(session_id: str, db: Session = Depends(get_db)):
    """
    获取实验计算状态，包括状态码(pending/running/completed/failed)和进度。
    """
    session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 不存在"
        )
    
    # TODO: 在实际应用中，验证用户是否有权限访问此会话
    
    # 构造响应
    response = ExperimentStatusResponse(
        session_id=session.id,
        status=session.status.value,
        progress=session.progress,
        error_message=session.results.error_message if session.results and session.status == ExperimentStatusEnum.FAILED else None
    )
    
    return response

# 取消实验
@router.post("/sessions/{session_id}/cancel", response_model=ExperimentStatusResponse)
async def cancel_experiment(session_id: str, db: Session = Depends(get_db)):
    """
    取消正在运行的实验。
    """
    session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 不存在"
        )
    
    # TODO: 在实际应用中，验证用户是否有权限取消此会话
    
    # 检查会话是否可以取消
    if session.status != ExperimentStatusEnum.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"只能取消正在运行的实验，当前状态: {session.status.value}"
        )
    
    # 如果有任务ID，尝试取消任务
    if session.job_id:
        # TODO: 使用Celery API取消任务
        # from app.tasks.worker import celery_app
        # celery_app.control.revoke(session.job_id, terminate=True)
        pass
    
    # 更新会话状态
    session.status = ExperimentStatusEnum.CANCELLED
    session.progress = 0
    db.commit()
    
    # 构造响应
    response = ExperimentStatusResponse(
        session_id=session.id,
        status=session.status.value,
        progress=session.progress
    )
    
    return response

# 获取实验结果
@router.get("/sessions/{session_id}/results")
async def get_experiment_results(session_id: str, request: Request, db: Session = Depends(get_db)):
    """
    获取实验的完整结果数据。
    """
    session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 不存在"
        )
    
    # TODO: 在实际应用中，验证用户是否有权限访问此会话
    
    # 检查是否有结果
    if not session.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 没有可用的结果"
        )
    
    # 如果状态不是已完成，返回错误
    if session.status != ExperimentStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"实验未完成，当前状态: {session.status.value}"
        )
    
    # 获取结果数据
    result_data = session.results.result_data or {}
    
    # 如果有存储路径，从MinIO获取详细结果
    if session.results.storage_path:
        try:
            # 使用request.app.state.storage_service
            storage_service = request.app.state.storage_service
            detailed_results = storage_service.get_json(session.results.storage_path)
            result_data.update({"detailed_results": detailed_results})
        except Exception as e:
            # 记录错误但继续返回基本结果
            pass
    
    return result_data

# 获取可视化数据
@router.get("/sessions/{session_id}/visualization/{data_type}")
async def get_visualization_data(session_id: str, data_type: str, request: Request, db: Session = Depends(get_db)):
    """
    获取特定类型的可视化数据(如velocity, pressure, streamlines等)。
    """
    session = db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 不存在"
        )
    
    # TODO: 在实际应用中，验证用户是否有权限访问此会话
    
    # 检查是否有结果
    if not session.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 '{session_id}' 没有可用的结果"
        )
    
    # 如果状态不是已完成，返回错误
    if session.status != ExperimentStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"实验未完成，当前状态: {session.status.value}"
        )
    
    # 查找指定类型的可视化数据
    visualization = None
    for vis in session.results.visualizations:
        if vis.type == data_type:
            visualization = vis
            break
    
    if not visualization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"找不到类型为 '{data_type}' 的可视化数据"
        )
    
    # 从存储服务获取可视化数据
    try:
        # 使用request.app.state.storage_service
        storage_service = request.app.state.storage_service
        vis_data = storage_service.get_json(visualization.storage_path)
        
        # 如果数据包含文件引用，生成预签名URL
        if "file_references" in vis_data:
            for ref_key, ref_path in vis_data["file_references"].items():
                vis_data["file_references"][ref_key] = storage_service.generate_presigned_url(ref_path)
        
        return vis_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取可视化数据时发生错误: {str(e)}"
        ) 

# 添加一个基本的测试端点
@router.get("/experiments-test")
async def get_available_experiments_test():
    """
    返回测试用的实验类型列表，不从数据库读取。
    """
    return [
        {
            "id": "1",
            "type": "hydrostatic_pressure",
            "name": "静水压强实验",
            "description": "研究液体深度与压强关系的实验",
            "category": "基础流体静力学",
            "parameters": [],
            "created_at": "2025-07-12T08:00:00Z",
            "updated_at": "2025-07-12T08:00:00Z"
        },
        {
            "id": "2",
            "type": "reynolds_experiment",
            "name": "雷诺实验",
            "description": "观察不同雷诺数下流体流动状态变化的实验",
            "category": "流体动力学",
            "parameters": [],
            "created_at": "2025-07-12T08:00:00Z",
            "updated_at": "2025-07-12T08:00:00Z"
        }
    ] 