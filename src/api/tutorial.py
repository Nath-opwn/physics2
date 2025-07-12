from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.database import get_db
from src.models.models import Tutorial, TutorialStep
from src.api.auth import get_current_active_user

router = APIRouter(
    prefix="/api/tutorial",
    tags=["tutorial"],
    responses={404: {"description": "Not found"}},
)

@router.get("/tutorials")
async def get_tutorials(
    difficulty: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """获取教程列表"""
    query = db.query(Tutorial)
    
    if difficulty:
        query = query.filter(Tutorial.difficulty == difficulty)
    
    tutorials = query.order_by(Tutorial.id).offset(offset).limit(limit).all()
    
    return tutorials

@router.get("/tutorials/{tutorial_id}")
async def get_tutorial(
    tutorial_id: int,
    db: Session = Depends(get_db)
):
    """获取教程详情"""
    tutorial = db.query(Tutorial).filter(Tutorial.id == tutorial_id).first()
    
    if not tutorial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="教程不存在"
        )
    
    return tutorial

@router.get("/tutorials/{tutorial_id}/steps")
async def get_tutorial_steps(
    tutorial_id: int,
    db: Session = Depends(get_db)
):
    """获取教程步骤"""
    tutorial = db.query(Tutorial).filter(Tutorial.id == tutorial_id).first()
    
    if not tutorial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="教程不存在"
        )
    
    steps = db.query(TutorialStep).filter(
        TutorialStep.tutorial_id == tutorial_id
    ).order_by(TutorialStep.step_number).all()
    
    return steps

@router.post("/tutorials")
async def create_tutorial(
    title: str,
    description: str,
    difficulty: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """创建教程"""
    tutorial = Tutorial(
        title=title,
        description=description,
        difficulty=difficulty
    )
    
    db.add(tutorial)
    db.commit()
    db.refresh(tutorial)
    
    return tutorial

@router.post("/tutorials/{tutorial_id}/steps")
async def create_tutorial_step(
    tutorial_id: int,
    step_number: int,
    title: str,
    content: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """创建教程步骤"""
    tutorial = db.query(Tutorial).filter(Tutorial.id == tutorial_id).first()
    
    if not tutorial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="教程不存在"
        )
    
    step = TutorialStep(
        tutorial_id=tutorial_id,
        step_number=step_number,
        title=title,
        content=content
    )
    
    db.add(step)
    db.commit()
    db.refresh(step)
    
    return step

@router.put("/tutorials/{tutorial_id}")
async def update_tutorial(
    tutorial_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    difficulty: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """更新教程"""
    tutorial = db.query(Tutorial).filter(Tutorial.id == tutorial_id).first()
    
    if not tutorial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="教程不存在"
        )
    
    if title:
        tutorial.title = title
    
    if description:
        tutorial.description = description
    
    if difficulty:
        tutorial.difficulty = difficulty
    
    db.commit()
    db.refresh(tutorial)
    
    return tutorial

@router.put("/tutorials/steps/{step_id}")
async def update_tutorial_step(
    step_id: int,
    step_number: Optional[int] = None,
    title: Optional[str] = None,
    content: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """更新教程步骤"""
    step = db.query(TutorialStep).filter(TutorialStep.id == step_id).first()
    
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="教程步骤不存在"
        )
    
    if step_number:
        step.step_number = step_number
    
    if title:
        step.title = title
    
    if content:
        step.content = content
    
    db.commit()
    db.refresh(step)
    
    return step

@router.delete("/tutorials/{tutorial_id}")
async def delete_tutorial(
    tutorial_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """删除教程"""
    tutorial = db.query(Tutorial).filter(Tutorial.id == tutorial_id).first()
    
    if not tutorial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="教程不存在"
        )
    
    # 删除关联的步骤
    db.query(TutorialStep).filter(TutorialStep.tutorial_id == tutorial_id).delete()
    
    # 删除教程
    db.delete(tutorial)
    db.commit()
    
    return {"message": "教程已删除"}

@router.delete("/tutorials/steps/{step_id}")
async def delete_tutorial_step(
    step_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """删除教程步骤"""
    step = db.query(TutorialStep).filter(TutorialStep.id == step_id).first()
    
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="教程步骤不存在"
        )
    
    db.delete(step)
    db.commit()
    
    return {"message": "教程步骤已删除"}

@router.get("/difficulties")
async def get_difficulties():
    """获取教程难度级别"""
    return ["beginner", "intermediate", "advanced"] 