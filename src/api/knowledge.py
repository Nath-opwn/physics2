from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.database import get_db
from src.models.models import KnowledgeItem
from src.api.auth import get_current_active_user

router = APIRouter(
    prefix="/api/knowledge",
    tags=["knowledge"],
    responses={404: {"description": "Not found"}},
)

@router.get("/items")
async def get_knowledge_items(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """获取知识项列表"""
    query = db.query(KnowledgeItem)
    
    if category:
        query = query.filter(KnowledgeItem.category == category)
    
    if tag:
        query = query.filter(KnowledgeItem.tags.contains(tag))
    
    items = query.order_by(KnowledgeItem.id).offset(offset).limit(limit).all()
    
    return items

@router.get("/items/{item_id}")
async def get_knowledge_item(
    item_id: int,
    db: Session = Depends(get_db)
):
    """获取知识项详情"""
    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识项不存在"
        )
    
    return item

@router.post("/items")
async def create_knowledge_item(
    title: str,
    category: str,
    content: str,
    tags: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """创建新知识项"""
    item = KnowledgeItem(
        title=title,
        category=category,
        content=content,
        tags=tags
    )
    
    db.add(item)
    db.commit()
    db.refresh(item)
    
    return item

@router.put("/items/{item_id}")
async def update_knowledge_item(
    item_id: int,
    title: Optional[str] = None,
    category: Optional[str] = None,
    content: Optional[str] = None,
    tags: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """更新知识项"""
    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识项不存在"
        )
    
    if title:
        item.title = title
    
    if category:
        item.category = category
    
    if content:
        item.content = content
    
    if tags:
        item.tags = tags
    
    db.commit()
    db.refresh(item)
    
    return item

@router.delete("/items/{item_id}")
async def delete_knowledge_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """删除知识项"""
    item = db.query(KnowledgeItem).filter(KnowledgeItem.id == item_id).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识项不存在"
        )
    
    db.delete(item)
    db.commit()
    
    return {"message": "知识项已删除"}

@router.get("/categories")
async def get_categories(
    db: Session = Depends(get_db)
):
    """获取所有分类"""
    categories = db.query(KnowledgeItem.category).distinct().all()
    return [category[0] for category in categories]

@router.get("/tags")
async def get_tags(
    db: Session = Depends(get_db)
):
    """获取所有标签"""
    items = db.query(KnowledgeItem.tags).all()
    all_tags = set()
    
    for item in items:
        if item[0]:
            tags = item[0].split(",")
            for tag in tags:
                all_tags.add(tag.strip())
    
    return list(all_tags) 