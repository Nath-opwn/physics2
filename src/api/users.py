from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Dict, List

from src.database.database import get_db
from src.models.models import User, UserPreference
from src.schemas.schemas import UserCreate, User as UserSchema, Token
from src.api.auth import (
    get_user, authenticate_user, create_access_token,
    get_password_hash, get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter(
    prefix="/api",
    tags=["users"],
    responses={404: {"description": "未找到"}},
)

@router.post("/auth/register", response_model=UserSchema)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """注册新用户"""
    # 检查用户名是否已存在
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已被注册"
        )
    
    # 检查邮箱是否已存在
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已被注册"
        )
    
    # 创建新用户
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.post("/auth/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """用户登录"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 创建访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """获取当前用户信息"""
    return current_user

@router.get("/user/preferences")
async def get_user_preferences(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """获取用户偏好设置"""
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).all()
    
    # 转换为字典格式
    result = {}
    for pref in preferences:
        result[pref.key] = pref.value
    
    return result

@router.post("/user/preferences")
async def update_user_preferences(
    preferences: Dict[str, str],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """更新用户偏好设置"""
    # 更新或创建偏好设置
    for key, value in preferences.items():
        # 查找是否已存在
        pref = db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id,
            UserPreference.key == key
        ).first()
        
        if pref:
            # 更新现有偏好
            pref.value = value
        else:
            # 创建新偏好
            new_pref = UserPreference(
                user_id=current_user.id,
                key=key,
                value=value
            )
            db.add(new_pref)
    
    db.commit()
    
    return {"status": "preferences updated"}

@router.post("/user/activity")
async def log_user_activity(
    activity: Dict[str, str],
    current_user: User = Depends(get_current_active_user)
):
    """记录用户活动"""
    # 实际应用中应该将活动记录到数据库
    # 这里简化为返回成功
    return {"status": "activity logged"} 