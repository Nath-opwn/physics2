import sys
import traceback
import functools
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Type

from src.utils.logger import get_logger

# 创建错误处理专用日志记录器
logger = get_logger("error_handler")

# 错误类型分类
SIMULATION_ERRORS = {
    "BOUNDARY_ERROR": "边界条件错误",
    "NUMERICAL_INSTABILITY": "数值不稳定",
    "DIVERGENCE_ERROR": "散度约束错误",
    "MEMORY_ERROR": "内存不足",
    "CUDA_ERROR": "CUDA错误",
    "CPP_EXT_ERROR": "C++扩展错误",
    "IO_ERROR": "输入输出错误",
    "CONFIG_ERROR": "配置错误"
}

# 错误记录
error_history: List[Dict[str, Any]] = []
error_stats: Dict[str, int] = {}

def record_error(error_type: str, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    记录错误信息
    
    参数:
        error_type: 错误类型
        message: 错误消息
        exception: 异常对象
        context: 错误上下文信息
        
    返回:
        错误记录字典
    """
    error_record = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "message": message,
        "traceback": traceback.format_exc() if exception else None,
        "context": context or {}
    }
    
    # 记录到历史
    error_history.append(error_record)
    if len(error_history) > 1000:  # 限制历史记录数量
        error_history.pop(0)
    
    # 更新统计
    error_stats[error_type] = error_stats.get(error_type, 0) + 1
    
    # 记录到日志
    if exception:
        logger.exception(f"{error_type}: {message}", exc_info=exception)
    else:
        logger.error(f"{error_type}: {message}")
    
    return error_record

def get_error_history(limit: int = 100, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取错误历史记录"""
    if error_type:
        filtered = [e for e in error_history if e["error_type"] == error_type]
        return filtered[-limit:]
    return error_history[-limit:]

def get_error_stats() -> Dict[str, int]:
    """获取错误统计信息"""
    return error_stats

def clear_error_history() -> None:
    """清除错误历史记录"""
    error_history.clear()
    error_stats.clear()

def handle_simulation_error(error_type: str, context: Optional[Dict[str, Any]] = None) -> Callable:
    """
    装饰器: 处理模拟过程中的错误
    
    参数:
        error_type: 错误类型
        context: 错误上下文信息
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_ctx = context or {}
                # 添加函数参数信息到上下文
                error_ctx.update({
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                record_error(error_type, str(e), e, error_ctx)
                # 根据错误类型决定是否重新抛出
                if error_type in ["MEMORY_ERROR", "CUDA_ERROR", "CPP_EXT_ERROR"]:
                    raise
                # 返回默认值或错误状态
                return None
        return wrapper
    return decorator

def try_with_fallback(fallback_value: Any, error_type: str = "GENERAL_ERROR", 
                      max_retries: int = 0, retry_delay: float = 1.0) -> Callable:
    """
    装饰器: 尝试执行函数，失败时返回回退值
    
    参数:
        fallback_value: 失败时返回的值
        error_type: 错误类型
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    record_error(error_type, str(e), e, {
                        "function": func.__name__,
                        "retries": retries
                    })
                    
                    if retries >= max_retries:
                        logger.warning(f"函数 {func.__name__} 执行失败，返回回退值")
                        return fallback_value
                    
                    retries += 1
                    logger.info(f"重试 {func.__name__} ({retries}/{max_retries})...")
                    time.sleep(retry_delay)
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, error_type: str = "EXECUTION_ERROR", 
                 fallback_value: Any = None, **kwargs) -> Any:
    """
    安全执行函数
    
    参数:
        func: 要执行的函数
        *args: 函数参数
        error_type: 错误类型
        fallback_value: 失败时返回的值
        **kwargs: 函数关键字参数
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        record_error(error_type, str(e), e, {
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        })
        return fallback_value

# 注册全局异常处理器
def setup_global_exception_handler():
    """设置全局异常处理器"""
    def global_exception_handler(exctype, value, tb):
        record_error("UNHANDLED_EXCEPTION", str(value), value)
        # 调用原始处理器
        sys.__excepthook__(exctype, value, tb)
    
    # 替换全局异常处理器
    sys.excepthook = global_exception_handler 