"""
实验模块包，包含所有可用的流体力学实验实现
"""
from typing import Dict, Any, Callable, Optional, Tuple, List

from app.experiments.karman_vortex import calculate_karman_vortex

# 实验计算函数映射
EXPERIMENT_CALCULATORS = {
    "karman_vortex": calculate_karman_vortex,
    # 其他实验将在此添加
}

def get_experiment_calculator(experiment_type: str):
    """
    根据实验类型获取对应的计算函数
    
    参数:
        experiment_type: 实验类型标识
        
    返回:
        对应的计算函数
        
    异常:
        KeyError: 如果实验类型不存在
    """
    if experiment_type not in EXPERIMENT_CALCULATORS:
        raise KeyError(f"未知实验类型: {experiment_type}")
    
    return EXPERIMENT_CALCULATORS[experiment_type] 