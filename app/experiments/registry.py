from typing import Dict, List, Any, Optional, Tuple
from app.models.schemas import ExperimentTypeBase, ParameterDefinition

# 实验注册表，用于存储可用的实验类型和配置
_EXPERIMENT_REGISTRY = {}

class ExperimentType(ExperimentTypeBase):
    """扩展的实验类型类，包含参数定义"""
    parameters: List[ParameterDefinition]

def register_experiment(
    type: str, 
    name: str, 
    description: Optional[str] = None, 
    category: Optional[str] = None,
    parameters: Optional[List[ParameterDefinition]] = None
) -> None:
    """
    注册一个新的实验类型
    
    参数:
        type: 实验类型标识
        name: 实验名称
        description: 实验描述
        category: 实验分类
        parameters: 实验参数定义
    """
    if parameters is None:
        parameters = []
    
    _EXPERIMENT_REGISTRY[type] = ExperimentType(
        type=type,
        name=name,
        description=description,
        category=category,
        parameters=parameters
    )

def get_experiment_types() -> List[ExperimentType]:
    """获取所有注册的实验类型"""
    return list(_EXPERIMENT_REGISTRY.values())

def get_experiment_config(experiment_type: str) -> Dict[str, Any]:
    """获取特定实验类型的配置"""
    if experiment_type not in _EXPERIMENT_REGISTRY:
        raise KeyError(f"实验类型 '{experiment_type}' 不存在")
    
    experiment = _EXPERIMENT_REGISTRY[experiment_type]
    return {
        "type": experiment.type,
        "name": experiment.name,
        "description": experiment.description,
        "category": experiment.category,
        "parameters": [param.model_dump() for param in experiment.parameters]
    }

# 注册默认实验
# 1. 静水压强
register_experiment(
    type="hydrostatic_pressure",
    name="静水压强实验",
    description="研究液体深度与压强关系的实验",
    category="基础流体静力学",
    parameters=[
        ParameterDefinition(
            name="fluid_density",
            type="number",
            default=1000.0,
            range=(800.0, 1200.0),
            unit="kg/m³",
            description="流体密度"
        ),
        ParameterDefinition(
            name="container_height",
            type="number",
            default=1.0,
            range=(0.1, 5.0),
            unit="m",
            description="容器高度"
        ),
        ParameterDefinition(
            name="gravity",
            type="number",
            default=9.8,
            range=(9.0, 10.0),
            unit="m/s²",
            description="重力加速度"
        ),
        ParameterDefinition(
            name="measurement_points",
            type="number",
            default=10,
            range=(5, 50),
            unit="points",
            description="测量点数量"
        )
    ]
)

# 2. 雷诺实验
register_experiment(
    type="reynolds_experiment",
    name="雷诺实验",
    description="观察不同雷诺数下流体流动状态变化的实验",
    category="流体动力学",
    parameters=[
        ParameterDefinition(
            name="fluid_viscosity",
            type="number",
            default=0.001,
            range=(0.0001, 0.01),
            unit="Pa·s",
            description="流体粘度"
        ),
        ParameterDefinition(
            name="pipe_diameter",
            type="number",
            default=0.02,
            range=(0.005, 0.1),
            unit="m",
            description="管道直径"
        ),
        ParameterDefinition(
            name="flow_velocity",
            type="number",
            default=0.5,
            range=(0.01, 5.0),
            unit="m/s",
            description="流体速度"
        ),
        ParameterDefinition(
            name="fluid_density",
            type="number",
            default=1000.0,
            range=(800.0, 1200.0),
            unit="kg/m³",
            description="流体密度"
        ),
        ParameterDefinition(
            name="simulation_time",
            type="number",
            default=10.0,
            range=(1.0, 60.0),
            unit="s",
            description="模拟时间"
        )
    ]
)

# 3. 伯努利方程验证
register_experiment(
    type="bernoulli_equation",
    name="伯努利方程验证",
    description="验证不同截面积管道中流体压强与速度关系的实验",
    category="流体动力学",
    parameters=[
        ParameterDefinition(
            name="inlet_diameter",
            type="number",
            default=0.05,
            range=(0.01, 0.2),
            unit="m",
            description="入口直径"
        ),
        ParameterDefinition(
            name="outlet_diameter",
            type="number",
            default=0.025,
            range=(0.005, 0.1),
            unit="m",
            description="出口直径"
        ),
        ParameterDefinition(
            name="flow_rate",
            type="number",
            default=0.001,
            range=(0.0001, 0.01),
            unit="m³/s",
            description="体积流量"
        ),
        ParameterDefinition(
            name="fluid_density",
            type="number",
            default=1000.0,
            range=(800.0, 1200.0),
            unit="kg/m³",
            description="流体密度"
        ),
        ParameterDefinition(
            name="height_difference",
            type="number",
            default=0.0,
            range=(0.0, 1.0),
            unit="m",
            description="高度差"
        )
    ]
)

# 4. 卡门涡街
register_experiment(
    type="karman_vortex",
    name="卡门涡街实验",
    description="研究流体绕圆柱流动时形成涡街的实验",
    category="计算流体动力学",
    parameters=[
        ParameterDefinition(
            name="cylinder_diameter",
            type="number",
            default=0.1,
            range=(0.01, 0.5),
            unit="m",
            description="圆柱直径"
        ),
        ParameterDefinition(
            name="flow_velocity",
            type="number",
            default=1.0,
            range=(0.1, 10.0),
            unit="m/s",
            description="来流速度"
        ),
        ParameterDefinition(
            name="fluid_density",
            type="number",
            default=1.0,
            range=(0.5, 2.0),
            unit="kg/m³",
            description="流体密度"
        ),
        ParameterDefinition(
            name="fluid_viscosity",
            type="number",
            default=0.00001,
            range=(0.000001, 0.001),
            unit="Pa·s",
            description="流体粘度"
        ),
        ParameterDefinition(
            name="domain_width",
            type="number",
            default=2.0,
            range=(1.0, 10.0),
            unit="m",
            description="计算域宽度"
        ),
        ParameterDefinition(
            name="domain_height",
            type="number",
            default=1.0,
            range=(0.5, 5.0),
            unit="m",
            description="计算域高度"
        ),
        ParameterDefinition(
            name="simulation_time",
            type="number",
            default=20.0,
            range=(5.0, 100.0),
            unit="s",
            description="模拟时间"
        ),
        ParameterDefinition(
            name="time_step",
            type="number",
            default=0.01,
            range=(0.001, 0.1),
            unit="s",
            description="时间步长"
        ),
        ParameterDefinition(
            name="mesh_resolution",
            type="select",
            default="medium",
            options=["very_coarse", "coarse", "medium", "fine", "very_fine"],
            description="网格分辨率"
        ),
        ParameterDefinition(
            name="save_frequency",
            type="number",
            default=10,
            range=(1, 100),
            unit="steps",
            description="结果保存频率"
        )
    ]
) 