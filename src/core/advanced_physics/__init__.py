"""高级物理模型

包含用于流体动力学模拟的高级物理模型。
"""

from .surface_tension import SurfaceTensionModel
from .thermal import ThermalModel
from .multiphase import MultiphaseModel, VOFModel, LevelSetModel

__all__ = [
    'SurfaceTensionModel',
    'ThermalModel',
    'MultiphaseModel',
    'VOFModel',
    'LevelSetModel',
] 