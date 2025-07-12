# 多相流模型

本文档介绍了流体动力学模拟系统中的多相流模型实现，包括体积流体法(VOF)和水平集方法(Level Set Method)。

## 概述

多相流模型用于模拟两种或多种不同流体之间的相互作用，如气液界面、液液界面等。我们实现了两种主流的多相流模拟方法：

1. **体积流体法(VOF)**：使用体积分数来追踪不同相的分布，适合于质量守恒要求高的场景
2. **水平集方法**：使用有符号距离函数来表示界面，适合于需要精确界面位置和曲率的场景

## 模型架构

多相流模型采用以下层次结构：

```
MultiphaseModel (基类)
├── VOFModel
└── LevelSetModel
```

### 基类功能

`MultiphaseModel` 基类提供以下核心功能：

- 初始化相场
- 设置相的物理属性（密度、粘度）
- 计算密度场和粘度场
- 定义模拟步进接口

## 体积流体法(VOF)

VOF方法通过追踪每个网格单元中各相的体积分数来模拟多相流。

### 主要特点

- 天然保持质量守恒
- 能够处理复杂拓扑变化（如液滴分裂、合并）
- 界面重构相对简单

### 核心算法

VOF模型的主要计算步骤包括：

1. **初始化**：设置初始体积分数分布
2. **平流**：使用半拉格朗日方法计算体积分数的平流
3. **归一化**：确保每个单元的体积分数和为1
4. **边界条件**：应用零梯度边界条件

### 使用示例

```python
# 创建VOF模型实例
vof = VOFModel(width, height, depth, num_phases=2)

# 设置相的物理属性
vof.set_phase_properties(0, density=1000.0, viscosity=1.0)  # 水
vof.set_phase_properties(1, density=1.0, viscosity=0.01)    # 空气

# 初始化相分布（球形液滴）
def sphere_region(x, y, z):
    return ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < radius**2

vof.initialize_phase(0, sphere_region)

# 执行模拟步骤
vof.step(dt, velocity_field)

# 获取结果
phase_field = vof.get_phase_field(0)
density_field = vof.get_density_field()
```

## 水平集方法

水平集方法使用有符号距离函数来隐式表示界面，界面被定义为距离函数的零等值面。

### 主要特点

- 精确的界面表示
- 便于计算界面法向和曲率
- 能够自然处理拓扑变化
- 需要重初始化以保持有符号距离特性

### 核心算法

水平集方法的主要计算步骤包括：

1. **初始化**：创建初始有符号距离函数
2. **平流**：使用半拉格朗日方法计算水平集函数的平流
3. **重初始化**：恢复水平集函数的有符号距离特性
4. **相场更新**：根据水平集函数更新相场

### 使用示例

```python
# 创建水平集模型实例
ls = LevelSetModel(width, height, depth, num_phases=2)

# 设置相的物理属性
ls.set_phase_properties(0, density=1000.0, viscosity=1.0)  # 水
ls.set_phase_properties(1, density=1.0, viscosity=0.01)    # 空气

# 初始化相分布
def sphere_region(x, y, z):
    return ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < radius**2

ls.initialize_phase(0, sphere_region)

# 执行模拟步骤
ls.step(dt, velocity_field)

# 获取结果
phase_field = ls.get_phase_field(0)
interface_field = ls.get_interface_field()
```

## 与其他物理模型的耦合

多相流模型可以与其他物理模型结合使用，例如：

### 与表面张力模型结合

```python
# 创建表面张力模型
surface_tension = SurfaceTensionModel(width, height, depth)
surface_tension.set_surface_tension_coefficient(0.07)

# 在模拟循环中
surface_tension.update_interface_properties(ls.phi)
surface_tension_force = surface_tension.compute_surface_tension_force()

# 更新速度场
velocity_field += dt * surface_tension_force / density_field
```

### 与热传导模型结合

```python
# 创建热传导模型
thermal = ThermalModel(width, height, depth)
thermal.set_thermal_diffusivity(0.1)
thermal.set_specific_heat(4200)

# 在模拟循环中
density_field = multiphase.get_density_field()
thermal.step(dt, velocity_field, density_field)
```

## 性能比较

VOF方法和水平集方法各有优缺点：

| 方法 | 优点 | 缺点 |
|------|------|------|
| VOF | 质量守恒性好<br>计算效率较高 | 界面重构复杂<br>曲率计算不精确 |
| 水平集 | 界面表示精确<br>曲率计算方便 | 质量不严格守恒<br>需要重初始化 |

## 未来改进方向

1. **混合方法**：实现结合VOF和水平集优点的混合方法(CLSVOF)
2. **自适应网格**：在界面附近使用更细的网格提高精度
3. **并行计算**：使用CUDA加速多相流计算
4. **多相流-固体耦合**：实现流体与固体的相互作用 