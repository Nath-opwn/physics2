#!/usr/bin/env python3
"""
添加湍流分析相关知识到知识库
"""
import os
import sys
import sqlite3
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.database import SessionLocal
from src.models.models import KnowledgeItem

def add_turbulence_knowledge():
    """添加湍流分析相关知识"""
    print("添加湍流分析相关知识...")
    
    # 创建数据库会话
    db = SessionLocal()
    
    # 湍流基础知识
    turbulence_basics = KnowledgeItem(
        title="湍流流动基础",
        category="advanced",
        content="""
# 湍流流动基础

湍流是流体力学中最复杂的现象之一，表现为流体运动的不规则、混沌和高度非线性特征。

## 湍流的基本特征

1. **不规则性**：湍流流动具有高度不规则和随机的特性，难以精确预测
2. **扩散性**：湍流极大地增强了动量、热量和质量的传递
3. **旋涡性**：包含多尺度的旋涡结构，从大尺度到小尺度
4. **耗散性**：湍流通过能量级联过程将动能转化为热能
5. **三维性**：真正的湍流总是三维的，即使平均流动是二维的

## 湍流的数学描述

湍流流动通常使用统计方法描述。雷诺分解将流动变量分解为平均部分和脉动部分：

$$u_i = \overline{u_i} + u_i'$$

其中，$\overline{u_i}$是平均速度，$u_i'$是速度脉动。

## 湍流强度

湍流强度是衡量湍流水平的重要参数，定义为速度脉动的均方根与平均速度的比值：

$$I = \\frac{\\sqrt{\\frac{1}{3}(u'^2 + v'^2 + w'^2)}}{U}$$

其中，$U$是平均速度大小。

## 湍流尺度

湍流包含多种尺度的结构：

- **积分尺度**：包含大部分湍流能量的大尺度结构
- **泰勒微尺度**：介于大尺度和小尺度之间的中间尺度
- **科尔莫哥洛夫尺度**：最小的湍流尺度，在此尺度上能量通过粘性耗散

## 湍流能量级联

湍流能量从大尺度结构传递到小尺度结构的过程称为能量级联。这一过程可以通过能量谱来描述，能量谱显示了不同波数（或频率）下湍流能量的分布。
        """,
        tags="湍流,流体力学,湍流强度,湍流尺度,能量级联",
        created_at=datetime.now()
    )
    
    # 湍流模型知识
    turbulence_models = KnowledgeItem(
        title="湍流模型概述",
        category="advanced",
        content="""
# 湍流模型概述

湍流模型是用于近似模拟湍流流动的数学模型，广泛应用于计算流体力学（CFD）中。

## 湍流模型分类

### 1. 雷诺平均Navier-Stokes方程（RANS）模型

RANS模型是最广泛使用的湍流模型，基于对Navier-Stokes方程进行时间平均。

#### 零方程模型
- **代数模型**：如混合长度模型
- **Baldwin-Lomax模型**：适用于边界层流动

#### 一方程模型
- **Spalart-Allmaras模型**：求解湍流粘性的输运方程

#### 两方程模型
- **k-ε模型**：求解湍流动能k和耗散率ε
- **k-ω模型**：求解湍流动能k和比耗散率ω
- **SST模型**：结合k-ε和k-ω模型的优点

### 2. 大涡模拟（LES）

LES模型直接模拟大尺度涡结构，而对小尺度涡采用亚格子模型。常用的亚格子模型包括：

- **Smagorinsky模型**
- **动态Smagorinsky模型**
- **WALE模型**

### 3. 直接数值模拟（DNS）

DNS直接求解Navier-Stokes方程，不使用任何湍流模型，但计算成本极高。

## 湍流模型选择

选择合适的湍流模型需要考虑以下因素：

1. **流动类型**：自由剪切流、壁面流、分离流等
2. **计算资源**：DNS > LES > RANS
3. **精度要求**：DNS > LES > RANS
4. **物理现象**：热传递、化学反应等

## 湍流模型的局限性

所有湍流模型都有其适用范围和局限性：

- **RANS模型**：难以准确预测强分离流动和非定常效应
- **LES模型**：壁面处理困难，计算成本较高
- **DNS模型**：仅适用于低雷诺数和简单几何形状
        """,
        tags="湍流模型,RANS,LES,DNS,CFD",
        created_at=datetime.now()
    )
    
    # 雷诺应力知识
    reynolds_stress = KnowledgeItem(
        title="雷诺应力与湍流能量",
        category="advanced",
        content="""
# 雷诺应力与湍流能量

## 雷诺应力张量

雷诺应力是湍流流动中由于速度脉动引起的额外应力项，在雷诺平均Navier-Stokes方程中表现为：

$$\\tau_{ij} = -\\rho \\overline{u_i' u_j'}$$

其中，$\\rho$是流体密度，$u_i'$和$u_j'$是速度脉动分量。

雷诺应力张量是一个对称张量，包含六个独立分量：

$$\\tau = 
\\begin{pmatrix} 
\\tau_{xx} & \\tau_{xy} & \\tau_{xz} \\\\ 
\\tau_{xy} & \\tau_{yy} & \\tau_{yz} \\\\ 
\\tau_{xz} & \\tau_{yz} & \\tau_{zz} 
\\end{pmatrix}
= -\\rho
\\begin{pmatrix} 
\\overline{u'^2} & \\overline{u'v'} & \\overline{u'w'} \\\\ 
\\overline{u'v'} & \\overline{v'^2} & \\overline{v'w'} \\\\ 
\\overline{u'w'} & \\overline{v'w'} & \\overline{w'^2} 
\\end{pmatrix}$$

## 湍流动能

湍流动能（TKE）定义为单位质量流体的速度脉动动能：

$$k = \\frac{1}{2}(\\overline{u'^2} + \\overline{v'^2} + \\overline{w'^2})$$

湍流动能是雷诺应力张量对角线元素的一半之和。

## 湍流动能方程

湍流动能的输运方程描述了湍流动能的产生、输运和耗散过程：

$$\\frac{\\partial k}{\\partial t} + U_j \\frac{\\partial k}{\\partial x_j} = P_k - \\varepsilon + \\frac{\\partial}{\\partial x_j}\\left[\\nu \\frac{\\partial k}{\\partial x_j} - \\overline{u_j' \\left(\\frac{p'}{\\rho} + \\frac{u_i' u_i'}{2}\\right)}\\right]$$

其中：
- $P_k = -\\overline{u_i' u_j'} \\frac{\\partial U_i}{\\partial x_j}$ 是湍流产生项
- $\\varepsilon = \\nu \\overline{\\frac{\\partial u_i'}{\\partial x_j} \\frac{\\partial u_i'}{\\partial x_j}}$ 是湍流耗散率

## 湍流能量谱

湍流能量谱描述了湍流动能在不同波数（或频率）下的分布。典型的湍流能量谱包含三个区域：

1. **能量包含区**：大尺度结构，能量输入区域
2. **惯性子区**：能量级联区域，遵循 $E(k) \\propto k^{-5/3}$ 的关系
3. **耗散区**：小尺度结构，能量通过粘性耗散

能量谱分析是研究湍流特性的重要工具，可以揭示湍流的尺度分布和能量传递机制。
        """,
        tags="雷诺应力,湍流动能,能量谱,湍流耗散",
        created_at=datetime.now()
    )
    
    # 涡结构识别知识
    vortex_identification = KnowledgeItem(
        title="涡结构识别方法",
        category="advanced",
        content="""
# 涡结构识别方法

涡结构是湍流流动中的基本组成部分，识别和分析涡结构对于理解湍流动力学至关重要。

## 涡的定义

涡的严格定义在流体力学中仍有争议，但通常认为涡是流体旋转运动的区域。常用的涡结构识别方法包括：

## 1. 涡量准则

涡量是速度场的旋度，定义为：

$$\\vec{\\omega} = \\nabla \\times \\vec{u}$$

涡量大小可以用于识别涡核区域，但在剪切流动中可能会产生误判。

## 2. Q准则

Q准则基于速度梯度张量的第二不变量，定义为：

$$Q = \\frac{1}{2}(\\|\\Omega\\|^2 - \\|S\\|^2)$$

其中，$\\Omega$是速度梯度张量的反对称部分（旋转张量），$S$是对称部分（应变率张量）。

当Q > 0时，旋转占主导，表示涡核区域。

## 3. λ₂准则

λ₂准则基于速度梯度张量的特征值，定义为：

$$\\lambda_2(S^2 + \\Omega^2) < 0$$

其中，λ₂是对称张量$S^2 + \\Omega^2$的第二小特征值。

## 4. Δ准则

Δ准则使用速度梯度张量的特征值来识别涡结构：

$$\\Delta = (Q/3)^3 + (R/2)^2 > 0$$

其中，$R = -\\det(\\nabla u)$是速度梯度张量的行列式。

## 5. 螺旋度准则

螺旋度定义为速度和涡量的点积：

$$H = \\vec{u} \\cdot \\vec{\\omega}$$

高螺旋度区域通常对应于强涡结构。

## 涡结构可视化

识别出的涡结构通常通过以下方式可视化：

1. **等值面**：绘制Q、λ₂等物理量的等值面
2. **流线**：沿涡核绘制流线
3. **粒子追踪**：释放示踪粒子观察其运动轨迹
4. **切片**：在特定平面上显示涡量或其他标量场

## 涡结构分析

涡结构分析通常关注以下特性：

1. **涡核位置**：涡结构的中心位置
2. **涡强度**：通常用涡量大小或Q值表示
3. **涡尺寸**：涡结构的特征尺寸
4. **涡轴方向**：涡结构的主要旋转轴方向
        """,
        tags="涡结构,Q准则,lambda2准则,涡量,涡识别",
        created_at=datetime.now()
    )
    
    # 添加到数据库
    db.add(turbulence_basics)
    db.add(turbulence_models)
    db.add(reynolds_stress)
    db.add(vortex_identification)
    
    # 提交更改
    db.commit()
    db.close()
    
    print("湍流分析相关知识添加完成！")

if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs("data", exist_ok=True)
    
    # 添加湍流分析知识
    add_turbulence_knowledge() 