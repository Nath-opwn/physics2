from sqlalchemy.orm import Session
import json
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from src.database.database import engine, SessionLocal, Base
from src.models.models import User, KnowledgeItem, Tutorial, TutorialStep, Experiment
from src.models.simulation_data import SimulationData, SurfaceTensionConfig, ContactAngleConfig, PerformanceMetric
from src.api.auth import get_password_hash

def create_database():
    """创建PostgreSQL数据库"""
    try:
        # 连接到默认的postgres数据库
        conn = psycopg2.connect(
            host="ps-0-postgresql.ns-dt9r90wb.svc",
            port="5432",
            user="postgres",
            password="t662ghf5"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # 检查数据库是否存在
        cursor.execute("SELECT 1 FROM pg_database WHERE datname='fluiddb'")
        exists = cursor.fetchone()
        
        if not exists:
            print("创建数据库 fluiddb...")
            cursor.execute("CREATE DATABASE fluiddb")
            print("数据库创建成功")
        else:
            print("数据库 fluiddb 已存在")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"创建数据库时出错: {e}")
        raise

def init_db():
    """初始化数据库"""
    # 创建数据库
    create_database()
    
    # 创建表
    print("创建数据库表...")
    Base.metadata.create_all(bind=engine)
    
    # 创建会话
    db = SessionLocal()
    
    # 检查是否已有用户
    user_count = db.query(User).count()
    if user_count == 0:
        print("创建初始用户...")
        # 创建管理员用户
        admin_user = User(
            username="admin",
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            is_active=True
        )
        db.add(admin_user)
        
        # 创建测试用户
        test_user = User(
            username="test",
            email="test@example.com",
            hashed_password=get_password_hash("test123"),
            is_active=True
        )
        db.add(test_user)
        
        db.commit()
        print("用户创建完成")
    
    # 检查是否已有知识项
    knowledge_count = db.query(KnowledgeItem).count()
    if knowledge_count == 0:
        print("创建初始知识库内容...")
        # 创建基础知识项
        knowledge_items = [
            {
                "title": "流体力学基本概念",
                "category": "fluid_basics",
                "content": """
                流体力学是研究流体（液体和气体）运动规律的科学。

                基本概念包括：
                - 密度：单位体积内的质量
                - 压力：单位面积上的力
                - 粘度：流体内部的摩擦力
                - 流速：流体质点的运动速度
                - 雷诺数：惯性力与粘性力的比值，用于判断流动状态

                流体可以分为理想流体和实际流体。理想流体无粘性、不可压缩，而实际流体具有粘性和可压缩性。
                """,
                "tags": "基础,概念,流体"
            },
            {
                "title": "Navier-Stokes方程",
                "category": "navier_stokes",
                "content": """
                Navier-Stokes方程是描述流体运动的基本方程，由法国物理学家Navier和英国物理学家Stokes分别独立导出。

                对于不可压缩流体，Navier-Stokes方程可表示为：

                ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F

                其中：
                - ρ是流体密度
                - u是速度矢量
                - p是压力
                - μ是动力粘度
                - F是外力

                这个方程表达了牛顿第二定律在流体中的应用，左侧表示单位体积流体的加速度，右侧表示作用在单位体积流体上的力。
                """,
                "tags": "方程,Navier-Stokes,数学"
            },
            {
                "title": "边界条件类型",
                "category": "boundary_conditions",
                "content": """
                在流体动力学模拟中，边界条件定义了流体与边界的相互作用方式。常见的边界条件包括：

                1. 固定边界（No-slip）：流体在边界处的速度为零，适用于流体与固体壁面的接触
                2. 周期性边界：出口处的流体状态等于入口处的流体状态，适用于模拟无限大的流场
                3. 开放边界：允许流体自由流入或流出，适用于模拟开放系统
                4. 对称边界：用于利用问题的对称性减少计算量

                正确设置边界条件对于获得准确的模拟结果至关重要。
                """,
                "tags": "边界条件,模拟,计算"
            },
            {
                "title": "表面张力现象",
                "category": "surface_tension",
                "content": """
                表面张力是液体表面表现出的类似于弹性薄膜的性质，它使液体表面积尽可能小。

                表面张力的产生是由于液体分子间的相互吸引力。在液体内部，分子受到四周分子的均匀拉力，合力为零；
                而在表面，分子只受到液体内部和侧面分子的拉力，导致表面分子有向内的拉力，形成表面张力。

                表面张力的大小与液体的性质、温度以及与其接触的物质有关。温度升高时，表面张力减小。

                在数值模拟中，常用的表面张力模型包括：
                1. 连续表面力模型(CSF)
                2. 锐界面表面张力模型(SSF)
                3. 相场法(PF)
                """,
                "tags": "表面张力,界面,模拟"
            },
            {
                "title": "接触角与润湿性",
                "category": "contact_angle",
                "content": """
                接触角是液体与固体表面接触时，液体表面与固体表面的夹角。它是表征固体表面润湿性的重要参数。

                接触角的大小取决于固体表面的性质和液体的性质：
                - 接触角 < 90°：液体润湿固体表面，如水在干净的玻璃上
                - 接触角 > 90°：液体不润湿固体表面，如水在疏水表面上
                - 接触角 = 0°：完全润湿
                - 接触角 = 180°：完全不润湿

                在数值模拟中，常用的接触角模型包括：
                1. 静态接触角模型
                2. 动态接触角模型
                3. 接触角滞后模型

                正确模拟接触角对于多相流中的流体-固体界面行为至关重要。
                """,
                "tags": "接触角,润湿性,界面"
            }
        ]
        
        for item_data in knowledge_items:
            item = KnowledgeItem(
                title=item_data["title"],
                category=item_data["category"],
                content=item_data["content"],
                tags=item_data["tags"]
            )
            db.add(item)
        
        db.commit()
        print("知识库内容创建完成")
    
    # 检查是否已有教程
    tutorial_count = db.query(Tutorial).count()
    if tutorial_count == 0:
        print("创建初始教程...")
        # 创建基础教程
        tutorial = Tutorial(
            title="流体模拟入门教程",
            description="学习如何使用流体动力学模拟系统进行基本模拟",
            difficulty="beginner"
        )
        db.add(tutorial)
        db.flush()  # 获取教程ID
        
        # 创建教程步骤
        steps = [
            {
                "step_number": 1,
                "title": "创建新模拟",
                "content": '点击"新建模拟"按钮，输入模拟名称和基本参数，如网格大小、流体粘度和密度。'
            },
            {
                "step_number": 2,
                "title": "设置边界条件",
                "content": '选择合适的边界条件类型，如固定边界、周期性边界或开放边界。'
            },
            {
                "step_number": 3,
                "title": "添加初始条件",
                "content": '设置流体的初始状态，如速度场和压力场的初始分布。'
            },
            {
                "step_number": 4,
                "title": "运行模拟",
                "content": '点击"开始"按钮运行模拟，观察流体的运动情况。'
            },
            {
                "step_number": 5,
                "title": "分析结果",
                "content": '使用数据探针和统计工具分析模拟结果，如速度分布、压力分布和涡量分布。'
            }
        ]
        
        for step_data in steps:
            step = TutorialStep(
                tutorial_id=tutorial.id,
                step_number=step_data["step_number"],
                title=step_data["title"],
                content=step_data["content"]
            )
            db.add(step)
        
        db.commit()
        print("教程创建完成")
    
    # 检查是否已有实验
    experiment_count = db.query(Experiment).count()
    if experiment_count == 0:
        print("创建初始实验...")
        # 创建示例实验
        experiments = [
            {
                "title": "卡门涡街实验",
                "description": "模拟流体绕过圆柱体产生的卡门涡街现象",
                "parameters": {
                    "width": 200,
                    "height": 100,
                    "depth": 50,
                    "viscosity": 0.01,
                    "density": 1.0,
                    "boundary_type": 1,
                    "cylinder_radius": 10,
                    "cylinder_position_x": 50,
                    "cylinder_position_y": 50,
                    "inlet_velocity": 1.0
                }
            },
            {
                "title": "隧道气流实验",
                "description": "模拟空气在隧道中的流动情况",
                "parameters": {
                    "width": 300,
                    "height": 80,
                    "depth": 80,
                    "viscosity": 0.001,
                    "density": 1.2,
                    "boundary_type": 2,
                    "inlet_velocity": 5.0,
                    "obstacle_positions": [
                        {"x": 100, "y": 40, "z": 40, "size": 10},
                        {"x": 200, "y": 30, "z": 50, "size": 15}
                    ]
                }
            },
            {
                "title": "表面张力实验",
                "description": "模拟液滴在不同表面上的表面张力和接触角效应",
                "parameters": {
                    "width": 100,
                    "height": 100,
                    "depth": 100,
                    "viscosity": 0.01,
                    "density": 1.0,
                    "boundary_type": 1,
                    "surface_tension": 0.072,
                    "contact_angle": 60,
                    "droplet_radius": 20,
                    "droplet_position": {"x": 50, "y": 50, "z": 20}
                }
            }
        ]
        
        for exp_data in experiments:
            exp = Experiment(
                title=exp_data["title"],
                description=exp_data["description"],
                parameters=exp_data["parameters"]
            )
            db.add(exp)
        
        db.commit()
        print("实验创建完成")
    
    # 关闭会话
    db.close()
    print("数据库初始化完成")

if __name__ == "__main__":
    init_db() 