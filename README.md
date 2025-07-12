# 流体力学仿真系统

高性能流体力学模拟系统的后端API，处理前端提交的模拟请求，执行计算密集型任务，并提供结果数据。系统支持多种流体力学实验，包括静水压强、雷诺实验、伯努利方程验证、卡门涡街等。

## 技术栈

- **编程语言**: Python 3.10+
- **Web框架**: FastAPI
- **计算库**: NumPy, SciPy, PyTorch
- **异步任务队列**: Celery 配合 Redis
- **数据存储**: PostgreSQL (实验元数据), MinIO (大型结果数据)
- **容器化**: Docker 和 Docker Compose

## 系统架构

系统由以下几个主要部分组成：

1. **API服务**: 处理前端请求，提供RESTful API
2. **任务队列**: 管理长时间运行的计算任务
3. **计算引擎**: 执行流体动力学计算
4. **存储服务**: 存储计算结果和可视化数据

## 快速开始

### 环境要求

- Docker 和 Docker Compose
- (可选) CUDA支持的GPU (用于加速计算)

### 安装与运行

1. 克隆项目

```bash
git clone https://github.com/yourusername/fluid-dynamics-simulation.git
cd fluid-dynamics-simulation
```

2. 创建环境变量文件

```bash
cp config/.env.example .env
# 根据需要编辑.env文件
```

3. 启动服务

```bash
docker-compose up -d
```

4. 访问API文档

打开浏览器访问 http://localhost:8000/docs 查看API文档

### 开发环境设置

1. 创建Python虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行开发服务器

```bash
uvicorn app.main:app --reload
```

## API概览

系统提供完整的RESTful API，详细文档可在以下位置查看：

- [API文档](API_DOCUMENTATION.md) - 详细的API参考文档
- Swagger UI: http://localhost:8000/docs (系统运行时)
- ReDoc: http://localhost:8000/redoc (系统运行时)

主要API端点包括：

### 实验管理

- `GET /api/lab/experiments`: 获取可用实验列表
- `GET /api/lab/experiments/{experiment_type}/configuration`: 获取特定实验配置

### 会话管理

- `POST /api/lab/sessions`: 创建新实验会话
- `GET /api/lab/sessions/user`: 获取用户会话列表
- `GET /api/lab/sessions/{session_id}`: 获取会话详情
- `PUT /api/lab/sessions/{session_id}/parameters`: 更新会话参数

### 实验执行

- `POST /api/lab/sessions/{session_id}/run`: 启动实验计算
- `GET /api/lab/sessions/{session_id}/status`: 获取计算状态
- `POST /api/lab/sessions/{session_id}/cancel`: 取消计算

### 结果获取

- `GET /api/lab/sessions/{session_id}/results`: 获取实验结果
- `GET /api/lab/sessions/{session_id}/visualization/{data_type}`: 获取可视化数据

## 支持的实验

1. **静水压强实验**: 研究液体深度与压强关系的实验
2. **雷诺实验**: 观察不同雷诺数下流体流动状态变化的实验
3. **伯努利方程验证**: 验证不同截面积管道中流体压强与速度关系的实验
4. **卡门涡街**: 研究流体绕圆柱流动时形成涡街的实验

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[MIT License](LICENSE) 