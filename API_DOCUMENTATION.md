# 流体力学仿真系统API文档

本文档详细描述了流体力学仿真系统提供的API端点。

## API访问

系统启动后，可以通过访问以下URL查看完整的交互式API文档：
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 认证

目前API不需要认证，但在生产环境中应该实现适当的认证机制。

## 基础URL

所有API端点都基于基础URL: `/api/lab`

## 端点概览

### 实验管理API

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/experiments` | 获取可用实验列表 |
| GET | `/experiments/{experiment_type}/configuration` | 获取特定实验类型的配置 |

### 会话管理API

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/sessions` | 创建新的实验会话 |
| GET | `/sessions/user` | 获取用户的所有会话 |
| GET | `/sessions/{session_id}` | 获取特定会话的详情 |
| PUT | `/sessions/{session_id}/parameters` | 更新会话参数 |

### 实验运行API

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/sessions/{session_id}/run` | 启动实验计算 |
| GET | `/sessions/{session_id}/status` | 获取计算状态 |
| POST | `/sessions/{session_id}/cancel` | 取消正在运行的实验 |

### 结果获取API

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/sessions/{session_id}/results` | 获取实验结果 |
| GET | `/sessions/{session_id}/visualization/{data_type}` | 获取可视化数据 |

### 系统端点

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 系统健康检查 |
| GET | `/` | API根端点信息 |

## 详细API规范

### 实验管理API

#### 获取可用的实验列表

```
GET /api/lab/experiments
```

获取系统中所有可用的流体力学实验列表，包括实验类型、名称和描述。

**响应**:

```json
[
  {
    "id": "uuid-string",
    "type": "karman_vortex",
    "name": "卡门涡街实验",
    "description": "研究流体绕圆柱流动时形成涡街的实验",
    "category": "计算流体动力学",
    "parameters": [
      {
        "name": "cylinder_diameter",
        "type": "number",
        "default": 0.1,
        "range": [0.01, 0.5],
        "unit": "m",
        "description": "圆柱直径"
      },
      // ... 其他参数
    ],
    "created_at": "2023-07-01T10:00:00Z",
    "updated_at": "2023-07-01T10:00:00Z"
  },
  // ... 其他实验类型
]
```

#### 获取特定实验类型的配置

```
GET /api/lab/experiments/{experiment_type}/configuration
```

获取特定实验类型的配置参数列表，包括参数名称、类型、默认值、范围等。

**路径参数**:
- `experiment_type`: 实验类型标识符，如 `karman_vortex`

**响应**:

```json
{
  "type": "karman_vortex",
  "name": "卡门涡街实验",
  "description": "研究流体绕圆柱流动时形成涡街的实验",
  "category": "计算流体动力学",
  "parameters": [
    {
      "name": "cylinder_diameter",
      "type": "number",
      "default": 0.1,
      "range": [0.01, 0.5],
      "unit": "m",
      "description": "圆柱直径"
    },
    // ... 其他参数
  ]
}
```

### 会话管理API

#### 创建实验会话

```
POST /api/lab/sessions
```

创建新的实验会话，保存用户选择的参数。

**请求体**:

```json
{
  "experiment_type": "karman_vortex",
  "name": "测试涡街实验",
  "description": "测试不同雷诺数下的涡街形成",
  "parameters": {
    "cylinder_diameter": 0.1,
    "flow_velocity": 1.0,
    "fluid_density": 1.0,
    "fluid_viscosity": 0.00001,
    "domain_width": 2.0,
    "domain_height": 1.0,
    "simulation_time": 20.0,
    "time_step": 0.01,
    "mesh_resolution": "medium",
    "save_frequency": 10
  }
}
```

**响应**:

```json
{
  "id": "uuid-string",
  "experiment_type": "karman_vortex",
  "user_id": "user-uuid",
  "name": "测试涡街实验",
  "description": "测试不同雷诺数下的涡街形成",
  "parameters": {
    "cylinder_diameter": 0.1,
    "flow_velocity": 1.0,
    // ... 其他参数
  },
  "status": "pending",
  "progress": 0,
  "created_at": "2023-07-10T15:30:00Z",
  "updated_at": "2023-07-10T15:30:00Z"
}
```

#### 获取用户会话

```
GET /api/lab/sessions/user
```

获取当前用户创建的所有实验会话。

**响应**:

```json
[
  {
    "id": "uuid-string",
    "experiment_type": "karman_vortex",
    "user_id": "user-uuid",
    "name": "测试涡街实验",
    "description": "测试不同雷诺数下的涡街形成",
    "parameters": { /* ... */ },
    "status": "pending",
    "progress": 0,
    "created_at": "2023-07-10T15:30:00Z",
    "updated_at": "2023-07-10T15:30:00Z"
  },
  // ... 其他会话
]
```

#### 获取特定会话详情

```
GET /api/lab/sessions/{session_id}
```

获取特定实验会话的详情。

**路径参数**:
- `session_id`: 会话ID

**响应**:

```json
{
  "id": "uuid-string",
  "experiment_type": "karman_vortex",
  "user_id": "user-uuid",
  "name": "测试涡街实验",
  "description": "测试不同雷诺数下的涡街形成",
  "parameters": { /* ... */ },
  "status": "pending",
  "progress": 0,
  "created_at": "2023-07-10T15:30:00Z",
  "updated_at": "2023-07-10T15:30:00Z"
}
```

#### 更新会话参数

```
PUT /api/lab/sessions/{session_id}/parameters
```

更新实验会话的参数。

**路径参数**:
- `session_id`: 会话ID

**请求体**:

```json
{
  "name": "更新的实验名称",
  "description": "更新的描述",
  "parameters": {
    "cylinder_diameter": 0.2,
    "flow_velocity": 2.0
    // ... 其他参数
  }
}
```

**响应**:

```json
{
  "id": "uuid-string",
  "experiment_type": "karman_vortex",
  "user_id": "user-uuid",
  "name": "更新的实验名称",
  "description": "更新的描述",
  "parameters": { /* 更新后的参数 */ },
  "status": "pending",
  "progress": 0,
  "created_at": "2023-07-10T15:30:00Z",
  "updated_at": "2023-07-10T15:45:00Z"
}
```

### 实验运行API

#### 启动实验

```
POST /api/lab/sessions/{session_id}/run
```

启动实验计算，将任务提交到异步队列。

**路径参数**:
- `session_id`: 会话ID

**响应**:

```json
{
  "job_id": "celery-task-id",
  "session_id": "uuid-string",
  "status": "running"
}
```

#### 获取实验状态

```
GET /api/lab/sessions/{session_id}/status
```

获取实验计算状态，包括状态码(pending/running/completed/failed)和进度。

**路径参数**:
- `session_id`: 会话ID

**响应**:

```json
{
  "session_id": "uuid-string",
  "status": "running",
  "progress": 45,
  "error_message": null
}
```

#### 取消实验

```
POST /api/lab/sessions/{session_id}/cancel
```

取消正在运行的实验。

**路径参数**:
- `session_id`: 会话ID

**响应**:

```json
{
  "session_id": "uuid-string",
  "status": "cancelled",
  "progress": 0
}
```

### 结果获取API

#### 获取实验结果

```
GET /api/lab/sessions/{session_id}/results
```

获取实验的完整结果数据。

**路径参数**:
- `session_id`: 会话ID

**响应**:

```json
{
  "reynolds_number": 10000.0,
  "shedding_frequency": 0.16,
  "strouhal_number": 0.21,
  "avg_drag_coefficient": 1.2,
  "computation_time": 145.8,
  "mesh_size": {
    "nx": 200,
    "ny": 100
  },
  "time_steps": 2000,
  "parameters": {
    // 原始实验参数
  },
  "detailed_results": {
    // 详细结果数据，从MinIO存储中检索
  }
}
```

#### 获取可视化数据

```
GET /api/lab/sessions/{session_id}/visualization/{data_type}
```

获取特定类型的可视化数据(如velocity, pressure, streamlines等)。

**路径参数**:
- `session_id`: 会话ID
- `data_type`: 可视化数据类型，如 `velocity`, `pressure`, `vorticity`

**响应**:

可视化数据，格式根据数据类型不同而变化。可能包含图像URL、数据数组等。

```json
{
  "time": 10.0,
  "step": 1000,
  "data": { /* 可视化数据 */ },
  "file_references": {
    "image_url": "预签名URL，用于直接访问图像"
  }
}
```

### 系统端点

#### 健康检查

```
GET /health
```

系统健康检查端点。

**响应**:

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

#### 根端点

```
GET /
```

API根端点，提供API基本信息。

**响应**:

```json
{
  "name": "流体力学仿真系统 API",
  "version": "0.1.0",
  "documentation": "/docs",
  "status": "运行中"
}
```

## 错误处理

所有API端点在出错时会返回适当的HTTP状态码和错误信息：

**示例错误响应**:

```json
{
  "detail": "找不到会话 'invalid-uuid'"
}
```

常见的错误状态码：

- `400 Bad Request`: 请求参数无效
- `404 Not Found`: 请求的资源不存在
- `500 Internal Server Error`: 服务器内部错误

## 支持的实验类型

系统目前支持以下实验类型：

1. **静水压强实验** (`hydrostatic_pressure`): 研究液体深度与压强关系的实验
2. **雷诺实验** (`reynolds_experiment`): 观察不同雷诺数下流体流动状态变化的实验
3. **伯努利方程验证** (`bernoulli_equation`): 验证不同截面积管道中流体压强与速度关系的实验
4. **卡门涡街** (`karman_vortex`): 研究流体绕圆柱流动时形成涡街的实验 