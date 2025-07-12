# 流体动力学模拟系统 - 错误处理与日志系统

本文档介绍了流体动力学模拟系统中新增的错误处理和日志系统功能。

## 功能概述

新增的错误处理和日志系统提供以下功能：

1. **集中式日志记录**
   - 多级别日志（debug, info, warning, error, critical）
   - 日志自动轮转和保留
   - 控制台和文件双重输出

2. **错误处理与跟踪**
   - 错误类型分类与统计
   - 详细错误上下文记录
   - 错误历史查询
   - 装饰器支持的自动错误处理

3. **系统性能监控**
   - CPU、内存、磁盘使用率实时监控
   - 进程资源使用情况跟踪
   - 性能指标历史记录和统计
   - 可视化监控面板

4. **API集成**
   - RESTful API接口提供监控数据
   - 健康检查端点
   - 错误统计和查询接口

## 目录结构

```
project/
├── logs/                  # 日志文件目录
├── metrics/               # 性能指标存储目录
├── src/
│   ├── utils/
│   │   ├── logger.py      # 日志记录模块
│   │   ├── error_handler.py # 错误处理模块
│   │   └── monitoring.py  # 系统监控模块
│   ├── api/
│   │   └── monitoring.py  # 监控API接口
│   └── main.py            # 主应用程序（已集成监控功能）
└── static/
    ├── js/
    │   └── monitoring-dashboard.js # 前端监控面板脚本
    └── monitoring.html    # 监控面板页面
```

## 使用方法

### 日志记录

```python
from src.utils.logger import get_logger

# 创建日志记录器
logger = get_logger("my_module")

# 记录不同级别的日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
logger.exception("异常信息", exc_info=exception)
```

### 错误处理

```python
from src.utils.error_handler import record_error, handle_simulation_error, try_with_fallback

# 记录错误
record_error("ERROR_TYPE", "错误消息", exception, {"key": "value"})

# 使用装饰器自动处理错误
@handle_simulation_error("NUMERICAL_INSTABILITY")
def unstable_calculation():
    # 可能会出错的代码
    pass

# 使用回退值装饰器
@try_with_fallback(fallback_value=[], error_type="IO_ERROR", max_retries=3)
def read_data():
    # 可能会出错的IO操作
    pass
```

### 系统监控

```python
from src.utils.monitoring import start_monitoring, stop_monitoring, get_latest_metrics

# 启动监控
start_monitoring()

# 获取最新指标
metrics = get_latest_metrics()
print(f"CPU使用率: {metrics['system']['cpu_percent']}%")
print(f"内存使用率: {metrics['system']['memory_percent']}%")

# 停止监控
stop_monitoring()
```

## API端点

系统提供以下API端点用于监控和错误处理：

- `GET /api/monitoring/health` - 健康检查（无需认证）
- `GET /api/monitoring/metrics/current` - 获取当前系统指标
- `GET /api/monitoring/metrics/history` - 获取历史系统指标
- `GET /api/monitoring/metrics/average` - 获取平均系统指标
- `GET /api/monitoring/errors/recent` - 获取最近错误
- `GET /api/monitoring/errors/stats` - 获取错误统计
- `POST /api/monitoring/errors/clear` - 清除错误历史

## 前端监控面板

系统提供了一个基于Web的监控面板，可通过以下URL访问：

```
http://localhost:8000/static/monitoring.html
```

监控面板提供以下功能：

- 系统资源使用情况实时图表
- 进程资源使用情况实时图表
- 错误统计图表
- 最近错误列表和详情查看

## 配置选项

可以通过环境变量或配置文件调整以下设置：

- `LOG_LEVEL` - 日志级别（默认：info）
- `LOG_DIR` - 日志目录（默认：./logs）
- `METRICS_DIR` - 指标存储目录（默认：./metrics）
- `METRICS_INTERVAL` - 指标收集间隔（默认：5秒）
- `MAX_ERROR_HISTORY` - 最大错误历史记录数（默认：1000）

## 依赖

- Python 3.8+
- psutil>=5.9.0
- fastapi>=0.95.0
- Chart.js (前端)

## 注意事项

- 在生产环境中，建议将日志和指标存储在持久化存储中
- 监控数据可能会占用大量磁盘空间，请定期清理或配置适当的保留策略
- 系统监控会消耗一定的CPU和内存资源，可以通过调整收集间隔来平衡性能和监控精度 