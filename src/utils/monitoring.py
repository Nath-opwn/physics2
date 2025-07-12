import os
import time
import threading
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from src.utils.logger import get_logger

# 创建监控专用日志记录器
logger = get_logger("monitoring")

class SystemMonitor:
    """系统监控类，用于收集系统性能指标"""
    
    def __init__(self, metrics_dir=None, interval=5.0):
        """
        初始化系统监控
        
        参数:
            metrics_dir: 指标存储目录
            interval: 收集间隔(秒)
        """
        # 设置指标存储目录
        if metrics_dir is None:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.metrics_dir = base_dir / "metrics"
        else:
            self.metrics_dir = Path(metrics_dir)
        
        # 确保目录存在
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        
        # 性能指标
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000  # 最大历史记录数量
        
        # 注册的回调函数
        self.callbacks: List[Callable] = []
        
        # 当前进程
        self.process = psutil.Process()
    
    def start(self):
        """启动监控"""
        if self.running:
            logger.warning("监控已在运行")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"系统监控已启动，间隔 {self.interval} 秒")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = self.collect_metrics()
                self._store_metrics(metrics)
                
                # 调用回调函数
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"监控回调函数执行失败: {str(e)}")
            
            except Exception as e:
                logger.error(f"收集系统指标失败: {str(e)}")
            
            time.sleep(self.interval)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """收集系统性能指标"""
        timestamp = datetime.now().isoformat()
        
        # 系统CPU和内存
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # 当前进程资源使用
        process_cpu = self.process.cpu_percent(interval=0.1) / psutil.cpu_count()
        process_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        
        # 网络IO
        net_io = psutil.net_io_counters()
        
        metrics = {
            "timestamp": timestamp,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "net_sent_mb": net_io.bytes_sent / (1024 * 1024),
                "net_recv_mb": net_io.bytes_recv / (1024 * 1024)
            },
            "process": {
                "cpu_percent": process_cpu,
                "memory_mb": process_memory,
                "threads": self.process.num_threads(),
                "open_files": len(self.process.open_files()),
                "connections": len(self.process.connections())
            }
        }
        
        return metrics
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """存储性能指标"""
        # 添加到历史记录
        self.metrics_history.append(metrics)
        
        # 限制历史记录大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        # 每小时保存一次指标到文件
        current_hour = datetime.now().strftime("%Y%m%d_%H")
        if len(self.metrics_history) % 60 == 0:  # 每60个采样点保存一次
            self._save_metrics_to_file(current_hour)
    
    def _save_metrics_to_file(self, timestamp_prefix: str):
        """将指标保存到文件"""
        if not self.metrics_history:
            return
        
        filename = self.metrics_dir / f"metrics_{timestamp_prefix}.json"
        
        try:
            # 如果文件已存在，追加数据
            existing_data = []
            if filename.exists():
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            
            # 合并数据
            all_data = existing_data + self.metrics_history[-60:]
            
            # 写入文件
            with open(filename, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            logger.debug(f"性能指标已保存到 {filename}")
        
        except Exception as e:
            logger.error(f"保存性能指标失败: {str(e)}")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """注册指标回调函数"""
        self.callbacks.append(callback)
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """获取最新的性能指标"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取历史性能指标"""
        return self.metrics_history[-limit:]
    
    def get_average_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """获取过去几分钟的平均性能指标"""
        # 计算需要的样本数
        samples_needed = int(minutes * 60 / self.interval)
        recent_metrics = self.metrics_history[-samples_needed:] if samples_needed <= len(self.metrics_history) else self.metrics_history
        
        if not recent_metrics:
            return {}
        
        # 计算平均值
        avg_metrics = {
            "system": {
                "cpu_percent": sum(m["system"]["cpu_percent"] for m in recent_metrics) / len(recent_metrics),
                "memory_percent": sum(m["system"]["memory_percent"] for m in recent_metrics) / len(recent_metrics),
                "disk_percent": sum(m["system"]["disk_percent"] for m in recent_metrics) / len(recent_metrics)
            },
            "process": {
                "cpu_percent": sum(m["process"]["cpu_percent"] for m in recent_metrics) / len(recent_metrics),
                "memory_mb": sum(m["process"]["memory_mb"] for m in recent_metrics) / len(recent_metrics)
            }
        }
        
        return avg_metrics

# 创建全局监控实例
system_monitor = SystemMonitor()

# 导出函数
start_monitoring = system_monitor.start
stop_monitoring = system_monitor.stop
get_latest_metrics = system_monitor.get_latest_metrics
get_metrics_history = system_monitor.get_metrics_history
get_average_metrics = system_monitor.get_average_metrics
register_callback = system_monitor.register_callback 