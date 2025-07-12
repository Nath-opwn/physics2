import os
import sys
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

class FluidSimLogger:
    """流体模拟系统的日志记录器"""
    
    def __init__(self, name="fluidsim", level="info", log_dir=None):
        """
        初始化日志记录器
        
        参数:
            name: 日志记录器名称
            level: 日志级别 (debug, info, warning, error, critical)
            log_dir: 日志文件保存目录，如果为None则使用默认目录
        """
        self.name = name
        self.level = LOG_LEVELS.get(level.lower(), logging.INFO)
        
        # 设置日志目录
        if log_dir is None:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.log_dir = base_dir / "logs"
        else:
            self.log_dir = Path(log_dir)
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # 清除现有处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        log_file = self.log_dir / f"{name}_{time.strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(self.level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message, *args, **kwargs):
        """记录调试级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """记录信息级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """记录警告级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """记录错误级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """记录严重错误级别日志"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message, *args, exc_info=True, **kwargs):
        """记录异常信息"""
        self.logger.exception(message, *args, exc_info=exc_info, **kwargs)

# 创建默认日志记录器实例
default_logger = FluidSimLogger()

# 导出函数，方便直接调用
debug = default_logger.debug
info = default_logger.info
warning = default_logger.warning
error = default_logger.error
critical = default_logger.critical
exception = default_logger.exception

def get_logger(name=None, level=None, log_dir=None):
    """获取自定义日志记录器"""
    if name is None:
        return default_logger
    return FluidSimLogger(name=name, level=level or "info", log_dir=log_dir) 