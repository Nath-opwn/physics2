#!/usr/bin/env python
"""
流体力学仿真系统服务启动脚本
"""
import os
import sys
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description='启动流体力学仿真系统服务')
    parser.add_argument('--host', default='0.0.0.0', help='绑定的主机地址')
    parser.add_argument('--port', type=int, default=8000, help='监听的端口')
    parser.add_argument('--reload', action='store_true', help='启用自动重载（开发模式）')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数量')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], help='日志级别')
    
    args = parser.parse_args()
    
    print(f"正在启动流体力学仿真系统服务，监听 {args.host}:{args.port}...")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main() 