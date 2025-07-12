#!/bin/bash
# 流体力学仿真系统启动脚本

# 检查是否安装了Docker和Docker Compose
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "错误: 需要安装Docker和Docker Compose"
    exit 1
fi

# 检查.env文件是否存在
if [ ! -f ".env" ]; then
    echo "警告: 没有找到.env文件, 复制.env.example..."
    if [ -f "config/.env.example" ]; then
        cp config/.env.example .env
    else
        echo "错误: 找不到config/.env.example文件"
        exit 1
    fi
fi

# 启动服务
echo "正在启动流体力学仿真系统..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 检查API服务是否正常运行
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "系统已成功启动!"
    echo "访问 http://localhost:8000/docs 查看API文档"
else
    echo "警告: 系统可能未正确启动，请检查docker-compose日志"
    docker-compose logs api
fi 