from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from src.api import auth, simulation, knowledge, tutorial
from src.database.database import engine, Base

# 创建FastAPI应用
app = FastAPI(
    title="流体动力学模拟系统API",
    description="多相流模型的API接口",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router)
app.include_router(simulation.router)
app.include_router(knowledge.router)
app.include_router(tutorial.router)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "欢迎使用流体动力学模拟系统API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # 获取端口，默认为8080
    port = int(os.environ.get("PORT", 8080))
    
    # 启动服务器
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)










