from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
import numpy as np
import json
import os
import time
import io
import zipfile
import csv
from typing import Dict, List, Optional
import uuid

from src.database.database import get_db
from src.models.models import User, SimulationSession
from src.api.auth import get_current_active_user
from src.api.simulation import active_simulations

# 尝试导入VTK库，如果不可用则提供警告
try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("警告: VTK库不可用，VTK格式导出将被禁用")

router = APIRouter(
    prefix="/api/export",
    tags=["export"],
    responses={404: {"description": "未找到"}},
)

# 导出任务状态
export_tasks = {}

# 确保导出目录存在
os.makedirs("exports", exist_ok=True)

@router.get("/formats")
async def get_export_formats():
    """获取支持的导出格式"""
    formats = [
        {
            "id": "json",
            "name": "JSON",
            "description": "JavaScript对象表示法，适用于Web应用"
        },
        {
            "id": "csv",
            "name": "CSV",
            "description": "逗号分隔值，适用于电子表格软件"
        },
        {
            "id": "numpy",
            "name": "NumPy",
            "description": "NumPy二进制格式，适用于Python科学计算"
        },
        {
            "id": "hdf5",
            "name": "HDF5",
            "description": "分层数据格式，适用于大型科学数据集"
        },
        {
            "id": "zip",
            "name": "ZIP",
            "description": "压缩包格式，包含多种格式的数据文件"
        }
    ]
    
    # 如果VTK可用，添加VTK格式
    if VTK_AVAILABLE:
        formats.append({
            "id": "vtk",
            "name": "VTK",
            "description": "Visualization Toolkit格式，适用于科学可视化软件"
        })
    
    return formats

def export_data_task(
    session_id: str,
    format: str,
    include_velocity: bool,
    include_pressure: bool,
    include_vorticity: bool,
    task_id: str,
    time_steps: Optional[List[int]] = None
):
    """后台导出数据任务"""
    try:
        # 更新任务状态
        export_tasks[task_id]["status"] = "processing"
        
        # 获取模拟实例
        sim = active_simulations[session_id]
        solver = sim["solver"]
        
        # 准备导出数据
        export_data = {
            "metadata": {
                "session_id": session_id,
                "step": sim["step"],
                "parameters": sim["params"],
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "grid_dimensions": [solver.width, solver.height, solver.depth]
            }
        }
        
        # 根据请求添加数据
        if include_velocity:
            velocity = solver.get_velocity_field()
            export_data["velocity"] = velocity
        
        if include_pressure:
            pressure = solver.get_pressure_field()
            export_data["pressure"] = pressure
        
        if include_vorticity:
            vorticity = solver.get_vorticity_field()
            export_data["vorticity"] = vorticity
        
        # 创建文件名
        filename = f"exports/{session_id}_{sim['step']}_{task_id}"
        
        # 根据格式导出数据
        if format == "json":
            # 转换NumPy数组为列表
            for key in export_data:
                if key != "metadata" and isinstance(export_data[key], np.ndarray):
                    export_data[key] = export_data[key].tolist()
            
            # 导出为JSON
            with open(f"{filename}.json", "w") as f:
                json.dump(export_data, f)
            
            export_tasks[task_id]["file_path"] = f"{filename}.json"
            export_tasks[task_id]["file_name"] = f"{session_id}_{sim['step']}.json"
        
        elif format == "numpy":
            # 导出为NumPy格式
            np.savez(filename, **export_data)
            
            export_tasks[task_id]["file_path"] = f"{filename}.npz"
            export_tasks[task_id]["file_name"] = f"{session_id}_{sim['step']}.npz"
        
        elif format == "csv":
            # 创建CSV文件
            csv_files = []
            
            # 为每个场创建CSV
            if include_velocity:
                velocity = export_data["velocity"]
                v_file = f"{filename}_velocity.csv"
                
                with open(v_file, "w") as f:
                    f.write("z,y,x,vx,vy,vz\n")
                    for z in range(velocity.shape[0]):
                        for y in range(velocity.shape[1]):
                            for x in range(velocity.shape[2]):
                                vx, vy, vz = velocity[z, y, x]
                                f.write(f"{z},{y},{x},{vx},{vy},{vz}\n")
                
                csv_files.append(v_file)
            
            if include_pressure:
                pressure = export_data["pressure"]
                p_file = f"{filename}_pressure.csv"
                
                with open(p_file, "w") as f:
                    f.write("z,y,x,pressure\n")
                    for z in range(pressure.shape[0]):
                        for y in range(pressure.shape[1]):
                            for x in range(pressure.shape[2]):
                                p = pressure[z, y, x]
                                f.write(f"{z},{y},{x},{p}\n")
                
                csv_files.append(p_file)
            
            if include_vorticity:
                vorticity = export_data["vorticity"]
                v_file = f"{filename}_vorticity.csv"
                
                with open(v_file, "w") as f:
                    f.write("z,y,x,wx,wy,wz\n")
                    for z in range(vorticity.shape[0]):
                        for y in range(vorticity.shape[1]):
                            for x in range(vorticity.shape[2]):
                                wx, wy, wz = vorticity[z, y, x]
                                f.write(f"{z},{y},{x},{wx},{wy},{wz}\n")
                
                csv_files.append(v_file)
            
            # 创建元数据CSV
            meta_file = f"{filename}_metadata.csv"
            with open(meta_file, "w") as f:
                f.write("key,value\n")
                for key, value in export_data["metadata"].items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            f.write(f"{key}.{k},{v}\n")
                    else:
                        f.write(f"{key},{value}\n")
            
            csv_files.append(meta_file)
            
            # 创建ZIP文件包含所有CSV
            zip_path = f"{filename}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for csv_file in csv_files:
                    zipf.write(csv_file, os.path.basename(csv_file))
                    # 添加完成后删除临时CSV文件
                    os.remove(csv_file)
            
            export_tasks[task_id]["file_path"] = zip_path
            export_tasks[task_id]["file_name"] = f"{session_id}_{sim['step']}_csv.zip"
        
        elif format == "vtk" and VTK_AVAILABLE:
            # 导出为VTK格式
            vtk_files = []
            
            # 网格尺寸
            nx, ny, nz = solver.width, solver.height, solver.depth
            
            # 创建结构化网格
            grid = vtk.vtkStructuredGrid()
            grid.SetDimensions(nx, ny, nz)
            
            # 创建点
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(nx * ny * nz)
            
            # 设置网格点坐标
            point_id = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        points.SetPoint(point_id, i, j, k)
                        point_id += 1
            
            grid.SetPoints(points)
            
            # 添加速度场
            if include_velocity:
                velocity = export_data["velocity"]
                
                # 重塑数据以匹配VTK期望的格式
                v_flat = np.reshape(velocity, (nx * ny * nz, 3))
                
                # 创建VTK数组
                v_array = numpy_support.numpy_to_vtk(v_flat)
                v_array.SetName("velocity")
                
                # 添加到网格
                grid.GetPointData().AddArray(v_array)
            
            # 添加压力场
            if include_pressure:
                pressure = export_data["pressure"]
                
                # 重塑数据
                p_flat = pressure.flatten()
                
                # 创建VTK数组
                p_array = numpy_support.numpy_to_vtk(p_flat)
                p_array.SetName("pressure")
                
                # 添加到网格
                grid.GetPointData().AddArray(p_array)
            
            # 添加涡量场
            if include_vorticity:
                vorticity = export_data["vorticity"]
                
                # 重塑数据
                w_flat = np.reshape(vorticity, (nx * ny * nz, 3))
                
                # 创建VTK数组
                w_array = numpy_support.numpy_to_vtk(w_flat)
                w_array.SetName("vorticity")
                
                # 添加到网格
                grid.GetPointData().AddArray(w_array)
            
            # 写入VTK文件
            vtk_file = f"{filename}.vts"
            writer = vtk.vtkXMLStructuredGridWriter()
            writer.SetFileName(vtk_file)
            writer.SetInputData(grid)
            writer.Write()
            
            vtk_files.append(vtk_file)
            
            export_tasks[task_id]["file_path"] = vtk_file
            export_tasks[task_id]["file_name"] = f"{session_id}_{sim['step']}.vts"
        
        elif format == "hdf5":
            try:
                import h5py
                
                # 创建HDF5文件
                h5_file = f"{filename}.h5"
                with h5py.File(h5_file, 'w') as f:
                    # 添加元数据
                    meta_group = f.create_group('metadata')
                    for key, value in export_data["metadata"].items():
                        if isinstance(value, dict):
                            sub_group = meta_group.create_group(key)
                            for k, v in value.items():
                                if isinstance(v, (int, float, str)):
                                    sub_group.attrs[k] = v
                        else:
                            meta_group.attrs[key] = value
                    
                    # 添加场数据
                    if include_velocity:
                        f.create_dataset('velocity', data=export_data["velocity"])
                    
                    if include_pressure:
                        f.create_dataset('pressure', data=export_data["pressure"])
                    
                    if include_vorticity:
                        f.create_dataset('vorticity', data=export_data["vorticity"])
                
                export_tasks[task_id]["file_path"] = h5_file
                export_tasks[task_id]["file_name"] = f"{session_id}_{sim['step']}.h5"
            
            except ImportError:
                export_tasks[task_id]["status"] = "error"
                export_tasks[task_id]["error"] = "HDF5库不可用，无法导出HDF5格式"
                return
        
        elif format == "zip":
            # 创建包含多种格式的ZIP文件
            zip_path = f"{filename}_multi.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # 添加JSON格式
                json_data = {}
                for key in export_data:
                    if key != "metadata":
                        if isinstance(export_data[key], np.ndarray):
                            json_data[key] = export_data[key].tolist()
                    else:
                        json_data[key] = export_data[key]
                
                json_file = f"{filename}_data.json"
                with open(json_file, 'w') as f:
                    json.dump(json_data, f)
                zipf.write(json_file, os.path.basename(json_file))
                os.remove(json_file)
                
                # 添加CSV格式
                if include_velocity:
                    velocity = export_data["velocity"]
                    v_file = f"{filename}_velocity.csv"
                    
                    with open(v_file, "w") as f:
                        f.write("z,y,x,vx,vy,vz\n")
                        for z in range(velocity.shape[0]):
                            for y in range(velocity.shape[1]):
                                for x in range(velocity.shape[2]):
                                    vx, vy, vz = velocity[z, y, x]
                                    f.write(f"{z},{y},{x},{vx},{vy},{vz}\n")
                    
                    zipf.write(v_file, os.path.basename(v_file))
                    os.remove(v_file)
                
                if include_pressure:
                    pressure = export_data["pressure"]
                    p_file = f"{filename}_pressure.csv"
                    
                    with open(p_file, "w") as f:
                        f.write("z,y,x,pressure\n")
                        for z in range(pressure.shape[0]):
                            for y in range(pressure.shape[1]):
                                for x in range(pressure.shape[2]):
                                    p = pressure[z, y, x]
                                    f.write(f"{z},{y},{x},{p}\n")
                    
                    zipf.write(p_file, os.path.basename(p_file))
                    os.remove(p_file)
                
                # 添加NumPy格式
                np_file = f"{filename}.npz"
                np.savez(np_file, **export_data)
                zipf.write(np_file, os.path.basename(np_file))
                os.remove(np_file)
                
                # 添加README
                readme_file = f"{filename}_README.txt"
                with open(readme_file, "w") as f:
                    f.write("流体模拟数据导出\n")
                    f.write("=================\n\n")
                    f.write(f"会话ID: {session_id}\n")
                    f.write(f"步骤: {sim['step']}\n")
                    f.write(f"导出时间: {export_data['metadata']['export_time']}\n\n")
                    f.write("文件说明:\n")
                    f.write("- *_data.json: JSON格式的完整数据\n")
                    f.write("- *_velocity.csv: CSV格式的速度场数据\n")
                    f.write("- *_pressure.csv: CSV格式的压力场数据\n")
                    f.write("- *.npz: NumPy格式的完整数据\n")
                
                zipf.write(readme_file, os.path.basename(readme_file))
                os.remove(readme_file)
            
            export_tasks[task_id]["file_path"] = zip_path
            export_tasks[task_id]["file_name"] = f"{session_id}_{sim['step']}_multi.zip"
        
        else:
            export_tasks[task_id]["status"] = "error"
            export_tasks[task_id]["error"] = f"不支持的导出格式: {format}"
            return
        
        # 更新任务状态为完成
        export_tasks[task_id]["status"] = "completed"
    
    except Exception as e:
        # 更新任务状态为错误
        export_tasks[task_id]["status"] = "error"
        export_tasks[task_id]["error"] = str(e)

@router.post("/data")
async def export_data(
    session_id: str,
    format: str = "json",
    include_velocity: bool = True,
    include_pressure: bool = True,
    include_vorticity: bool = False,
    time_steps: Optional[List[int]] = None,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """导出数据"""
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 检查会话所有权
    session = db.query(SimulationSession).filter(
        SimulationSession.id == session_id,
        SimulationSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    
    # 检查格式是否支持
    if format == "vtk" and not VTK_AVAILABLE:
        raise HTTPException(status_code=400, detail="VTK格式不可用，请安装VTK库")
    
    if format == "hdf5":
        try:
            import h5py
        except ImportError:
            raise HTTPException(status_code=400, detail="HDF5格式不可用，请安装h5py库")
    
    # 创建任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    export_tasks[task_id] = {
        "id": task_id,
        "session_id": session_id,
        "format": format,
        "status": "pending",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": current_user.id
    }
    
    # 启动后台任务
    background_tasks.add_task(
        export_data_task,
        session_id,
        format,
        include_velocity,
        include_pressure,
        include_vorticity,
        task_id,
        time_steps
    )
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.get("/status/{task_id}")
async def get_export_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """获取导出任务状态"""
    if task_id not in export_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = export_tasks[task_id]
    
    # 检查任务所有权
    if task["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问此任务")
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "created_at": task["created_at"],
        "error": task.get("error", None),
        "file_name": task.get("file_name", None) if task["status"] == "completed" else None
    }

@router.get("/download/{task_id}")
async def download_export(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """下载导出文件"""
    if task_id not in export_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = export_tasks[task_id]
    
    # 检查任务所有权
    if task["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问此任务")
    
    # 检查任务状态
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    # 检查文件是否存在
    if not os.path.exists(task["file_path"]):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=task["file_path"],
        filename=task["file_name"],
        media_type="application/octet-stream"
    )

@router.get("/history")
async def get_export_history(
    current_user: User = Depends(get_current_active_user)
):
    """获取导出历史"""
    user_tasks = []
    
    for task_id, task in export_tasks.items():
        if task["user_id"] == current_user.id:
            user_tasks.append({
                "task_id": task_id,
                "session_id": task["session_id"],
                "format": task["format"],
                "status": task["status"],
                "created_at": task["created_at"],
                "file_name": task.get("file_name", None) if task["status"] == "completed" else None
            })
    
    return user_tasks

@router.post("/batch")
async def batch_export(
    session_id: str,
    formats: List[str],
    include_velocity: bool = True,
    include_pressure: bool = True,
    include_vorticity: bool = False,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """批量导出多种格式"""
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 检查会话所有权
    session = db.query(SimulationSession).filter(
        SimulationSession.id == session_id,
        SimulationSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    
    # 创建批处理任务
    batch_tasks = []
    
    for format in formats:
        # 检查格式是否支持
        if format == "vtk" and not VTK_AVAILABLE:
            continue
        
        if format == "hdf5":
            try:
                import h5py
            except ImportError:
                continue
        
        # 创建任务ID
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        export_tasks[task_id] = {
            "id": task_id,
            "session_id": session_id,
            "format": format,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": current_user.id
        }
        
        # 启动后台任务
        background_tasks.add_task(
            export_data_task,
            session_id,
            format,
            include_velocity,
            include_pressure,
            include_vorticity,
            task_id
        )
        
        batch_tasks.append({
            "task_id": task_id,
            "format": format,
            "status": "pending"
        })
    
    return {
        "batch_size": len(batch_tasks),
        "tasks": batch_tasks
    }

@router.get("/preview/{session_id}")
async def preview_data(
    session_id: str,
    field: str,
    current_user: User = Depends(get_current_active_user)
):
    """获取数据预览"""
    # 检查会话是否存在
    if session_id not in active_simulations:
        raise HTTPException(status_code=404, detail="模拟会话不存在")
    
    # 获取模拟实例
    sim = active_simulations[session_id]
    solver = sim["solver"]
    
    # 获取场数据
    if field == "velocity":
        data = solver.get_velocity_field()
        # 获取中心切片
        center_z = data.shape[0] // 2
        preview = data[center_z, :, :, :].tolist()
    elif field == "pressure":
        data = solver.get_pressure_field()
        center_z = data.shape[0] // 2
        preview = data[center_z, :, :].tolist()
    elif field == "vorticity":
        data = solver.get_vorticity_field()
        center_z = data.shape[0] // 2
        preview = data[center_z, :, :, :].tolist()
    else:
        raise HTTPException(status_code=400, detail="不支持的场类型")
    
    return {
        "session_id": session_id,
        "field": field,
        "preview": preview
    } 