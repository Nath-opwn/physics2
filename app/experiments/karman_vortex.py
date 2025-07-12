import numpy as np
from typing import Dict, Any, Callable, Optional, List, Tuple
import time
import io
import logging
import os
import json
import h5py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

class KarmanVortexCalculator:
    """卡门涡街计算器"""
    
    def __init__(self, parameters: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None):
        """
        初始化计算器
        
        参数:
            parameters: 实验参数
            progress_callback: 进度回调函数
        """
        self.parameters = parameters
        self.progress_callback = progress_callback
        self.time_steps = []
        self.results = {}
        self.visualization_data = {}
        
        # 提取参数
        self.cylinder_diameter = parameters.get("cylinder_diameter", 0.1)  # 圆柱直径 (m)
        self.flow_velocity = parameters.get("flow_velocity", 1.0)  # 来流速度 (m/s)
        self.fluid_density = parameters.get("fluid_density", 1.0)  # 流体密度 (kg/m³)
        self.fluid_viscosity = parameters.get("fluid_viscosity", 0.00001)  # 流体粘度 (Pa·s)
        self.domain_width = parameters.get("domain_width", 2.0)  # 计算域宽度 (m)
        self.domain_height = parameters.get("domain_height", 1.0)  # 计算域高度 (m)
        self.simulation_time = parameters.get("simulation_time", 20.0)  # 模拟时间 (s)
        self.time_step = parameters.get("time_step", 0.01)  # 时间步长 (s)
        self.save_frequency = parameters.get("save_frequency", 10)  # 结果保存频率
        
        # 计算雷诺数
        self.reynolds_number = (self.fluid_density * self.flow_velocity * 
                                self.cylinder_diameter / self.fluid_viscosity)
        
        # 根据网格分辨率设置网格大小
        mesh_resolution = parameters.get("mesh_resolution", "medium")
        self.nx, self.ny = self._get_mesh_resolution(mesh_resolution)
        
        # 计算网格步长
        self.dx = self.domain_width / self.nx
        self.dy = self.domain_height / self.ny
        
        # 计算圆柱位置 (圆柱中心位于计算域左侧1/3处，高度居中)
        self.cylinder_x = self.domain_width / 3
        self.cylinder_y = self.domain_height / 2
        self.cylinder_radius = self.cylinder_diameter / 2
        
        # 初始化计算网格
        self.x = np.linspace(0, self.domain_width, self.nx)
        self.y = np.linspace(0, self.domain_height, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 初始化流场
        self.u = np.ones((self.ny, self.nx)) * self.flow_velocity  # x方向速度
        self.v = np.zeros((self.ny, self.nx))  # y方向速度
        self.p = np.zeros((self.ny, self.nx))  # 压力场
        self.vorticity = np.zeros((self.ny, self.nx))  # 涡量场
        
        # 创建掩码数组，标记固体区域
        self.mask = self._create_cylinder_mask()
        
        # 在固体区域将速度设为零
        self.u = self.u * (1 - self.mask)
        self.v = self.v * (1 - self.mask)
        
        logger.info(f"初始化完成，雷诺数: {self.reynolds_number:.2f}, 网格大小: {self.nx}x{self.ny}")
    
    def _get_mesh_resolution(self, resolution: str) -> Tuple[int, int]:
        """根据分辨率设置选择网格尺寸"""
        resolution_map = {
            "very_coarse": (50, 25),
            "coarse": (100, 50),
            "medium": (200, 100),
            "fine": (400, 200),
            "very_fine": (800, 400)
        }
        return resolution_map.get(resolution, (200, 100))
    
    def _create_cylinder_mask(self) -> np.ndarray:
        """创建表示圆柱体位置的掩码"""
        mask = np.zeros((self.ny, self.nx))
        for j in range(self.ny):
            for i in range(self.nx):
                if ((self.x[i] - self.cylinder_x)**2 + 
                    (self.y[j] - self.cylinder_y)**2) < self.cylinder_radius**2:
                    mask[j, i] = 1
        return mask
    
    def _apply_boundary_conditions(self):
        """应用边界条件"""
        # 入口边界: 固定速度
        self.u[:, 0] = self.flow_velocity
        self.v[:, 0] = 0
        
        # 出口边界: 零梯度
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        
        # 上下边界: 滑移条件
        self.u[0, :] = self.u[1, :]
        self.u[-1, :] = self.u[-2, :]
        self.v[0, :] = 0
        self.v[-1, :] = 0
        
        # 圆柱边界: 应用掩码，速度为零
        self.u = self.u * (1 - self.mask)
        self.v = self.v * (1 - self.mask)
    
    def _calculate_vorticity(self) -> np.ndarray:
        """计算涡量场 (速度的旋度)"""
        vorticity = np.zeros((self.ny, self.nx))
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                if self.mask[j, i] == 0:  # 只在流体区域计算
                    dvdx = (self.v[j, i+1] - self.v[j, i-1]) / (2 * self.dx)
                    dudy = (self.u[j+1, i] - self.u[j-1, i]) / (2 * self.dy)
                    vorticity[j, i] = dvdx - dudy
        return vorticity
    
    def _solve_step(self):
        """求解一个时间步"""
        # 保存当前速度用于更新
        u_old = self.u.copy()
        v_old = self.v.copy()
        
        # 计算压力梯度 (简化求解)
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                if self.mask[j, i] == 0:  # 只在流体区域计算
                    # 压力拉普拉斯方程 (简化求解)
                    self.p[j, i] = 0.25 * (self.p[j+1, i] + self.p[j-1, i] + 
                                          self.p[j, i+1] + self.p[j, i-1])
        
        # 求解动量方程 (简化的Navier-Stokes方程)
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                if self.mask[j, i] == 0:  # 只在流体区域计算
                    # x方向动量方程
                    convection_u = (u_old[j, i] * (u_old[j, i+1] - u_old[j, i-1]) / (2 * self.dx) +
                                   v_old[j, i] * (u_old[j+1, i] - u_old[j-1, i]) / (2 * self.dy))
                    
                    diffusion_u = (u_old[j, i+1] - 2*u_old[j, i] + u_old[j, i-1]) / self.dx**2 + \
                                 (u_old[j+1, i] - 2*u_old[j, i] + u_old[j-1, i]) / self.dy**2
                    
                    pressure_grad_u = (self.p[j, i+1] - self.p[j, i-1]) / (2 * self.dx)
                    
                    self.u[j, i] = u_old[j, i] + self.time_step * (
                        -convection_u + 
                        self.fluid_viscosity/self.fluid_density * diffusion_u - 
                        (1/self.fluid_density) * pressure_grad_u
                    )
                    
                    # y方向动量方程
                    convection_v = (u_old[j, i] * (v_old[j, i+1] - v_old[j, i-1]) / (2 * self.dx) +
                                   v_old[j, i] * (v_old[j+1, i] - v_old[j-1, i]) / (2 * self.dy))
                    
                    diffusion_v = (v_old[j, i+1] - 2*v_old[j, i] + v_old[j, i-1]) / self.dx**2 + \
                                 (v_old[j+1, i] - 2*v_old[j, i] + v_old[j-1, i]) / self.dy**2
                    
                    pressure_grad_v = (self.p[j+1, i] - self.p[j-1, i]) / (2 * self.dy)
                    
                    self.v[j, i] = v_old[j, i] + self.time_step * (
                        -convection_v + 
                        self.fluid_viscosity/self.fluid_density * diffusion_v - 
                        (1/self.fluid_density) * pressure_grad_v
                    )
        
        # 应用边界条件
        self._apply_boundary_conditions()
    
    def _calculate_forces(self) -> Tuple[float, float]:
        """计算圆柱上的力 (阻力和升力)"""
        drag = 0.0
        lift = 0.0
        
        # 遍历靠近圆柱的网格点
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                if self.mask[j, i] == 1 and np.any(self.mask[j-1:j+2, i-1:i+2] == 0):
                    # 压力分量
                    p_force = self.p[j, i]
                    
                    # 计算法向量 (从圆心指向该点)
                    nx = (self.x[i] - self.cylinder_x) / self.cylinder_radius
                    ny = (self.y[j] - self.cylinder_y) / self.cylinder_radius
                    
                    # 累加力
                    drag += p_force * nx * self.dx * self.dy
                    lift += p_force * ny * self.dx * self.dy
        
        # 归一化
        reference_force = 0.5 * self.fluid_density * self.flow_velocity**2 * self.cylinder_diameter
        drag_coefficient = drag / reference_force
        lift_coefficient = lift / reference_force
        
        return drag_coefficient, lift_coefficient
    
    def _generate_visualization(self, step: int) -> Dict[str, Any]:
        """生成特定步骤的可视化数据"""
        # 计算速度幅值
        velocity_magnitude = np.sqrt(self.u**2 + self.v**2)
        
        # 计算涡量
        self.vorticity = self._calculate_vorticity()
        
        # 获取时间
        current_time = step * self.time_step
        
        # 创建可视化数据字典
        vis_data = {
            "time": current_time,
            "step": step,
            "velocity": {
                "u": self.u.tolist(),
                "v": self.v.tolist(),
                "magnitude": velocity_magnitude.tolist()
            },
            "pressure": self.p.tolist(),
            "vorticity": self.vorticity.tolist()
        }
        
        return vis_data
    
    def _generate_plots(self, step: int) -> Dict[str, io.BytesIO]:
        """生成可视化图像"""
        plot_buffers = {}
        
        # 速度幅值图
        fig_vel = Figure(figsize=(10, 6))
        ax_vel = fig_vel.add_subplot(111)
        velocity_magnitude = np.sqrt(self.u**2 + self.v**2)
        
        # 创建掩码版本的数据用于绘图
        masked_velocity = np.ma.array(velocity_magnitude, mask=self.mask)
        
        im = ax_vel.imshow(
            masked_velocity, 
            extent=[0, self.domain_width, 0, self.domain_height],
            origin='lower', 
            aspect='equal',
            cmap=cm.viridis
        )
        
        # 绘制速度矢量
        stride = max(1, self.nx // 25)
        ax_vel.quiver(
            self.X[::stride, ::stride], 
            self.Y[::stride, ::stride],
            self.u[::stride, ::stride], 
            self.v[::stride, ::stride],
            color='white', 
            scale=25
        )
        
        # 添加圆柱
        circle = plt.Circle(
            (self.cylinder_x, self.cylinder_y), 
            self.cylinder_radius, 
            color='white', 
            fill=True
        )
        ax_vel.add_patch(circle)
        
        ax_vel.set_title(f'速度场 (t = {step * self.time_step:.2f} s)')
        ax_vel.set_xlabel('x (m)')
        ax_vel.set_ylabel('y (m)')
        
        divider = make_axes_locatable(ax_vel)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig_vel.colorbar(im, cax=cax, label='速度 (m/s)')
        
        fig_vel.tight_layout()
        buf_vel = io.BytesIO()
        fig_vel.savefig(buf_vel, format='png', dpi=150)
        buf_vel.seek(0)
        plot_buffers['velocity'] = buf_vel
        
        # 涡量图
        fig_vort = Figure(figsize=(10, 6))
        ax_vort = fig_vort.add_subplot(111)
        
        # 创建掩码版本的涡量用于绘图
        masked_vorticity = np.ma.array(self.vorticity, mask=self.mask)
        
        im = ax_vort.imshow(
            masked_vorticity, 
            extent=[0, self.domain_width, 0, self.domain_height],
            origin='lower', 
            aspect='equal',
            cmap=cm.RdBu_r, 
            vmin=-5, 
            vmax=5
        )
        
        # 添加圆柱
        circle = plt.Circle(
            (self.cylinder_x, self.cylinder_y), 
            self.cylinder_radius, 
            color='white', 
            fill=True
        )
        ax_vort.add_patch(circle)
        
        ax_vort.set_title(f'涡量场 (t = {step * self.time_step:.2f} s)')
        ax_vort.set_xlabel('x (m)')
        ax_vort.set_ylabel('y (m)')
        
        divider = make_axes_locatable(ax_vort)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig_vort.colorbar(im, cax=cax, label='涡量 (1/s)')
        
        fig_vort.tight_layout()
        buf_vort = io.BytesIO()
        fig_vort.savefig(buf_vort, format='png', dpi=150)
        buf_vort.seek(0)
        plot_buffers['vorticity'] = buf_vort
        
        # 压力图
        fig_press = Figure(figsize=(10, 6))
        ax_press = fig_press.add_subplot(111)
        
        # 创建掩码版本的压力用于绘图
        masked_pressure = np.ma.array(self.p, mask=self.mask)
        
        im = ax_press.imshow(
            masked_pressure, 
            extent=[0, self.domain_width, 0, self.domain_height],
            origin='lower', 
            aspect='equal',
            cmap=cm.coolwarm
        )
        
        # 添加圆柱
        circle = plt.Circle(
            (self.cylinder_x, self.cylinder_y), 
            self.cylinder_radius, 
            color='white', 
            fill=True
        )
        ax_press.add_patch(circle)
        
        ax_press.set_title(f'压力场 (t = {step * self.time_step:.2f} s)')
        ax_press.set_xlabel('x (m)')
        ax_press.set_ylabel('y (m)')
        
        divider = make_axes_locatable(ax_press)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig_press.colorbar(im, cax=cax, label='压力')
        
        fig_press.tight_layout()
        buf_press = io.BytesIO()
        fig_press.savefig(buf_press, format='png', dpi=150)
        buf_press.seek(0)
        plot_buffers['pressure'] = buf_press
        
        return plot_buffers
    
    def _calculate_shedding_frequency(self) -> float:
        """计算涡脱落频率"""
        # 使用y方向速度在圆柱下游位置的时间序列
        probe_x_index = int((self.cylinder_x + 2 * self.cylinder_diameter) / self.dx)
        probe_y_index = int(self.cylinder_y / self.dy)
        
        # 获取后半段时间的数据 (跳过初始阶段)
        start_idx = len(self.time_steps) // 2
        times = np.array(self.time_steps[start_idx:])
        velocities = np.array([step_data['probe_v'] for step_data in self.time_steps_data[start_idx:]])
        
        if len(times) < 4:  # 需要至少几个周期来计算频率
            logger.warning("数据点不足以计算脱落频率")
            return 0.0
        
        # 使用FFT计算频率
        n = len(velocities)
        dt = times[1] - times[0]
        fft_freqs = np.fft.fftfreq(n, dt)
        fft_vals = np.abs(np.fft.fft(velocities))
        
        # 获取正频率的最大幅值
        positive_idx = np.where(fft_freqs > 0)[0]
        dominant_idx = positive_idx[np.argmax(fft_vals[positive_idx])]
        
        return fft_freqs[dominant_idx]
    
    def _calculate_strouhal_number(self, shedding_frequency: float) -> float:
        """计算斯特劳哈尔数"""
        if shedding_frequency > 0 and self.flow_velocity > 0:
            return shedding_frequency * self.cylinder_diameter / self.flow_velocity
        return 0.0
    
    def solve(self) -> Dict[str, Any]:
        """
        运行完整求解过程
        
        返回:
            包含计算结果、时间步数据和可视化数据的字典
        """
        start_time = time.time()
        
        # 计算总时间步数
        total_steps = int(self.simulation_time / self.time_step)
        
        # 存储每个时间步的探针数据
        self.time_steps = []
        self.time_steps_data = []
        
        # 存储力系数历史
        drag_history = []
        lift_history = []
        
        # 在圆柱下游设置探针点
        probe_x_index = int((self.cylinder_x + 2 * self.cylinder_diameter) / self.dx)
        probe_y_index = int(self.cylinder_y / self.dy)
        
        # 主求解循环
        for step in range(total_steps):
            current_time = step * self.time_step
            
            # 求解当前时间步
            self._solve_step()
            
            # 计算阻力和升力系数
            drag_coefficient, lift_coefficient = self._calculate_forces()
            drag_history.append(drag_coefficient)
            lift_history.append(lift_coefficient)
            
            # 记录探针点数据
            if 0 <= probe_x_index < self.nx and 0 <= probe_y_index < self.ny:
                probe_u = self.u[probe_y_index, probe_x_index]
                probe_v = self.v[probe_y_index, probe_x_index]
            else:
                probe_u = 0.0
                probe_v = 0.0
            
            self.time_steps.append(current_time)
            self.time_steps_data.append({
                'time': current_time,
                'probe_u': float(probe_u),
                'probe_v': float(probe_v),
                'drag': float(drag_coefficient),
                'lift': float(lift_coefficient)
            })
            
            # 每隔一定步数保存详细数据
            if step % self.save_frequency == 0 or step == total_steps - 1:
                # 保存可视化数据
                self.visualization_data[f"step_{step}"] = self._generate_visualization(step)
                
                # 生成图像
                try:
                    self.visualization_data[f"step_{step}"]["plots"] = self._generate_plots(step)
                except Exception as e:
                    logger.error(f"生成可视化图像失败: {str(e)}")
            
            # 更新进度
            if self.progress_callback and step % 10 == 0:
                progress = min(99, int((step + 1) / total_steps * 100))
                self.progress_callback(progress)
        
        # 计算涡脱落频率
        shedding_frequency = self._calculate_shedding_frequency()
        strouhal_number = self._calculate_strouhal_number(shedding_frequency)
        
        # 计算平均阻力系数
        avg_drag_coefficient = np.mean(drag_history[total_steps//2:]) if drag_history else 0.0
        
        # 计算总计算时间
        computation_time = time.time() - start_time
        
        # 完成进度
        if self.progress_callback:
            self.progress_callback(100)
        
        # 构造结果
        self.results = {
            "reynolds_number": float(self.reynolds_number),
            "shedding_frequency": float(shedding_frequency),
            "strouhal_number": float(strouhal_number),
            "avg_drag_coefficient": float(avg_drag_coefficient),
            "computation_time": float(computation_time),
            "mesh_size": {"nx": self.nx, "ny": self.ny},
            "time_steps": len(self.time_steps),
            "parameters": self.parameters
        }
        
        logger.info(f"计算完成，雷诺数: {self.reynolds_number:.2f}, 斯特劳哈尔数: {strouhal_number:.4f}")
        
        return {
            "results": self.results,
            "time_steps_data": self.time_steps_data,
            "visualization_data": self.visualization_data
        }

def calculate_karman_vortex(
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    卡门涡街计算核心逻辑
    
    参数:
        parameters: 实验参数，包含流速、流体属性、圆柱直径等
        progress_callback: 进度回调函数
    
    返回:
        计算结果，包含涡街频率、斯特劳哈尔数等
        时序数据，包含不同时间点的流场数据
        可视化数据，包含流线、压力场、速度场等
    """
    calculator = KarmanVortexCalculator(parameters, progress_callback)
    result = calculator.solve()
    
    return result["results"], result["time_steps_data"], result["visualization_data"] 