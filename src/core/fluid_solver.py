import numpy as np
from scipy.ndimage import convolve

# 尝试导入C++扩展
try:
    from .cpp_ext import diffuse as cpp_diffuse
    from .cpp_ext import advect as cpp_advect
    from .cpp_ext import project as cpp_project
    from .cpp_ext import compute_vorticity as cpp_compute_vorticity
    from .cpp_ext import EXTENSION_LOADED
except ImportError:
    EXTENSION_LOADED = False

class FluidSolver:
    """
    简化版流体求解器，实现基本的流体动力学模拟
    支持C++/OpenMP加速版本
    """
    
    def __init__(self, width, height, depth, viscosity=0.1, density=1.0, use_cpp_ext=True):
        """
        初始化流体求解器
        
        参数:
            width (int): 网格宽度
            height (int): 网格高度
            depth (int): 网格深度
            viscosity (float): 流体粘度
            density (float): 流体密度
            use_cpp_ext (bool): 是否使用C++扩展
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.viscosity = viscosity
        self.density = density
        
        # 是否使用C++扩展
        self.use_cpp_ext = use_cpp_ext and EXTENSION_LOADED
        if self.use_cpp_ext:
            print("使用C++扩展进行流体模拟计算")
        else:
            print("使用纯Python进行流体模拟计算")
        
        # 初始化速度场 (u, v, w)
        self.u = np.zeros((depth, height, width))
        self.v = np.zeros((depth, height, width))
        self.w = np.zeros((depth, height, width))
        
        # 上一步的速度场
        self.u_prev = np.zeros((depth, height, width))
        self.v_prev = np.zeros((depth, height, width))
        self.w_prev = np.zeros((depth, height, width))
        
        # 压力场
        self.pressure = np.zeros((depth, height, width))
        self.div = np.zeros((depth, height, width))  # 用于C++扩展的散度场
        
        # 涡量场
        self.vorticity = np.zeros((depth, height, width))
        
        # 边界条件类型
        self.boundary_type = 0
        
        # 障碍物标记 (0表示流体，1表示障碍物)
        self.obstacles = np.zeros((depth, height, width), dtype=np.int8)
        
        # 初始化卷积核
        self._init_kernels()
        
        # 性能统计
        self.performance_stats = {
            'diffuse_time': 0.0,
            'advect_time': 0.0,
            'project_time': 0.0,
            'total_time': 0.0,
            'steps': 0
        }
    
    def reset_performance_stats(self):
        """重置性能统计数据"""
        self.performance_stats = {
            'diffuse_time': 0.0,
            'advect_time': 0.0,
            'project_time': 0.0,
            'total_time': 0.0,
            'steps': 0
        }
    
    def _init_kernels(self):
        """初始化用于计算的卷积核"""
        # 拉普拉斯算子
        self.laplacian_kernel = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ])
    
    def set_boundary_condition(self, boundary_type):
        """
        设置边界条件类型
        
        参数:
            boundary_type (int): 边界条件类型
                0: 固定边界
                1: 周期性边界
                2: 开放边界
                3: 自定义边界
        """
        self.boundary_type = boundary_type
    
    def add_force(self, x, y, z, fx, fy, fz):
        """
        在指定位置添加力
        
        参数:
            x, y, z (int): 力的位置
            fx, fy, fz (float): 力的分量
        """
        # 确保坐标在网格范围内
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            # 在3x3x3区域内添加力，使用高斯分布
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                            # 高斯权重
                            weight = np.exp(-(dx**2 + dy**2 + dz**2) / 2) / (2 * np.pi * np.sqrt(2 * np.pi))
                            self.u[nz, ny, nx] += fx * weight
                            self.v[nz, ny, nx] += fy * weight
                            self.w[nz, ny, nx] += fz * weight
    
    def add_obstacle(self, shape, params):
        """
        添加障碍物
        
        参数:
            shape (str): 障碍物形状，支持'sphere', 'cylinder', 'box'
            params (dict): 障碍物参数
                - sphere: {'center': (x,y,z), 'radius': r}
                - cylinder: {'center': (x,y,z), 'radius': r, 'height': h, 'axis': 'x'|'y'|'z'}
                - box: {'min': (x1,y1,z1), 'max': (x2,y2,z2)}
        """
        if shape == 'sphere':
            center = params.get('center', (self.width//2, self.height//2, self.depth//2))
            radius = params.get('radius', min(self.width, self.height, self.depth) // 8)
            
            # 在球体内部设置障碍物
            for z in range(self.depth):
                for y in range(self.height):
                    for x in range(self.width):
                        dx = x - center[0]
                        dy = y - center[1]
                        dz = z - center[2]
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        if dist < radius:
                            self.obstacles[z, y, x] = 1
        
        elif shape == 'cylinder':
            center = params.get('center', (self.width//2, self.height//2, self.depth//2))
            radius = params.get('radius', min(self.width, self.height, self.depth) // 8)
            height = params.get('height', self.height // 2)
            axis = params.get('axis', 'y')
            
            # 在圆柱体内部设置障碍物
            for z in range(self.depth):
                for y in range(self.height):
                    for x in range(self.width):
                        if axis == 'y':
                            dx = x - center[0]
                            dz = z - center[2]
                            dist = np.sqrt(dx**2 + dz**2)
                            if dist < radius and abs(y - center[1]) < height/2:
                                self.obstacles[z, y, x] = 1
                        elif axis == 'x':
                            dy = y - center[1]
                            dz = z - center[2]
                            dist = np.sqrt(dy**2 + dz**2)
                            if dist < radius and abs(x - center[0]) < height/2:
                                self.obstacles[z, y, x] = 1
                        elif axis == 'z':
                            dx = x - center[0]
                            dy = y - center[1]
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < radius and abs(z - center[2]) < height/2:
                                self.obstacles[z, y, x] = 1
        
        elif shape == 'box':
            min_point = params.get('min', (0, 0, 0))
            max_point = params.get('max', (self.width//4, self.height//4, self.depth//4))
            
            # 在盒子内部设置障碍物
            x_min, y_min, z_min = min_point
            x_max, y_max, z_max = max_point
            
            for z in range(max(0, z_min), min(self.depth, z_max + 1)):
                for y in range(max(0, y_min), min(self.height, y_max + 1)):
                    for x in range(max(0, x_min), min(self.width, x_max + 1)):
                        self.obstacles[z, y, x] = 1
    
    def clear_obstacles(self):
        """清除所有障碍物"""
        self.obstacles.fill(0)
    
    def setup_preset_simulation(self, preset_type, params=None):
        """
        设置预设模拟类型
        
        参数:
            preset_type (str): 预设类型，支持'cylinder_flow', 'karman_vortex', 'channel_flow', 'lid_driven_cavity'
            params (dict): 可选参数
        """
        if params is None:
            params = {}
        
        # 清除现有状态
        self.u.fill(0)
        self.v.fill(0)
        self.w.fill(0)
        self.pressure.fill(0)
        self.obstacles.fill(0)
        
        if preset_type == 'cylinder_flow':
            # 圆柱体绕流
            # 设置入口速度
            inlet_velocity = params.get('inlet_velocity', 1.0)
            for z in range(self.depth):
                for y in range(self.height):
                    self.u[z, y, 0:5] = inlet_velocity
            
            # 添加圆柱障碍物
            cylinder_params = {
                'center': (self.width // 3, self.height // 2, self.depth // 2),
                'radius': min(self.height, self.depth) // 6,
                'height': self.depth,
                'axis': 'x'
            }
            self.add_obstacle('cylinder', cylinder_params)
            
            # 设置边界条件
            self.set_boundary_condition(0)  # 固定边界
        
        elif preset_type == 'karman_vortex':
            # 卡门涡街
            # 设置恒定入口流速
            inlet_velocity = params.get('inlet_velocity', 1.0)
            for z in range(self.depth):
                for y in range(self.height):
                    self.u[z, y, 0:5] = inlet_velocity
            
            # 添加小扰动，促进涡街形成
            center_y = self.height // 2
            center_z = self.depth // 2
            for z in range(center_z - 5, center_z + 5):
                for y in range(center_y - 5, center_y + 5):
                    if 0 <= z < self.depth and 0 <= y < self.height:
                        self.v[z, y, 10:20] = 0.1 * np.sin((y - center_y) * 0.5)
            
            # 添加圆柱障碍物
            cylinder_params = {
                'center': (self.width // 4, self.height // 2, self.depth // 2),
                'radius': min(self.height, self.depth) // 10,
                'height': self.depth,
                'axis': 'x'
            }
            self.add_obstacle('cylinder', cylinder_params)
            
            # 设置边界条件
            self.set_boundary_condition(2)  # 开放边界
        
        elif preset_type == 'channel_flow':
            # 通道流
            # 设置抛物线入口速度剖面
            inlet_velocity = params.get('inlet_velocity', 1.0)
            for z in range(self.depth):
                for y in range(self.height):
                    # 抛物线速度分布
                    normalized_y = 2 * (y / self.height - 0.5)  # -1 到 1
                    normalized_z = 2 * (z / self.depth - 0.5)   # -1 到 1
                    
                    # 抛物线分布，在中心最大，边缘为0
                    profile = (1 - normalized_y**2) * (1 - normalized_z**2)
                    self.u[z, y, 0:5] = inlet_velocity * profile
            
            # 添加通道障碍物（上下壁）
            wall_thickness = params.get('wall_thickness', self.height // 10)
            
            # 上壁
            for z in range(self.depth):
                for y in range(self.height - wall_thickness, self.height):
                    for x in range(self.width):
                        self.obstacles[z, y, x] = 1
            
            # 下壁
            for z in range(self.depth):
                for y in range(0, wall_thickness):
                    for x in range(self.width):
                        self.obstacles[z, y, x] = 1
            
            # 设置边界条件
            self.set_boundary_condition(0)  # 固定边界
        
        elif preset_type == 'lid_driven_cavity':
            # 盖驱动腔流
            # 设置顶部移动壁面
            lid_velocity = params.get('lid_velocity', 1.0)
            for z in range(self.depth):
                for x in range(self.width):
                    self.u[z, self.height-1, x] = lid_velocity
            
            # 添加腔体障碍物（三面壁）
            for z in range(self.depth):
                # 左壁
                for y in range(self.height):
                    self.obstacles[z, y, 0] = 1
                
                # 右壁
                for y in range(self.height):
                    self.obstacles[z, y, self.width-1] = 1
                
                # 底壁
                for x in range(self.width):
                    self.obstacles[z, 0, x] = 1
            
            # 设置边界条件
            self.set_boundary_condition(0)  # 固定边界
    
    def _diffuse(self, field, prev_field, dt):
        """
        扩散步骤
        
        参数:
            field: 当前场
            prev_field: 上一步的场
            dt: 时间步长
        
        返回:
            扩散后的场
        """
        # 如果启用了C++扩展，则使用C++实现
        if self.use_cpp_ext:
            # 创建场的副本
            result = field.copy()
            
            # 调用C++扩展
            cpp_diffuse(result, prev_field, self.viscosity, dt, 20)
            
            # 应用边界条件
            self._apply_boundary_condition(result)
            
            return result
        else:
            # 原始Python实现
            diff = self.viscosity * dt
            
            # 创建场的副本
            result = field.copy()
            
            # 迭代求解扩散方程
            for _ in range(20):
                # 使用拉普拉斯算子计算扩散
                laplacian = convolve(result, self.laplacian_kernel, mode='constant', cval=0.0)
                result = (prev_field + diff * laplacian) / (1 + 6 * diff)
                
                # 应用边界条件
                self._apply_boundary_condition(result)
            
            return result
    
    def _advect(self, field, u, v, w, dt):
        """
        平流步骤
        
        参数:
            field: 要平流的场
            u, v, w: 速度场分量
            dt: 时间步长
        
        返回:
            平流后的场
        """
        # 如果启用了C++扩展，则使用C++实现
        if self.use_cpp_ext:
            # 创建场的副本
            result = np.zeros_like(field)
            
            # 调用C++扩展
            cpp_advect(result, field, u, v, w, dt)
            
            # 应用边界条件
            self._apply_boundary_condition(result)
            
            return result
        else:
            # 原始Python实现
            depth, height, width = field.shape
            result = np.zeros_like(field)
            
            # 对每个网格点进行回溯
            for z in range(1, depth-1):
                for y in range(1, height-1):
                    for x in range(1, width-1):
                        # 计算回溯位置
                        pos_x = x - dt * u[z, y, x] * width
                        pos_y = y - dt * v[z, y, x] * height
                        pos_z = z - dt * w[z, y, x] * depth
                        
                        # 确保位置在网格内
                        pos_x = max(0.5, min(width - 1.5, pos_x))
                        pos_y = max(0.5, min(height - 1.5, pos_y))
                        pos_z = max(0.5, min(depth - 1.5, pos_z))
                        
                        # 计算插值索引
                        i0 = int(pos_x)
                        i1 = i0 + 1
                        j0 = int(pos_y)
                        j1 = j0 + 1
                        k0 = int(pos_z)
                        k1 = k0 + 1
                        
                        # 计算插值权重
                        s1 = pos_x - i0
                        s0 = 1 - s1
                        t1 = pos_y - j0
                        t0 = 1 - t1
                        u1 = pos_z - k0
                        u0 = 1 - u1
                        
                        # 三线性插值
                        result[z, y, x] = (
                            u0 * (t0 * (s0 * field[k0, j0, i0] + s1 * field[k0, j0, i1]) +
                                 t1 * (s0 * field[k0, j1, i0] + s1 * field[k0, j1, i1])) +
                            u1 * (t0 * (s0 * field[k1, j0, i0] + s1 * field[k1, j0, i1]) +
                                 t1 * (s0 * field[k1, j1, i0] + s1 * field[k1, j1, i1]))
                        )
            
            # 应用边界条件
            self._apply_boundary_condition(result)
            
            return result
    
    def _project(self, u, v, w):
        """
        投影步骤，确保速度场是无散度的
        
        参数:
            u, v, w: 速度场分量
        
        返回:
            投影后的速度场分量
        """
        # 如果启用了C++扩展，则使用C++实现
        if self.use_cpp_ext:
            # 创建速度场的副本
            u_result = u.copy()
            v_result = v.copy()
            w_result = w.copy()
            
            # 调用C++扩展
            cpp_project(u_result, v_result, w_result, self.pressure, self.div, 20)
            
            # 应用边界条件
            self._apply_boundary_condition(u_result)
            self._apply_boundary_condition(v_result)
            self._apply_boundary_condition(w_result)
            
            return u_result, v_result, w_result
        else:
            # 原始Python实现
            depth, height, width = u.shape
            
            # 创建压力场和散度场
            pressure = np.zeros((depth, height, width))
            div = np.zeros((depth, height, width))
            
            # 计算速度场的散度
            for z in range(1, depth-1):
                for y in range(1, height-1):
                    for x in range(1, width-1):
                        div[z, y, x] = -0.5 * (
                            (u[z, y, x+1] - u[z, y, x-1]) / width +
                            (v[z, y+1, x] - v[z, y-1, x]) / height +
                            (w[z+1, y, x] - w[z-1, y, x]) / depth
                        )
            
            # 求解泊松方程
            for _ in range(20):
                for z in range(1, depth-1):
                    for y in range(1, height-1):
                        for x in range(1, width-1):
                            pressure[z, y, x] = (div[z, y, x] +
                                pressure[z, y, x+1] + pressure[z, y, x-1] +
                                pressure[z, y+1, x] + pressure[z, y-1, x] +
                                pressure[z+1, y, x] + pressure[z-1, y, x]
                            ) / 6.0
                
                # 应用边界条件
                self._apply_boundary_condition(pressure)
            
            # 应用压力梯度
            u_result = u.copy()
            v_result = v.copy()
            w_result = w.copy()
            
            for z in range(1, depth-1):
                for y in range(1, height-1):
                    for x in range(1, width-1):
                        u_result[z, y, x] -= 0.5 * width * (pressure[z, y, x+1] - pressure[z, y, x-1])
                        v_result[z, y, x] -= 0.5 * height * (pressure[z, y+1, x] - pressure[z, y-1, x])
                        w_result[z, y, x] -= 0.5 * depth * (pressure[z+1, y, x] - pressure[z-1, y, x])
            
            # 应用边界条件
            self._apply_boundary_condition(u_result)
            self._apply_boundary_condition(v_result)
            self._apply_boundary_condition(w_result)
            
            # 保存压力场
            self.pressure = pressure
            
            return u_result, v_result, w_result
    
    def _apply_boundary_condition(self, field):
        """
        应用边界条件
        
        参数:
            field (ndarray): 要应用边界条件的场
        """
        if self.boundary_type == 0:  # 固定边界
            # 设置边界值为0
            field[0, :, :] = 0
            field[-1, :, :] = 0
            field[:, 0, :] = 0
            field[:, -1, :] = 0
            field[:, :, 0] = 0
            field[:, :, -1] = 0
        
        elif self.boundary_type == 1:  # 周期性边界
            # 复制对面的值
            field[0, :, :] = field[-2, :, :]
            field[-1, :, :] = field[1, :, :]
            field[:, 0, :] = field[:, -2, :]
            field[:, -1, :] = field[:, 1, :]
            field[:, :, 0] = field[:, :, -2]
            field[:, :, -1] = field[:, :, 1]
        
        elif self.boundary_type == 2:  # 开放边界
            # 使用内部值
            field[0, :, :] = field[1, :, :]
            field[-1, :, :] = field[-2, :, :]
            field[:, 0, :] = field[:, 1, :]
            field[:, -1, :] = field[:, -2, :]
            field[:, :, 0] = field[:, :, 1]
            field[:, :, -1] = field[:, :, -2]
        
        # 应用障碍物边界条件
        # 在障碍物位置设置场值为0
        field[self.obstacles == 1] = 0
    
    def _apply_obstacles(self):
        """处理障碍物，确保障碍物内部速度为零"""
        depth, height, width = self.obstacles.shape
        
        # 对每个障碍物网格点设置速度为零
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if self.obstacles[z, y, x] == 1:
                        self.u[z, y, x] = 0
                        self.v[z, y, x] = 0
                        self.w[z, y, x] = 0
                        
                        # 处理障碍物边界
                        if x > 0:
                            self.u[z, y, x-1] = 0
                        if x < width - 1:
                            self.u[z, y, x+1] = 0
                        if y > 0:
                            self.v[z, y-1, x] = 0
                        if y < height - 1:
                            self.v[z, y+1, x] = 0
                        if z > 0:
                            self.w[z-1, y, x] = 0
                        if z < depth - 1:
                            self.w[z+1, y, x] = 0
    
    def _compute_vorticity(self):
        """计算涡量场"""
        # 如果启用了C++扩展，则使用C++实现
        if self.use_cpp_ext:
            # 创建涡量场
            vorticity = np.zeros_like(self.vorticity)
            
            # 调用C++扩展
            cpp_compute_vorticity(self.u, self.v, self.w, vorticity)
            
            self.vorticity = vorticity
        else:
            # 原始Python实现
            depth, height, width = self.u.shape
            vorticity = np.zeros((depth, height, width))
            
            # 计算涡量
            for z in range(1, depth-1):
                for y in range(1, height-1):
                    for x in range(1, width-1):
                        # 计算速度梯度
                        du_dy = (self.u[z, y+1, x] - self.u[z, y-1, x]) / (2.0 * height)
                        du_dz = (self.u[z+1, y, x] - self.u[z-1, y, x]) / (2.0 * depth)
                        dv_dx = (self.v[z, y, x+1] - self.v[z, y, x-1]) / (2.0 * width)
                        dv_dz = (self.v[z+1, y, x] - self.v[z-1, y, x]) / (2.0 * depth)
                        dw_dx = (self.w[z, y, x+1] - self.w[z, y, x-1]) / (2.0 * width)
                        dw_dy = (self.w[z, y+1, x] - self.w[z, y-1, x]) / (2.0 * height)
                        
                        # 计算涡量分量
                        curl_x = dw_dy - dv_dz
                        curl_y = du_dz - dw_dx
                        curl_z = dv_dx - du_dy
                        
                        # 计算涡量大小
                        vorticity[z, y, x] = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
            
            self.vorticity = vorticity
    
    def step(self, dt):
        """
        执行一个时间步长的模拟
        
        参数:
            dt: 时间步长
        """
        # 保存当前状态
        self.u_prev = self.u.copy()
        self.v_prev = self.v.copy()
        self.w_prev = self.w.copy()
        
        # 处理障碍物
        if np.any(self.obstacles):
            self._apply_obstacles()
        
        # 扩散步骤
        self.u = self._diffuse(self.u, self.u_prev, dt)
        self.v = self._diffuse(self.v, self.v_prev, dt)
        self.w = self._diffuse(self.w, self.w_prev, dt)
        
        # 投影步骤（确保速度场无散度）
        self.u, self.v, self.w = self._project(self.u, self.v, self.w)
        
        # 平流步骤
        self.u = self._advect(self.u, self.u, self.v, self.w, dt)
        self.v = self._advect(self.v, self.u, self.v, self.w, dt)
        self.w = self._advect(self.w, self.u, self.v, self.w, dt)
        
        # 再次投影
        self.u, self.v, self.w = self._project(self.u, self.v, self.w)
        
        # 计算涡量场
        self._compute_vorticity()
        
        # 处理障碍物
        if np.any(self.obstacles):
            self._apply_obstacles()
    
    def get_velocity_field(self):
        """获取速度场"""
        # 合并三个分量为一个向量场
        velocity = np.zeros((self.depth, self.height, self.width, 3))
        velocity[:, :, :, 0] = self.u
        velocity[:, :, :, 1] = self.v
        velocity[:, :, :, 2] = self.w
        return velocity
    
    def get_pressure_field(self):
        """获取压力场"""
        return self.pressure
    
    def get_vorticity_field(self):
        """获取涡量场"""
        return self.vorticity
    
    def get_data_at_point(self, x, y, z):
        """
        获取指定点的数据
        
        参数:
            x, y, z (float): 查询点坐标
            
        返回:
            dict: 包含速度、压力和涡量的字典
        """
        # 确保坐标在网格范围内
        x = max(0, min(self.width - 1, int(x)))
        y = max(0, min(self.height - 1, int(y)))
        z = max(0, min(self.depth - 1, int(z)))
        
        return {
            "velocity": [self.u[z, y, x], self.v[z, y, x], self.w[z, y, x]],
            "pressure": self.pressure[z, y, x],
            "vorticity": self.vorticity[z, y, x].tolist()
        } 

def analyze_vortex_structures(vorticity_data, width, height, depth, threshold=0.1, method="q_criterion"):
    """
    分析涡量场中的涡结构
    
    参数:
        vorticity_data: 涡量场数据，列表形式 [[wx, wy, wz], ...]
        width, height, depth: 网格尺寸
        threshold: 涡结构识别阈值
        method: 涡结构识别方法，可选值: "q_criterion", "lambda2", "vorticity_magnitude"
        
    返回:
        涡结构列表，每个结构包含位置、强度、大小和方向
    """
    import numpy as np
    from scipy import ndimage
    
    # 将涡量数据转换为3D NumPy数组
    vort_mag = np.zeros((width, height, depth))
    vort_x = np.zeros((width, height, depth))
    vort_y = np.zeros((width, height, depth))
    vort_z = np.zeros((width, height, depth))
    
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                idx = x + y * width + z * width * height
                if idx < len(vorticity_data):
                    wx, wy, wz = vorticity_data[idx]
                    vort_x[x, y, z] = wx
                    vort_y[x, y, z] = wy
                    vort_z[x, y, z] = wz
                    vort_mag[x, y, z] = np.sqrt(wx**2 + wy**2 + wz**2)
    
    # 根据选择的方法识别涡结构
    if method == "vorticity_magnitude":
        # 使用涡量大小直接识别
        vortex_field = vort_mag
    elif method == "q_criterion":
        # 计算Q准则 (需要速度梯度张量，这里简化处理)
        # 实际Q准则需要计算速度梯度张量的第二不变量
        vortex_field = vort_mag  # 简化，实际应基于速度梯度计算
    elif method == "lambda2":
        # Lambda2准则 (需要速度梯度张量，这里简化处理)
        vortex_field = vort_mag  # 简化，实际应基于速度梯度计算
    else:
        vortex_field = vort_mag
    
    # 应用阈值，识别涡核区域
    vortex_regions = vortex_field > threshold
    
    # 标记连通区域
    labeled_regions, num_regions = ndimage.label(vortex_regions)
    
    # 分析每个涡结构
    vortex_structures = []
    
    for region_idx in range(1, num_regions + 1):
        # 获取当前区域的坐标
        region_coords = np.where(labeled_regions == region_idx)
        
        if len(region_coords[0]) > 0:
            # 计算涡核中心 (加权平均)
            weights = np.array([vortex_field[x, y, z] for x, y, z in zip(*region_coords)])
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                center_x = np.sum(region_coords[0] * weights) / total_weight
                center_y = np.sum(region_coords[1] * weights) / total_weight
                center_z = np.sum(region_coords[2] * weights) / total_weight
                
                # 计算涡结构大小 (估计为等效球体半径)
                size = np.power(len(region_coords[0]) * 3/4 / np.pi, 1/3)
                
                # 计算涡结构强度 (最大涡量)
                strength = np.max([vortex_field[x, y, z] for x, y, z in zip(*region_coords)])
                
                # 计算涡轴方向 (使用区域内平均涡量向量)
                avg_vort_x = np.mean([vort_x[x, y, z] for x, y, z in zip(*region_coords)])
                avg_vort_y = np.mean([vort_y[x, y, z] for x, y, z in zip(*region_coords)])
                avg_vort_z = np.mean([vort_z[x, y, z] for x, y, z in zip(*region_coords)])
                
                # 归一化方向向量
                norm = np.sqrt(avg_vort_x**2 + avg_vort_y**2 + avg_vort_z**2)
                if norm > 0:
                    orientation = [avg_vort_x/norm, avg_vort_y/norm, avg_vort_z/norm]
                else:
                    orientation = [0, 0, 1]  # 默认方向
                
                # 添加到结果列表
                vortex_structures.append({
                    "position": [float(center_x), float(center_y), float(center_z)],
                    "strength": float(strength),
                    "size": float(size),
                    "orientation": [float(orientation[0]), float(orientation[1]), float(orientation[2])]
                })
    
    return vortex_structures 

def analyze_turbulence(velocity_data, width, height, depth, region=None):
    """
    分析流体模拟中的湍流特性
    
    参数:
        velocity_data: 速度场数据，列表形式 [[u, v, w], ...]
        width, height, depth: 网格尺寸
        region: 分析区域，字典形式 {"x_min": x1, "y_min": y1, "z_min": z1, 
                                 "x_max": x2, "y_max": y2, "z_max": z2}
        
    返回:
        包含湍流强度、雷诺应力和能量谱的字典
    """
    import numpy as np
    from scipy import signal
    
    # 设置默认区域为整个计算域
    if region is None:
        region = {
            "x_min": 0, "y_min": 0, "z_min": 0,
            "x_max": width, "y_max": height, "z_max": depth
        }
    
    # 提取区域边界
    x_min = max(0, region["x_min"])
    y_min = max(0, region["y_min"])
    z_min = max(0, region["z_min"])
    x_max = min(width, region["x_max"])
    y_max = min(height, region["y_max"])
    z_max = min(depth, region["z_max"])
    
    # 将速度数据转换为3D NumPy数组
    u = np.zeros((width, height, depth))
    v = np.zeros((width, height, depth))
    w = np.zeros((width, height, depth))
    
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                idx = x + y * width + z * width * height
                if idx < len(velocity_data):
                    u[x, y, z] = velocity_data[idx][0]
                    v[x, y, z] = velocity_data[idx][1]
                    w[x, y, z] = velocity_data[idx][2]
    
    # 提取指定区域的速度数据
    u_region = u[x_min:x_max, y_min:y_max, z_min:z_max]
    v_region = v[x_min:x_max, y_min:y_max, z_min:z_max]
    w_region = w[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # 计算平均速度
    u_mean = np.mean(u_region)
    v_mean = np.mean(v_region)
    w_mean = np.mean(w_region)
    
    # 计算速度脉动
    u_fluc = u_region - u_mean
    v_fluc = v_region - v_mean
    w_fluc = w_region - w_mean
    
    # 计算湍流强度 (脉动速度RMS / 平均速度)
    u_rms = np.sqrt(np.mean(u_fluc**2))
    v_rms = np.sqrt(np.mean(v_fluc**2))
    w_rms = np.sqrt(np.mean(w_fluc**2))
    
    mean_velocity_mag = np.sqrt(u_mean**2 + v_mean**2 + w_mean**2)
    if mean_velocity_mag > 0:
        turbulence_intensity = (u_rms + v_rms + w_rms) / (3 * mean_velocity_mag)
    else:
        turbulence_intensity = 0
    
    # 计算雷诺应力张量
    uu = np.mean(u_fluc * u_fluc)
    uv = np.mean(u_fluc * v_fluc)
    uw = np.mean(u_fluc * w_fluc)
    vv = np.mean(v_fluc * v_fluc)
    vw = np.mean(v_fluc * w_fluc)
    ww = np.mean(w_fluc * w_fluc)
    
    reynolds_stresses = [
        [float(uu), float(uv), float(uw)],
        [float(uv), float(vv), float(vw)],
        [float(uw), float(vw), float(ww)]
    ]
    
    # 计算能量谱 (简化版，使用FFT)
    # 为简化计算，我们只计算x方向的1D能量谱
    energy_spectrum = {}
    
    # 确保有足够的点进行FFT
    if x_max - x_min > 4:
        # 对每个y,z位置计算x方向的能量谱，然后平均
        spectra = []
        
        for y_idx in range(y_max - y_min):
            for z_idx in range(z_max - z_min):
                # 获取一维速度分布
                u_line = u_fluc[:, y_idx, z_idx]
                v_line = v_fluc[:, y_idx, z_idx]
                w_line = w_fluc[:, y_idx, z_idx]
                
                # 计算功率谱密度
                f_u, psd_u = signal.welch(u_line, nperseg=min(len(u_line), 8))
                f_v, psd_v = signal.welch(v_line, nperseg=min(len(v_line), 8))
                f_w, psd_w = signal.welch(w_line, nperseg=min(len(w_line), 8))
                
                # 合并三个方向的谱
                psd_total = psd_u + psd_v + psd_w
                spectra.append((f_u, psd_total))
        
        # 平均所有位置的谱
        if spectra:
            # 使用第一个谱的频率作为基准
            freqs = spectra[0][0]
            avg_psd = np.zeros_like(spectra[0][1])
            
            for _, psd in spectra:
                avg_psd += psd
            
            avg_psd /= len(spectra)
            
            # 转换为列表格式
            energy_spectrum = {
                "frequencies": freqs.tolist(),
                "energy_density": avg_psd.tolist()
            }
    else:
        # 如果点数不足，返回空谱
        energy_spectrum = {
            "frequencies": [],
            "energy_density": []
        }
    
    return {
        "turbulence_intensity": float(turbulence_intensity),
        "reynolds_stresses": reynolds_stresses,
        "energy_spectrum": energy_spectrum
    } 