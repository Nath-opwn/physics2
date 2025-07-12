// 全局变量
const API_URL = 'http://localhost:8000';
let token = localStorage.getItem('token');
let currentSession = null;
let simulationRunning = false;
let ws = null;

// 3D渲染相关变量
let scene, camera, renderer, controls;
let particles, particleGeometry, particleMaterial;
let isThreeDInitialized = false;
let currentVisType = 'velocity';

// 图表相关变量
let analysisChart = null;
let chartData = {
    velocity: [],
    pressure: [],
    vorticity: [],
    energy: []
};

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 检查登录状态
    checkLoginStatus();
    
    // 注册事件监听器
    registerEventListeners();
    
    // 初始化画布
    initCanvas();
    
    // 加载知识库分类
    loadKnowledgeCategories();
    
    // 初始化暗黑模式
    initDarkMode();
    
    // 初始化高级可视化
    initAdvancedVisualization();
    
    // 创建预设场景列表
    createPresetsList();
});

// 检查登录状态
function checkLoginStatus() {
    if (token) {
        // 显示主内容，隐藏登录表单
        document.getElementById('loginSection').classList.add('d-none');
        document.getElementById('mainContent').classList.remove('d-none');
        document.getElementById('loginNav').classList.add('d-none');
        document.getElementById('userNav').classList.remove('d-none');
        document.getElementById('logoutNav').classList.remove('d-none');
        
        // 加载知识库分类
        loadKnowledgeCategories();
    } else {
        // 显示登录表单，隐藏主内容
        document.getElementById('loginSection').classList.remove('d-none');
        document.getElementById('mainContent').classList.add('d-none');
        document.getElementById('loginNav').classList.remove('d-none');
        document.getElementById('userNav').classList.add('d-none');
        document.getElementById('logoutNav').classList.add('d-none');
    }
}

// 注册事件监听器
function registerEventListeners() {
    // 登录和注册标签切换
    document.getElementById('loginTab').addEventListener('click', function(e) {
        e.preventDefault();
        this.classList.add('active');
        document.getElementById('registerTab').classList.remove('active');
        document.getElementById('loginForm').classList.remove('d-none');
        document.getElementById('registerForm').classList.add('d-none');
    });
    
    document.getElementById('registerTab').addEventListener('click', function(e) {
        e.preventDefault();
        this.classList.add('active');
        document.getElementById('loginTab').classList.remove('active');
        document.getElementById('loginForm').classList.add('d-none');
        document.getElementById('registerForm').classList.remove('d-none');
    });
    
    // 登录表单提交
    document.getElementById('loginForm').addEventListener('submit', function(e) {
        e.preventDefault();
        login();
    });
    
    // 注册表单提交
    document.getElementById('registerForm').addEventListener('submit', function(e) {
        e.preventDefault();
        register();
    });
    
    // 退出登录
    document.getElementById('logoutBtn').addEventListener('click', function(e) {
        e.preventDefault();
        logout();
    });
    
    // 模拟表单提交
    document.getElementById('simulationForm').addEventListener('submit', function(e) {
        e.preventDefault();
        initializeSimulation();
    });
    
    // 模拟控制按钮
    document.getElementById('startBtn').addEventListener('click', startSimulation);
    document.getElementById('pauseBtn').addEventListener('click', pauseSimulation);
    document.getElementById('stopBtn').addEventListener('click', stopSimulation);
    document.getElementById('stepBtn').addEventListener('click', stepSimulation);
    
    // 数据探针表单提交
    document.getElementById('probeForm').addEventListener('submit', function(e) {
        e.preventDefault();
        probeData();
    });
    
    // 获取统计数据
    document.getElementById('statsBtn').addEventListener('click', getStatistics);
    
    // 搜索知识库
    document.getElementById('searchBtn').addEventListener('click', searchKnowledge);
    
    // 按Enter键搜索
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchKnowledge();
        }
    });
    
    // 视图切换
    document.getElementById('2dViewTab').addEventListener('click', function(e) {
        e.preventDefault();
        this.classList.add('active');
        document.getElementById('3dViewTab').classList.remove('active');
        document.getElementById('2dView').classList.add('active');
        document.getElementById('3dView').classList.remove('active');
    });
    
    document.getElementById('3dViewTab').addEventListener('click', function(e) {
        e.preventDefault();
        this.classList.add('active');
        document.getElementById('2dViewTab').classList.remove('active');
        document.getElementById('2dView').classList.remove('active');
        document.getElementById('3dView').classList.add('active');
        
        // 初始化3D视图（如果尚未初始化）
        if (!isThreeDInitialized && currentSession) {
            init3DView();
        }
    });
    
    // 可视化类型切换
    document.querySelectorAll('input[name="visualizationType"]').forEach(radio => {
        radio.addEventListener('change', function() {
            currentVisType = this.value;
            if (currentSession) {
                updateParticleColors();
            }
        });
    });

    // 力的大小滑块
    document.getElementById('forceMagnitude').addEventListener('input', function() {
        document.getElementById('forceMagnitudeValue').textContent = this.value;
    });

    // 添加力表单提交
    document.getElementById('forceForm').addEventListener('submit', function(e) {
        e.preventDefault();
        addForce();
    });

    // 添加障碍物按钮
    document.getElementById('addObstacleBtn').addEventListener('click', addObstacle);
    
    // 清除障碍物按钮
    document.getElementById('clearObstaclesBtn').addEventListener('click', clearObstacles);
    
    // 设置预设模拟按钮
    document.getElementById('setupPresetBtn').addEventListener('click', setupPresetSimulation);

    // 模板选择变化
    document.getElementById('templateSelect').addEventListener('change', function() {
        applyTemplate(this.value);
    });

    // 图表标签切换
    document.getElementById('velocityChartTab').addEventListener('click', function(e) {
        e.preventDefault();
        setActiveChartTab(this, 'velocity');
    });
    
    document.getElementById('pressureChartTab').addEventListener('click', function(e) {
        e.preventDefault();
        setActiveChartTab(this, 'pressure');
    });
    
    document.getElementById('vorticityChartTab').addEventListener('click', function(e) {
        e.preventDefault();
        setActiveChartTab(this, 'vorticity');
    });
    
    document.getElementById('energyChartTab').addEventListener('click', function(e) {
        e.preventDefault();
        setActiveChartTab(this, 'energy');
    });
    
    // 时间序列分析按钮
    document.getElementById('timeSeriesBtn').addEventListener('click', getTimeSeriesData);
    
    // 导出数据按钮
    document.getElementById('exportDataBtn').addEventListener('click', exportData);
}

// 应用模板
function applyTemplate(templateName) {
    // 如果是自定义，不做任何改变
    if (templateName === 'custom') {
        return;
    }
    
    // 定义模板参数
    const templates = {
        waterTank: {
            width: 100,
            height: 50,
            depth: 100,
            viscosity: 0.05,
            density: 1.0,
            boundaryType: 0
        },
        smoke: {
            width: 80,
            height: 120,
            depth: 80,
            viscosity: 0.01,
            density: 0.3,
            boundaryType: 2
        },
        vortex: {
            width: 64,
            height: 64,
            depth: 64,
            viscosity: 0.001,
            density: 1.0,
            boundaryType: 1
        },
        fountain: {
            width: 80,
            height: 100,
            depth: 80,
            viscosity: 0.02,
            density: 1.0,
            boundaryType: 0
        },
        cylinder_flow: {
            width: 120,
            height: 60,
            depth: 60,
            viscosity: 0.01,
            density: 1.0,
            boundaryType: 0
        },
        karman_vortex: {
            width: 150,
            height: 60,
            depth: 60,
            viscosity: 0.005,
            density: 1.0,
            boundaryType: 2
        },
        channel_flow: {
            width: 120,
            height: 60,
            depth: 40,
            viscosity: 0.02,
            density: 1.0,
            boundaryType: 0
        },
        lid_driven_cavity: {
            width: 64,
            height: 64,
            depth: 64,
            viscosity: 0.03,
            density: 1.0,
            boundaryType: 0
        }
    };
    
    // 获取选定的模板
    const template = templates[templateName];
    if (!template) return;
    
    // 应用模板参数到表单
    document.getElementById('width').value = template.width;
    document.getElementById('height').value = template.height;
    document.getElementById('depth').value = template.depth;
    document.getElementById('viscosity').value = template.viscosity;
    document.getElementById('density').value = template.density;
    document.getElementById('boundaryType').value = template.boundaryType;
}

// 登录
async function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    try {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        
        const response = await fetch(`${API_URL}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            token = data.access_token;
            localStorage.setItem('token', token);
            checkLoginStatus();
        } else {
            const error = await response.json();
            alert(`登录失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('登录错误:', error);
        alert('登录失败，请检查网络连接');
    }
}

// 注册
async function register() {
    const username = document.getElementById('regUsername').value;
    const email = document.getElementById('regEmail').value;
    const password = document.getElementById('regPassword').value;
    
    try {
        const response = await fetch(`${API_URL}/api/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                email,
                password
            })
        });
        
        if (response.ok) {
            alert('注册成功，请登录');
            // 切换到登录表单
            document.getElementById('loginTab').click();
        } else {
            const error = await response.json();
            alert(`注册失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('注册错误:', error);
        alert('注册失败，请检查网络连接');
    }
}

// 退出登录
function logout() {
    token = null;
    localStorage.removeItem('token');
    checkLoginStatus();
    
    // 关闭WebSocket连接
    if (ws) {
        ws.close();
        ws = null;
    }
    
    // 重置模拟状态
    currentSession = null;
    simulationRunning = false;
    document.getElementById('currentSimulation').textContent = '无';
    document.getElementById('currentStep').textContent = '0';
    
    // 禁用按钮
    document.getElementById('startBtn').disabled = true;
    document.getElementById('pauseBtn').disabled = true;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('stepBtn').disabled = true;
    document.getElementById('probeBtn').disabled = true;
    document.getElementById('statsBtn').disabled = true;

    // 销毁3D场景
    if (isThreeDInitialized) {
        if (renderer) {
            renderer.dispose();
        }
        if (scene) {
            scene.traverse(function(object) {
                if (object.isMesh) {
                    object.geometry.dispose();
                    object.material.dispose();
                }
            });
            scene.remove(particles);
            particles = null;
            particleGeometry = null;
            particleMaterial = null;
        }
        isThreeDInitialized = false;
    }
}

// 初始化模拟
async function initializeSimulation() {
    if (!token) {
        alert('请先登录');
        return;
    }
    
    const name = document.getElementById('simName').value;
    const template = document.getElementById('templateSelect').value;
    const width = parseInt(document.getElementById('width').value);
    const height = parseInt(document.getElementById('height').value);
    const depth = parseInt(document.getElementById('depth').value);
    const viscosity = parseFloat(document.getElementById('viscosity').value);
    const density = parseFloat(document.getElementById('density').value);
    const boundaryType = parseInt(document.getElementById('boundaryType').value);
    
    try {
        const response = await fetch(`${API_URL}/api/simulation/initialize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                name,
                template,
                width,
                height,
                depth,
                viscosity,
                density,
                boundary_type: boundaryType
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            currentSession = data.session_id;
            document.getElementById('currentSimulation').textContent = name;
            
            // 启用控制按钮
            document.getElementById('startBtn').disabled = false;
            document.getElementById('pauseBtn').disabled = false;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('stepBtn').disabled = false;
            
            // 启用交互按钮
            document.getElementById('addForceBtn').disabled = false;
            document.getElementById('addObstacleBtn').disabled = false;
            document.getElementById('clearObstaclesBtn').disabled = false;
            document.getElementById('setupPresetBtn').disabled = false;
            
            // 启用分析按钮
            document.getElementById('probeBtn').disabled = false;
            document.getElementById('statsBtn').disabled = false;
            document.getElementById('timeSeriesBtn').disabled = false;
            document.getElementById('exportDataBtn').disabled = false;
            
            // 初始化3D视图
            if (document.getElementById('3dViewTab').classList.contains('active') && !isThreeDInitialized) {
                init3DView();
            }
            
            alert('模拟初始化成功');
        } else {
            const error = await response.json();
            alert(`初始化失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('初始化错误:', error);
        alert('初始化失败，请检查网络连接');
    }
}

// 开始模拟
function startSimulation() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    // 控制API
    controlSimulation('start');
    
    // 建立WebSocket连接
    connectWebSocket();
}

// 暂停模拟
function pauseSimulation() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    controlSimulation('pause');
    
    // 关闭WebSocket连接
    if (ws) {
        ws.close();
        ws = null;
    }
}

// 停止模拟
function stopSimulation() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    controlSimulation('stop');
    
    // 关闭WebSocket连接
    if (ws) {
        ws.close();
        ws = null;
    }
    
    // 重置步数
    document.getElementById('currentStep').textContent = '0';
}

// 单步模拟
async function stepSimulation() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/simulation/step`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: currentSession,
                dt: 0.01
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            document.getElementById('currentStep').textContent = data.step;
            
            // 更新画布
            updateCanvas(data.velocity, data.pressure);
        } else {
            const error = await response.json();
            alert(`单步模拟失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('单步模拟错误:', error);
        alert('单步模拟失败，请检查网络连接');
    }
}

// 控制模拟
async function controlSimulation(action) {
    try {
        const response = await fetch(`${API_URL}/api/simulation/control`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: currentSession,
                action: action
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log(`模拟${action}成功:`, data);
            simulationRunning = (action === 'start');
        } else {
            const error = await response.json();
            alert(`控制模拟失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('控制模拟错误:', error);
        alert('控制模拟失败，请检查网络连接');
    }
}

// 建立WebSocket连接
function connectWebSocket() {
    if (!currentSession) return;
    
    const wsUrl = `ws://localhost:8000/api/simulation/stream/${currentSession}`;
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket连接已建立');
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.error) {
            console.error('WebSocket错误:', data.error);
            return;
        }
        
        document.getElementById('currentStep').textContent = data.step;
        
        // 更新画布
        updateCanvas(data.velocity, data.pressure);
    };
    
    ws.onclose = function() {
        console.log('WebSocket连接已关闭');
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket错误:', error);
    };
}

// 初始化画布
function initCanvas() {
    const canvas = document.getElementById('simulationCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// 更新画布
function updateCanvas(velocity, pressure) {
    const canvas = document.getElementById('simulationCanvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // 清空画布
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, width, height);
    
    if (!velocity || !pressure) return;
    
    // 获取数据维度
    const depth = velocity.length;
    const rows = velocity[0].length;
    const cols = velocity[0][0].length;
    
    // 选择中间层显示
    const midDepth = Math.floor(depth / 2);
    
    // 计算单元格大小
    const cellWidth = width / cols;
    const cellHeight = height / rows;
    
    // 绘制压力场
    for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
            // 获取压力值
            const p = pressure[midDepth][y][x];
            
            // 映射到颜色
            const r = Math.min(255, Math.max(0, Math.floor(p * 255)));
            const b = Math.min(255, Math.max(0, Math.floor((1 - p) * 255)));
            
            // 设置填充颜色
            ctx.fillStyle = `rgb(${r}, 0, ${b})`;
            
            // 绘制单元格
            ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
        }
    }
    
    // 绘制速度场
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    
    for (let y = 0; y < rows; y += 2) {
        for (let x = 0; x < cols; x += 2) {
            // 获取速度分量
            const u = velocity[midDepth][y][x][0];
            const v = velocity[midDepth][y][x][1];
            
            // 计算速度大小和方向
            const speed = Math.sqrt(u*u + v*v);
            const scale = 5; // 缩放因子
            
            // 计算起点和终点
            const startX = x * cellWidth + cellWidth / 2;
            const startY = y * cellHeight + cellHeight / 2;
            const endX = startX + u * scale;
            const endY = startY + v * scale;
            
            // 绘制速度向量
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
            ctx.stroke();
        }
    }

    // 如果3D视图已初始化，更新粒子颜色
    if (isThreeDInitialized) {
        updateParticleColors();
    }
}

// 数据探针
async function probeData() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    const probeData = {
        session_id: currentSession,
        x: parseInt(document.getElementById('probeX').value),
        y: parseInt(document.getElementById('probeY').value),
        z: parseInt(document.getElementById('probeZ').value)
    };
    
    try {
        const response = await fetch(`${API_URL}/api/simulation/probe`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(probeData)
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // 更新探针结果
            const resultsDiv = document.getElementById('probeResults');
            resultsDiv.innerHTML = `
                <p>位置: (${data.position.x}, ${data.position.y}, ${data.position.z})</p>
                <p>速度: [${data.velocity.map(v => v.toFixed(4)).join(', ')}]</p>
                <p>压力: ${data.pressure.toFixed(4)}</p>
            `;
        } else {
            const error = await response.json();
            alert(`获取探针数据失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('探针数据错误:', error);
        alert('获取探针数据失败，请检查网络连接');
    }
}

// 获取统计数据
async function getStatistics() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/analysis/statistics?session_id=${currentSession}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // 更新图表数据
            chartData.velocity = data.velocity_histogram || [];
            chartData.pressure = data.pressure_histogram || [];
            chartData.vorticity = data.vorticity_histogram || [];
            
            // 显示当前活动标签的图表
            const activeTab = document.querySelector('#chartTabs .nav-link.active');
            if (activeTab.id === 'velocityChartTab') {
                updateChart('velocity');
            } else if (activeTab.id === 'pressureChartTab') {
                updateChart('pressure');
            } else if (activeTab.id === 'vorticityChartTab') {
                updateChart('vorticity');
            } else if (activeTab.id === 'energyChartTab') {
                updateChart('energy');
            }
            
            // 显示统计结果
            document.getElementById('statsResult').innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>速度统计</h6>
                        <p>平均值: ${data.velocity_mean.toFixed(4)}</p>
                        <p>最大值: ${data.velocity_max.toFixed(4)}</p>
                    </div>
                    <div class="col-md-6">
                        <h6>压力统计</h6>
                        <p>平均值: ${data.pressure_mean.toFixed(4)}</p>
                        <p>最小值: ${data.pressure_min.toFixed(4)}</p>
                        <p>最大值: ${data.pressure_max.toFixed(4)}</p>
                    </div>
                </div>
            `;
        } else {
            const error = await response.json();
            alert(`获取统计数据失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('统计数据错误:', error);
        alert('获取统计数据失败，请检查网络连接');
    }
}

// 加载知识库分类
async function loadKnowledgeCategories() {
    try {
        const response = await fetch(`${API_URL}/api/knowledge/categories`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const categories = await response.json();
            
            // 更新分类列表
            const categoryList = document.getElementById('categoryList');
            categoryList.innerHTML = '';
            
            categories.forEach(category => {
                const item = document.createElement('a');
                item.href = '#';
                item.className = 'list-group-item list-group-item-action';
                item.textContent = category.name;
                item.dataset.categoryId = category.id;
                item.addEventListener('click', function(e) {
                    e.preventDefault();
                    loadKnowledgeByCategory(category.id);
                    
                    // 高亮选中的分类
                    document.querySelectorAll('#categoryList a').forEach(a => {
                        a.classList.remove('active');
                    });
                    item.classList.add('active');
                });
                categoryList.appendChild(item);
            });
            
            // 同时更新搜索下拉框
            const searchCategory = document.getElementById('searchCategory');
            searchCategory.innerHTML = '<option value="">所有分类</option>';
            
            categories.forEach(category => {
                const option = document.createElement('option');
                option.value = category.id;
                option.textContent = category.name;
                searchCategory.appendChild(option);
            });
        } else {
            console.error('加载知识分类失败');
        }
    } catch (error) {
        console.error('加载知识分类错误:', error);
    }
}

// 按分类加载知识内容
async function loadKnowledgeByCategory(category) {
    try {
        const response = await fetch(`${API_URL}/api/knowledge/content?category=${category}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const content = await response.json();
            displayKnowledgeContent(content);
        } else {
            console.error('加载知识内容失败');
        }
    } catch (error) {
        console.error('加载知识内容错误:', error);
    }
}

// 搜索知识库
async function searchKnowledge() {
    const query = document.getElementById('searchInput').value.trim();
    const category = document.getElementById('searchCategory').value;
    const tags = document.getElementById('searchTags').value.trim();
    
    if (!query && !category && !tags) {
        alert('请输入搜索关键词、选择分类或输入标签');
        return;
    }
    
    try {
        // 构建查询URL
        let url = `${API_URL}/api/knowledge/search?`;
        
        if (query) {
            url += `query=${encodeURIComponent(query)}`;
        }
        
        if (category) {
            url += `&category=${encodeURIComponent(category)}`;
        }
        
        if (tags) {
            url += `&tags=${encodeURIComponent(tags)}`;
        }
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const results = await response.json();
            displaySearchResults(results);
        } else {
            console.error('搜索知识库失败');
        }
    } catch (error) {
        console.error('搜索知识库错误:', error);
    }
}

// 显示搜索结果
function displaySearchResults(items) {
    const contentDiv = document.getElementById('knowledgeContent');
    
    if (!items || items.length === 0) {
        contentDiv.innerHTML = '<div class="alert alert-warning">未找到相关内容</div>';
        return;
    }
    
    contentDiv.innerHTML = `<h4>搜索结果 (${items.length})</h4>`;
    
    items.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'card mb-3';
        
        // 将标签转换为徽章
        const tagHtml = item.tags.split(',').map(tag => 
            `<span class="badge bg-secondary me-1">${tag.trim()}</span>`
        ).join('');
        
        itemDiv.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">${item.title}</h5>
                <span class="badge bg-primary">${item.category}</span>
            </div>
            <div class="card-body">
                <p>${item.snippet}</p>
                <div class="mt-2">${tagHtml}</div>
                <button class="btn btn-sm btn-outline-primary mt-2 view-full-content" data-id="${item.id}">查看全文</button>
            </div>
        `;
        
        contentDiv.appendChild(itemDiv);
    });
    
    // 为"查看全文"按钮添加事件监听器
    document.querySelectorAll('.view-full-content').forEach(button => {
        button.addEventListener('click', function() {
            const id = this.dataset.id;
            loadKnowledgeItem(id);
        });
    });
}

// 显示知识内容
function displayKnowledgeContent(items) {
    const contentDiv = document.getElementById('knowledgeContent');
    
    if (!items || items.length === 0) {
        contentDiv.innerHTML = '<div class="alert alert-warning">未找到相关内容</div>';
        return;
    }
    
    contentDiv.innerHTML = '';
    
    items.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'card mb-3';
        
        // 将标签转换为徽章
        const tagHtml = item.tags.split(',').map(tag => 
            `<span class="badge bg-secondary me-1">${tag.trim()}</span>`
        ).join('');
        
        // 使用Markdown渲染内容
        const contentHtml = renderMarkdown(item.content);
        
        itemDiv.innerHTML = `
            <div class="card-header">
                <h5 class="mb-0">${item.title}</h5>
            </div>
            <div class="card-body">
                <div class="knowledge-content">${contentHtml}</div>
                <div class="mt-3">${tagHtml}</div>
            </div>
        `;
        
        contentDiv.appendChild(itemDiv);
    });
}

// 加载单个知识条目
async function loadKnowledgeItem(id) {
    try {
        const response = await fetch(`${API_URL}/api/knowledge/content/${id}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const item = await response.json();
            
            // 显示单个条目
            const contentDiv = document.getElementById('knowledgeContent');
            
            // 将标签转换为徽章
            const tagHtml = item.tags.split(',').map(tag => 
                `<span class="badge bg-secondary me-1">${tag.trim()}</span>`
            ).join('');
            
            // 使用Markdown渲染内容
            const contentHtml = renderMarkdown(item.content);
            
            contentDiv.innerHTML = `
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">${item.title}</h5>
                        <span class="badge bg-primary">${item.category}</span>
                    </div>
                    <div class="card-body">
                        <div class="knowledge-content">${contentHtml}</div>
                        <div class="mt-3">${tagHtml}</div>
                    </div>
                </div>
                <button class="btn btn-secondary mt-3" id="backToResults">返回</button>
            `;
            
            // 添加返回按钮事件
            document.getElementById('backToResults').addEventListener('click', function() {
                history.back();
            });
        } else {
            console.error('加载知识条目失败');
        }
    } catch (error) {
        console.error('加载知识条目错误:', error);
    }
}

// 简单的Markdown渲染函数
function renderMarkdown(markdown) {
    if (!markdown) return '';
    
    // 处理标题
    let html = markdown
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
        .replace(/^##### (.*$)/gm, '<h5>$1</h5>')
        .replace(/^###### (.*$)/gm, '<h6>$1</h6>');
    
    // 处理粗体
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 处理斜体
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // 处理列表
    html = html.replace(/^\s*-\s*(.*$)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // 处理段落
    html = html.replace(/^(?!<[uh]).+$/gm, function(m) {
        return '<p>' + m + '</p>';
    });
    
    // 处理换行
    html = html.replace(/\n/g, '');
    
    return html;
}

// 初始化3D视图
function init3DView() {
    const container = document.getElementById('threeDContainer');
    
    // 创建场景
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // 创建相机
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 100;
    
    // 创建渲染器
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // 添加轨道控制
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // 添加灯光
    const ambientLight = new THREE.AmbientLight(0xcccccc, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // 创建粒子系统
    createParticleSystem();
    
    // 标记为已初始化
    isThreeDInitialized = true;
    
    // 开始动画循环
    animate();
}

// 创建粒子系统
function createParticleSystem() {
    // 获取模拟参数
    const width = parseInt(document.getElementById('width').value) || 50;
    const height = parseInt(document.getElementById('height').value) || 50;
    const depth = parseInt(document.getElementById('depth').value) || 50;
    
    // 创建粒子
    const particleCount = Math.min(width * height * depth / 8, 10000); // 限制粒子数量
    particleGeometry = new THREE.BufferGeometry();
    
    // 随机分布粒子
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
        // 位置
        positions[i * 3] = (Math.random() - 0.5) * width;
        positions[i * 3 + 1] = (Math.random() - 0.5) * height;
        positions[i * 3 + 2] = (Math.random() - 0.5) * depth;
        
        // 颜色 (默认白色)
        colors[i * 3] = 1.0;
        colors[i * 3 + 1] = 1.0;
        colors[i * 3 + 2] = 1.0;
        
        // 大小
        sizes[i] = 2.0;
    }
    
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    // 创建材质
    particleMaterial = new THREE.PointsMaterial({
        size: 1,
        vertexColors: true,
        transparent: true,
        opacity: 0.8
    });
    
    // 创建粒子系统
    particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);
}

// 更新粒子颜色
function updateParticleColors() {
    if (!particles || !currentSession) return;
    
    // 获取当前数据
    fetch(`${API_URL}/api/simulation/region`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
            session_id: currentSession,
            x_min: 0,
            y_min: 0,
            z_min: 0,
            x_max: parseInt(document.getElementById('width').value) || 50,
            y_max: parseInt(document.getElementById('height').value) || 50,
            z_max: parseInt(document.getElementById('depth').value) || 50
        })
    })
    .then(response => response.json())
    .then(data => {
        const colors = particleGeometry.attributes.color.array;
        const positions = particleGeometry.attributes.position.array;
        const particleCount = positions.length / 3;
        
        // 根据可视化类型更新颜色
        for (let i = 0; i < particleCount; i++) {
            // 获取粒子位置
            const x = positions[i * 3];
            const y = positions[i * 3 + 1];
            const z = positions[i * 3 + 2];
            
            // 根据位置查找最近的数据点
            // 简化版：使用数据的平均值
            let value = 0;
            
            if (currentVisType === 'velocity') {
                // 速度场可视化 - 使用速度大小
                const velocityMean = data.statistics.velocity_mean || 0.1;
                value = Math.random() * velocityMean; // 简化版，实际应该使用插值
            } else if (currentVisType === 'pressure') {
                // 压力场可视化
                const pressureMin = data.statistics.pressure_min || -1;
                const pressureMax = data.statistics.pressure_max || 1;
                const pressureRange = pressureMax - pressureMin;
                value = (Math.random() * pressureRange + pressureMin) / pressureRange; // 归一化
            } else if (currentVisType === 'vorticity') {
                // 涡量场可视化
                value = Math.random(); // 简化版
            }
            
            // 设置颜色 (使用热力图颜色映射)
            if (value < 0.33) {
                // 蓝到青
                colors[i * 3] = 0;
                colors[i * 3 + 1] = value * 3;
                colors[i * 3 + 2] = 1;
            } else if (value < 0.66) {
                // 青到黄
                colors[i * 3] = (value - 0.33) * 3;
                colors[i * 3 + 1] = 1;
                colors[i * 3 + 2] = 1 - (value - 0.33) * 3;
            } else {
                // 黄到红
                colors[i * 3] = 1;
                colors[i * 3 + 1] = 1 - (value - 0.66) * 3;
                colors[i * 3 + 2] = 0;
            }
        }
        
        // 更新颜色属性
        particleGeometry.attributes.color.needsUpdate = true;
    })
    .catch(error => console.error('获取区域数据失败:', error));
}

// 动画循环
function animate() {
    requestAnimationFrame(animate);
    
    if (controls) {
        controls.update();
    }
    
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
} 

// 添加力
async function addForce() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    const x = parseInt(document.getElementById('forceX').value);
    const y = parseInt(document.getElementById('forceY').value);
    const z = parseInt(document.getElementById('forceZ').value);
    
    const dirX = parseFloat(document.getElementById('forceDirX').value);
    const dirY = parseFloat(document.getElementById('forceDirY').value);
    const dirZ = parseFloat(document.getElementById('forceDirZ').value);
    
    const magnitude = parseFloat(document.getElementById('forceMagnitude').value);
    
    // 计算力的分量
    const fx = dirX * magnitude;
    const fy = dirY * magnitude;
    const fz = dirZ * magnitude;
    
    try {
        const response = await fetch(`${API_URL}/api/simulation/force`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: currentSession,
                x: x,
                y: y,
                z: z,
                fx: fx,
                fy: fy,
                fz: fz
            })
        });
        
        if (response.ok) {
            console.log('力添加成功');
        } else {
            const error = await response.json();
            alert(`添加力失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('添加力错误:', error);
        alert('添加力失败，请检查网络连接');
    }
}

// 添加障碍物
async function addObstacle() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    // 创建一个简单的对话框来选择障碍物类型和参数
    const obstacleType = prompt('请选择障碍物类型 (sphere, cylinder, box):', 'sphere');
    if (!obstacleType) return;
    
    let params = {};
    
    if (obstacleType === 'sphere') {
        const x = parseInt(prompt('中心 X 坐标:', Math.floor(document.getElementById('width').value / 2)));
        const y = parseInt(prompt('中心 Y 坐标:', Math.floor(document.getElementById('height').value / 2)));
        const z = parseInt(prompt('中心 Z 坐标:', Math.floor(document.getElementById('depth').value / 2)));
        const radius = parseInt(prompt('半径:', 5));
        
        params = {
            center: [x, y, z],
            radius: radius
        };
    } else if (obstacleType === 'cylinder') {
        const x = parseInt(prompt('中心 X 坐标:', Math.floor(document.getElementById('width').value / 2)));
        const y = parseInt(prompt('中心 Y 坐标:', Math.floor(document.getElementById('height').value / 2)));
        const z = parseInt(prompt('中心 Z 坐标:', Math.floor(document.getElementById('depth').value / 2)));
        const radius = parseInt(prompt('半径:', 5));
        const height = parseInt(prompt('高度:', 20));
        const axis = prompt('轴向 (x, y, z):', 'y');
        
        params = {
            center: [x, y, z],
            radius: radius,
            height: height,
            axis: axis
        };
    } else if (obstacleType === 'box') {
        const x1 = parseInt(prompt('最小 X 坐标:', 10));
        const y1 = parseInt(prompt('最小 Y 坐标:', 10));
        const z1 = parseInt(prompt('最小 Z 坐标:', 10));
        const x2 = parseInt(prompt('最大 X 坐标:', 20));
        const y2 = parseInt(prompt('最大 Y 坐标:', 20));
        const z2 = parseInt(prompt('最大 Z 坐标:', 20));
        
        params = {
            min: [x1, y1, z1],
            max: [x2, y2, z2]
        };
    } else {
        alert('不支持的障碍物类型');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/simulation/obstacle`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: currentSession,
                shape: obstacleType,
                params: params
            })
        });
        
        if (response.ok) {
            alert('障碍物添加成功');
        } else {
            const error = await response.json();
            alert(`添加障碍物失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('添加障碍物错误:', error);
        alert('添加障碍物失败，请检查网络连接');
    }
}

// 清除障碍物
async function clearObstacles() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    if (!confirm('确定要清除所有障碍物吗？')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/simulation/obstacle/${currentSession}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            alert('障碍物已清除');
        } else {
            const error = await response.json();
            alert(`清除障碍物失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('清除障碍物错误:', error);
        alert('清除障碍物失败，请检查网络连接');
    }
}

// 设置预设模拟
async function setupPresetSimulation() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    const presetType = document.getElementById('templateSelect').value;
    if (presetType === 'custom') {
        alert('请选择一个预设模拟类型');
        return;
    }
    
    // 检查是否是新增的预设类型
    const newPresets = ['cylinder_flow', 'karman_vortex', 'channel_flow', 'lid_driven_cavity'];
    if (!newPresets.includes(presetType)) {
        alert('此模板不支持直接设置预设模拟，请使用初始化功能');
        return;
    }
    
    // 获取参数
    let params = {};
    if (presetType === 'cylinder_flow' || presetType === 'karman_vortex') {
        const velocity = parseFloat(prompt('入口速度:', '1.0'));
        if (velocity) {
            params.inlet_velocity = velocity;
        }
    } else if (presetType === 'channel_flow') {
        const velocity = parseFloat(prompt('入口速度:', '1.0'));
        const thickness = parseInt(prompt('壁厚:', '5'));
        if (velocity) params.inlet_velocity = velocity;
        if (thickness) params.wall_thickness = thickness;
    } else if (presetType === 'lid_driven_cavity') {
        const velocity = parseFloat(prompt('盖板速度:', '1.0'));
        if (velocity) params.lid_velocity = velocity;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/simulation/preset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: currentSession,
                preset_type: presetType,
                params: params
            })
        });
        
        if (response.ok) {
            alert(`预设模拟 ${presetType} 已设置`);
            // 重置步数显示
            document.getElementById('currentStep').textContent = '0';
        } else {
            const error = await response.json();
            alert(`设置预设模拟失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('设置预设模拟错误:', error);
        alert('设置预设模拟失败，请检查网络连接');
    }
}

// 设置活动的图表标签
function setActiveChartTab(tab, chartType) {
    // 移除所有标签的active类
    document.querySelectorAll('#chartTabs .nav-link').forEach(item => {
        item.classList.remove('active');
    });
    
    // 添加active类到当前标签
    tab.classList.add('active');
    
    // 更新图表
    updateChart(chartType);
}

// 更新图表
function updateChart(chartType) {
    const ctx = document.getElementById('analysisChart').getContext('2d');
    
    // 如果图表已存在，销毁它
    if (analysisChart) {
        analysisChart.destroy();
    }
    
    // 准备图表数据
    let data = [];
    let labels = [];
    let title = '';
    let color = '';
    
    switch (chartType) {
        case 'velocity':
            data = chartData.velocity;
            title = '速度分布';
            color = 'rgba(54, 162, 235, 0.6)';
            break;
        case 'pressure':
            data = chartData.pressure;
            title = '压力分布';
            color = 'rgba(255, 99, 132, 0.6)';
            break;
        case 'vorticity':
            data = chartData.vorticity;
            title = '涡量分布';
            color = 'rgba(75, 192, 192, 0.6)';
            break;
        case 'energy':
            data = chartData.energy;
            title = '能量变化';
            color = 'rgba(153, 102, 255, 0.6)';
            break;
    }
    
    // 如果没有数据，显示提示信息
    if (data.length === 0) {
        // 创建空图表
        analysisChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['无数据'],
                datasets: [{
                    label: title,
                    data: [0],
                    backgroundColor: color
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: '请先获取统计数据'
                    }
                }
            }
        });
        return;
    }
    
    // 为图表创建标签
    for (let i = 0; i < data.length; i++) {
        labels.push(i.toString());
    }
    
    // 创建图表
    analysisChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: title,
                data: data,
                backgroundColor: color
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: title
                }
            }
        }
    });
}

// 获取时间序列数据
async function getTimeSeriesData() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/analysis/timeseries?session_id=${currentSession}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // 更新能量图表数据
            chartData.energy = data.energy_series || [];
            
            // 显示能量图表
            setActiveChartTab(document.getElementById('energyChartTab'), 'energy');
        } else {
            const error = await response.json();
            alert(`获取时间序列数据失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('获取时间序列数据错误:', error);
        alert('获取时间序列数据失败，请检查网络连接');
    }
}

// 导出数据
async function exportData() {
    if (!currentSession) {
        alert('请先初始化模拟');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/export/data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: currentSession,
                format: "json",
                include_velocity: true,
                include_pressure: true,
                include_vorticity: true
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            alert(`数据导出成功，文件路径: ${data.file_path}`);
            
            // 创建下载链接
            const downloadLink = document.createElement('a');
            downloadLink.href = `${API_URL}/api/export/download?file_path=${encodeURIComponent(data.file_path)}`;
            downloadLink.download = data.file_name;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        } else {
            const error = await response.json();
            alert(`导出数据失败: ${error.detail}`);
        }
    } catch (error) {
        console.error('导出数据错误:', error);
        alert('导出数据失败，请检查网络连接');
    }
} 

// 涡环碰撞模拟场景
function setupVortexRingCollision() {
    // 显示加载指示器
    showLoadingIndicator();
    
    // 准备参数
    const params = {
        grid_size: [128, 128, 128],
        ring1_center: [32, 64, 64],
        ring2_center: [96, 64, 64],
        ring_radius: 20,
        ring_thickness: 4,
        ring_strength: parseFloat(document.getElementById('vortexRingStrength').value || 1.0),
        viscosity: parseFloat(document.getElementById('vortexRingViscosity').value || 0.001),
        dt: 0.01
    };
    
    // 发送请求到后端
    fetch('/api/simulation/presets/vortex-ring-collision', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getToken()}`
        },
        body: JSON.stringify({
            session_id: currentSessionId,
            params: params
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('设置涡环碰撞场景失败');
        }
        return response.json();
    })
    .then(data => {
        console.log('涡环碰撞场景设置成功:', data);
        hideLoadingIndicator();
        showNotification('涡环碰撞场景设置成功', 'success');
        
        // 更新UI
        updateSimulationStatus('ready');
        document.getElementById('startSimulationBtn').disabled = false;
        
        // 重置可视化
        resetVisualization();
    })
    .catch(error => {
        console.error('设置涡环碰撞场景错误:', error);
        hideLoadingIndicator();
        showNotification(`设置涡环碰撞场景失败: ${error.message}`, 'error');
    });
}

// 在预设场景列表中添加涡环碰撞选项
function createPresetsList() {
    const presetsList = document.getElementById('presetsList');
    if (!presetsList) return;
    
    // 清空现有内容
    presetsList.innerHTML = '';
    
    // 定义预设场景
    const presets = [
        {
            id: 'cylinder-flow',
            name: '圆柱体绕流',
            description: '模拟流体绕过圆柱体的流动',
            setupFunction: setupCylinderFlow,
            configFields: [
                { id: 'cylinderRadius', label: '圆柱半径', type: 'range', min: 5, max: 30, value: 10, step: 1 },
                { id: 'cylinderViscosity', label: '流体粘度', type: 'range', min: 0.0001, max: 0.01, value: 0.001, step: 0.0001 }
            ]
        },
        {
            id: 'karman-vortex',
            name: '卡门涡街',
            description: '模拟卡门涡街现象',
            setupFunction: setupKarmanVortex,
            configFields: [
                { id: 'karmanObstacleSize', label: '障碍物尺寸', type: 'range', min: 5, max: 30, value: 10, step: 1 },
                { id: 'karmanVelocity', label: '入口速度', type: 'range', min: 0.1, max: 2, value: 1, step: 0.1 }
            ]
        },
        {
            id: 'lid-driven-cavity',
            name: '盖驱动腔流',
            description: '模拟盖驱动腔内的流动',
            setupFunction: setupLidDrivenCavity,
            configFields: [
                { id: 'lidVelocity', label: '盖板速度', type: 'range', min: 0.1, max: 2, value: 1, step: 0.1 },
                { id: 'lidViscosity', label: '流体粘度', type: 'range', min: 0.0001, max: 0.01, value: 0.001, step: 0.0001 }
            ]
        },
        {
            id: 'vortex-ring-collision',
            name: '涡环碰撞',
            description: '模拟两个涡环相撞的动态过程',
            setupFunction: setupVortexRingCollision,
            configFields: [
                { id: 'vortexRingStrength', label: '涡环强度', type: 'range', min: 0.1, max: 2, value: 1, step: 0.1 },
                { id: 'vortexRingViscosity', label: '流体粘度', type: 'range', min: 0.0001, max: 0.01, value: 0.001, step: 0.0001 }
            ]
        }
    ];
    
    // 创建预设场景卡片
    presets.forEach(preset => {
        const presetCard = document.createElement('div');
        presetCard.className = 'preset-card';
        presetCard.setAttribute('data-preset-id', preset.id);
        
        const presetTitle = document.createElement('h3');
        presetTitle.textContent = preset.name;
        
        const presetDesc = document.createElement('p');
        presetDesc.textContent = preset.description;
        
        const configContainer = document.createElement('div');
        configContainer.className = 'preset-config';
        
        // 添加配置字段
        preset.configFields.forEach(field => {
            const fieldContainer = document.createElement('div');
            fieldContainer.className = 'config-field';
            
            const label = document.createElement('label');
            label.setAttribute('for', field.id);
            label.textContent = field.label;
            
            const input = document.createElement('input');
            input.setAttribute('type', field.type);
            input.setAttribute('id', field.id);
            input.setAttribute('min', field.min);
            input.setAttribute('max', field.max);
            input.setAttribute('step', field.step);
            input.setAttribute('value', field.value);
            
            const valueDisplay = document.createElement('span');
            valueDisplay.className = 'range-value';
            valueDisplay.textContent = field.value;
            
            // 更新显示的值
            input.addEventListener('input', () => {
                valueDisplay.textContent = input.value;
            });
            
            fieldContainer.appendChild(label);
            fieldContainer.appendChild(input);
            fieldContainer.appendChild(valueDisplay);
            configContainer.appendChild(fieldContainer);
        });
        
        const applyBtn = document.createElement('button');
        applyBtn.className = 'btn btn-primary';
        applyBtn.textContent = '应用';
        applyBtn.addEventListener('click', () => {
            if (currentSessionId) {
                preset.setupFunction();
            } else {
                showNotification('请先创建或选择一个模拟会话', 'warning');
            }
        });
        
        presetCard.appendChild(presetTitle);
        presetCard.appendChild(presetDesc);
        presetCard.appendChild(configContainer);
        presetCard.appendChild(applyBtn);
        
        presetsList.appendChild(presetCard);
    });
} 

// 暗黑模式切换
function initDarkMode() {
    const darkModeSwitch = document.getElementById('darkModeSwitch');
    
    // 检查本地存储中的暗黑模式设置
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    
    // 应用初始设置
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        darkModeSwitch.checked = true;
    }
    
    // 监听切换事件
    darkModeSwitch.addEventListener('change', () => {
        if (darkModeSwitch.checked) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('darkMode', 'true');
        } else {
            document.body.classList.remove('dark-mode');
            localStorage.setItem('darkMode', 'false');
        }
    });
}

// 初始化高级可视化
function initAdvancedVisualization() {
    // 获取UI元素
    const advancedViewTab = document.getElementById('advancedViewTab');
    const advancedView = document.getElementById('advancedView');
    const particleToggle = document.getElementById('particleToggle');
    const streamlineToggle = document.getElementById('streamlineToggle');
    const particleCount = document.getElementById('particleCount');
    const particleSpeed = document.getElementById('particleSpeed');
    const streamlineCount = document.getElementById('streamlineCount');
    const streamlineLength = document.getElementById('streamlineLength');
    const regenerateStreamlines = document.getElementById('regenerateStreamlines');
    
    // 显示值标签
    const particleCountValue = document.getElementById('particleCountValue');
    const particleSpeedValue = document.getElementById('particleSpeedValue');
    const streamlineCountValue = document.getElementById('streamlineCountValue');
    const streamlineLengthValue = document.getElementById('streamlineLengthValue');
    
    // 初始化可视化
    let isVisualizationInitialized = false;
    
    // 切换到高级可视化标签时初始化
    advancedViewTab.addEventListener('click', (e) => {
        e.preventDefault();
        
        // 激活标签
        document.querySelectorAll('#viewTabs .nav-link').forEach(tab => {
            tab.classList.remove('active');
        });
        advancedViewTab.classList.add('active');
        
        // 显示内容
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        advancedView.classList.add('active');
        
        // 初始化可视化（如果尚未初始化）
        if (!isVisualizationInitialized && window.FluidVisualization) {
            window.FluidVisualization.init('advancedVisContainer');
            isVisualizationInitialized = true;
            
            // 如果有模拟数据，更新速度场
            if (currentSessionId) {
                fetchVelocityFieldForVisualization();
            }
        }
    });
    
    // 粒子切换
    particleToggle.addEventListener('change', () => {
        if (window.FluidVisualization) {
            window.FluidVisualization.toggleParticles(particleToggle.checked);
        }
    });
    
    // 流线切换
    streamlineToggle.addEventListener('change', () => {
        if (window.FluidVisualization) {
            window.FluidVisualization.toggleStreamlines(streamlineToggle.checked);
        }
    });
    
    // 粒子数量滑块
    particleCount.addEventListener('input', () => {
        particleCountValue.textContent = particleCount.value;
        // 粒子数量变化需要重新初始化粒子系统，暂不实现
    });
    
    // 粒子速度滑块
    particleSpeed.addEventListener('input', () => {
        particleSpeedValue.textContent = particleSpeed.value;
        // 可以在这里调整粒子动画速度，暂不实现
    });
    
    // 流线数量滑块
    streamlineCount.addEventListener('input', () => {
        streamlineCountValue.textContent = streamlineCount.value;
    });
    
    // 流线长度滑块
    streamlineLength.addEventListener('input', () => {
        streamlineLengthValue.textContent = streamlineLength.value;
    });
    
    // 重新生成流线按钮
    regenerateStreamlines.addEventListener('click', () => {
        if (window.FluidVisualization) {
            const count = parseInt(streamlineCount.value);
            const steps = parseInt(streamlineLength.value);
            window.FluidVisualization.generateStreamlines(count, steps);
        }
    });
}

// 获取速度场数据用于可视化
function fetchVelocityFieldForVisualization() {
    if (!currentSessionId) return;
    
    fetch(`/api/simulation/${currentSessionId}/data`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getToken()}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('获取模拟数据失败');
        }
        return response.json();
    })
    .then(data => {
        // 提取速度场数据
        const velocityField = {
            nx: data.dimensions.width,
            ny: data.dimensions.height,
            nz: data.dimensions.depth,
            u: data.velocity.u,
            v: data.velocity.v,
            w: data.velocity.w
        };
        
        // 更新可视化
        if (window.FluidVisualization) {
            window.FluidVisualization.updateVelocityField(velocityField);
        }
    })
    .catch(error => {
        console.error('获取速度场数据错误:', error);
    });
}

// 初始化函数
document.addEventListener('DOMContentLoaded', function() {
    // 检查登录状态
    checkLoginStatus();
    
    // 注册事件监听器
    registerEventListeners();
    
    // 初始化画布
    initCanvas();
    
    // 加载知识库分类
    loadKnowledgeCategories();
    
    // 初始化暗黑模式
    initDarkMode();
    
    // 初始化高级可视化
    initAdvancedVisualization();
    
    // 创建预设场景列表
    createPresetsList();
});

// 更新模拟数据时同步更新可视化
function updateSimulationData(data) {
    // 更新高级可视化
    if (window.FluidVisualization) {
        const velocityField = {
            nx: data.dimensions.width,
            ny: data.dimensions.height,
            nz: data.dimensions.depth,
            u: data.velocity.u,
            v: data.velocity.v,
            w: data.velocity.w
        };
        window.FluidVisualization.updateVelocityField(velocityField);
    }
} 