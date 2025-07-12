/**
 * 高级流体可视化模块
 * 提供流线和粒子追踪等高级可视化功能
 */

// 全局变量
let threeScene, threeCamera, threeRenderer;
let particleSystem, streamlines;
let velocityField = null;
let animationId = null;
let isAnimating = false;

// 初始化Three.js场景
function initVisualization(containerId) {
    // 创建场景
    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x111111);
    
    // 创建相机
    threeCamera = new THREE.PerspectiveCamera(
        60, // 视角
        1, // 宽高比（将在调整大小时更新）
        0.1, // 近平面
        1000 // 远平面
    );
    threeCamera.position.set(100, 100, 100);
    threeCamera.lookAt(0, 0, 0);
    
    // 创建渲染器
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    threeRenderer = new THREE.WebGLRenderer({ antialias: true });
    threeRenderer.setSize(width, height);
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    
    // 清除容器内容并添加渲染器
    container.innerHTML = '';
    container.appendChild(threeRenderer.domElement);
    
    // 添加轨道控制
    const controls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // 添加坐标轴辅助
    const axesHelper = new THREE.AxesHelper(50);
    threeScene.add(axesHelper);
    
    // 添加环境光和方向光
    const ambientLight = new THREE.AmbientLight(0x404040);
    threeScene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    threeScene.add(directionalLight);
    
    // 添加网格
    addGrid();
    
    // 初始化粒子系统
    initParticleSystem();
    
    // 初始化流线
    initStreamlines();
    
    // 渲染场景
    render();
    
    // 处理窗口大小变化
    window.addEventListener('resize', () => {
        if (!container) return;
        
        const newWidth = container.clientWidth;
        const newHeight = container.clientHeight;
        
        threeCamera.aspect = newWidth / newHeight;
        threeCamera.updateProjectionMatrix();
        
        threeRenderer.setSize(newWidth, newHeight);
    });
}

// 添加网格
function addGrid() {
    const gridHelper = new THREE.GridHelper(100, 10, 0x444444, 0x222222);
    threeScene.add(gridHelper);
    
    // 添加包围盒
    const boxGeometry = new THREE.BoxGeometry(100, 100, 100);
    const boxMaterial = new THREE.MeshBasicMaterial({
        color: 0x888888,
        wireframe: true,
        transparent: true,
        opacity: 0.1
    });
    const box = new THREE.Mesh(boxGeometry, boxMaterial);
    threeScene.add(box);
}

// 初始化粒子系统
function initParticleSystem() {
    const particleCount = 1000;
    const particles = new THREE.BufferGeometry();
    
    // 粒子位置
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    // 初始化粒子属性
    for (let i = 0; i < particleCount; i++) {
        // 随机位置
        positions[i * 3] = (Math.random() - 0.5) * 100;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 100;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 100;
        
        // 基于位置设置颜色（可视化效果）
        colors[i * 3] = 0.5 + positions[i * 3] / 200;
        colors[i * 3 + 1] = 0.5 + positions[i * 3 + 1] / 200;
        colors[i * 3 + 2] = 0.5 + positions[i * 3 + 2] / 200;
        
        // 粒子大小
        sizes[i] = 2.0;
    }
    
    particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particles.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    // 创建粒子材质
    const particleMaterial = new THREE.PointsMaterial({
        size: 2,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        sizeAttenuation: true
    });
    
    // 创建粒子系统
    particleSystem = new THREE.Points(particles, particleMaterial);
    particleSystem.visible = false; // 默认隐藏
    threeScene.add(particleSystem);
}

// 初始化流线
function initStreamlines() {
    streamlines = new THREE.Group();
    threeScene.add(streamlines);
}

// 更新速度场数据
function updateVelocityField(data) {
    velocityField = data;
}

// 生成流线
function generateStreamlines(count = 20, steps = 100, stepSize = 0.5) {
    // 清除现有流线
    while (streamlines.children.length > 0) {
        streamlines.remove(streamlines.children[0]);
    }
    
    if (!velocityField) return;
    
    const { nx, ny, nz, u, v, w } = velocityField;
    
    for (let i = 0; i < count; i++) {
        // 随机起点
        let x = Math.floor(Math.random() * nx);
        let y = Math.floor(Math.random() * ny);
        let z = Math.floor(Math.random() * nz);
        
        const points = [];
        points.push(new THREE.Vector3(x, y, z));
        
        // 追踪流线
        for (let step = 0; step < steps; step++) {
            // 获取当前位置的速度
            const vx = interpolateVelocity(x, y, z, u, nx, ny, nz);
            const vy = interpolateVelocity(x, y, z, v, nx, ny, nz);
            const vz = interpolateVelocity(x, y, z, w, nx, ny, nz);
            
            // 速度太小则停止
            const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            if (speed < 0.01) break;
            
            // 更新位置
            x += vx * stepSize;
            y += vy * stepSize;
            z += vz * stepSize;
            
            // 检查是否超出边界
            if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) break;
            
            // 添加点
            points.push(new THREE.Vector3(x, y, z));
        }
        
        // 如果流线太短，跳过
        if (points.length < 5) continue;
        
        // 创建流线几何体
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        // 创建流线材质
        const material = new THREE.LineBasicMaterial({
            color: new THREE.Color(Math.random(), Math.random(), Math.random()),
            linewidth: 1
        });
        
        // 创建流线
        const line = new THREE.Line(geometry, material);
        streamlines.add(line);
    }
}

// 在速度场中插值获取速度
function interpolateVelocity(x, y, z, velocityComponent, nx, ny, nz) {
    // 简单的最近邻插值
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const zi = Math.floor(z);
    
    if (xi < 0 || xi >= nx - 1 || yi < 0 || yi >= ny - 1 || zi < 0 || zi >= nz - 1) {
        return 0;
    }
    
    // 获取索引
    const idx = (zi * ny + yi) * nx + xi;
    return velocityComponent[idx];
}

// 更新粒子位置
function updateParticles(deltaTime = 0.1) {
    if (!particleSystem || !velocityField) return;
    
    const { nx, ny, nz, u, v, w } = velocityField;
    
    const positions = particleSystem.geometry.attributes.position.array;
    const colors = particleSystem.geometry.attributes.color.array;
    const count = positions.length / 3;
    
    for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        
        // 当前位置
        let x = positions[i3];
        let y = positions[i3 + 1];
        let z = positions[i3 + 2];
        
        // 归一化到网格坐标
        const gridX = (x + 50) * (nx / 100);
        const gridY = (y + 50) * (ny / 100);
        const gridZ = (z + 50) * (nz / 100);
        
        // 获取速度
        const vx = interpolateVelocity(gridX, gridY, gridZ, u, nx, ny, nz);
        const vy = interpolateVelocity(gridX, gridY, gridZ, v, nx, ny, nz);
        const vz = interpolateVelocity(gridX, gridY, gridZ, w, nx, ny, nz);
        
        // 更新位置
        positions[i3] += vx * deltaTime;
        positions[i3 + 1] += vy * deltaTime;
        positions[i3 + 2] += vz * deltaTime;
        
        // 检查边界并重置
        if (positions[i3] < -50 || positions[i3] > 50 ||
            positions[i3 + 1] < -50 || positions[i3 + 1] > 50 ||
            positions[i3 + 2] < -50 || positions[i3 + 2] > 50) {
            positions[i3] = (Math.random() - 0.5) * 100;
            positions[i3 + 1] = (Math.random() - 0.5) * 100;
            positions[i3 + 2] = (Math.random() - 0.5) * 100;
        }
        
        // 更新颜色（基于速度）
        const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
        colors[i3] = Math.min(1, speed * 2); // R - 速度映射到红色
        colors[i3 + 1] = Math.min(1, speed); // G - 速度映射到绿色
        colors[i3 + 2] = Math.max(0, 1 - speed * 2); // B - 速度反向映射到蓝色
    }
    
    particleSystem.geometry.attributes.position.needsUpdate = true;
    particleSystem.geometry.attributes.color.needsUpdate = true;
}

// 启动粒子动画
function startParticleAnimation() {
    if (isAnimating) return;
    
    isAnimating = true;
    particleSystem.visible = true;
    
    function animate() {
        if (!isAnimating) return;
        
        updateParticles(0.2);
        render();
        
        animationId = requestAnimationFrame(animate);
    }
    
    animate();
}

// 停止粒子动画
function stopParticleAnimation() {
    isAnimating = false;
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

// 切换粒子系统可见性
function toggleParticles(visible) {
    if (!particleSystem) return;
    
    particleSystem.visible = visible;
    
    if (visible) {
        startParticleAnimation();
    } else {
        stopParticleAnimation();
    }
}

// 切换流线可见性
function toggleStreamlines(visible) {
    if (!streamlines) return;
    
    streamlines.visible = visible;
    
    if (visible && velocityField) {
        generateStreamlines();
    }
}

// 渲染场景
function render() {
    if (threeRenderer && threeScene && threeCamera) {
        threeRenderer.render(threeScene, threeCamera);
    }
}

// 清理资源
function disposeVisualization() {
    stopParticleAnimation();
    
    if (threeScene) {
        // 清理场景中的对象
        threeScene.traverse(object => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
    }
    
    if (threeRenderer) {
        threeRenderer.dispose();
    }
}

// 导出函数
window.FluidVisualization = {
    init: initVisualization,
    updateVelocityField: updateVelocityField,
    generateStreamlines: generateStreamlines,
    toggleParticles: toggleParticles,
    toggleStreamlines: toggleStreamlines,
    dispose: disposeVisualization
}; 