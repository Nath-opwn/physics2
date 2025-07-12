/**
 * 监控面板脚本
 * 用于显示系统监控和错误信息
 */

class MonitoringDashboard {
    constructor(containerId, apiBaseUrl = '') {
        this.containerId = containerId;
        this.apiBaseUrl = apiBaseUrl;
        this.container = document.getElementById(containerId);
        this.token = localStorage.getItem('auth_token');
        this.updateInterval = 5000; // 5秒更新一次
        this.metricsHistory = [];
        this.errorsHistory = [];
        this.charts = {};
        
        if (!this.container) {
            console.error(`Container with id ${containerId} not found`);
            return;
        }
        
        this.init();
    }
    
    async init() {
        this.createDashboardStructure();
        await this.fetchInitialData();
        this.renderDashboard();
        this.startAutoRefresh();
    }
    
    createDashboardStructure() {
        this.container.innerHTML = `
            <div class="monitoring-dashboard">
                <div class="dashboard-header">
                    <h2>系统监控面板</h2>
                    <div class="dashboard-controls">
                        <button id="refresh-btn" class="btn btn-primary">刷新</button>
                        <select id="update-interval" class="form-select">
                            <option value="5000">5秒</option>
                            <option value="10000">10秒</option>
                            <option value="30000">30秒</option>
                            <option value="60000">1分钟</option>
                        </select>
                    </div>
                </div>
                
                <div class="dashboard-content">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">系统资源使用</div>
                                <div class="card-body">
                                    <canvas id="system-resources-chart"></canvas>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">进程资源使用</div>
                                <div class="card-body">
                                    <canvas id="process-resources-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    错误统计
                                    <button id="clear-errors-btn" class="btn btn-sm btn-danger float-end">清除错误</button>
                                </div>
                                <div class="card-body">
                                    <canvas id="error-stats-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">最近错误</div>
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table id="errors-table" class="table table-striped">
                                            <thead>
                                                <tr>
                                                    <th>时间</th>
                                                    <th>类型</th>
                                                    <th>消息</th>
                                                    <th>操作</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <!-- 错误列表将在这里动态生成 -->
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 添加事件监听器
        document.getElementById('refresh-btn').addEventListener('click', () => this.refreshData());
        document.getElementById('update-interval').addEventListener('change', (e) => {
            this.updateInterval = parseInt(e.target.value);
            this.startAutoRefresh();
        });
        document.getElementById('clear-errors-btn').addEventListener('click', () => this.clearErrors());
    }
    
    async fetchInitialData() {
        try {
            const [metricsHistory, errorsHistory, errorStats] = await Promise.all([
                this.fetchData('/api/monitoring/metrics/history?limit=20'),
                this.fetchData('/api/monitoring/errors/recent?limit=10'),
                this.fetchData('/api/monitoring/errors/stats')
            ]);
            
            this.metricsHistory = metricsHistory || [];
            this.errorsHistory = errorsHistory || [];
            this.errorStats = errorStats || { stats: {}, total: 0 };
        } catch (error) {
            console.error('Error fetching initial data:', error);
        }
    }
    
    async fetchData(endpoint) {
        try {
            const response = await fetch(this.apiBaseUrl + endpoint, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`Error fetching data from ${endpoint}:`, error);
            return null;
        }
    }
    
    async refreshData() {
        await this.fetchInitialData();
        this.renderDashboard();
    }
    
    startAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(() => this.refreshData(), this.updateInterval);
    }
    
    renderDashboard() {
        this.renderSystemResourcesChart();
        this.renderProcessResourcesChart();
        this.renderErrorStatsChart();
        this.renderErrorsTable();
    }
    
    renderSystemResourcesChart() {
        const ctx = document.getElementById('system-resources-chart').getContext('2d');
        
        // 提取数据
        const labels = this.metricsHistory.map(m => {
            const date = new Date(m.timestamp);
            return date.toLocaleTimeString();
        });
        
        const cpuData = this.metricsHistory.map(m => m.system.cpu_percent);
        const memoryData = this.metricsHistory.map(m => m.system.memory_percent);
        const diskData = this.metricsHistory.map(m => m.system.disk_percent);
        
        if (this.charts.systemResources) {
            this.charts.systemResources.destroy();
        }
        
        this.charts.systemResources = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'CPU (%)',
                        data: cpuData,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4
                    },
                    {
                        label: '内存 (%)',
                        data: memoryData,
                        borderColor: 'rgba(153, 102, 255, 1)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        tension: 0.4
                    },
                    {
                        label: '磁盘 (%)',
                        data: diskData,
                        borderColor: 'rgba(255, 159, 64, 1)',
                        backgroundColor: 'rgba(255, 159, 64, 0.2)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
    
    renderProcessResourcesChart() {
        const ctx = document.getElementById('process-resources-chart').getContext('2d');
        
        // 提取数据
        const labels = this.metricsHistory.map(m => {
            const date = new Date(m.timestamp);
            return date.toLocaleTimeString();
        });
        
        const cpuData = this.metricsHistory.map(m => m.process.cpu_percent);
        const memoryData = this.metricsHistory.map(m => m.process.memory_mb);
        
        if (this.charts.processResources) {
            this.charts.processResources.destroy();
        }
        
        this.charts.processResources = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'CPU (%)',
                        data: cpuData,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: '内存 (MB)',
                        data: memoryData,
                        borderColor: 'rgba(153, 102, 255, 1)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'CPU (%)'
                        }
                    },
                    y1: {
                        beginAtZero: true,
                        position: 'right',
                        grid: {
                            drawOnChartArea: false
                        },
                        title: {
                            display: true,
                            text: '内存 (MB)'
                        }
                    }
                }
            }
        });
    }
    
    renderErrorStatsChart() {
        const ctx = document.getElementById('error-stats-chart').getContext('2d');
        
        if (!this.errorStats || !this.errorStats.stats) {
            return;
        }
        
        const labels = Object.keys(this.errorStats.stats);
        const data = Object.values(this.errorStats.stats);
        const backgroundColors = labels.map((_, i) => {
            const hue = (i * 137) % 360; // 使用黄金角来生成不同的颜色
            return `hsl(${hue}, 70%, 60%)`;
        });
        
        if (this.charts.errorStats) {
            this.charts.errorStats.destroy();
        }
        
        if (labels.length === 0) {
            // 没有错误数据时显示空状态
            const noDataDiv = document.createElement('div');
            noDataDiv.className = 'text-center py-5';
            noDataDiv.innerHTML = '<p class="text-muted">没有错误数据</p>';
            ctx.canvas.parentNode.appendChild(noDataDiv);
            ctx.canvas.style.display = 'none';
            return;
        }
        
        ctx.canvas.style.display = 'block';
        
        this.charts.errorStats = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: '错误数量',
                        data: data,
                        backgroundColor: backgroundColors
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    }
    
    renderErrorsTable() {
        const tableBody = document.querySelector('#errors-table tbody');
        
        if (!this.errorsHistory || this.errorsHistory.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center">没有错误记录</td>
                </tr>
            `;
            return;
        }
        
        tableBody.innerHTML = this.errorsHistory.map(error => {
            const date = new Date(error.timestamp);
            const formattedDate = date.toLocaleString();
            
            return `
                <tr>
                    <td>${formattedDate}</td>
                    <td><span class="badge bg-danger">${error.error_type}</span></td>
                    <td>${error.message}</td>
                    <td>
                        <button class="btn btn-sm btn-info view-error-details" data-error-id="${error.timestamp}">
                            详情
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
        
        // 添加详情按钮点击事件
        document.querySelectorAll('.view-error-details').forEach(button => {
            button.addEventListener('click', () => {
                const errorId = button.getAttribute('data-error-id');
                const error = this.errorsHistory.find(e => e.timestamp === errorId);
                this.showErrorDetails(error);
            });
        });
    }
    
    showErrorDetails(error) {
        // 创建模态框
        const modalId = 'error-details-modal';
        let modal = document.getElementById(modalId);
        
        if (!modal) {
            const modalDiv = document.createElement('div');
            modalDiv.innerHTML = `
                <div class="modal fade" id="${modalId}" tabindex="-1" aria-hidden="true">
                    <div class="modal-dialog modal-lg">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">错误详情</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body">
                                <div class="error-details-content"></div>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modalDiv.firstElementChild);
            modal = document.getElementById(modalId);
        }
        
        // 填充错误详情
        const content = modal.querySelector('.error-details-content');
        content.innerHTML = `
            <div class="mb-3">
                <strong>时间:</strong> ${new Date(error.timestamp).toLocaleString()}
            </div>
            <div class="mb-3">
                <strong>类型:</strong> <span class="badge bg-danger">${error.error_type}</span>
            </div>
            <div class="mb-3">
                <strong>消息:</strong> ${error.message}
            </div>
            ${error.traceback ? `
                <div class="mb-3">
                    <strong>堆栈跟踪:</strong>
                    <pre class="bg-light p-3 mt-2" style="max-height: 300px; overflow-y: auto;">${error.traceback}</pre>
                </div>
            ` : ''}
            ${error.context && Object.keys(error.context).length > 0 ? `
                <div class="mb-3">
                    <strong>上下文:</strong>
                    <pre class="bg-light p-3 mt-2">${JSON.stringify(error.context, null, 2)}</pre>
                </div>
            ` : ''}
        `;
        
        // 显示模态框
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
    
    async clearErrors() {
        try {
            const response = await fetch(this.apiBaseUrl + '/api/monitoring/errors/clear', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // 刷新数据
            this.refreshData();
        } catch (error) {
            console.error('Error clearing errors:', error);
        }
    }
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MonitoringDashboard;
} 