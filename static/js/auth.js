/**
 * 认证脚本
 * 处理用户登录、注册和会话管理
 */

class AuthManager {
    constructor() {
        this.token = localStorage.getItem('auth_token');
        this.username = localStorage.getItem('username');
        this.apiBaseUrl = ''; // 如果API在不同域名，请在此设置
    }

    /**
     * 检查用户是否已登录
     * @returns {boolean} 是否已登录
     */
    isLoggedIn() {
        return !!this.token;
    }

    /**
     * 获取当前用户名
     * @returns {string|null} 用户名
     */
    getCurrentUsername() {
        return this.username;
    }

    /**
     * 获取认证令牌
     * @returns {string|null} 认证令牌
     */
    getToken() {
        return this.token;
    }

    /**
     * 用户登录
     * @param {string} username 用户名
     * @param {string} password 密码
     * @returns {Promise<Object>} 登录结果
     */
    async login(username, password) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/users/token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: new URLSearchParams({
                    'username': username,
                    'password': password
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '登录失败');
            }

            const data = await response.json();
            this.token = data.access_token;
            this.username = username;
            
            // 保存到本地存储
            localStorage.setItem('auth_token', this.token);
            localStorage.setItem('username', this.username);
            
            return { success: true, message: '登录成功' };
        } catch (error) {
            console.error('登录错误:', error);
            return { success: false, message: error.message || '登录失败' };
        }
    }

    /**
     * 用户注册
     * @param {string} username 用户名
     * @param {string} email 邮箱
     * @param {string} password 密码
     * @returns {Promise<Object>} 注册结果
     */
    async register(username, email, password) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/users/register`, {
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

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '注册失败');
            }

            const data = await response.json();
            return { success: true, message: '注册成功' };
        } catch (error) {
            console.error('注册错误:', error);
            return { success: false, message: error.message || '注册失败' };
        }
    }

    /**
     * 用户退出登录
     */
    logout() {
        this.token = null;
        this.username = null;
        localStorage.removeItem('auth_token');
        localStorage.removeItem('username');
        
        // 重定向到登录页面
        window.location.href = 'login.html';
    }

    /**
     * 获取用户信息
     * @returns {Promise<Object>} 用户信息
     */
    async getUserInfo() {
        if (!this.isLoggedIn()) {
            return { success: false, message: '未登录' };
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/users/me`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (!response.ok) {
                throw new Error('获取用户信息失败');
            }

            const data = await response.json();
            return { success: true, data };
        } catch (error) {
            console.error('获取用户信息错误:', error);
            return { success: false, message: error.message };
        }
    }

    /**
     * 更新用户信息
     * @param {Object} userData 用户数据
     * @returns {Promise<Object>} 更新结果
     */
    async updateUserInfo(userData) {
        if (!this.isLoggedIn()) {
            return { success: false, message: '未登录' };
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/users/me`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify(userData)
            });

            if (!response.ok) {
                throw new Error('更新用户信息失败');
            }

            const data = await response.json();
            return { success: true, data, message: '更新成功' };
        } catch (error) {
            console.error('更新用户信息错误:', error);
            return { success: false, message: error.message };
        }
    }

    /**
     * 检查令牌是否有效
     * @returns {Promise<boolean>} 令牌是否有效
     */
    async validateToken() {
        if (!this.isLoggedIn()) {
            return false;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/status`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            return response.ok;
        } catch (error) {
            console.error('验证令牌错误:', error);
            return false;
        }
    }
}

// 创建全局认证管理器实例
const authManager = new AuthManager();

// 导出认证管理器
if (typeof module !== 'undefined' && module.exports) {
    module.exports = authManager;
} 