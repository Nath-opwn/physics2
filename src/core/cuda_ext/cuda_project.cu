#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// 定义CUDA错误检查宏
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误在 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

// 计算散度核函数
__global__ void computeDivergenceKernel(float* velocity, float* divergence,
                                      int width, int height, int depth, float dt) {
    // 计算3D网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // 检查边界
    if (x <= 0 || y <= 0 || z <= 0 || x >= width-1 || y >= height-1 || z >= depth-1)
        return;
    
    // 计算线性索引
    int idx = x * height * depth + y * depth + z;
    
    // 计算速度场的散度
    float vx_plus = velocity[(x+1) * height * depth + y * depth + z + 0];
    float vx_minus = velocity[(x-1) * height * depth + y * depth + z + 0];
    float vy_plus = velocity[x * height * depth + (y+1) * depth + z + 1];
    float vy_minus = velocity[x * height * depth + (y-1) * depth + z + 1];
    float vz_plus = velocity[x * height * depth + y * depth + (z+1) + 2];
    float vz_minus = velocity[x * height * depth + y * depth + (z-1) + 2];
    
    divergence[idx] = ((vx_plus - vx_minus) + (vy_plus - vy_minus) + (vz_plus - vz_minus)) * 0.5f / dt;
}

// 求解泊松方程核函数
__global__ void solvePressureKernel(float* pressure, float* divergence,
                                  int width, int height, int depth) {
    // 计算3D网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // 检查边界
    if (x <= 0 || y <= 0 || z <= 0 || x >= width-1 || y >= height-1 || z >= depth-1)
        return;
    
    // 计算线性索引
    int idx = x * height * depth + y * depth + z;
    
    // 获取相邻单元格的压力
    float p_xp1 = pressure[(x+1) * height * depth + y * depth + z];
    float p_xm1 = pressure[(x-1) * height * depth + y * depth + z];
    float p_yp1 = pressure[x * height * depth + (y+1) * depth + z];
    float p_ym1 = pressure[x * height * depth + (y-1) * depth + z];
    float p_zp1 = pressure[x * height * depth + y * depth + (z+1)];
    float p_zm1 = pressure[x * height * depth + y * depth + (z-1)];
    
    // 雅可比迭代求解泊松方程
    pressure[idx] = (p_xp1 + p_xm1 + p_yp1 + p_ym1 + p_zp1 + p_zm1 - divergence[idx]) / 6.0f;
}

// 应用压力梯度核函数
__global__ void applyPressureGradientKernel(float* velocity, float* pressure,
                                          int width, int height, int depth, float dt) {
    // 计算3D网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // 检查边界
    if (x <= 0 || y <= 0 || z <= 0 || x >= width-1 || y >= height-1 || z >= depth-1)
        return;
    
    // 计算线性索引
    int vel_idx = (x * height * depth + y * depth + z) * 3;
    
    // 计算压力梯度
    float p_xp1 = pressure[(x+1) * height * depth + y * depth + z];
    float p_xm1 = pressure[(x-1) * height * depth + y * depth + z];
    float p_yp1 = pressure[x * height * depth + (y+1) * depth + z];
    float p_ym1 = pressure[x * height * depth + (y-1) * depth + z];
    float p_zp1 = pressure[x * height * depth + y * depth + (z+1)];
    float p_zm1 = pressure[x * height * depth + y * depth + (z-1)];
    
    // 更新速度场
    velocity[vel_idx + 0] -= dt * (p_xp1 - p_xm1) * 0.5f;
    velocity[vel_idx + 1] -= dt * (p_yp1 - p_ym1) * 0.5f;
    velocity[vel_idx + 2] -= dt * (p_zp1 - p_zm1) * 0.5f;
}

// 导出的C函数
extern "C" void cuda_project(float* velocity, float* pressure, float* divergence,
                           int width, int height, int depth, float dt, int iterations) {
    // 分配设备内存
    float *d_velocity, *d_pressure, *d_divergence;
    size_t vel_size = width * height * depth * 3 * sizeof(float);
    size_t scalar_size = width * height * depth * sizeof(float);
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_velocity, vel_size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_pressure, scalar_size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_divergence, scalar_size));
    
    // 复制数据到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_velocity, velocity, vel_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_pressure, pressure, scalar_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_divergence, divergence, scalar_size, cudaMemcpyHostToDevice));
    
    // 设置线程块和网格大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    );
    
    // 计算散度
    computeDivergenceKernel<<<gridSize, blockSize>>>(d_velocity, d_divergence, width, height, depth, dt);
    CUDA_CHECK_ERROR(cudaGetLastError());
    
    // 初始化压力场
    CUDA_CHECK_ERROR(cudaMemset(d_pressure, 0, scalar_size));
    
    // 迭代求解泊松方程
    for (int i = 0; i < iterations; i++) {
        solvePressureKernel<<<gridSize, blockSize>>>(d_pressure, d_divergence, width, height, depth);
        CUDA_CHECK_ERROR(cudaGetLastError());
    }
    
    // 应用压力梯度
    applyPressureGradientKernel<<<gridSize, blockSize>>>(d_velocity, d_pressure, width, height, depth, dt);
    CUDA_CHECK_ERROR(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK_ERROR(cudaMemcpy(velocity, d_velocity, vel_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(pressure, d_pressure, scalar_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(divergence, d_divergence, scalar_size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    cudaFree(d_velocity);
    cudaFree(d_pressure);
    cudaFree(d_divergence);
} 