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

// 三线性插值函数
__device__ float trilinearInterpolate(float* field, int width, int height, int depth, 
                                    float x, float y, float z, int component, int components) {
    // 计算网格坐标
    int x0 = floorf(x);
    int y0 = floorf(y);
    int z0 = floorf(z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    
    // 确保坐标在有效范围内
    x0 = max(0, min(width - 1, x0));
    y0 = max(0, min(height - 1, y0));
    z0 = max(0, min(depth - 1, z0));
    x1 = max(0, min(width - 1, x1));
    y1 = max(0, min(height - 1, y1));
    z1 = max(0, min(depth - 1, z1));
    
    // 计算插值权重
    float sx = x - x0;
    float sy = y - y0;
    float sz = z - z0;
    
    // 获取场值
    float c000 = field[(x0 * height * depth + y0 * depth + z0) * components + component];
    float c001 = field[(x0 * height * depth + y0 * depth + z1) * components + component];
    float c010 = field[(x0 * height * depth + y1 * depth + z0) * components + component];
    float c011 = field[(x0 * height * depth + y1 * depth + z1) * components + component];
    float c100 = field[(x1 * height * depth + y0 * depth + z0) * components + component];
    float c101 = field[(x1 * height * depth + y0 * depth + z1) * components + component];
    float c110 = field[(x1 * height * depth + y1 * depth + z0) * components + component];
    float c111 = field[(x1 * height * depth + y1 * depth + z1) * components + component];
    
    // 三线性插值
    float c00 = c000 * (1 - sx) + c100 * sx;
    float c01 = c001 * (1 - sx) + c101 * sx;
    float c10 = c010 * (1 - sx) + c110 * sx;
    float c11 = c011 * (1 - sx) + c111 * sx;
    
    float c0 = c00 * (1 - sy) + c10 * sy;
    float c1 = c01 * (1 - sy) + c11 * sy;
    
    return c0 * (1 - sz) + c1 * sz;
}

// 平流核函数
__global__ void advectKernel(float* field, float* field_prev, float* velocity,
                           int width, int height, int depth, float dt, int components) {
    // 计算3D网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // 检查边界
    if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth)
        return;
    
    // 计算线性索引基址
    int idx = (x * height * depth + y * depth + z) * components;
    int vel_idx = (x * height * depth + y * depth + z) * 3;
    
    // 获取速度
    float vx = velocity[vel_idx];
    float vy = velocity[vel_idx + 1];
    float vz = velocity[vel_idx + 2];
    
    // 计算回溯位置
    float pos_x = x - dt * vx;
    float pos_y = y - dt * vy;
    float pos_z = z - dt * vz;
    
    // 确保位置在有效范围内
    pos_x = max(0.0f, min(float(width - 1.01f), pos_x));
    pos_y = max(0.0f, min(float(height - 1.01f), pos_y));
    pos_z = max(0.0f, min(float(depth - 1.01f), pos_z));
    
    // 对每个分量进行插值
    for (int c = 0; c < components; c++) {
        field[idx + c] = trilinearInterpolate(field_prev, width, height, depth, pos_x, pos_y, pos_z, c, components);
    }
}

// 导出的C函数
extern "C" void cuda_advect(float* field, float* field_prev, float* velocity, 
                          int width, int height, int depth, float dt, int components) {
    // 分配设备内存
    float *d_field, *d_field_prev, *d_velocity;
    size_t field_size = width * height * depth * components * sizeof(float);
    size_t vel_size = width * height * depth * 3 * sizeof(float);
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_field, field_size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_field_prev, field_size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_velocity, vel_size));
    
    // 复制数据到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_field, field, field_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_field_prev, field_prev, field_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_velocity, velocity, vel_size, cudaMemcpyHostToDevice));
    
    // 设置线程块和网格大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    );
    
    // 执行平流计算
    advectKernel<<<gridSize, blockSize>>>(d_field, d_field_prev, d_velocity, width, height, depth, dt, components);
    CUDA_CHECK_ERROR(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK_ERROR(cudaMemcpy(field, d_field, field_size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    cudaFree(d_field);
    cudaFree(d_field_prev);
    cudaFree(d_velocity);
} 