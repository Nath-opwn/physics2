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

// 扩散核函数 - 标量场版本
__global__ void diffuseKernel(float* field, float* field_prev, int width, int height, int depth, 
                             float alpha, float beta, int components) {
    // 计算3D网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // 检查边界
    if (x <= 0 || y <= 0 || z <= 0 || x >= width-1 || y >= height-1 || z >= depth-1)
        return;
    
    // 计算线性索引
    int idx = (x * height * depth + y * depth + z) * components;
    
    // 对每个分量进行扩散计算
    for (int c = 0; c < components; c++) {
        int i = idx + c;
        
        // 获取相邻单元格的索引
        int i_xp1 = ((x+1) * height * depth + y * depth + z) * components + c;
        int i_xm1 = ((x-1) * height * depth + y * depth + z) * components + c;
        int i_yp1 = (x * height * depth + (y+1) * depth + z) * components + c;
        int i_ym1 = (x * height * depth + (y-1) * depth + z) * components + c;
        int i_zp1 = (x * height * depth + y * depth + (z+1)) * components + c;
        int i_zm1 = (x * height * depth + y * depth + (z-1)) * components + c;
        
        // 雅可比迭代计算
        field[i] = (field_prev[i] + alpha * (
            field[i_xp1] + field[i_xm1] +
            field[i_yp1] + field[i_ym1] +
            field[i_zp1] + field[i_zm1]
        )) * beta;
    }
}

// 应用边界条件核函数
__global__ void applyBoundaryKernel(float* field, int width, int height, int depth, int components, int boundary_type) {
    // 计算2D网格中的位置 (用于处理边界面)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 检查边界
    if (x >= width || y >= height)
        return;
    
    // 固定边界条件
    if (boundary_type == 0) {
        // 处理z方向边界
        for (int c = 0; c < components; c++) {
            // z = 0 边界
            field[(x * height * depth + y * depth) * components + c] = 0.0f;
            // z = depth-1 边界
            field[(x * height * depth + y * depth + depth-1) * components + c] = 0.0f;
        }
        
        // 处理y方向边界
        for (int z = 0; z < depth; z++) {
            for (int c = 0; c < components; c++) {
                // y = 0 边界
                field[(x * height * depth + z) * components + c] = 0.0f;
                // y = height-1 边界
                field[(x * height * depth + (height-1) * depth + z) * components + c] = 0.0f;
            }
        }
        
        // 处理x方向边界
        if (x == 0 || x == width-1) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < depth; z++) {
                    for (int c = 0; c < components; c++) {
                        field[(x * height * depth + y * depth + z) * components + c] = 0.0f;
                    }
                }
            }
        }
    }
    // 周期性边界条件
    else if (boundary_type == 1) {
        // 这里简化处理，实际应用中需要更复杂的周期性边界处理
    }
}

// 导出的C函数
extern "C" void cuda_diffuse(float* field, float* field_prev, int width, int height, int depth, 
                           float alpha, float beta, int iterations, int components) {
    // 分配设备内存
    float *d_field, *d_field_prev;
    size_t size = width * height * depth * components * sizeof(float);
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_field, size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_field_prev, size));
    
    // 复制数据到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_field, field, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_field_prev, field_prev, size, cudaMemcpyHostToDevice));
    
    // 设置线程块和网格大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    );
    
    // 设置2D边界处理的线程块和网格大小
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D(
        (width + blockSize2D.x - 1) / blockSize2D.x,
        (height + blockSize2D.y - 1) / blockSize2D.y
    );
    
    // 执行迭代
    for (int i = 0; i < iterations; i++) {
        // 执行扩散计算
        diffuseKernel<<<gridSize, blockSize>>>(d_field, d_field_prev, width, height, depth, alpha, beta, components);
        CUDA_CHECK_ERROR(cudaGetLastError());
        
        // 应用边界条件
        applyBoundaryKernel<<<gridSize2D, blockSize2D>>>(d_field, width, height, depth, components, 0);
        CUDA_CHECK_ERROR(cudaGetLastError());
    }
    
    // 复制结果回主机
    CUDA_CHECK_ERROR(cudaMemcpy(field, d_field, size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    cudaFree(d_field);
    cudaFree(d_field_prev);
} 