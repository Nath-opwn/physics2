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

// 涡量计算核函数
__global__ void computeVorticityKernel(float* velocity, float* vorticity,
                                     int width, int height, int depth) {
    // 计算3D网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // 检查边界
    if (x <= 0 || y <= 0 || z <= 0 || x >= width-1 || y >= height-1 || z >= depth-1)
        return;
    
    // 计算线性索引
    int idx = (x * height * depth + y * depth + z) * 3;
    
    // 获取相邻单元格的速度
    // x方向
    float vx_yp1 = velocity[(x * height * depth + (y+1) * depth + z) * 3 + 0];
    float vx_ym1 = velocity[(x * height * depth + (y-1) * depth + z) * 3 + 0];
    float vx_zp1 = velocity[(x * height * depth + y * depth + (z+1)) * 3 + 0];
    float vx_zm1 = velocity[(x * height * depth + y * depth + (z-1)) * 3 + 0];
    
    // y方向
    float vy_xp1 = velocity[((x+1) * height * depth + y * depth + z) * 3 + 1];
    float vy_xm1 = velocity[((x-1) * height * depth + y * depth + z) * 3 + 1];
    float vy_zp1 = velocity[(x * height * depth + y * depth + (z+1)) * 3 + 1];
    float vy_zm1 = velocity[(x * height * depth + y * depth + (z-1)) * 3 + 1];
    
    // z方向
    float vz_xp1 = velocity[((x+1) * height * depth + y * depth + z) * 3 + 2];
    float vz_xm1 = velocity[((x-1) * height * depth + y * depth + z) * 3 + 2];
    float vz_yp1 = velocity[(x * height * depth + (y+1) * depth + z) * 3 + 2];
    float vz_ym1 = velocity[(x * height * depth + (y-1) * depth + z) * 3 + 2];
    
    // 计算速度梯度
    float dvx_dy = (vx_yp1 - vx_ym1) * 0.5f;
    float dvx_dz = (vx_zp1 - vx_zm1) * 0.5f;
    float dvy_dx = (vy_xp1 - vy_xm1) * 0.5f;
    float dvy_dz = (vy_zp1 - vy_zm1) * 0.5f;
    float dvz_dx = (vz_xp1 - vz_xm1) * 0.5f;
    float dvz_dy = (vz_yp1 - vz_ym1) * 0.5f;
    
    // 计算涡量 (curl = ∇ × v)
    vorticity[idx + 0] = dvz_dy - dvy_dz;  // ω_x = ∂v_z/∂y - ∂v_y/∂z
    vorticity[idx + 1] = dvx_dz - dvz_dx;  // ω_y = ∂v_x/∂z - ∂v_z/∂x
    vorticity[idx + 2] = dvy_dx - dvx_dy;  // ω_z = ∂v_y/∂x - ∂v_x/∂y
}

// 导出的C函数
extern "C" void cuda_compute_vorticity(float* velocity, float* vorticity,
                                     int width, int height, int depth) {
    // 分配设备内存
    float *d_velocity, *d_vorticity;
    size_t vel_size = width * height * depth * 3 * sizeof(float);
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_velocity, vel_size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_vorticity, vel_size));
    
    // 复制数据到设备
    CUDA_CHECK_ERROR(cudaMemcpy(d_velocity, velocity, vel_size, cudaMemcpyHostToDevice));
    
    // 设置线程块和网格大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    );
    
    // 计算涡量
    computeVorticityKernel<<<gridSize, blockSize>>>(d_velocity, d_vorticity, width, height, depth);
    CUDA_CHECK_ERROR(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK_ERROR(cudaMemcpy(vorticity, d_vorticity, vel_size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    cudaFree(d_velocity);
    cudaFree(d_vorticity);
} 