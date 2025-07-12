#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

// 错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// 三线性插值核函数
__device__ float trilinear_interpolate_cuda(const float* field, int width, int height, int depth,
                                         float x, float y, float z) {
    // 边界检查
    x = fmaxf(0.0f, fminf(float(width - 1.001f), x));
    y = fmaxf(0.0f, fminf(float(height - 1.001f), y));
    z = fmaxf(0.0f, fminf(float(depth - 1.001f), z));
    
    // 计算整数索引和插值系数
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int z0 = static_cast<int>(floorf(z));
    
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    int z1 = min(z0 + 1, depth - 1);
    
    float sx = x - x0;
    float sy = y - y0;
    float sz = z - z0;
    
    // 获取8个顶点的值
    float c000 = field[x0 * height * depth + y0 * depth + z0];
    float c001 = field[x0 * height * depth + y0 * depth + z1];
    float c010 = field[x0 * height * depth + y1 * depth + z0];
    float c011 = field[x0 * height * depth + y1 * depth + z1];
    float c100 = field[x1 * height * depth + y0 * depth + z0];
    float c101 = field[x1 * height * depth + y0 * depth + z1];
    float c110 = field[x1 * height * depth + y1 * depth + z0];
    float c111 = field[x1 * height * depth + y1 * depth + z1];
    
    // 三线性插值
    float c00 = c000 * (1 - sx) + c100 * sx;
    float c01 = c001 * (1 - sx) + c101 * sx;
    float c10 = c010 * (1 - sx) + c110 * sx;
    float c11 = c011 * (1 - sx) + c111 * sx;
    
    float c0 = c00 * (1 - sy) + c10 * sy;
    float c1 = c01 * (1 - sy) + c11 * sy;
    
    return c0 * (1 - sz) + c1 * sz;
}

// VOF方法的平流计算核函数
__global__ void advect_vof_kernel(float* volume_fractions_out, const float* volume_fractions_in,
                                const float* velocity_field, int width, int height, int depth,
                                int phase_idx, int num_phases, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        // 计算当前位置索引
        int idx = x * height * depth + y * depth + z;
        
        // 计算回溯位置
        float pos_x = x - dt * velocity_field[idx * 3 + 0];
        float pos_y = y - dt * velocity_field[idx * 3 + 1];
        float pos_z = z - dt * velocity_field[idx * 3 + 2];
        
        // 三线性插值
        int phase_offset = phase_idx * width * height * depth;
        volume_fractions_out[phase_offset + idx] = trilinear_interpolate_cuda(
            volume_fractions_in + phase_offset, width, height, depth,
            pos_x, pos_y, pos_z
        );
    }
}

// 应用边界条件核函数
__global__ void apply_vof_boundary_kernel(float* volume_fractions, int width, int height, int depth,
                                        int phase_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // x方向边界
    if (tid < height * depth) {
        int y = tid / depth;
        int z = tid % depth;
        
        volume_fractions[phase_idx * width * height * depth + 0 * height * depth + y * depth + z] =
            volume_fractions[phase_idx * width * height * depth + 1 * height * depth + y * depth + z];
        
        volume_fractions[phase_idx * width * height * depth + (width-1) * height * depth + y * depth + z] =
            volume_fractions[phase_idx * width * height * depth + (width-2) * height * depth + y * depth + z];
    }
    
    // y方向边界
    if (tid < width * depth) {
        int x = tid / depth;
        int z = tid % depth;
        
        volume_fractions[phase_idx * width * height * depth + x * height * depth + 0 * depth + z] =
            volume_fractions[phase_idx * width * height * depth + x * height * depth + 1 * depth + z];
        
        volume_fractions[phase_idx * width * height * depth + x * height * depth + (height-1) * depth + z] =
            volume_fractions[phase_idx * width * height * depth + x * height * depth + (height-2) * depth + z];
    }
    
    // z方向边界
    if (tid < width * height) {
        int x = tid / height;
        int y = tid % height;
        
        volume_fractions[phase_idx * width * height * depth + x * height * depth + y * depth + 0] =
            volume_fractions[phase_idx * width * height * depth + x * height * depth + y * depth + 1];
        
        volume_fractions[phase_idx * width * height * depth + x * height * depth + y * depth + (depth-1)] =
            volume_fractions[phase_idx * width * height * depth + x * height * depth + y * depth + (depth-2)];
    }
}

// 归一化体积分数核函数
__global__ void normalize_volume_fractions_kernel(float* volume_fractions, int width, int height, int depth,
                                                int num_phases) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = x * height * depth + y * depth + z;
        
        // 计算体积分数和
        float total = 0.0f;
        for (int p = 0; p < num_phases; ++p) {
            total += volume_fractions[p * width * height * depth + idx];
        }
        
        // 避免除以零
        total = fmaxf(total, 1e-6f);
        
        // 归一化
        for (int p = 0; p < num_phases; ++p) {
            volume_fractions[p * width * height * depth + idx] /= total;
        }
    }
}

// 计算梯度核函数
__global__ void compute_gradient_kernel(const float* field, float* grad_x, float* grad_y, float* grad_z,
                                      int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (x < width - 1 && y < height - 1 && z < depth - 1) {
        int idx = x * height * depth + y * depth + z;
        
        // 中心差分
        grad_x[idx] = (field[(x+1) * height * depth + y * depth + z] -
                      field[(x-1) * height * depth + y * depth + z]) * 0.5f;
        
        grad_y[idx] = (field[x * height * depth + (y+1) * depth + z] -
                      field[x * height * depth + (y-1) * depth + z]) * 0.5f;
        
        grad_z[idx] = (field[x * height * depth + y * depth + (z+1)] -
                      field[x * height * depth + y * depth + (z-1)]) * 0.5f;
    }
}

// 计算符号函数核函数
__global__ void compute_sign_phi_kernel(const float* phi, float* sign_phi, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        sign_phi[idx] = phi[idx] / sqrtf(phi[idx] * phi[idx] + 1e-6f);
    }
}

// 水平集重初始化步骤核函数
__global__ void reinitialize_step_kernel(const float* phi, float* phi_temp, const float* sign_phi,
                                       const float* grad_x, const float* grad_y, const float* grad_z,
                                       int size, float dt_reinit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad_mag = sqrtf(grad_x[idx]*grad_x[idx] + grad_y[idx]*grad_y[idx] + grad_z[idx]*grad_z[idx]);
        
        // 更新水平集函数
        phi_temp[idx] = phi[idx] - dt_reinit * sign_phi[idx] * (grad_mag - 1.0f);
    }
}

// 应用水平集边界条件核函数
__global__ void apply_levelset_boundary_kernel(float* phi, int width, int height, int depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // x方向边界
    if (tid < height * depth) {
        int y = tid / depth;
        int z = tid % depth;
        
        phi[0 * height * depth + y * depth + z] = phi[1 * height * depth + y * depth + z];
        phi[(width-1) * height * depth + y * depth + z] = phi[(width-2) * height * depth + y * depth + z];
    }
    
    // y方向边界
    if (tid < width * depth) {
        int x = tid / depth;
        int z = tid % depth;
        
        phi[x * height * depth + 0 * depth + z] = phi[x * height * depth + 1 * depth + z];
        phi[x * height * depth + (height-1) * depth + z] = phi[x * height * depth + (height-2) * depth + z];
    }
    
    // z方向边界
    if (tid < width * height) {
        int x = tid / height;
        int y = tid % height;
        
        phi[x * height * depth + y * depth + 0] = phi[x * height * depth + y * depth + 1];
        phi[x * height * depth + y * depth + (depth-1)] = phi[x * height * depth + y * depth + (depth-2)];
    }
}

// 水平集方法的平流计算核函数
__global__ void advect_levelset_kernel(float* phi_out, const float* phi_in, const float* velocity_field,
                                     int width, int height, int depth, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        // 计算当前位置索引
        int idx = x * height * depth + y * depth + z;
        
        // 计算回溯位置
        float pos_x = x - dt * velocity_field[idx * 3 + 0];
        float pos_y = y - dt * velocity_field[idx * 3 + 1];
        float pos_z = z - dt * velocity_field[idx * 3 + 2];
        
        // 三线性插值
        phi_out[idx] = trilinear_interpolate_cuda(phi_in, width, height, depth, pos_x, pos_y, pos_z);
    }
}

// 根据水平集函数更新相场核函数
__global__ void update_phase_fields_kernel(const float* phi, float* phase_fields,
                                         int width, int height, int depth, int num_phases, float epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = x * height * depth + y * depth + z;
        
        // 使用平滑Heaviside函数
        phase_fields[0 * width * height * depth + idx] = 0.5f * (1.0f - tanhf(phi[idx] / epsilon));
        
        // 更新其他相 (简化为二相情况)
        if (num_phases > 1) {
            phase_fields[1 * width * height * depth + idx] = 1.0f - phase_fields[0 * width * height * depth + idx];
        }
    }
}

// 计算界面场核函数
__global__ void compute_interface_field_kernel(const float* phi, float* interface_field,
                                            int width, int height, int depth, float epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = x * height * depth + y * depth + z;
        
        // 使用平滑Delta函数
        float phi_val = phi[idx];
        if (fabsf(phi_val) < epsilon) {
            interface_field[idx] = 0.5f * (1.0f + cosf(M_PI * phi_val / epsilon));
        } else {
            interface_field[idx] = 0.0f;
        }
    }
}

// 计算曲率核函数
__global__ void compute_curvature_kernel(const float* phi, float* curvature,
                                      const float* grad_x, const float* grad_y, const float* grad_z,
                                      int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (x < width - 1 && y < height - 1 && z < depth - 1) {
        int idx = x * height * depth + y * depth + z;
        
        // 计算梯度大小
        float grad_mag = sqrtf(
            grad_x[idx] * grad_x[idx] + 
            grad_y[idx] * grad_y[idx] + 
            grad_z[idx] * grad_z[idx]
        );
        
        // 避免除以零
        if (grad_mag < 1e-6f) {
            curvature[idx] = 0.0f;
            return;
        }
        
        // 归一化梯度
        float nx = grad_x[idx] / grad_mag;
        float ny = grad_y[idx] / grad_mag;
        float nz = grad_z[idx] / grad_mag;
        
        // 计算梯度的散度 (中心差分)
        float dnx_dx = (grad_x[(x+1) * height * depth + y * depth + z] / 
                       fmaxf(1e-6f, sqrtf(grad_x[(x+1) * height * depth + y * depth + z] * grad_x[(x+1) * height * depth + y * depth + z] + 
                                         grad_y[(x+1) * height * depth + y * depth + z] * grad_y[(x+1) * height * depth + y * depth + z] + 
                                         grad_z[(x+1) * height * depth + y * depth + z] * grad_z[(x+1) * height * depth + y * depth + z])) - 
                       grad_x[(x-1) * height * depth + y * depth + z] / 
                       fmaxf(1e-6f, sqrtf(grad_x[(x-1) * height * depth + y * depth + z] * grad_x[(x-1) * height * depth + y * depth + z] + 
                                         grad_y[(x-1) * height * depth + y * depth + z] * grad_y[(x-1) * height * depth + y * depth + z] + 
                                         grad_z[(x-1) * height * depth + y * depth + z] * grad_z[(x-1) * height * depth + y * depth + z]))) * 0.5f;
        
        float dny_dy = (grad_y[x * height * depth + (y+1) * depth + z] / 
                       fmaxf(1e-6f, sqrtf(grad_x[x * height * depth + (y+1) * depth + z] * grad_x[x * height * depth + (y+1) * depth + z] + 
                                         grad_y[x * height * depth + (y+1) * depth + z] * grad_y[x * height * depth + (y+1) * depth + z] + 
                                         grad_z[x * height * depth + (y+1) * depth + z] * grad_z[x * height * depth + (y+1) * depth + z])) - 
                       grad_y[x * height * depth + (y-1) * depth + z] / 
                       fmaxf(1e-6f, sqrtf(grad_x[x * height * depth + (y-1) * depth + z] * grad_x[x * height * depth + (y-1) * depth + z] + 
                                         grad_y[x * height * depth + (y-1) * depth + z] * grad_y[x * height * depth + (y-1) * depth + z] + 
                                         grad_z[x * height * depth + (y-1) * depth + z] * grad_z[x * height * depth + (y-1) * depth + z]))) * 0.5f;
        
        float dnz_dz = (grad_z[x * height * depth + y * depth + (z+1)] / 
                       fmaxf(1e-6f, sqrtf(grad_x[x * height * depth + y * depth + (z+1)] * grad_x[x * height * depth + y * depth + (z+1)] + 
                                         grad_y[x * height * depth + y * depth + (z+1)] * grad_y[x * height * depth + y * depth + (z+1)] + 
                                         grad_z[x * height * depth + y * depth + (z+1)] * grad_z[x * height * depth + y * depth + (z+1)])) - 
                       grad_z[x * height * depth + y * depth + (z-1)] / 
                       fmaxf(1e-6f, sqrtf(grad_x[x * height * depth + y * depth + (z-1)] * grad_x[x * height * depth + y * depth + (z-1)] + 
                                         grad_y[x * height * depth + y * depth + (z-1)] * grad_y[x * height * depth + y * depth + (z-1)] + 
                                         grad_z[x * height * depth + y * depth + (z-1)] * grad_z[x * height * depth + y * depth + (z-1)]))) * 0.5f;
        
        // 曲率是梯度场的散度
        curvature[idx] = dnx_dx + dny_dy + dnz_dz;
    }
}

// 计算表面张力力核函数
__global__ void compute_surface_tension_kernel(const float* phi, const float* curvature, 
                                            const float* interface_field, const float* grad_x, 
                                            const float* grad_y, const float* grad_z,
                                            float* force_field, int width, int height, int depth, 
                                            float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = x * height * depth + y * depth + z;
        
        // 计算梯度大小
        float grad_mag = sqrtf(
            grad_x[idx] * grad_x[idx] + 
            grad_y[idx] * grad_y[idx] + 
            grad_z[idx] * grad_z[idx]
        );
        
        // 避免除以零
        grad_mag = fmaxf(1e-6f, grad_mag);
        
        // 计算法向量
        float nx = grad_x[idx] / grad_mag;
        float ny = grad_y[idx] / grad_mag;
        float nz = grad_z[idx] / grad_mag;
        
        // 计算表面张力力
        float force_magnitude = sigma * curvature[idx] * interface_field[idx];
        
        force_field[idx * 3 + 0] = force_magnitude * nx;
        force_field[idx * 3 + 1] = force_magnitude * ny;
        force_field[idx * 3 + 2] = force_magnitude * nz;
    }
}

// 以下是CUDA函数的包装器，供外部调用

// VOF方法的平流计算
extern "C" void advect_vof_cuda(float* volume_fractions_out, const float* volume_fractions_in,
                              const float* velocity_field, int width, int height, int depth,
                              int phase_idx, int num_phases, float dt) {
    // 设置网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y,
                 (depth + blockSize.z - 1) / blockSize.z);
    
    // 启动核函数
    advect_vof_kernel<<<gridSize, blockSize>>>(
        volume_fractions_out, volume_fractions_in, velocity_field,
        width, height, depth, phase_idx, num_phases, dt
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 应用边界条件
    int max_threads = 256;
    int max_blocks = (height * depth + max_threads - 1) / max_threads;
    apply_vof_boundary_kernel<<<max_blocks, max_threads>>>(
        volume_fractions_out, width, height, depth, phase_idx
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 归一化体积分数
extern "C" void normalize_volume_fractions_cuda(float* volume_fractions, int width, int height, int depth,
                                              int num_phases) {
    // 设置网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y,
                 (depth + blockSize.z - 1) / blockSize.z);
    
    // 启动核函数
    normalize_volume_fractions_kernel<<<gridSize, blockSize>>>(
        volume_fractions, width, height, depth, num_phases
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 水平集方法的重初始化
extern "C" void reinitialize_levelset_cuda(float* phi, int width, int height, int depth,
                                         int iterations, float dt_reinit) {
    int size = width * height * depth;
    
    // 分配设备内存
    float *d_phi_temp, *d_sign_phi, *d_grad_x, *d_grad_y, *d_grad_z;
    CUDA_CHECK(cudaMalloc(&d_phi_temp, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sign_phi, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_x, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_z, size * sizeof(float)));
    
    // 设置网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y,
                 (depth + blockSize.z - 1) / blockSize.z);
    
    dim3 gradBlockSize(8, 8, 8);
    dim3 gradGridSize((width - 2 + gradBlockSize.x - 1) / gradBlockSize.x,
                     (height - 2 + gradBlockSize.y - 1) / gradBlockSize.y,
                     (depth - 2 + gradBlockSize.z - 1) / gradBlockSize.z);
    
    int max_threads = 256;
    int max_blocks = (size + max_threads - 1) / max_threads;
    
    // 计算符号函数
    compute_sign_phi_kernel<<<max_blocks, max_threads>>>(phi, d_sign_phi, size);
    CUDA_CHECK(cudaGetLastError());
    
    // 迭代重初始化
    for (int iter = 0; iter < iterations; ++iter) {
        // 计算梯度
        compute_gradient_kernel<<<gradGridSize, gradBlockSize>>>(
            phi, d_grad_x, d_grad_y, d_grad_z, width, height, depth
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 更新水平集函数
        reinitialize_step_kernel<<<max_blocks, max_threads>>>(
            phi, d_phi_temp, d_sign_phi, d_grad_x, d_grad_y, d_grad_z, size, dt_reinit
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 复制更新后的水平集函数
        CUDA_CHECK(cudaMemcpy(phi, d_phi_temp, size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // 应用边界条件
        int boundary_blocks = max((height * depth + max_threads - 1) / max_threads,
                                max((width * depth + max_threads - 1) / max_threads,
                                    (width * height + max_threads - 1) / max_threads));
        apply_levelset_boundary_kernel<<<boundary_blocks, max_threads>>>(phi, width, height, depth);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(d_phi_temp));
    CUDA_CHECK(cudaFree(d_sign_phi));
    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_grad_z));
    
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 水平集方法的平流计算
extern "C" void advect_levelset_cuda(float* phi_out, const float* phi_in, const float* velocity_field,
                                   int width, int height, int depth, float dt) {
    // 设置网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y,
                 (depth + blockSize.z - 1) / blockSize.z);
    
    // 启动核函数
    advect_levelset_kernel<<<gridSize, blockSize>>>(
        phi_out, phi_in, velocity_field, width, height, depth, dt
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 应用边界条件
    int max_threads = 256;
    int boundary_blocks = max((height * depth + max_threads - 1) / max_threads,
                            max((width * depth + max_threads - 1) / max_threads,
                                (width * height + max_threads - 1) / max_threads));
    apply_levelset_boundary_kernel<<<boundary_blocks, max_threads>>>(phi_out, width, height, depth);
    CUDA_CHECK(cudaGetLastError());
    
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 根据水平集函数更新相场
extern "C" void update_phase_fields_cuda(const float* phi, float* phase_fields,
                                       int width, int height, int depth, int num_phases, float epsilon) {
    // 设置网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y,
                 (depth + blockSize.z - 1) / blockSize.z);
    
    // 启动核函数
    update_phase_fields_kernel<<<gridSize, blockSize>>>(
        phi, phase_fields, width, height, depth, num_phases, epsilon
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 计算界面场
extern "C" void compute_interface_field_cuda(const float* phi, float* interface_field,
                                          int width, int height, int depth, float epsilon) {
    // 设置网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                 (height + blockSize.y - 1) / blockSize.y,
                 (depth + blockSize.z - 1) / blockSize.z);
    
    // 启动核函数
    compute_interface_field_kernel<<<gridSize, blockSize>>>(
        phi, interface_field, width, height, depth, epsilon
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
} 

// 计算曲率的CUDA接口函数
extern "C" void compute_curvature_cuda(const float* phi, float* curvature,
                                     int width, int height, int depth) {
    // 设置CUDA网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (width - 2 + blockSize.x - 1) / blockSize.x,
        (height - 2 + blockSize.y - 1) / blockSize.y,
        (depth - 2 + blockSize.z - 1) / blockSize.z
    );
    
    // 分配设备内存
    float *d_phi, *d_curvature, *d_grad_x, *d_grad_y, *d_grad_z;
    size_t size = width * height * depth * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_phi, size));
    CUDA_CHECK(cudaMalloc(&d_curvature, size));
    CUDA_CHECK(cudaMalloc(&d_grad_x, size));
    CUDA_CHECK(cudaMalloc(&d_grad_y, size));
    CUDA_CHECK(cudaMalloc(&d_grad_z, size));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_phi, phi, size, cudaMemcpyHostToDevice));
    
    // 计算梯度
    dim3 gradBlockSize(8, 8, 8);
    dim3 gradGridSize(
        (width - 2 + gradBlockSize.x - 1) / gradBlockSize.x,
        (height - 2 + gradBlockSize.y - 1) / gradBlockSize.y,
        (depth - 2 + gradBlockSize.z - 1) / gradBlockSize.z
    );
    
    compute_gradient_kernel<<<gradGridSize, gradBlockSize>>>(d_phi, d_grad_x, d_grad_y, d_grad_z, width, height, depth);
    CUDA_CHECK(cudaGetLastError());
    
    // 计算曲率
    compute_curvature_kernel<<<gridSize, blockSize>>>(d_phi, d_curvature, d_grad_x, d_grad_y, d_grad_z, width, height, depth);
    CUDA_CHECK(cudaGetLastError());
    
    // 应用边界条件
    int maxThreads = 512;
    int numThreads = min(maxThreads, max(height * depth, max(width * depth, width * height)));
    int numBlocks = (max(height * depth, max(width * depth, width * height)) + numThreads - 1) / numThreads;
    
    apply_levelset_boundary_kernel<<<numBlocks, numThreads>>>(d_curvature, width, height, depth);
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(curvature, d_curvature, size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFree(d_curvature));
    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_grad_z));
}

// 计算表面张力力的CUDA接口函数
extern "C" void compute_surface_tension_cuda(const float* phi, const float* curvature, float* force_field,
                                          int width, int height, int depth, float sigma, float epsilon) {
    // 设置CUDA网格和块大小
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    );
    
    // 分配设备内存
    float *d_phi, *d_curvature, *d_force_field, *d_interface_field, *d_grad_x, *d_grad_y, *d_grad_z;
    size_t size = width * height * depth * sizeof(float);
    size_t force_size = width * height * depth * 3 * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_phi, size));
    CUDA_CHECK(cudaMalloc(&d_curvature, size));
    CUDA_CHECK(cudaMalloc(&d_force_field, force_size));
    CUDA_CHECK(cudaMalloc(&d_interface_field, size));
    CUDA_CHECK(cudaMalloc(&d_grad_x, size));
    CUDA_CHECK(cudaMalloc(&d_grad_y, size));
    CUDA_CHECK(cudaMalloc(&d_grad_z, size));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_phi, phi, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_curvature, curvature, size, cudaMemcpyHostToDevice));
    
    // 计算界面场
    compute_interface_field_kernel<<<gridSize, blockSize>>>(d_phi, d_interface_field, width, height, depth, epsilon);
    CUDA_CHECK(cudaGetLastError());
    
    // 计算梯度
    dim3 gradBlockSize(8, 8, 8);
    dim3 gradGridSize(
        (width - 2 + gradBlockSize.x - 1) / gradBlockSize.x,
        (height - 2 + gradBlockSize.y - 1) / gradBlockSize.y,
        (depth - 2 + gradBlockSize.z - 1) / gradBlockSize.z
    );
    
    compute_gradient_kernel<<<gradGridSize, gradBlockSize>>>(d_phi, d_grad_x, d_grad_y, d_grad_z, width, height, depth);
    CUDA_CHECK(cudaGetLastError());
    
    // 计算表面张力力
    compute_surface_tension_kernel<<<gridSize, blockSize>>>(d_phi, d_curvature, d_interface_field, 
                                                         d_grad_x, d_grad_y, d_grad_z,
                                                         d_force_field, width, height, depth, sigma);
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(force_field, d_force_field, force_size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFree(d_curvature));
    CUDA_CHECK(cudaFree(d_force_field));
    CUDA_CHECK(cudaFree(d_interface_field));
    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_grad_z));
} 