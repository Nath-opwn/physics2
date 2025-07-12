#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <omp.h>

// 三线性插值函数
float trilinear_interpolate(const float* field, int width, int height, int depth,
                           float x, float y, float z) {
    // 边界检查
    x = std::max(0.0f, std::min(float(width - 1.001f), x));
    y = std::max(0.0f, std::min(float(height - 1.001f), y));
    z = std::max(0.0f, std::min(float(depth - 1.001f), z));
    
    // 计算整数索引和插值系数
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int z0 = static_cast<int>(std::floor(z));
    
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);
    int z1 = std::min(z0 + 1, depth - 1);
    
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

// VOF方法的平流计算
void advect_vof(float* volume_fractions_out, const float* volume_fractions_in, 
                const float* velocity_field, int width, int height, int depth, 
                int phase_idx, int num_phases, float dt) {
    
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                // 计算当前位置索引
                int idx = x * height * depth + y * depth + z;
                
                // 计算回溯位置
                float pos_x = x - dt * velocity_field[idx * 3 + 0];
                float pos_y = y - dt * velocity_field[idx * 3 + 1];
                float pos_z = z - dt * velocity_field[idx * 3 + 2];
                
                // 三线性插值
                int phase_offset = phase_idx * width * height * depth;
                volume_fractions_out[phase_offset + idx] = trilinear_interpolate(
                    volume_fractions_in + phase_offset, width, height, depth, 
                    pos_x, pos_y, pos_z
                );
            }
        }
    }
    
    // 应用边界条件
    for (int y = 0; y < height; ++y) {
        for (int z = 0; z < depth; ++z) {
            // x方向边界
            volume_fractions_out[phase_idx * width * height * depth + 0 * height * depth + y * depth + z] = 
                volume_fractions_out[phase_idx * width * height * depth + 1 * height * depth + y * depth + z];
            volume_fractions_out[phase_idx * width * height * depth + (width-1) * height * depth + y * depth + z] = 
                volume_fractions_out[phase_idx * width * height * depth + (width-2) * height * depth + y * depth + z];
        }
    }
    
    for (int x = 0; x < width; ++x) {
        for (int z = 0; z < depth; ++z) {
            // y方向边界
            volume_fractions_out[phase_idx * width * height * depth + x * height * depth + 0 * depth + z] = 
                volume_fractions_out[phase_idx * width * height * depth + x * height * depth + 1 * depth + z];
            volume_fractions_out[phase_idx * width * height * depth + x * height * depth + (height-1) * depth + z] = 
                volume_fractions_out[phase_idx * width * height * depth + x * height * depth + (height-2) * depth + z];
        }
    }
    
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            // z方向边界
            volume_fractions_out[phase_idx * width * height * depth + x * height * depth + y * depth + 0] = 
                volume_fractions_out[phase_idx * width * height * depth + x * height * depth + y * depth + 1];
            volume_fractions_out[phase_idx * width * height * depth + x * height * depth + y * depth + (depth-1)] = 
                volume_fractions_out[phase_idx * width * height * depth + x * height * depth + y * depth + (depth-2)];
        }
    }
}

// 归一化体积分数
void normalize_volume_fractions(float* volume_fractions, int width, int height, int depth, int num_phases) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                int idx = x * height * depth + y * depth + z;
                
                // 计算体积分数和
                float total = 0.0f;
                for (int p = 0; p < num_phases; ++p) {
                    total += volume_fractions[p * width * height * depth + idx];
                }
                
                // 避免除以零
                total = std::max(total, 1e-6f);
                
                // 归一化
                for (int p = 0; p < num_phases; ++p) {
                    volume_fractions[p * width * height * depth + idx] /= total;
                }
            }
        }
    }
}

// 计算梯度
void compute_gradient(const float* field, float* grad_x, float* grad_y, float* grad_z,
                     int width, int height, int depth) {
    #pragma omp parallel for collapse(3)
    for (int x = 1; x < width - 1; ++x) {
        for (int y = 1; y < height - 1; ++y) {
            for (int z = 1; z < depth - 1; ++z) {
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
    }
}

// 水平集方法的重初始化
void reinitialize_levelset(float* phi, int width, int height, int depth, 
                          int iterations, float dt_reinit) {
    // 创建临时数组
    std::vector<float> phi_temp(width * height * depth);
    std::vector<float> grad_x(width * height * depth);
    std::vector<float> grad_y(width * height * depth);
    std::vector<float> grad_z(width * height * depth);
    
    // 计算符号函数
    std::vector<float> sign_phi(width * height * depth);
    #pragma omp parallel for
    for (int i = 0; i < width * height * depth; ++i) {
        sign_phi[i] = phi[i] / std::sqrt(phi[i] * phi[i] + 1e-6f);
    }
    
    // 迭代重初始化
    for (int iter = 0; iter < iterations; ++iter) {
        // 计算梯度
        compute_gradient(phi, grad_x.data(), grad_y.data(), grad_z.data(), width, height, depth);
        
        // 计算梯度大小
        #pragma omp parallel for
        for (int i = 0; i < width * height * depth; ++i) {
            float grad_mag = std::sqrt(grad_x[i]*grad_x[i] + grad_y[i]*grad_y[i] + grad_z[i]*grad_z[i]);
            
            // 更新水平集函数
            phi_temp[i] = phi[i] - dt_reinit * sign_phi[i] * (grad_mag - 1.0f);
        }
        
        // 更新水平集函数
        std::memcpy(phi, phi_temp.data(), width * height * depth * sizeof(float));
        
        // 应用边界条件
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                phi[0 * height * depth + y * depth + z] = phi[1 * height * depth + y * depth + z];
                phi[(width-1) * height * depth + y * depth + z] = phi[(width-2) * height * depth + y * depth + z];
            }
        }
        
        for (int x = 0; x < width; ++x) {
            for (int z = 0; z < depth; ++z) {
                phi[x * height * depth + 0 * depth + z] = phi[x * height * depth + 1 * depth + z];
                phi[x * height * depth + (height-1) * depth + z] = phi[x * height * depth + (height-2) * depth + z];
            }
        }
        
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                phi[x * height * depth + y * depth + 0] = phi[x * height * depth + y * depth + 1];
                phi[x * height * depth + y * depth + (depth-1)] = phi[x * height * depth + y * depth + (depth-2)];
            }
        }
    }
}

// 水平集方法的平流计算
void advect_levelset(float* phi_out, const float* phi_in, const float* velocity_field,
                    int width, int height, int depth, float dt) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                // 计算当前位置索引
                int idx = x * height * depth + y * depth + z;
                
                // 计算回溯位置
                float pos_x = x - dt * velocity_field[idx * 3 + 0];
                float pos_y = y - dt * velocity_field[idx * 3 + 1];
                float pos_z = z - dt * velocity_field[idx * 3 + 2];
                
                // 三线性插值
                phi_out[idx] = trilinear_interpolate(phi_in, width, height, depth, pos_x, pos_y, pos_z);
            }
        }
    }
    
    // 应用边界条件
    for (int y = 0; y < height; ++y) {
        for (int z = 0; z < depth; ++z) {
            phi_out[0 * height * depth + y * depth + z] = phi_out[1 * height * depth + y * depth + z];
            phi_out[(width-1) * height * depth + y * depth + z] = phi_out[(width-2) * height * depth + y * depth + z];
        }
    }
    
    for (int x = 0; x < width; ++x) {
        for (int z = 0; z < depth; ++z) {
            phi_out[x * height * depth + 0 * depth + z] = phi_out[x * height * depth + 1 * depth + z];
            phi_out[x * height * depth + (height-1) * depth + z] = phi_out[x * height * depth + (height-2) * depth + z];
        }
    }
    
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            phi_out[x * height * depth + y * depth + 0] = phi_out[x * height * depth + y * depth + 1];
            phi_out[x * height * depth + y * depth + (depth-1)] = phi_out[x * height * depth + y * depth + (depth-2)];
        }
    }
}

// 根据水平集函数更新相场
void update_phase_fields(const float* phi, float* phase_fields, 
                        int width, int height, int depth, int num_phases, float epsilon) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                int idx = x * height * depth + y * depth + z;
                
                // 使用平滑Heaviside函数
                phase_fields[0 * width * height * depth + idx] = 0.5f * (1.0f - std::tanh(phi[idx] / epsilon));
                
                // 更新其他相 (简化为二相情况)
                if (num_phases > 1) {
                    phase_fields[1 * width * height * depth + idx] = 1.0f - phase_fields[0 * width * height * depth + idx];
                }
            }
        }
    }
}

// 计算界面场
void compute_interface_field(const float* phi, float* interface_field,
                           int width, int height, int depth, float epsilon) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                int idx = x * height * depth + y * depth + z;
                
                // 计算界面场 (Heaviside函数的导数)
                float phi_val = phi[idx];
                if (phi_val < -epsilon) {
                    interface_field[idx] = 0.0f;
                } else if (phi_val > epsilon) {
                    interface_field[idx] = 0.0f;
                } else {
                    interface_field[idx] = 0.5f * (1.0f + cosf(M_PI * phi_val / epsilon)) / epsilon;
                }
            }
        }
    }
}

// 计算曲率 (用于表面张力)
void compute_curvature(const float* phi, float* curvature,
                      int width, int height, int depth) {
    // 创建临时梯度场
    std::vector<float> grad_x(width * height * depth);
    std::vector<float> grad_y(width * height * depth);
    std::vector<float> grad_z(width * height * depth);
    
    // 计算梯度
    compute_gradient(phi, grad_x.data(), grad_y.data(), grad_z.data(), width, height, depth);
    
    // 计算梯度的散度 (即曲率)
    #pragma omp parallel for collapse(3)
    for (int x = 1; x < width - 1; ++x) {
        for (int y = 1; y < height - 1; ++y) {
            for (int z = 1; z < depth - 1; ++z) {
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
                    continue;
                }
                
                // 归一化梯度
                float nx = grad_x[idx] / grad_mag;
                float ny = grad_y[idx] / grad_mag;
                float nz = grad_z[idx] / grad_mag;
                
                // 计算梯度的散度 (中心差分)
                float dnx_dx = (grad_x[(x+1) * height * depth + y * depth + z] / 
                               std::max(1e-6f, sqrtf(grad_x[(x+1) * height * depth + y * depth + z] * grad_x[(x+1) * height * depth + y * depth + z] + 
                                                   grad_y[(x+1) * height * depth + y * depth + z] * grad_y[(x+1) * height * depth + y * depth + z] + 
                                                   grad_z[(x+1) * height * depth + y * depth + z] * grad_z[(x+1) * height * depth + y * depth + z])) - 
                               grad_x[(x-1) * height * depth + y * depth + z] / 
                               std::max(1e-6f, sqrtf(grad_x[(x-1) * height * depth + y * depth + z] * grad_x[(x-1) * height * depth + y * depth + z] + 
                                                   grad_y[(x-1) * height * depth + y * depth + z] * grad_y[(x-1) * height * depth + y * depth + z] + 
                                                   grad_z[(x-1) * height * depth + y * depth + z] * grad_z[(x-1) * height * depth + y * depth + z]))) * 0.5f;
                
                float dny_dy = (grad_y[x * height * depth + (y+1) * depth + z] / 
                               std::max(1e-6f, sqrtf(grad_x[x * height * depth + (y+1) * depth + z] * grad_x[x * height * depth + (y+1) * depth + z] + 
                                                   grad_y[x * height * depth + (y+1) * depth + z] * grad_y[x * height * depth + (y+1) * depth + z] + 
                                                   grad_z[x * height * depth + (y+1) * depth + z] * grad_z[x * height * depth + (y+1) * depth + z])) - 
                               grad_y[x * height * depth + (y-1) * depth + z] / 
                               std::max(1e-6f, sqrtf(grad_x[x * height * depth + (y-1) * depth + z] * grad_x[x * height * depth + (y-1) * depth + z] + 
                                                   grad_y[x * height * depth + (y-1) * depth + z] * grad_y[x * height * depth + (y-1) * depth + z] + 
                                                   grad_z[x * height * depth + (y-1) * depth + z] * grad_z[x * height * depth + (y-1) * depth + z]))) * 0.5f;
                
                float dnz_dz = (grad_z[x * height * depth + y * depth + (z+1)] / 
                               std::max(1e-6f, sqrtf(grad_x[x * height * depth + y * depth + (z+1)] * grad_x[x * height * depth + y * depth + (z+1)] + 
                                                   grad_y[x * height * depth + y * depth + (z+1)] * grad_y[x * height * depth + y * depth + (z+1)] + 
                                                   grad_z[x * height * depth + y * depth + (z+1)] * grad_z[x * height * depth + y * depth + (z+1)])) - 
                               grad_z[x * height * depth + y * depth + (z-1)] / 
                               std::max(1e-6f, sqrtf(grad_x[x * height * depth + y * depth + (z-1)] * grad_x[x * height * depth + y * depth + (z-1)] + 
                                                   grad_y[x * height * depth + y * depth + (z-1)] * grad_y[x * height * depth + y * depth + (z-1)] + 
                                                   grad_z[x * height * depth + y * depth + (z-1)] * grad_z[x * height * depth + y * depth + (z-1)]))) * 0.5f;
                
                // 曲率是梯度场的散度
                curvature[idx] = dnx_dx + dny_dy + dnz_dz;
            }
        }
    }
    
    // 应用边界条件
    for (int y = 0; y < height; ++y) {
        for (int z = 0; z < depth; ++z) {
            curvature[0 * height * depth + y * depth + z] = curvature[1 * height * depth + y * depth + z];
            curvature[(width-1) * height * depth + y * depth + z] = curvature[(width-2) * height * depth + y * depth + z];
        }
    }
    
    for (int x = 0; x < width; ++x) {
        for (int z = 0; z < depth; ++z) {
            curvature[x * height * depth + 0 * depth + z] = curvature[x * height * depth + 1 * depth + z];
            curvature[x * height * depth + (height-1) * depth + z] = curvature[x * height * depth + (height-2) * depth + z];
        }
    }
    
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            curvature[x * height * depth + y * depth + 0] = curvature[x * height * depth + y * depth + 1];
            curvature[x * height * depth + y * depth + (depth-1)] = curvature[x * height * depth + y * depth + (depth-2)];
        }
    }
}

// 计算表面张力力
void compute_surface_tension(const float* phi, const float* curvature, float* force_field,
                           int width, int height, int depth, float sigma, float epsilon) {
    // 创建临时界面场
    std::vector<float> interface_field(width * height * depth);
    compute_interface_field(phi, interface_field.data(), width, height, depth, epsilon);
    
    // 创建临时梯度场
    std::vector<float> grad_x(width * height * depth);
    std::vector<float> grad_y(width * height * depth);
    std::vector<float> grad_z(width * height * depth);
    
    // 计算梯度
    compute_gradient(phi, grad_x.data(), grad_y.data(), grad_z.data(), width, height, depth);
    
    // 计算表面张力力
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                int idx = x * height * depth + y * depth + z;
                
                // 表面张力力 = sigma * kappa * delta(phi) * n
                // 其中 sigma 是表面张力系数，kappa 是曲率，delta(phi) 是界面场，n 是法向量
                
                // 计算梯度大小
                float grad_mag = sqrtf(
                    grad_x[idx] * grad_x[idx] + 
                    grad_y[idx] * grad_y[idx] + 
                    grad_z[idx] * grad_z[idx]
                );
                
                // 避免除以零
                grad_mag = std::max(1e-6f, grad_mag);
                
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
    }
} 