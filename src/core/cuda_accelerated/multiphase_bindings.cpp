#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <string>

namespace py = pybind11;

// 声明C++函数
void advect_vof(float* volume_fractions_out, const float* volume_fractions_in, 
                const float* velocity_field, int width, int height, int depth, 
                int phase_idx, int num_phases, float dt);

void normalize_volume_fractions(float* volume_fractions, int width, int height, int depth, int num_phases);

void reinitialize_levelset(float* phi, int width, int height, int depth, 
                          int iterations, float dt_reinit);

void advect_levelset(float* phi_out, const float* phi_in, const float* velocity_field,
                    int width, int height, int depth, float dt);

void update_phase_fields(const float* phi, float* phase_fields, 
                        int width, int height, int depth, int num_phases, float epsilon);

void compute_interface_field(const float* phi, float* interface_field,
                           int width, int height, int depth, float epsilon);

// 添加表面张力相关函数声明
void compute_curvature(const float* phi, float* curvature,
                      int width, int height, int depth);

void compute_surface_tension(const float* phi, const float* curvature, float* force_field,
                           int width, int height, int depth, float sigma, float epsilon);

// 声明CUDA函数
#ifdef CUDA_AVAILABLE
extern "C" {
    void advect_vof_cuda(float* volume_fractions_out, const float* volume_fractions_in,
                       const float* velocity_field, int width, int height, int depth,
                       int phase_idx, int num_phases, float dt);
    
    void normalize_volume_fractions_cuda(float* volume_fractions, int width, int height, int depth,
                                       int num_phases);
    
    void reinitialize_levelset_cuda(float* phi, int width, int height, int depth,
                                  int iterations, float dt_reinit);
    
    void advect_levelset_cuda(float* phi_out, const float* phi_in, const float* velocity_field,
                            int width, int height, int depth, float dt);
    
    void update_phase_fields_cuda(const float* phi, float* phase_fields,
                                int width, int height, int depth, int num_phases, float epsilon);
    
    void compute_interface_field_cuda(const float* phi, float* interface_field,
                                   int width, int height, int depth, float epsilon);
    
    // 添加表面张力相关CUDA函数声明
    void compute_curvature_cuda(const float* phi, float* curvature,
                              int width, int height, int depth);
    
    void compute_surface_tension_cuda(const float* phi, const float* curvature, float* force_field,
                                    int width, int height, int depth, float sigma, float epsilon);
}
#endif

// 检查CUDA是否可用
bool is_cuda_available() {
#ifdef CUDA_AVAILABLE
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

// Python包装函数 - VOF平流
py::array_t<float> py_advect_vof(py::array_t<float> volume_fractions,
                                py::array_t<float> velocity_field,
                                int phase_idx, float dt, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info vf_buf = volume_fractions.request();
    py::buffer_info vel_buf = velocity_field.request();
    
    if (vf_buf.ndim != 4) {
        throw std::runtime_error("Volume fractions array must be 4-dimensional");
    }
    
    if (vel_buf.ndim != 4) {
        throw std::runtime_error("Velocity field array must be 4-dimensional");
    }
    
    if (vel_buf.shape[3] != 3) {
        throw std::runtime_error("Velocity field must have 3 components (x, y, z)");
    }
    
    // 获取维度信息
    int num_phases = vf_buf.shape[0];
    int width = vf_buf.shape[1];
    int height = vf_buf.shape[2];
    int depth = vf_buf.shape[3];
    
    // 创建输出数组
    py::array_t<float> result = py::array_t<float>({num_phases, width, height, depth});
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* vf_in = static_cast<float*>(vf_buf.ptr);
    float* vf_out = static_cast<float*>(result_buf.ptr);
    float* vel = static_cast<float*>(vel_buf.ptr);
    
    // 复制输入到输出（以保留其他相的数据）
    std::memcpy(vf_out, vf_in, num_phases * width * height * depth * sizeof(float));
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        advect_vof_cuda(vf_out, vf_in, vel, width, height, depth, phase_idx, num_phases, dt);
#endif
    } else {
        // 使用CPU实现
        advect_vof(vf_out, vf_in, vel, width, height, depth, phase_idx, num_phases, dt);
    }
    
    return result;
}

// Python包装函数 - 归一化体积分数
py::array_t<float> py_normalize_volume_fractions(py::array_t<float> volume_fractions, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info vf_buf = volume_fractions.request();
    
    if (vf_buf.ndim != 4) {
        throw std::runtime_error("Volume fractions array must be 4-dimensional");
    }
    
    // 获取维度信息
    int num_phases = vf_buf.shape[0];
    int width = vf_buf.shape[1];
    int height = vf_buf.shape[2];
    int depth = vf_buf.shape[3];
    
    // 创建输出数组（复制输入）
    py::array_t<float> result = volume_fractions.attr("copy")();
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* vf = static_cast<float*>(result_buf.ptr);
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        normalize_volume_fractions_cuda(vf, width, height, depth, num_phases);
#endif
    } else {
        // 使用CPU实现
        normalize_volume_fractions(vf, width, height, depth, num_phases);
    }
    
    return result;
}

// Python包装函数 - 水平集重初始化
py::array_t<float> py_reinitialize_levelset(py::array_t<float> phi,
                                          int iterations, float dt_reinit, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info phi_buf = phi.request();
    
    if (phi_buf.ndim != 3) {
        throw std::runtime_error("Phi array must be 3-dimensional");
    }
    
    // 获取维度信息
    int width = phi_buf.shape[0];
    int height = phi_buf.shape[1];
    int depth = phi_buf.shape[2];
    
    // 创建输出数组（复制输入）
    py::array_t<float> result = phi.attr("copy")();
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* phi_data = static_cast<float*>(result_buf.ptr);
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        reinitialize_levelset_cuda(phi_data, width, height, depth, iterations, dt_reinit);
#endif
    } else {
        // 使用CPU实现
        reinitialize_levelset(phi_data, width, height, depth, iterations, dt_reinit);
    }
    
    return result;
}

// Python包装函数 - 水平集平流
py::array_t<float> py_advect_levelset(py::array_t<float> phi,
                                    py::array_t<float> velocity_field,
                                    float dt, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info phi_buf = phi.request();
    py::buffer_info vel_buf = velocity_field.request();
    
    if (phi_buf.ndim != 3) {
        throw std::runtime_error("Phi array must be 3-dimensional");
    }
    
    if (vel_buf.ndim != 4) {
        throw std::runtime_error("Velocity field array must be 4-dimensional");
    }
    
    if (vel_buf.shape[3] != 3) {
        throw std::runtime_error("Velocity field must have 3 components (x, y, z)");
    }
    
    // 获取维度信息
    int width = phi_buf.shape[0];
    int height = phi_buf.shape[1];
    int depth = phi_buf.shape[2];
    
    // 创建输出数组
    py::array_t<float> result = py::array_t<float>({width, height, depth});
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* phi_in = static_cast<float*>(phi_buf.ptr);
    float* phi_out = static_cast<float*>(result_buf.ptr);
    float* vel = static_cast<float*>(vel_buf.ptr);
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        advect_levelset_cuda(phi_out, phi_in, vel, width, height, depth, dt);
#endif
    } else {
        // 使用CPU实现
        advect_levelset(phi_out, phi_in, vel, width, height, depth, dt);
    }
    
    return result;
}

// Python包装函数 - 更新相场
py::array_t<float> py_update_phase_fields(py::array_t<float> phi, int num_phases, float epsilon, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info phi_buf = phi.request();
    
    if (phi_buf.ndim != 3) {
        throw std::runtime_error("Phi array must be 3-dimensional");
    }
    
    // 获取维度信息
    int width = phi_buf.shape[0];
    int height = phi_buf.shape[1];
    int depth = phi_buf.shape[2];
    
    // 创建输出数组
    py::array_t<float> result = py::array_t<float>({num_phases, width, height, depth});
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* phi_data = static_cast<float*>(phi_buf.ptr);
    float* phase_fields = static_cast<float*>(result_buf.ptr);
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        update_phase_fields_cuda(phi_data, phase_fields, width, height, depth, num_phases, epsilon);
#endif
    } else {
        // 使用CPU实现
        update_phase_fields(phi_data, phase_fields, width, height, depth, num_phases, epsilon);
    }
    
    return result;
}

// Python包装函数 - 计算界面场
py::array_t<float> py_compute_interface_field(py::array_t<float> phi, float epsilon, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info phi_buf = phi.request();
    
    if (phi_buf.ndim != 3) {
        throw std::runtime_error("Phi array must be 3-dimensional");
    }
    
    // 获取维度信息
    int width = phi_buf.shape[0];
    int height = phi_buf.shape[1];
    int depth = phi_buf.shape[2];
    
    // 创建输出数组
    py::array_t<float> result = py::array_t<float>({width, height, depth});
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* phi_data = static_cast<float*>(phi_buf.ptr);
    float* interface_field = static_cast<float*>(result_buf.ptr);
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        compute_interface_field_cuda(phi_data, interface_field, width, height, depth, epsilon);
#endif
    } else {
        // 使用CPU实现
        compute_interface_field(phi_data, interface_field, width, height, depth, epsilon);
    }
    
    return result;
}

// Python包装函数 - 计算曲率
py::array_t<float> py_compute_curvature(py::array_t<float> phi, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info phi_buf = phi.request();
    
    if (phi_buf.ndim != 3) {
        throw std::runtime_error("Phi array must be 3-dimensional");
    }
    
    // 获取维度信息
    int width = phi_buf.shape[0];
    int height = phi_buf.shape[1];
    int depth = phi_buf.shape[2];
    
    // 创建输出数组
    py::array_t<float> result = py::array_t<float>({width, height, depth});
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* phi_data = static_cast<float*>(phi_buf.ptr);
    float* result_data = static_cast<float*>(result_buf.ptr);
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        compute_curvature_cuda(phi_data, result_data, width, height, depth);
#endif
    } else {
        // 使用CPU实现
        compute_curvature(phi_data, result_data, width, height, depth);
    }
    
    return result;
}

// Python包装函数 - 计算表面张力力
py::array_t<float> py_compute_surface_tension(py::array_t<float> phi, py::array_t<float> curvature, 
                                            float sigma, float epsilon, bool use_cuda = true) {
    // 获取NumPy数组信息
    py::buffer_info phi_buf = phi.request();
    py::buffer_info curv_buf = curvature.request();
    
    if (phi_buf.ndim != 3) {
        throw std::runtime_error("Phi array must be 3-dimensional");
    }
    
    if (curv_buf.ndim != 3) {
        throw std::runtime_error("Curvature array must be 3-dimensional");
    }
    
    // 获取维度信息
    int width = phi_buf.shape[0];
    int height = phi_buf.shape[1];
    int depth = phi_buf.shape[2];
    
    // 创建输出数组
    py::array_t<float> result = py::array_t<float>({width, height, depth, 3});
    py::buffer_info result_buf = result.request();
    
    // 获取数据指针
    float* phi_data = static_cast<float*>(phi_buf.ptr);
    float* curv_data = static_cast<float*>(curv_buf.ptr);
    float* result_data = static_cast<float*>(result_buf.ptr);
    
    // 调用计算函数
    bool cuda_available = is_cuda_available();
    if (use_cuda && cuda_available) {
#ifdef CUDA_AVAILABLE
        // 使用CUDA实现
        compute_surface_tension_cuda(phi_data, curv_data, result_data, width, height, depth, sigma, epsilon);
#endif
    } else {
        // 使用CPU实现
        compute_surface_tension(phi_data, curv_data, result_data, width, height, depth, sigma, epsilon);
    }
    
    return result;
}

PYBIND11_MODULE(multiphase_core, m) {
    m.doc() = "C++/CUDA accelerated multiphase flow model";
    
    m.def("is_cuda_available", &is_cuda_available, "Check if CUDA is available");
    
    m.def("advect_vof", &py_advect_vof, 
          py::arg("volume_fractions"), 
          py::arg("velocity_field"), 
          py::arg("phase_idx"), 
          py::arg("dt"), 
          py::arg("use_cuda") = true,
          "Advect volume fractions using VOF method");
    
    m.def("normalize_volume_fractions", &py_normalize_volume_fractions, 
          py::arg("volume_fractions"), 
          py::arg("use_cuda") = true,
          "Normalize volume fractions");
    
    m.def("reinitialize_levelset", &py_reinitialize_levelset, 
          py::arg("phi"), 
          py::arg("iterations"), 
          py::arg("dt_reinit"), 
          py::arg("use_cuda") = true,
          "Reinitialize level set function");
    
    m.def("advect_levelset", &py_advect_levelset, 
          py::arg("phi"), 
          py::arg("velocity_field"), 
          py::arg("dt"), 
          py::arg("use_cuda") = true,
          "Advect level set function");
    
    m.def("update_phase_fields", &py_update_phase_fields, 
          py::arg("phi"), 
          py::arg("num_phases"), 
          py::arg("epsilon"), 
          py::arg("use_cuda") = true,
          "Update phase fields from level set function");
    
    m.def("compute_interface_field", &py_compute_interface_field, 
          py::arg("phi"), 
          py::arg("epsilon"), 
          py::arg("use_cuda") = true,
          "Compute interface field from level set function");
    
    // 添加表面张力相关函数
    m.def("compute_curvature", &py_compute_curvature, 
          py::arg("phi"), 
          py::arg("use_cuda") = true,
          "Compute curvature from level set function");
    
    m.def("compute_surface_tension", &py_compute_surface_tension, 
          py::arg("phi"), 
          py::arg("curvature"), 
          py::arg("sigma"), 
          py::arg("epsilon"), 
          py::arg("use_cuda") = true,
          "Compute surface tension force from level set function and curvature");
} 