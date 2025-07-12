#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// 声明CUDA函数
extern "C" {
    void cuda_diffuse(float* field, float* field_prev, int width, int height, int depth, 
                     float alpha, float beta, int iterations, int components);
    
    void cuda_advect(float* field, float* field_prev, float* velocity, 
                    int width, int height, int depth, float dt, int components);
    
    void cuda_project(float* velocity, float* pressure, float* divergence,
                     int width, int height, int depth, float dt, int iterations);
    
    void cuda_compute_vorticity(float* velocity, float* vorticity,
                              int width, int height, int depth);
}

// Python包装函数
py::array_t<float> py_cuda_diffuse(py::array_t<float> field, py::array_t<float> field_prev, 
                                 float alpha, float beta, int iterations) {
    auto buf_field = field.request();
    auto buf_field_prev = field_prev.request();
    
    if (buf_field.ndim < 3 || buf_field_prev.ndim < 3) {
        throw std::runtime_error("输入数组维度必须至少为3");
    }
    
    int components = 1;
    int width, height, depth;
    
    if (buf_field.ndim == 4) {
        // 矢量场
        width = buf_field.shape[0];
        height = buf_field.shape[1];
        depth = buf_field.shape[2];
        components = buf_field.shape[3];
    } else {
        // 标量场
        width = buf_field.shape[0];
        height = buf_field.shape[1];
        depth = buf_field.shape[2];
    }
    
    // 调用CUDA函数
    cuda_diffuse(
        static_cast<float*>(buf_field.ptr),
        static_cast<float*>(buf_field_prev.ptr),
        width, height, depth, alpha, beta, iterations, components
    );
    
    return field;
}

py::array_t<float> py_cuda_advect(py::array_t<float> field, py::array_t<float> field_prev, 
                                py::array_t<float> velocity, float dt) {
    auto buf_field = field.request();
    auto buf_field_prev = field_prev.request();
    auto buf_velocity = velocity.request();
    
    if (buf_field.ndim < 3 || buf_field_prev.ndim < 3 || buf_velocity.ndim != 4) {
        throw std::runtime_error("输入数组维度不正确");
    }
    
    int components = 1;
    if (buf_field.ndim == 4) {
        components = buf_field.shape[3];
    }
    
    int width = buf_field.shape[0];
    int height = buf_field.shape[1];
    int depth = buf_field.shape[2];
    
    // 调用CUDA函数
    cuda_advect(
        static_cast<float*>(buf_field.ptr),
        static_cast<float*>(buf_field_prev.ptr),
        static_cast<float*>(buf_velocity.ptr),
        width, height, depth, dt, components
    );
    
    return field;
}

py::array_t<float> py_cuda_project(py::array_t<float> velocity, py::array_t<float> pressure, 
                                 py::array_t<float> divergence, float dt, int iterations) {
    auto buf_velocity = velocity.request();
    auto buf_pressure = pressure.request();
    auto buf_divergence = divergence.request();
    
    if (buf_velocity.ndim != 4 || buf_pressure.ndim != 3 || buf_divergence.ndim != 3) {
        throw std::runtime_error("输入数组维度不正确");
    }
    
    int width = buf_velocity.shape[0];
    int height = buf_velocity.shape[1];
    int depth = buf_velocity.shape[2];
    
    // 调用CUDA函数
    cuda_project(
        static_cast<float*>(buf_velocity.ptr),
        static_cast<float*>(buf_pressure.ptr),
        static_cast<float*>(buf_divergence.ptr),
        width, height, depth, dt, iterations
    );
    
    return velocity;
}

py::array_t<float> py_cuda_compute_vorticity(py::array_t<float> velocity) {
    auto buf_velocity = velocity.request();
    
    if (buf_velocity.ndim != 4) {
        throw std::runtime_error("速度场必须是4维数组");
    }
    
    int width = buf_velocity.shape[0];
    int height = buf_velocity.shape[1];
    int depth = buf_velocity.shape[2];
    
    // 创建涡量场
    auto vorticity = py::array_t<float>({width, height, depth, 3});
    auto buf_vorticity = vorticity.request();
    
    // 调用CUDA函数
    cuda_compute_vorticity(
        static_cast<float*>(buf_velocity.ptr),
        static_cast<float*>(buf_vorticity.ptr),
        width, height, depth
    );
    
    return vorticity;
}

// 模块定义
PYBIND11_MODULE(cuda_kernels, m) {
    m.doc() = "CUDA加速的流体模拟核心函数";
    
    m.def("diffuse", &py_cuda_diffuse, "CUDA加速的扩散计算",
          py::arg("field"), py::arg("field_prev"), 
          py::arg("alpha"), py::arg("beta"), py::arg("iterations") = 20);
    
    m.def("advect", &py_cuda_advect, "CUDA加速的平流计算",
          py::arg("field"), py::arg("field_prev"), py::arg("velocity"), py::arg("dt"));
    
    m.def("project", &py_cuda_project, "CUDA加速的投影计算",
          py::arg("velocity"), py::arg("pressure"), py::arg("divergence"), 
          py::arg("dt"), py::arg("iterations") = 50);
    
    m.def("compute_vorticity", &py_cuda_compute_vorticity, "CUDA加速的涡量计算",
          py::arg("velocity"));
} 