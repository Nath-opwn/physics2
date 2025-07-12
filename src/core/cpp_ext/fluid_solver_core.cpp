#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

// 编译命令:
// g++ -O3 -fPIC -shared -o fluid_solver_core.so fluid_solver_core.cpp -I/usr/include/python3.x -I/usr/lib/python3.x/site-packages/numpy/core/include -fopenmp

// 线性求解器最大迭代次数
#define MAX_ITERATIONS 50
// 收敛阈值
#define TOLERANCE 1e-5

// 辅助函数：检查NumPy数组维度和类型
static bool check_array(PyArrayObject *array, int ndim) {
    if (array == NULL || !PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError, "参数必须是NumPy数组");
        return false;
    }
    
    if (PyArray_NDIM(array) != ndim) {
        PyErr_SetString(PyExc_ValueError, "数组维度不正确");
        return false;
    }
    
    if (PyArray_TYPE(array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "数组类型必须是float64");
        return false;
    }
    
    return true;
}

// 扩散步骤的C++实现
static PyObject* diffuse(PyObject* self, PyObject* args) {
    PyArrayObject *x, *x0;
    double diff, dt;
    int iterations;
    
    // 解析Python参数
    if (!PyArg_ParseTuple(args, "O!O!ddi", 
                          &PyArray_Type, &x,
                          &PyArray_Type, &x0,
                          &diff, &dt, &iterations)) {
        return NULL;
    }
    
    // 检查数组
    if (!check_array(x, 3) || !check_array(x0, 3)) {
        return NULL;
    }
    
    // 获取数组维度
    npy_intp *dims = PyArray_DIMS(x);
    int depth = (int)dims[0];
    int height = (int)dims[1];
    int width = (int)dims[2];
    
    // 计算扩散系数
    double a = dt * diff * width * height * depth;
    
    // 获取数据指针
    double *x_data = (double*)PyArray_DATA(x);
    double *x0_data = (double*)PyArray_DATA(x0);
    
    // 执行扩散计算
    for (int k = 0; k < iterations; k++) {
        #pragma omp parallel for collapse(3)
        for (int z = 1; z < depth - 1; z++) {
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    int idx = z * height * width + y * width + x;
                    int idx_up = (z+1) * height * width + y * width + x;
                    int idx_down = (z-1) * height * width + y * width + x;
                    int idx_left = z * height * width + y * width + (x-1);
                    int idx_right = z * height * width + y * width + (x+1);
                    int idx_front = z * height * width + (y-1) * width + x;
                    int idx_back = z * height * width + (y+1) * width + x;
                    
                    x_data[idx] = (x0_data[idx] + a * (
                        x_data[idx_up] + x_data[idx_down] +
                        x_data[idx_left] + x_data[idx_right] +
                        x_data[idx_front] + x_data[idx_back]
                    )) / (1 + 6 * a);
                }
            }
        }
    }
    
    Py_RETURN_NONE;
}

// 平流步骤的C++实现
static PyObject* advect(PyObject* self, PyObject* args) {
    PyArrayObject *d, *d0, *u, *v, *w;
    double dt;
    
    // 解析Python参数
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!d", 
                          &PyArray_Type, &d,
                          &PyArray_Type, &d0,
                          &PyArray_Type, &u,
                          &PyArray_Type, &v,
                          &PyArray_Type, &w,
                          &dt)) {
        return NULL;
    }
    
    // 检查数组
    if (!check_array(d, 3) || !check_array(d0, 3) || 
        !check_array(u, 3) || !check_array(v, 3) || !check_array(w, 3)) {
        return NULL;
    }
    
    // 获取数组维度
    npy_intp *dims = PyArray_DIMS(d);
    int depth = (int)dims[0];
    int height = (int)dims[1];
    int width = (int)dims[2];
    
    // 获取数据指针
    double *d_data = (double*)PyArray_DATA(d);
    double *d0_data = (double*)PyArray_DATA(d0);
    double *u_data = (double*)PyArray_DATA(u);
    double *v_data = (double*)PyArray_DATA(v);
    double *w_data = (double*)PyArray_DATA(w);
    
    // 执行平流计算
    #pragma omp parallel for collapse(3)
    for (int z = 1; z < depth - 1; z++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = z * height * width + y * width + x;
                
                // 回溯粒子位置
                double x_pos = x - dt * u_data[idx] * width;
                double y_pos = y - dt * v_data[idx] * height;
                double z_pos = z - dt * w_data[idx] * depth;
                
                // 确保位置在边界内
                x_pos = std::max(0.5, std::min((double)width - 1.5, x_pos));
                y_pos = std::max(0.5, std::min((double)height - 1.5, y_pos));
                z_pos = std::max(0.5, std::min((double)depth - 1.5, z_pos));
                
                // 计算插值索引
                int i0 = (int)x_pos;
                int i1 = i0 + 1;
                int j0 = (int)y_pos;
                int j1 = j0 + 1;
                int k0 = (int)z_pos;
                int k1 = k0 + 1;
                
                // 计算插值权重
                double s1 = x_pos - i0;
                double s0 = 1 - s1;
                double t1 = y_pos - j0;
                double t0 = 1 - t1;
                double u1 = z_pos - k0;
                double u0 = 1 - u1;
                
                // 三线性插值
                d_data[idx] = 
                    u0 * (t0 * (s0 * d0_data[k0 * height * width + j0 * width + i0] +
                                s1 * d0_data[k0 * height * width + j0 * width + i1]) +
                          t1 * (s0 * d0_data[k0 * height * width + j1 * width + i0] +
                                s1 * d0_data[k0 * height * width + j1 * width + i1])) +
                    u1 * (t0 * (s0 * d0_data[k1 * height * width + j0 * width + i0] +
                                s1 * d0_data[k1 * height * width + j0 * width + i1]) +
                          t1 * (s0 * d0_data[k1 * height * width + j1 * width + i0] +
                                s1 * d0_data[k1 * height * width + j1 * width + i1]));
            }
        }
    }
    
    Py_RETURN_NONE;
}

// 投影步骤的C++实现
static PyObject* project(PyObject* self, PyObject* args) {
    PyArrayObject *u, *v, *w, *p, *div;
    int iterations;
    
    // 解析Python参数
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!i", 
                          &PyArray_Type, &u,
                          &PyArray_Type, &v,
                          &PyArray_Type, &w,
                          &PyArray_Type, &p,
                          &PyArray_Type, &div,
                          &iterations)) {
        return NULL;
    }
    
    // 检查数组
    if (!check_array(u, 3) || !check_array(v, 3) || !check_array(w, 3) ||
        !check_array(p, 3) || !check_array(div, 3)) {
        return NULL;
    }
    
    // 获取数组维度
    npy_intp *dims = PyArray_DIMS(u);
    int depth = (int)dims[0];
    int height = (int)dims[1];
    int width = (int)dims[2];
    
    // 获取数据指针
    double *u_data = (double*)PyArray_DATA(u);
    double *v_data = (double*)PyArray_DATA(v);
    double *w_data = (double*)PyArray_DATA(w);
    double *p_data = (double*)PyArray_DATA(p);
    double *div_data = (double*)PyArray_DATA(div);
    
    // 计算速度场的散度
    #pragma omp parallel for collapse(3)
    for (int z = 1; z < depth - 1; z++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = z * height * width + y * width + x;
                int idx_right = z * height * width + y * width + (x+1);
                int idx_left = z * height * width + y * width + (x-1);
                int idx_back = z * height * width + (y+1) * width + x;
                int idx_front = z * height * width + (y-1) * width + x;
                int idx_up = (z+1) * height * width + y * width + x;
                int idx_down = (z-1) * height * width + y * width + x;
                
                div_data[idx] = -0.5 * (
                    (u_data[idx_right] - u_data[idx_left]) / width +
                    (v_data[idx_back] - v_data[idx_front]) / height +
                    (w_data[idx_up] - w_data[idx_down]) / depth
                );
                p_data[idx] = 0.0;
            }
        }
    }
    
    // 求解泊松方程
    for (int k = 0; k < iterations; k++) {
        #pragma omp parallel for collapse(3)
        for (int z = 1; z < depth - 1; z++) {
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    int idx = z * height * width + y * width + x;
                    int idx_right = z * height * width + y * width + (x+1);
                    int idx_left = z * height * width + y * width + (x-1);
                    int idx_back = z * height * width + (y+1) * width + x;
                    int idx_front = z * height * width + (y-1) * width + x;
                    int idx_up = (z+1) * height * width + y * width + x;
                    int idx_down = (z-1) * height * width + y * width + x;
                    
                    p_data[idx] = (div_data[idx] + 
                        p_data[idx_right] + p_data[idx_left] +
                        p_data[idx_back] + p_data[idx_front] +
                        p_data[idx_up] + p_data[idx_down]
                    ) / 6.0;
                }
            }
        }
    }
    
    // 应用压力梯度
    #pragma omp parallel for collapse(3)
    for (int z = 1; z < depth - 1; z++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = z * height * width + y * width + x;
                int idx_right = z * height * width + y * width + (x+1);
                int idx_left = z * height * width + y * width + (x-1);
                int idx_back = z * height * width + (y+1) * width + x;
                int idx_front = z * height * width + (y-1) * width + x;
                int idx_up = (z+1) * height * width + y * width + x;
                int idx_down = (z-1) * height * width + y * width + x;
                
                u_data[idx] -= 0.5 * width * (p_data[idx_right] - p_data[idx_left]);
                v_data[idx] -= 0.5 * height * (p_data[idx_back] - p_data[idx_front]);
                w_data[idx] -= 0.5 * depth * (p_data[idx_up] - p_data[idx_down]);
            }
        }
    }
    
    Py_RETURN_NONE;
}

// 涡量计算的C++实现
static PyObject* compute_vorticity(PyObject* self, PyObject* args) {
    PyArrayObject *u, *v, *w, *vorticity;
    
    // 解析Python参数
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                          &PyArray_Type, &u,
                          &PyArray_Type, &v,
                          &PyArray_Type, &w,
                          &PyArray_Type, &vorticity)) {
        return NULL;
    }
    
    // 检查数组
    if (!check_array(u, 3) || !check_array(v, 3) || 
        !check_array(w, 3) || !check_array(vorticity, 3)) {
        return NULL;
    }
    
    // 获取数组维度
    npy_intp *dims = PyArray_DIMS(u);
    int depth = (int)dims[0];
    int height = (int)dims[1];
    int width = (int)dims[2];
    
    // 获取数据指针
    double *u_data = (double*)PyArray_DATA(u);
    double *v_data = (double*)PyArray_DATA(v);
    double *w_data = (double*)PyArray_DATA(w);
    double *vorticity_data = (double*)PyArray_DATA(vorticity);
    
    // 计算涡量
    #pragma omp parallel for collapse(3)
    for (int z = 1; z < depth - 1; z++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = z * height * width + y * width + x;
                int idx_right = z * height * width + y * width + (x+1);
                int idx_left = z * height * width + y * width + (x-1);
                int idx_back = z * height * width + (y+1) * width + x;
                int idx_front = z * height * width + (y-1) * width + x;
                int idx_up = (z+1) * height * width + y * width + x;
                int idx_down = (z-1) * height * width + y * width + x;
                
                // 计算速度梯度
                double du_dy = (u_data[idx_back] - u_data[idx_front]) / (2.0 * height);
                double du_dz = (u_data[idx_up] - u_data[idx_down]) / (2.0 * depth);
                double dv_dx = (v_data[idx_right] - v_data[idx_left]) / (2.0 * width);
                double dv_dz = (v_data[idx_up] - v_data[idx_down]) / (2.0 * depth);
                double dw_dx = (w_data[idx_right] - w_data[idx_left]) / (2.0 * width);
                double dw_dy = (w_data[idx_back] - w_data[idx_front]) / (2.0 * height);
                
                // 计算涡量大小 (|ω| = |∇ × v|)
                double curl_x = dw_dy - dv_dz;
                double curl_y = du_dz - dw_dx;
                double curl_z = dv_dx - du_dy;
                
                vorticity_data[idx] = sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z);
            }
        }
    }
    
    Py_RETURN_NONE;
}

// 模块方法定义
static PyMethodDef FluidSolverMethods[] = {
    {"diffuse", diffuse, METH_VARARGS, "执行扩散步骤"},
    {"advect", advect, METH_VARARGS, "执行平流步骤"},
    {"project", project, METH_VARARGS, "执行投影步骤"},
    {"compute_vorticity", compute_vorticity, METH_VARARGS, "计算涡量场"},
    {NULL, NULL, 0, NULL} // 哨兵
};

// 模块定义
static struct PyModuleDef fluid_solver_module = {
    PyModuleDef_HEAD_INIT,
    "fluid_solver_core",   // 模块名
    "流体求解器的C++扩展", // 模块文档
    -1,                    // 每个解释器状态的大小
    FluidSolverMethods     // 方法表
};

// 初始化模块
PyMODINIT_FUNC PyInit_fluid_solver_core(void) {
    import_array(); // 初始化NumPy
    return PyModule_Create(&fluid_solver_module);
} 