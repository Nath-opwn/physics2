cmake_minimum_required(VERSION 3.10)
project(multiphase_core LANGUAGES CXX)

# 检查CUDA是否可用
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DCUDA_AVAILABLE)
    message(STATUS "CUDA found, enabling CUDA support")
else()
    message(STATUS "CUDA not found, disabling CUDA support")
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果CUDA可用，设置CUDA标准
if(CUDA_FOUND)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# 查找Python和pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 使用Python模块中的pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# 查找OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 如果CUDA可用，添加CUDA编译选项
if(CUDA_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    
    # 检测CUDA架构并设置适当的编译标志
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70 75 80 CACHE STRING "CUDA architectures" FORCE)
    endif()
endif()

# 添加C++源文件
set(CPP_SOURCES
    multiphase_core.cpp
)

# 如果CUDA可用，添加CUDA源文件
if(CUDA_FOUND)
    set(CUDA_SOURCES
        multiphase_cuda.cu
    )
    # 编译CUDA源文件为目标库
    add_library(multiphase_cuda_lib STATIC ${CUDA_SOURCES})
    target_compile_features(multiphase_cuda_lib PUBLIC cxx_std_14)
    set_target_properties(multiphase_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    # 创建一个空的CUDA库，以便在没有CUDA的情况下也能编译
    add_library(multiphase_cuda_lib INTERFACE)
endif()

# 添加Python绑定源文件
set(BINDING_SOURCES
    multiphase_bindings.cpp
)

# 编译C++源文件为目标库
add_library(multiphase_cpp_lib STATIC ${CPP_SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(multiphase_cpp_lib PRIVATE OpenMP::OpenMP_CXX)
endif()

# 创建Python模块
pybind11_add_module(multiphase_core ${BINDING_SOURCES})
target_link_libraries(multiphase_core PRIVATE multiphase_cpp_lib)
if(CUDA_FOUND)
    target_link_libraries(multiphase_core PRIVATE multiphase_cuda_lib)
endif()

# 设置输出目录
set_target_properties(multiphase_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../
)

# 安装目标
install(TARGETS multiphase_core DESTINATION ${CMAKE_SOURCE_DIR}/../../) 

# 添加测试目标
if(CUDA_FOUND)
    add_custom_target(run_tests
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../../test_acceleration.py
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../../test_surface_tension.py
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../../test_contact_angle.py
        DEPENDS multiphase_core
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../../
        COMMENT "Running tests with CUDA support"
    )
else()
    add_custom_target(run_tests
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../../test_acceleration.py
        DEPENDS multiphase_core
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../../
        COMMENT "Running tests without CUDA support"
    )
endif() 