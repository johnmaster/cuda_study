cmake_minimum_required(VERSION 3.18)
project(wmma_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# 检测 GPU 架构，默认 sm_70 (Volta) 或更高
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# 添加可执行文件
add_executable(wmma_gemm main.cu)

# 链接 cuBLAS
target_link_libraries(wmma_gemm cublas)

