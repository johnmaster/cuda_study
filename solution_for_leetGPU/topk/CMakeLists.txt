cmake_minimum_required(VERSION 3.18)
project(topk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构 (根据你的 GPU 调整)
set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(topk
    main.cu
    kernels.cu
)

target_compile_options(topk PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)


