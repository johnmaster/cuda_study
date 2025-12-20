cmake_minimum_required(VERSION 3.18)
project(Reduce2D_Benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 自动找 CUDA
find_package(CUDA REQUIRED)

add_executable(reduce_2d
    main.cu
    kernels.cu
)

target_include_directories(reduce_2d PRIVATE ${CUDA_INCLUDE_DIRS})
set_target_properties(reduce_2d PROPERTIES CUDA_ARCHITECTURES "native")

