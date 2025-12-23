cmake_minimum_required(VERSION 3.18)
project(LayerNorm_Benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

add_executable(layer_norm
    main.cu
    kernels.cu
)

target_include_directories(layer_norm PRIVATE ${CUDA_INCLUDE_DIRS})
set_target_properties(layer_norm PROPERTIES CUDA_ARCHITECTURES "native")

