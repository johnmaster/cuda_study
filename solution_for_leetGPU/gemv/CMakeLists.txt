cmake_minimum_required(VERSION 3.18)
project(SGEMV_Benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

add_executable(gemv
    main.cpp
    kernels.cu
)

target_include_directories(gemv PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(gemv cublas)
set_target_properties(gemv PROPERTIES CUDA_ARCHITECTURES "native")

