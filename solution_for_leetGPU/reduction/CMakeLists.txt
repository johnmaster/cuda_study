cmake_minimum_required(VERSION 3.18)
project(SGEMM_Benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 自动找 CUDA
find_package(CUDA REQUIRED)

add_executable(bench
    main.cu
    kernels.cu
)

target_include_directories(bench PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(bench cublas)
set_target_properties(bench PROPERTIES CUDA_ARCHITECTURES "native")