cmake_minimum_required(VERSION 3.18)
project(dot_product LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

#nvidia-smi --query-gpu=name,compute_cap --format=csv
set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(
    dot_product
    main.cu
    kernels.cu
)

target_compile_options(dot_product PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
