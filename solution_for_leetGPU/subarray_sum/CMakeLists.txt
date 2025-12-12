cmake_minimum_required(VERSION 3.18)
project(subarray_sum LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

#nvidia-smi --query-gpu=name,compute_cap --format=csv
set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(
    subarray_sum
    main.cu
    kernels.cu
)

target_compile_options(subarray_sum PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
