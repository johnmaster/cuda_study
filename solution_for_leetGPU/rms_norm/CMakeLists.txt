cmake_minimum_required(VERSION 3.18)
project(rms_norm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# nvidia-smi --query-gpu=name,compute_cap --format=csv
set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(
    rms_norm
    main.cu
    kernels.cu
)

target_compile_options(rms_norm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
