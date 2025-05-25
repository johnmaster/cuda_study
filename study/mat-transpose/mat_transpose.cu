#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// 将行优先排列的矩阵转变为列优先排列的矩阵
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y,
                                                 const int row, const int col) {
    const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int global_row = global_idx / col;
    const int global_col = global_idx % col;
    if (global_idx < row * col)
        y[global_col * row + global_row] = x[global_idx];
}

// 将列优先排列的矩阵转变为行优先排列的矩阵
__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y,
                                                 const int row, const int col) {
    const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int global_col = global_idx / row;
    const int global_row = global_idx % row;
    if (global_idx < row * col)
        y[global_idx] = x[global_row * col + global_col];
}

__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y, const int row,
                                                    const int col) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int global_col = (global_idx * 4) % col;
    int global_row = (global_idx * 4) / col;
    if (global_row < row && global_col + 3 < col) {
        float4 x_val = reinterpret_cast<float4 *>(x)[global_idx];
        y[global_col * row + global_row] = x_val.x;
        y[(global_col + 1) * row + global_row] = x_val.y;
        y[(global_col + 2) * row + global_row] = x_val.z;
        y[(global_col + 3) * row + global_row] = x_val.w;
    }
}

__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y,
                                                   const int row,
                                                   const int col) {
    const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int global_col = (global_idx * 4) / row;
    const int global_row = (global_idx * 4) % row;
    if (global_row + 3 < col && global_col < col) {
        float4 x_val;
        x_val.x = x[global_row * col + global_col];
        x_val.y = x[(global_row + 1) * col + global_col];
        x_val.z = x[(global_row + 2) * col + global_col];
        x_val.w = x[(global_row + 3) * col + global_col];
        reinterpret_cast<float4 *>(y)[global_idx] = FLOAT4(x_val);
    }
}



#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)    \
    m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if (((T).options().dtype() != (th_type))) {                                \
      std::cout << "Tensor Info:" << (T).options() << std::endl;               \
      throw std::runtime_error("values must be " #th_type);                    \
    }

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)         \
    void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                  \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                  \
        const int M = x.size(0);                                                \
        const int N = x.size(1);                                                \
        dim3 block(WARP_SIZE);                                                  \
        dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));              \
        mat_transpose_##tag##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                     \
            reinterpret_cast<element_type *>(y.data_ptr()), N                   \
        );                                                                      \
    }

