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
                                                 const int row,
                                                 const int col) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx / col;
    const int global_col = global_idx % col;
    if (global_idx < row * col)
        y[global_col * row + global_row] = x[global_row * col + global_col];
}

// 将列优先排列的矩阵转变为行优先排列的矩阵
__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y,
                                                 const int row,
                                                 const int col) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_col = global_idx / row;
    const int global_row = global_idx % row;
    if (global_idx < row * col)
        y[global_col * row + global_row] = x[global_row * col + global_col];
}

__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y,
                                                    const int row,
                                                    const int col) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_col = (global_idx * 4) % col;
    const int global_row = (global_idx * 4) / col;
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
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_col = (global_idx * 4) / row;
    const int global_row = (global_idx * 4) % row;
    if (global_row + 3 < row && global_col < col) {
        float4 x_val;
        x_val.x = x[global_row * col + global_col];
        x_val.y = x[(global_row + 1) * col + global_col];
        x_val.z = x[(global_row + 2) * col + global_col];
        x_val.w = x[(global_row + 3) * col + global_col];
        reinterpret_cast<float4 *>(y)[(global_col * row + global_row) / 4] = FLOAT4(x_val);
    }
}

/*
    对角线遍历
    block的执行顺序在整个二维网络中呈对角线跳跃的方式，避免多个block同时访问共享内存中
    邻近bank，从而降低或消除shared memory bank conflict

    (0,0)  (1,0)  (2,0)  (3,0)
    (0,1)  (1,1)  (2,1)  (3,1)
    (0,2)  (1,2)  (2,2)  (3,2)
    (0,3)  (1,3)  (2,3)  (3,3)

    (0,0)
        (1,1)
            (2,2)
                (3,3)
    (1,0)
        (2,1)
            (3,2)
                (0,3)
    (2,0)
        (3,1)
            (0,2)
                (1,3)
    (3,0)
        (0,1)
            (1,2)
                (2,3)
    
*/
__global__ void mat_transpose_f32_diagonal2d_kernel(float *x, float *y,
                                                    const int row,
                                                    const int col) {
    const int block_y = blockIdx.x;
    const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int global_col = threadIdx.x + blockDim.x * block_x;
    const int global_row = threadIdx.y + blockDim.y * block_y;
    if (global_col < col && global_row < row) {
        y[global_row * col + global_col] = x[global_col * row + global_row];
    }
}

__global__ void mat_transpose_f32_col2row2d_kernel(float *x, float *y,
                                                    const int row,
                                                    const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x < col && global_y < row)
        y[global_x * row + global_y] = x[global_y * col + global_x];
}

__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y,
                                                    const int row,
                                                    const int col) {
    const int global_y = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_x = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_y < col && global_x < row)
        y[global_y * row + global_x] = x[global_x * col + global_y];
}

/*
    在CUDA中，2D block中的threads是按照行优先排列，即同一行内的thread(threadIdx.x相同)连续。
    每32个连续的threads会组成一个warp。
*/
__global__ void mat_transpose_f32x4_col2row2d_kernel(float *x, float *y,
                                                     const int row,
                                                     const int col) {
    const int global_x = blockIdx.x * blockDim.y + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x * 4 + 3 < col && global_y < row) {
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        y[(global_x * 4) * row + global_y] = x_val.x;
        y[(global_x * 4 + 1) * row + global_y] = x_val.y;
        y[(global_x * 4 + 2) * row + global_y] = x_val.z;
        y[(global_x * 4 + 3) * row + global_y] = x_val.w;
    }
}

__global__ void mat_transpose_f32x4_row2col2d_kernel(float *x, float *y,
                                                     const int row,
                                                     const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_y * 4 + 3 < row && global_x < col) {
        float4 x_val;
        x_val.x = x[(global_y * 4) * col + global_x];
        x_val.y = x[(global_y * 4 + 1) * col + global_x];
        x_val.z = x[(global_y * 4 + 2) * col + global_x];
        x_val.w = x[(global_y * 4 + 3) * col + global_x];
        reinterpret_cast<float4 *>(y)[global_x * row / 4 + global_y] = FLOAT4(x_val);
    }
}

/*
    __shared__ float tile[H][M]是共享内存上的二维数组变量；
    是每一个线程块私有的，线程间共享的快速内存；
    位于每个SM上的片上内存
    block内所有线程可读写，但是block外线程不可访问；
*/
__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];
    if (global_x * 4 + 3 < col && global_y < row) {
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
        __syncthreads();
        
        float4 smem_val;
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
    }
}

__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    
    __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S];
    if (global_y * 4 + 3 < row && global_x < col) {
        float4 x_val;
        x_val.x = x[(global_y * 4) * col + global_x];
        x_val.y = x[(global_y * 4 + 1) * col + global_x];
        x_val.z = x[(global_y * 4 + 2) * col + global_x];
        x_val.w = x[(global_y * 4 + 3) * col + global_x];
        tile[local_y * 4][local_x] = x_val.x;
        tile[local_y * 4 + 1][local_x] = x_val.y;
        tile[local_y * 4 + 2][local_x] = x_val.z;
        tile[local_y * 4 + 3][local_x] = x_val.w;
        __syncthreads();
        
        float4 smem_val;
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
        smem_val.y = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
        smem_val.z = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
        smem_val.w = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
        
        const int bid_x = blockIdx.x * blockDim.x;
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = bid_x + (local_y % STRIDE) * 4;
        const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
        y[out_y * row + out_x] = smem_val.x;
        y[(out_y + 1) * row + out_x] = smem_val.y;
        y[(out_y + 2) * row + out_x] = smem_val.z;
        y[(out_y + 3) * row + out_x] = smem_val.w;
    }
}

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float *x, float *y,
                                                                const int row,
                                                                const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];
    if (global_x * 4 + 3 < col && global_y < row)
    {
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        tile[local_y][local_x * 4] = x_val.x;
        tile[local_y][local_x * 4 + 1] = x_val.y;
        tile[local_y][local_x * 4 + 2] = x_val.z;
        tile[local_y][local_x * 4 + 3] = x_val.w;
        __syncthreads();
        float4 smem_val;
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
    }
}

__global__ void mat_transpose_f32x4_shared_bcf_row2col2d_kernel(float *x, float *y,
                                                                const int row,
                                                                const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    
    __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];
    if (global_y * 4 + 3 < row && global_x < col) {
        float4 x_val;
        x_val.x = x[(global_y * 4) * col + global_x];
        x_val.y = x[(global_y * 4 + 1) * col + global_x];
        x_val.z = x[(global_y * 4 + 2) * col + global_x];
        x_val.w = x[(global_y * 4 + 3) * col + global_x];
        tile[local_y * 4][local_x] = x_val.x;
        tile[local_y * 4 + 1][local_x] = x_val.y;
        tile[local_y * 4 + 2][local_x] = x_val.z;
        tile[local_y * 4 + 3][local_x] = x_val.w;
        __syncthreads();
        float4 smem_val;
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
        smem_val.y = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
        smem_val.z = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
        smem_val.w = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
        
        const int bid_x = blockIdx.x * blockDim.x;
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = bid_x + (local_y % STRIDE) * 4;
        const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
        y[out_y * row + out_x] = smem_val.x;
        y[(out_y + 1) * row + out_x] = smem_val.y;
        y[(out_y + 2) * row + out_x] = smem_val.z;
        y[(out_y + 3) * row + out_x] = smem_val.w;
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

/*
    block(WARP_SIZE): 一个block中有多少个线程
    N * M: 总的元素数
    N * M / n_pack: 一个线程处理n_pack个数据, 总共需要多少个线程
    N * M / n_pack / WARP_SIZE: 总共需要多少个block
*/
#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)         \
    void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                  \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                  \
        const int M = x.size(0);                                                \
        const int N = x.size(1);                                                \
        dim3 block(WARP_SIZE);                                                  \
        dim3 grid((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE);                \
        mat_transpose_##tag##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                     \
            reinterpret_cast<element_type *>(y.data_ptr()), M, N);              \
    }

/*
    block(WARP_SIZE_S, WARP_SIZE_S): 表示在行，列方向上有WARP_SIZE_S个线程
    n_element_col: 表示在列方向上一个线程要处理的数据数
    n_element_row: 表示在行方向上一个线程要处理的数据数
    WARP_SIZE_S * n_element_col: 表示在列方向上总共要处理的数据数
    N / (WARP_SIZE_S * n_element_col): 表示在列方向上需要多少个block
*/
#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type,               \
                                      n_element_row, n_element_col)             \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y) {              \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                      \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                      \
    const int M = x.size(0);                                                    \
    const int N = x.size(1);                                                    \
    dim3 block(WARP_SIZE_S, WARP_SIZE_S);                                       \
    dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),            \
              (M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));           \
    mat_transpose_##tag##2d_kernel<<<grid,                                      \
        block>>>(reinterpret_cast<element_type *>(x.data_ptr()),                \
                 reinterpret_cast<element_type *>(y.data_ptr()), M, N);         \
  }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)
// 2d index. easier for diagonal
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_row2col, torch::kFloat32, float, 4, 1)
// diagonal index method.
TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)
// shared memory
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_col2row, torch::kFloat32, float, 1,
                              4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_row2col, torch::kFloat32, float, 4,
                              1)
// shared memory with bcf
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_col2row, torch::kFloat32, float,
                              1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_row2col, torch::kFloat32, float,
                              4, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)
  // 2d index. easier for diagonal
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col2d)
  // diagonal index method.
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_diagonal2d)
  // shared memory optimize
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_row2col2d)
  // shared memory optimize with bcf
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_row2col2d)
}