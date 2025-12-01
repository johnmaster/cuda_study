#include "kernels.cuh"

float* alloc_host(size_t n) {
    return new float[n];
}
float* alloc_device(size_t n) {
    float *p;
    CHECK_CUDA(cudaMalloc(&p, n * sizeof(float)));
    return p;
}
void free_host(float* p) {
    delete[] p;
}
void free_device(float* p) {
    CHECK_CUDA(cudaFree(p));
}

void rand_init(float* h_data, size_t n, unsigned seed = 0x2025) {
    std::mt19937 rng(seed);                    // 高级随机数引擎（比 rand() 好）
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);  // 直接生成 -1~1
    for (size_t i = 0; i < n; ++i) 
        h_data[i] = dist(rng);
}
void printf_format(const char* name, float time) {
    std::cout << std::left                           // 名字左对齐
              << std::setw(40) << name               // 方法名占 32 字符
              << std::right                          // 后面的数字右对齐
              << std::fixed << std::setprecision(3)
              << std::setw(12) << time << " us"   // 时间，带单位
              << "\n";
}

__global__ void mat_transpose_f32_col2row_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx / cols;
    const int global_col = global_idx % cols;

    if (global_idx < rows * cols) {
        device_out[global_col * rows + global_row] = device_data[global_idx];
    }
}
void mat_transpose_f32_col2row(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WARP_SIZE);
    dim3 grid((rows * cols + WARP_SIZE - 1) / WARP_SIZE);

    mat_transpose_f32_col2row_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32_row2col_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_cols = global_idx / rows;
    const int global_rows = global_idx % rows;

    if (global_idx < rows * cols) {
        device_out[global_idx] = device_data[global_rows * cols + global_cols];
    }
}
void mat_transpose_f32_row2col(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WARP_SIZE);
    dim3 grid((rows * cols + WARP_SIZE - 1) / WARP_SIZE);

    mat_transpose_f32_row2col_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32x4_col2row_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_col = global_idx * 4 % cols;
    const int global_row = global_idx * 4 / cols;
    
    if (global_row < rows && global_col + 3 < cols) {
        float4 x_val = reinterpret_cast<float4 *>(device_data)[global_idx];
        
        device_out[(global_col + 0) * rows + global_row] = x_val.x;
        device_out[(global_col + 1) * rows + global_row] = x_val.y;
        device_out[(global_col + 2) * rows + global_row] = x_val.z;
        device_out[(global_col + 3) * rows + global_row] = x_val.w;
    }
}
void mat_transpose_f32x4_col2row(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WARP_SIZE);
    dim3 grid((rows * cols + WARP_SIZE - 1) / WARP_SIZE / 4);

    mat_transpose_f32x4_col2row_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32x4_row2col_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = (global_idx * 4) % rows;
    const int global_col = (global_idx * 4) / rows;
    
    if (global_col < cols && global_row + 3 < rows) {
        float4 x_val;
        
        x_val.x = device_data[(global_row + 0) * cols + global_col];
        x_val.y = device_data[(global_row + 1) * cols + global_col];
        x_val.z = device_data[(global_row + 2) * cols + global_col];
        x_val.w = device_data[(global_row + 3) * cols + global_col];
        
        reinterpret_cast<float4 *>(device_out)[global_idx] = FLOAT4(x_val);
    }
}
void mat_transpose_f32x4_row2col(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WARP_SIZE);
    dim3 grid((rows * cols + WARP_SIZE - 1) / WARP_SIZE / 4);

    mat_transpose_f32x4_row2col_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32_col2row2d_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x < cols && global_y < rows) {
        device_out[global_x * rows + global_y] = device_data[global_y * cols + global_x];
    }
}
void mat_transpose_f32_col2row2d(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WAPR_SIZE_S, WAPR_SIZE_S);
    dim3 grid((cols + WAPR_SIZE_S - 1) / WAPR_SIZE_S, (rows + WAPR_SIZE_S - 1) / WAPR_SIZE_S);

    mat_transpose_f32_col2row2d_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}


__global__ void mat_transpose_f32_row2col2d_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_x = blockIdx.y * blockDim.y + threadIdx.y;
    const int global_y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_y < cols && global_x < rows) {
        device_out[global_y * rows + global_x] = device_data[global_x * cols + global_y];
    }
}
void mat_transpose_f32_row2col2d(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WAPR_SIZE_S, WAPR_SIZE_S);
    dim3 grid((cols + WAPR_SIZE_S - 1) / WAPR_SIZE_S, (rows + WAPR_SIZE_S - 1) / WAPR_SIZE_S);
    
    mat_transpose_f32_row2col2d_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32x4_col2row2d_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (global_x * 4 + 3 < cols && global_y < rows) {
        float4 x_val = reinterpret_cast<float4 *>(device_data)[global_y * cols / 4 + global_x];
        
        device_out[(global_x * 4 + 0) * rows + global_y] = x_val.x;
        device_out[(global_x * 4 + 1) * rows + global_y] = x_val.y;
        device_out[(global_x * 4 + 2) * rows + global_y] = x_val.z;
        device_out[(global_x * 4 + 3) * rows + global_y] = x_val.w;
    }
}
void mat_transpose_f32x4_col2row2d(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WAPR_SIZE_S, WAPR_SIZE_S);
    dim3 grid((cols + WAPR_SIZE_S - 1) / (WAPR_SIZE_S * 4), (rows + WAPR_SIZE_S - 1) / WAPR_SIZE_S);

    mat_transpose_f32x4_col2row2d_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32x4_row2col2d_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (global_x < cols && global_y * 4 + 3 < rows) {
        float4 x_val;
        
        x_val.x = device_data[(global_y * 4 + 0) * cols + global_x];
        x_val.y = device_data[(global_y * 4 + 1) * cols + global_x];
        x_val.z = device_data[(global_y * 4 + 2) * cols + global_x];
        x_val.w = device_data[(global_y * 4 + 3) * cols + global_x];
        
        reinterpret_cast<float4*>(device_out)[global_x * rows / 4 + global_y] = FLOAT4(x_val);
    }
}
void mat_transpose_f32x4_row2col2d(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WAPR_SIZE_S, WAPR_SIZE_S);
    dim3 grid((cols + WAPR_SIZE_S - 1) / WAPR_SIZE_S, (rows + WAPR_SIZE_S - 1) / (WAPR_SIZE_S * 4));
    
    mat_transpose_f32x4_row2col2d_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32x4_shared_col2row2_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WAPR_SIZE_S][WAPR_SIZE_S * 4];

    if (global_x * 4 < cols && global_y < rows) {
        float4 x_val = reinterpret_cast<float4*>(device_data)[global_y * cols / 4 + global_x];
        FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
        __syncthreads();

        float4 smem_val;
        constexpr int STRIDE = WAPR_SIZE_S / 4;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(device_out)[(out_y * rows + out_x) / 4] = FLOAT4(smem_val);
    }
}
void mat_transpose_f32x4_shared_col2row2d(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WAPR_SIZE_S, WAPR_SIZE_S);
    dim3 grid((cols + WAPR_SIZE_S - 1) / (WAPR_SIZE_S * 4), (rows + WAPR_SIZE_S - 1) / WAPR_SIZE_S);

    mat_transpose_f32x4_shared_col2row2_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(float* device_data, float* device_out, int rows, int cols) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WAPR_SIZE_S * 4][WAPR_SIZE_S];

    if (global_y * 4 < rows && global_x < cols) {
        float4 x_val;
        x_val.x = device_data[(global_y * 4) * cols + global_x];
        x_val.y = device_data[(global_y * 4 + 1) * cols + global_x];
        x_val.z = device_data[(global_y * 4 + 2) * cols + global_x];
        x_val.w = device_data[(global_y * 4 + 3) * cols + global_x];
        tile[local_y * 4][local_x] = x_val.x;
        tile[local_y * 4 + 1][local_x] = x_val.y;
        tile[local_y * 4 + 2][local_x] = x_val.z;
        tile[local_y * 4 + 3][local_x] = x_val.w;
        __syncthreads();

        float4 smem_val;
        const int STRIDE = WAPR_SIZE_S / 4;
        smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
        smem_val.y = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
        smem_val.z = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
        smem_val.w = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
        
        const int bid_x = blockIdx.x * blockDim.x;
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = bid_x + (local_y % STRIDE) * 4;
        const int out_x = bid_y * 4 + local_x * 4 + local_y / STRIDE;

        device_out[(out_y + 0) * rows + out_x] = smem_val.x;
        device_out[(out_y + 1) * rows + out_x] = smem_val.y;
        device_out[(out_y + 2) * rows + out_x] = smem_val.z;
        device_out[(out_y + 3) * rows + out_x] = smem_val.w;
    }
}
void mat_transpose_f32x4_shared_row2col2d(float* device_data, float* device_out, int rows, int cols) {
    dim3 block(WAPR_SIZE_S, WAPR_SIZE_S);
    dim3 grid((cols + WAPR_SIZE_S - 1) / WAPR_SIZE_S, (rows + WAPR_SIZE_S - 1) / (WAPR_SIZE_S * 4));
    
    mat_transpose_f32x4_shared_row2col2d_kernel<<<grid, block>>>(device_data, device_out, rows, cols);
}

void benchMark(int M, int N) {
    const size_t sizeA = M * N;
    float* host_data = alloc_host(sizeA);
    float* device_data = alloc_device(sizeA);
    float* device_out = alloc_device(sizeA);
    float time = 0.0;

    rand_init(host_data, sizeA);
    CHECK_CUDA(cudaMemcpy(device_data, host_data, sizeA * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << std::left
              << std::setw(20)
              << "Matrix: "
              << M << " x " << N << "\n";

    std::cout << "=== CUDA Matrix transpose Performance Comparison ===\n";

    std::vector<Kernel> kernels = {
        {"mat_transpose_f32_col2row",                   mat_transpose_f32_col2row},
        {"mat_transpose_f32_row2col",                   mat_transpose_f32_row2col},
        {"mat_transpose_f32x4_col2row",                 mat_transpose_f32x4_col2row},
        {"mat_transpose_f32x4_row2col",                 mat_transpose_f32x4_row2col},
        {"mat_transpose_f32_col2row2d",                 mat_transpose_f32_col2row2d},
        {"mat_transpose_f32_row2col2d",                 mat_transpose_f32_row2col2d},
        {"mat_transpose_f32x4_col2row2d",               mat_transpose_f32x4_col2row2d},
        {"mat_transpose_f32x4_row2col2d",               mat_transpose_f32x4_row2col2d},
        {"mat_transpose_f32x4_shared_col2row2d",        mat_transpose_f32x4_shared_col2row2d},
        {"mat_transpose_f32x4_shared_row2col2d",        mat_transpose_f32x4_shared_row2col2d},
    };

    for (auto k: kernels) {
        time = 0.0;
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        int warm = 20, runs = 200;

        CHECK_CUDA(cudaMemset(device_out, 0, sizeA * sizeof(float)));
        for(int i = 0; i < warm; i++) {
            k.func(device_data, device_out, M, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaEventRecord(start));
        for(int i = 0; i < runs; i++) {
            k.func(device_data, device_out, M, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        printf_format(k.name, time / runs);
        //std::cout << "\n";
    }

    free_host(host_data);
    free_device(device_data);
    free_device(device_out);
}