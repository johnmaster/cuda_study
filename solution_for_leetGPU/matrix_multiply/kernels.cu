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
float max_error(const float* ref, const float* test, size_t n) {
    float err = 0.0f;
    for (size_t i = 0; i < n; i++)
        err = std::max(err, std::abs(test[i] - ref[i]));
    return err;
}
double run_cublas(float* dA, float* dB, float* dC, int M, int N, int K, int repeats = 50) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    // warmup
    for (int i = 0; i < 10; ++i)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeats; ++i)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    CHECK_CUBLAS(cublasDestroy(handle));

    return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

void rand_init(float* h_data, size_t n, unsigned seed = 0x2025) {
    std::mt19937 rng(seed);                    // 高级随机数引擎（比 rand() 好）
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);  // 直接生成 -1~1
    for (size_t i = 0; i < n; ++i) 
        h_data[i] = dist(rng);
}

void printMatrixInformation(int M, int N, int K) {
    std::cout << "\n";
    std::cout << "┌";
    std::cout << std::string(70, '-');
    std::cout << "┐\n";

    std::cout << "│" 
              << std::setw(70) << std::left 
              << (" M = " + std::to_string(M) + 
                  ", N = " + std::to_string(N) + 
                  ", K = " + std::to_string(K) + " ").c_str()
              << "│\n";

    std::cout << "└";
    std::cout << std::string(70, '-');
    std::cout << "┘\n\n";

    std::cout << std::left
              << std::setw(45) << "Kernel"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "TFLOPS"
              << std::setw(14) << "Max Error"
              << "\n";
    std::cout << std::string(80, '-') << "\n";
}


__global__ void sgemm_native_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float psum = 0.0f;
#pragma unroll
        for (int k = 0; k < N; k++) {
            psum += a[m * N + k] * b[k * K + n];
        }
        c[m * N + n] = psum; 
    }
}
void sgemm_native_f32(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_native_f32_kernel<<<grid, block>>>(a, b, c, M, N, K);
}

template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
    __shared__ float s_a[BM][BK], s_b[BK][BN];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    int load_smem_a_m = tid / BK;
    int load_smem_a_k = tid % BK;
    int load_smem_b_k = tid / BN;
    int load_smem_b_n = tid % BN;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    
    float sum = 0.0f;
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
        
        for (int k = 0; k < BK; ++k) {
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }

    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;
}
void sgemm_sliced_k_f32(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(a, b, c, M, N, K);
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float* a, float* b, float* c, int M, int N, int K) {
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    int load_smem_a_m = tid / 2;
    int load_smem_a_k = tid % 2 == 0 ? 0 : 4;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32 * 4;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    
    float r_c[TM][TN] = {0.0f};
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    int comp_smem_a_m = ty * TM + m;
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; m++) {
        int store_gmem_c_m = by * BM + ty * TM + m;
        for (int n = 0; n < TN; n += 4) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}
void sgemm_t_8x8_sliced_k_f32x4(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(a, b, c, M, N, K);
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float* a, float* b, float* c, int M, int N, int K) {
    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = tid % 2 == 0 ? 0 : 4;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32 * 4;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_gmem_a_addr]);
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_gmem_b_addr]);

        s_a[load_smem_a_k][load_smem_a_m] = r_load_a[0];
        s_a[load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        
        FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(r_load_b[0]);
        __syncthreads();

        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
            
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);        
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM / 2; i++) {
        int store_gmem_c_m = by * BM + ty * TM / 2 + i;
        int store_gmem_c_n = bx * BN + tx * TN / 2;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_gmem_c_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    for (int i = 0; i < TM / 2; i++) {
        int store_gmem_c_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_gmem_c_n = bx * BN + tx * TN / 2;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_gmem_c_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
void sgemm_t_8x8_sliced_k_f32x4_bcf(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(a, b, c, M, N, K);
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset_kernel(
    float* a, float* b, float* c, int M, int N, int K) {
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = tid % 2 == 0 ? 0 : 4;
    int load_smem_b_k = tid / 32;
    int load_smem_b_n = tid % 32 * 4;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    
    {
        int load_gmem_a_k = load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_gmem_a_addr]);
        int load_gmem_b_k = load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_gmem_b_addr]);
        
        s_a[0][load_smem_a_k + 0][load_smem_a_m] = r_load_a[0];
        s_a[0][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[0][load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[0][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        FLOAT4(s_b[0][load_smem_b_k][load_smem_b_n]) = FLOAT4(r_load_b[0]);
    }
    __syncthreads();
    
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 0x1;
        int smem_sel_next = bk & 0x1;
        
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_gmem_a_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_gmem_b_addr]);

        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);
            
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

        s_a[smem_sel_next][load_smem_a_k + 0][load_smem_a_m] = r_load_a[0];
        s_a[smem_sel_next][load_smem_a_k + 1][load_smem_a_m] = r_load_a[1];
        s_a[smem_sel_next][load_smem_a_k + 2][load_smem_a_m] = r_load_a[2];
        s_a[smem_sel_next][load_smem_a_k + 3][load_smem_a_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]) =
            FLOAT4(r_load_b[0]);
        __syncthreads();
    }
    
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

        for (int tm = 0; tm < TM; tm++) {
            for (int tn = 0; tn < TN; tn++) {
            // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int OFFSET = 4;
    
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset_kernel<BM, BN, BK, TM, TN, OFFSET>
        <<<grid, block>>>(a, b, c, M, N, K);
}

void benchMark(int M, int N, int K) {
    const size_t sizeA = M * K, sizeB = K * N, sizeC = M * N;
    
    float* host_A = alloc_host(sizeA);
    float* host_B = alloc_host(sizeB);
    float* host_C_ref = alloc_host(sizeC);
    float* host_C_test = alloc_host(sizeC);

    rand_init(host_A, sizeA);
    rand_init(host_B, sizeB);

    float *device_A = alloc_device(sizeA);
    float *device_B = alloc_device(sizeB);
    float *device_C = alloc_device(sizeC);

    CHECK_CUDA(cudaMemcpy(device_A, host_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_B, host_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(device_C, 0, sizeC * sizeof(float)));
    double cublas_ms = run_cublas(device_A, device_B, device_C, M, N, K, 50);
    CHECK_CUDA(cudaMemcpy(host_C_ref, device_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double flops = 2.0 * M * N * K;
    double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;

    printMatrixInformation(M, N, K);

    std::cout << std::left
              << std::setw(45) << "cuBLAS (reference)"
              << std::setw(12) << std::fixed << std::setprecision(3) << cublas_ms
              << std::setw(12) << std::setprecision(2) << cublas_tflops
              << std::setw(14) << "0.0"
              << "\n";

    std::vector<Kernel> kernels = {
        {"sgemm_native_f32",                                sgemm_native_f32},
        {"sgemm_sliced_k_f32",                              sgemm_sliced_k_f32},
        {"sgemm_t_8x8_sliced_k_f32x4",                      sgemm_t_8x8_sliced_k_f32x4},
        {"sgemm_t_8x8_sliced_k_f32x4_bcf",                  sgemm_t_8x8_sliced_k_f32x4_bcf},
        {"sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset",      sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset},
    };

    for (const auto& k : kernels) {

        CHECK_CUDA(cudaMemset(device_C, 0, sizeC * sizeof(float)));
        // warm up
        for (int i = 0; i < 10; ++i) {
            k.func(device_A, device_B, device_C, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; ++i) {
            k.func(device_A, device_B, device_C, M, N, K);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count() / 50;
        double flops = 2.0 * M * N * K;
        double tflops = flops / (ms * 1e-3) / 1e12;

        // 检查误差
        CHECK_CUDA(cudaMemcpy(host_C_test, device_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (size_t i = 0; i < sizeC; ++i) {
            max_err = std::max(max_err, std::abs(host_C_test[i] - host_C_ref[i]));
        }

        std::cout << std::left << std::setw(45) << k.name
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << ms
                  << std::setw(12) << tflops
                  << std::setw(14) << max_err << "\n";
    }

    free_host(host_A);
    free_host(host_B);
    free_host(host_C_ref);
    free_host(host_C_test);
    free_device(device_A);
    free_device(device_B);
    free_device(device_C);
}