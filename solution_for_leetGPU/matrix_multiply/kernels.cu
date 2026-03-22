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
double run_cublas(float* dA, float* dB, float* dC, int M, int N, int K,
                  int repeats = 50, bool pedantic_fp32 = false) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    if (pedantic_fp32)
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    float alpha = 1.0f, beta = 0.0f;

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
        for (int k = 0; k < K; k++) {
            psum += a[m * K + k] * b[k * N + n];
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

// ============================================================================
// V6: BK=16 双缓冲 + BCF + 向量化 + Padding
// 将 BK 从 8 增大到 16，主循环迭代次数减半，syncthreads 开销减半
// ============================================================================
template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 4>
__global__ void sgemm_v6_bk16_bcf_dbuf_kernel(
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    int M, int N, int K) {
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // A: 128×16=2048 floats, 256 threads → 每线程 2 个 float4 (pass1 k=0..7, pass2 k=8..15)
    int load_a_m = tid / 2;         // 0..127
    int load_a_k = (tid % 2) * 4;   // 0 or 4
    int gmem_a_m = by * BM + load_a_m;

    // B: 16×128=2048 floats, 256 threads → 每线程 2 个 float4 (pass1 k=0..7, pass2 k=8..15)
    int load_b_k = tid / 32;        // 0..7
    int load_b_n = (tid % 32) * 4;  // 0..124
    int gmem_b_n = bx * BN + load_b_n;

    float r_load_a[4];
    float r_comp_a[TM], r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    // ---- 预加载 buffer 0 ----
    // A pass 1: k = load_a_k .. load_a_k+3
    FLOAT4(r_load_a[0]) = FLOAT4(a[gmem_a_m * K + load_a_k]);
    s_a[0][load_a_k + 0][load_a_m] = r_load_a[0];
    s_a[0][load_a_k + 1][load_a_m] = r_load_a[1];
    s_a[0][load_a_k + 2][load_a_m] = r_load_a[2];
    s_a[0][load_a_k + 3][load_a_m] = r_load_a[3];
    // A pass 2: k+8
    FLOAT4(r_load_a[0]) = FLOAT4(a[gmem_a_m * K + load_a_k + 8]);
    s_a[0][load_a_k + 8][load_a_m] = r_load_a[0];
    s_a[0][load_a_k + 9][load_a_m] = r_load_a[1];
    s_a[0][load_a_k + 10][load_a_m] = r_load_a[2];
    s_a[0][load_a_k + 11][load_a_m] = r_load_a[3];
    // B pass 1
    FLOAT4(s_b[0][load_b_k][load_b_n]) =
        FLOAT4(b[load_b_k * N + gmem_b_n]);
    // B pass 2
    FLOAT4(s_b[0][load_b_k + 8][load_b_n]) =
        FLOAT4(b[(load_b_k + 8) * N + gmem_b_n]);

    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int cur = (bk - 1) & 1;
        int nxt = bk & 1;

        // gmem → register (A 需要 scatter-store, 延迟到后面)
        int gk_a = bk * BK + load_a_k;
        FLOAT4(r_load_a[0]) = FLOAT4(a[gmem_a_m * K + gk_a]);
        float r_load_a2[4];
        FLOAT4(r_load_a2[0]) = FLOAT4(a[gmem_a_m * K + gk_a + 8]);

        int gk_b = bk * BK + load_b_k;
        float r_load_b[4];
        FLOAT4(r_load_b[0]) = FLOAT4(b[gk_b * N + gmem_b_n]);
        float r_load_b2[4];
        FLOAT4(r_load_b2[0]) = FLOAT4(b[(gk_b + 8) * N + gmem_b_n]);

        // 从 current buffer 计算
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[cur][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[cur][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[cur][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[cur][tk][tx * TN / 2 + BN / 2]);
            #pragma unroll
            for (int tm = 0; tm < TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TN; tn++)
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }

        // register → next smem buffer
        s_a[nxt][load_a_k + 0][load_a_m] = r_load_a[0];
        s_a[nxt][load_a_k + 1][load_a_m] = r_load_a[1];
        s_a[nxt][load_a_k + 2][load_a_m] = r_load_a[2];
        s_a[nxt][load_a_k + 3][load_a_m] = r_load_a[3];
        s_a[nxt][load_a_k + 8][load_a_m] = r_load_a2[0];
        s_a[nxt][load_a_k + 9][load_a_m] = r_load_a2[1];
        s_a[nxt][load_a_k + 10][load_a_m] = r_load_a2[2];
        s_a[nxt][load_a_k + 11][load_a_m] = r_load_a2[3];
        FLOAT4(s_b[nxt][load_b_k][load_b_n]) = FLOAT4(r_load_b[0]);
        FLOAT4(s_b[nxt][load_b_k + 8][load_b_n]) = FLOAT4(r_load_b2[0]);
        __syncthreads();
    }

    // 处理最后一个 buffer
    int last = ((K + BK - 1) / BK - 1) & 1;
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[last][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[last][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[last][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[last][tk][tx * TN / 2 + BN / 2]);
        #pragma unroll
        for (int tm = 0; tm < TM; tm++)
            #pragma unroll
            for (int tn = 0; tn < TN; tn++)
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int cm = by * BM + ty * TM / 2 + i;
        int cn = bx * BN + tx * TN / 2;
        FLOAT4(c[cm * N + cn]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[cm * N + cn + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int cm = by * BM + BM / 2 + ty * TM / 2 + i;
        int cn = bx * BN + tx * TN / 2;
        FLOAT4(c[cm * N + cn]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[cm * N + cn + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
void sgemm_v6_bk16_bcf_dbuf(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8, OFFSET = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v6_bk16_bcf_dbuf_kernel<BM, BN, BK, TM, TN, OFFSET>
        <<<grid, block>>>(a, b, c, M, N, K);
}

// ============================================================================
// V7: V6 基础上 + 内循环寄存器预取 (Software Pipelining)
// tk 循环中，在计算 tk 的同时预取 tk+1 的 smem 数据到寄存器
// ============================================================================
template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 4>
__global__ void sgemm_v7_reg_prefetch_kernel(
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    int M, int N, int K) {
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    int load_a_m = tid / 2;
    int load_a_k = (tid % 2) * 4;
    int gmem_a_m = by * BM + load_a_m;
    int load_b_k = tid / 32;
    int load_b_n = (tid % 32) * 4;
    int gmem_b_n = bx * BN + load_b_n;

    float r_load_a[4];
    float r_comp_a[TM], r_comp_b[TN];
    float r_next_a[TM], r_next_b[TN];
    float r_c[TM][TN] = {0.0f};

    // 预加载 buffer 0
    FLOAT4(r_load_a[0]) = FLOAT4(a[gmem_a_m * K + load_a_k]);
    s_a[0][load_a_k + 0][load_a_m] = r_load_a[0];
    s_a[0][load_a_k + 1][load_a_m] = r_load_a[1];
    s_a[0][load_a_k + 2][load_a_m] = r_load_a[2];
    s_a[0][load_a_k + 3][load_a_m] = r_load_a[3];
    FLOAT4(r_load_a[0]) = FLOAT4(a[gmem_a_m * K + load_a_k + 8]);
    s_a[0][load_a_k + 8][load_a_m] = r_load_a[0];
    s_a[0][load_a_k + 9][load_a_m] = r_load_a[1];
    s_a[0][load_a_k + 10][load_a_m] = r_load_a[2];
    s_a[0][load_a_k + 11][load_a_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_k][load_b_n]) = FLOAT4(b[load_b_k * N + gmem_b_n]);
    FLOAT4(s_b[0][load_b_k + 8][load_b_n]) = FLOAT4(b[(load_b_k + 8) * N + gmem_b_n]);
    __syncthreads();

    // 内循环的"计算+预取"宏，在计算 tk 的同时从 smem 预取 tk+1
    #define COMPUTE_AND_PREFETCH(buf, tk, prefetch_tk) do { \
        FLOAT4(r_next_a[0]) = FLOAT4(s_a[buf][prefetch_tk][ty * TM / 2]); \
        FLOAT4(r_next_a[4]) = FLOAT4(s_a[buf][prefetch_tk][ty * TM / 2 + BM / 2]); \
        FLOAT4(r_next_b[0]) = FLOAT4(s_b[buf][prefetch_tk][tx * TN / 2]); \
        FLOAT4(r_next_b[4]) = FLOAT4(s_b[buf][prefetch_tk][tx * TN / 2 + BN / 2]); \
        _Pragma("unroll") \
        for (int tm = 0; tm < TM; tm++) \
            _Pragma("unroll") \
            for (int tn = 0; tn < TN; tn++) \
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]); \
        _Pragma("unroll") \
        for (int i = 0; i < TM; i++) r_comp_a[i] = r_next_a[i]; \
        _Pragma("unroll") \
        for (int i = 0; i < TN; i++) r_comp_b[i] = r_next_b[i]; \
    } while(0)

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int cur = (bk - 1) & 1;
        int nxt = bk & 1;

        // gmem loads
        int gk_a = bk * BK + load_a_k;
        FLOAT4(r_load_a[0]) = FLOAT4(a[gmem_a_m * K + gk_a]);
        float r_load_a2[4];
        FLOAT4(r_load_a2[0]) = FLOAT4(a[gmem_a_m * K + gk_a + 8]);
        int gk_b = bk * BK + load_b_k;
        float r_load_b[4], r_load_b2[4];
        FLOAT4(r_load_b[0]) = FLOAT4(b[gk_b * N + gmem_b_n]);
        FLOAT4(r_load_b2[0]) = FLOAT4(b[(gk_b + 8) * N + gmem_b_n]);

        // 预取 tk=0 到 r_comp
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[cur][0][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[cur][0][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[cur][0][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[cur][0][tx * TN / 2 + BN / 2]);

        // tk=0..BK-2: 计算 tk，同时预取 tk+1
        #pragma unroll
        for (int tk = 0; tk < BK - 1; tk++) {
            COMPUTE_AND_PREFETCH(cur, tk, tk + 1);
        }
        // 最后一步 tk=BK-1: 只计算，不预取
        #pragma unroll
        for (int tm = 0; tm < TM; tm++)
            #pragma unroll
            for (int tn = 0; tn < TN; tn++)
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);

        // store to next buffer
        s_a[nxt][load_a_k + 0][load_a_m] = r_load_a[0];
        s_a[nxt][load_a_k + 1][load_a_m] = r_load_a[1];
        s_a[nxt][load_a_k + 2][load_a_m] = r_load_a[2];
        s_a[nxt][load_a_k + 3][load_a_m] = r_load_a[3];
        s_a[nxt][load_a_k + 8][load_a_m] = r_load_a2[0];
        s_a[nxt][load_a_k + 9][load_a_m] = r_load_a2[1];
        s_a[nxt][load_a_k + 10][load_a_m] = r_load_a2[2];
        s_a[nxt][load_a_k + 11][load_a_m] = r_load_a2[3];
        FLOAT4(s_b[nxt][load_b_k][load_b_n]) = FLOAT4(r_load_b[0]);
        FLOAT4(s_b[nxt][load_b_k + 8][load_b_n]) = FLOAT4(r_load_b2[0]);
        __syncthreads();
    }

    // 尾部
    int last = ((K + BK - 1) / BK - 1) & 1;
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[last][0][ty * TM / 2]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[last][0][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[last][0][tx * TN / 2]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[last][0][tx * TN / 2 + BN / 2]);
    #pragma unroll
    for (int tk = 0; tk < BK - 1; tk++) {
        COMPUTE_AND_PREFETCH(last, tk, tk + 1);
    }
    #pragma unroll
    for (int tm = 0; tm < TM; tm++)
        #pragma unroll
        for (int tn = 0; tn < TN; tn++)
            r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);

    #undef COMPUTE_AND_PREFETCH

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int cm = by * BM + ty * TM / 2 + i;
        int cn = bx * BN + tx * TN / 2;
        FLOAT4(c[cm * N + cn]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[cm * N + cn + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int cm = by * BM + BM / 2 + ty * TM / 2 + i;
        int cn = bx * BN + tx * TN / 2;
        FLOAT4(c[cm * N + cn]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[cm * N + cn + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
void sgemm_v7_reg_prefetch(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8, OFFSET = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v7_reg_prefetch_kernel<BM, BN, BK, TM, TN, OFFSET>
        <<<grid, block>>>(a, b, c, M, N, K);
}

// ============================================================================
// V8: TF32 Tensor Core SGEMM — 利用 Ampere Tensor Core 进行 FP32 GEMM
// TF32 精度: 输入 FP32 截断到 TF32 (19-bit, 10-bit mantissa) 后用 Tensor Core 计算
// Tensor Core 吞吐是 FP32 FMA 的 2-4 倍
// wmma::m16n16k8: 每个 warp 一次计算 16×16 的输出块，K 步长 8
// Block tile: 128×128, 每个 block 8 warps (256 threads)
// 每个 warp 负责 2 个 16×16 = 32×16 的输出区域
// ============================================================================
#include <mma.h>
using namespace nvcuda;

__global__ void sgemm_v8_tf32_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
    constexpr int PAD = 4;

    __shared__ float s_a[2][BK][BM + PAD];
    __shared__ float s_b[2][BK][BN + PAD];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;

    const int WARPS_PER_BLOCK = 8;
    const int warp_row = warpId / 2;
    const int warp_col = warpId % 2;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    int load_a_m = tid / 2;
    int load_a_k = (tid % 2) * 4;
    int gmem_a_m = by * BM + load_a_m;
    int load_b_k = tid / 32;
    int load_b_n = (tid % 32) * 4;
    int gmem_b_n = bx * BN + load_b_n;

    float r_load_a[4];

    auto load_tile = [&](int buf, int k_base) {
        FLOAT4(r_load_a[0]) = reinterpret_cast<const float4*>(&a[gmem_a_m * K + k_base + load_a_k])[0];
        s_a[buf][load_a_k + 0][load_a_m] = r_load_a[0];
        s_a[buf][load_a_k + 1][load_a_m] = r_load_a[1];
        s_a[buf][load_a_k + 2][load_a_m] = r_load_a[2];
        s_a[buf][load_a_k + 3][load_a_m] = r_load_a[3];
        FLOAT4(s_b[buf][load_b_k][load_b_n]) =
            reinterpret_cast<const float4*>(&b[(k_base + load_b_k) * N + gmem_b_n])[0];
    };

    load_tile(0, 0);
    __syncthreads();

    int num_k_blocks = (K + BK - 1) / BK;
    for (int bk = 0; bk < num_k_blocks; bk++) {
        int cur = bk & 1;
        if (bk + 1 < num_k_blocks) {
            load_tile((bk + 1) & 1, (bk + 1) * BK);
        }

        #pragma unroll
        for (int wk = 0; wk < BK / WMMA_K; wk++) {
            #pragma unroll
            for (int wi = 0; wi < 2; wi++) {
                int smem_a_row = warp_row * 32 + wi * WMMA_M;
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               wmma::precision::tf32, wmma::col_major> frag_a;
                wmma::load_matrix_sync(frag_a,
                    &s_a[cur][wk * WMMA_K][smem_a_row],
                    BM + PAD);

                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int smem_b_col = warp_col * 64 + wj * WMMA_N;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   wmma::precision::tf32, wmma::row_major> frag_b;
                    wmma::load_matrix_sync(frag_b,
                        &s_b[cur][wk * WMMA_K][smem_b_col],
                        BN + PAD);

                    wmma::mma_sync(acc[wi][wj], frag_a, frag_b, acc[wi][wj]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wi = 0; wi < 2; wi++) {
        #pragma unroll
        for (int wj = 0; wj < 4; wj++) {
            int c_row = by * BM + warp_row * 32 + wi * WMMA_M;
            int c_col = bx * BN + warp_col * 64 + wj * WMMA_N;
            wmma::store_matrix_sync(&c[c_row * N + c_col], acc[wi][wj], N,
                                    wmma::mem_row_major);
        }
    }
}
void sgemm_v8_tf32(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128;
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v8_tf32_kernel<<<grid, block>>>(a, b, c, M, N, K);
}

// ============================================================================
// V9: mma.sync PTX TF32 — inline PTX 替代 WMMA, 更精细的寄存器控制
// 使用 m16n8k8 指令, 每 warp 覆盖 32×64 输出 (2×8 个 m16n8k8)
// 单缓冲 BK=8, 与 V8 结构对齐以确保正确性
// ============================================================================
__device__ __forceinline__ uint32_t to_tf32_bits(float f) {
    uint32_t r;
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(f));
    return r;
}

__global__ void sgemm_v9_mma_ptx_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, PAD = 4;

    __shared__ float s_a[BK][BM + PAD];
    __shared__ float s_b[BK][BN + PAD];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;

    const int warp_row = warpId / 2;
    const int warp_col = warpId % 2;

    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int gid = laneId / 4;
    const int tid_in = laneId % 4;

    int load_a_m = tid / 2;
    int load_a_k = (tid % 2) * 4;
    int gmem_a_m = by * BM + load_a_m;
    int load_b_k = tid / 32;
    int load_b_n = (tid % 32) * 4;
    int gmem_b_n = bx * BN + load_b_n;

    float r_load_a[4];

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int k_base = bk * BK;

        FLOAT4(r_load_a[0]) = reinterpret_cast<const float4*>(
            &a[gmem_a_m * K + k_base + load_a_k])[0];
        s_a[load_a_k + 0][load_a_m] = r_load_a[0];
        s_a[load_a_k + 1][load_a_m] = r_load_a[1];
        s_a[load_a_k + 2][load_a_m] = r_load_a[2];
        s_a[load_a_k + 3][load_a_m] = r_load_a[3];
        FLOAT4(s_b[load_b_k][load_b_n]) = reinterpret_cast<const float4*>(
            &b[(k_base + load_b_k) * N + gmem_b_n])[0];
        __syncthreads();

        int a_base_m = warp_row * 32;
        int b_base_n = warp_col * 64;

        #pragma unroll
        for (int wi = 0; wi < 2; wi++) {
            int a_m_off = a_base_m + wi * 16;
            uint32_t frag_a[4];
            frag_a[0] = to_tf32_bits(s_a[tid_in    ][a_m_off + gid]);
            frag_a[1] = to_tf32_bits(s_a[tid_in    ][a_m_off + gid + 8]);
            frag_a[2] = to_tf32_bits(s_a[tid_in + 4][a_m_off + gid]);
            frag_a[3] = to_tf32_bits(s_a[tid_in + 4][a_m_off + gid + 8]);

            #pragma unroll
            for (int wj = 0; wj < 8; wj++) {
                int b_n_off = b_base_n + wj * 8;
                uint32_t frag_b[2];
                frag_b[0] = to_tf32_bits(s_b[tid_in    ][b_n_off + gid]);
                frag_b[1] = to_tf32_bits(s_b[tid_in + 4][b_n_off + gid]);

                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[wi][wj][0]), "+f"(acc[wi][wj][1]),
                      "+f"(acc[wi][wj][2]), "+f"(acc[wi][wj][3])
                    : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                      "r"(frag_b[0]), "r"(frag_b[1])
                );
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wi = 0; wi < 2; wi++) {
        #pragma unroll
        for (int wj = 0; wj < 8; wj++) {
            int c_row = by * BM + warp_row * 32 + wi * 16 + gid;
            int c_col = bx * BN + warp_col * 64 + wj * 8 + tid_in * 2;
            c[c_row * N + c_col]           = acc[wi][wj][0];
            c[c_row * N + c_col + 1]       = acc[wi][wj][1];
            c[(c_row + 8) * N + c_col]     = acc[wi][wj][2];
            c[(c_row + 8) * N + c_col + 1] = acc[wi][wj][3];
        }
    }
}
void sgemm_v9_mma_ptx(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128;
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v9_mma_ptx_kernel<<<grid, block>>>(
        (const float*)a, (const float*)b, c, M, N, K);
}

// ============================================================================
// V10: cp.async for B + register-transpose for A + 3-Stage Pipeline (FP32)
// A: register-mediated load with transpose → s_a[k][m+PAD] (optimal for compute)
// B: cp.async → s_b[k][n+PAD] (bypasses registers, hides latency)
// 3-stage pipeline with overlapped async B loads and register A loads
// ============================================================================
#include <cuda_pipeline_primitives.h>

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int PAD = 4>
__global__ __launch_bounds__(256, 2)
void sgemm_v10_cpasync_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int M, int N, int K) {
    constexpr int STAGES = 3;

    __shared__ float s_a[STAGES][BK][BM + PAD];
    __shared__ float s_b[STAGES][BK][BN + PAD];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    float r_comp_a[TM], r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    int load_a_m = tid / 2;
    int load_a_k = (tid % 2) * 4;
    int gmem_a_m = by * BM + load_a_m;

    int load_b_k = tid / 32;
    int load_b_n = (tid % 32) * 4;
    int gmem_b_n = bx * BN + load_b_n;

    float r_load_a[4];

    int num_k_blocks = (K + BK - 1) / BK;

    auto load_a_tile = [&](int stage, int k_base) {
        FLOAT4(r_load_a[0]) = reinterpret_cast<const float4*>(
            &a[gmem_a_m * K + k_base + load_a_k])[0];
        s_a[stage][load_a_k + 0][load_a_m] = r_load_a[0];
        s_a[stage][load_a_k + 1][load_a_m] = r_load_a[1];
        s_a[stage][load_a_k + 2][load_a_m] = r_load_a[2];
        s_a[stage][load_a_k + 3][load_a_m] = r_load_a[3];
    };

    auto load_b_async = [&](int stage, int k_base) {
        __pipeline_memcpy_async(
            &s_b[stage][load_b_k][load_b_n],
            &b[(k_base + load_b_k) * N + gmem_b_n],
            16);
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1 && s < num_k_blocks; s++) {
        int k_base = s * BK;
        load_a_tile(s, k_base);
        load_b_async(s, k_base);
        __pipeline_commit();
    }

    for (int bk = 0; bk < num_k_blocks; bk++) {
        int buf = bk % STAGES;

        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();

        int future = bk + STAGES - 1;
        if (future < num_k_blocks) {
            int fut_buf = future % STAGES;
            int k_base = future * BK;
            load_a_tile(fut_buf, k_base);
            load_b_async(fut_buf, k_base);
            __pipeline_commit();
        }

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                r_comp_a[m] = s_a[buf][k][ty * TM + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                r_comp_b[n] = s_b[buf][k][tx * TN + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    r_c[m][n] = __fmaf_rn(r_comp_a[m], r_comp_b[n], r_c[m][n]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int c_row = by * BM + ty * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int c_col = bx * BN + tx * TN + n;
            FLOAT4(c[c_row * N + c_col]) = FLOAT4(r_c[m][n]);
        }
    }
}
void sgemm_v10_cpasync(float* a, float* b, float* c, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, PAD = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v10_cpasync_kernel<BM, BN, BK, TM, TN, PAD>
        <<<grid, block>>>((const float*)a, (const float*)b, c, M, N, K);
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

    double flops = 2.0 * M * N * K;

    CHECK_CUDA(cudaMemset(device_C, 0, sizeC * sizeof(float)));
    double cublas_tf32_ms = run_cublas(device_A, device_B, device_C, M, N, K, 50, false);
    double cublas_tf32_tflops = flops / (cublas_tf32_ms * 1e-3) / 1e12;

    CHECK_CUDA(cudaMemset(device_C, 0, sizeC * sizeof(float)));
    double cublas_fp32_ms = run_cublas(device_A, device_B, device_C, M, N, K, 50, true);
    CHECK_CUDA(cudaMemcpy(host_C_ref, device_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    double cublas_fp32_tflops = flops / (cublas_fp32_ms * 1e-3) / 1e12;

    printMatrixInformation(M, N, K);

    std::cout << std::left
              << std::setw(45) << "cuBLAS (TF32, default)"
              << std::setw(12) << std::fixed << std::setprecision(3) << cublas_tf32_ms
              << std::setw(12) << std::setprecision(2) << cublas_tf32_tflops
              << std::setw(14) << "-"
              << "\n";
    std::cout << std::left
              << std::setw(45) << "cuBLAS (FP32, pedantic)"
              << std::setw(12) << std::fixed << std::setprecision(3) << cublas_fp32_ms
              << std::setw(12) << std::setprecision(2) << cublas_fp32_tflops
              << std::setw(14) << "0.0"
              << "\n";

    std::vector<Kernel> kernels = {
        {"sgemm_native_f32",                                sgemm_native_f32},
        {"sgemm_sliced_k_f32",                              sgemm_sliced_k_f32},
        {"sgemm_t_8x8_sliced_k_f32x4",                      sgemm_t_8x8_sliced_k_f32x4},
        {"sgemm_t_8x8_sliced_k_f32x4_bcf",                  sgemm_t_8x8_sliced_k_f32x4_bcf},
        {"sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset",      sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset},
        {"sgemm_v6_bk16_bcf_dbuf",                           sgemm_v6_bk16_bcf_dbuf},
        {"sgemm_v7_reg_prefetch",                             sgemm_v7_reg_prefetch},
        {"sgemm_v8_tf32_tensor_core",                          sgemm_v8_tf32},
        {"sgemm_v9_mma_ptx_tf32",                               sgemm_v9_mma_ptx},
        {"sgemm_v10_cpasync_3stage",                             sgemm_v10_cpasync},
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