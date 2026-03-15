#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include "kernels.cuh"

// 错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CPU 参考实现
void cpu_gemm_reference(const half* A, const half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// 验证结果
bool verify_result(const float* ref, const float* result, int size, float tolerance = 1e-2f) {
    int errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = std::abs(ref[i] - result[i]);
        max_error = std::max(max_error, error);
        float relative_error = error / (std::abs(ref[i]) + 1e-6f);
        if (relative_error > tolerance && error > tolerance) {
            if (errors < 5) {
                std::cout << "Mismatch at index " << i << ": ref=" << ref[i] 
                          << ", result=" << result[i] << ", error=" << error << std::endl;
            }
            errors++;
        }
    }
    std::cout << "Max error: " << max_error << ", Total errors: " << errors << "/" << size << std::endl;
    return errors == 0;
}

// 计时函数
template<typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, const half* d_A, const half* d_B, float* d_C, 
                       int M, int N, int K, int warmup = 5, int repeat = 20) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
        kernel(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count() / repeat;
    return elapsed_ms;
}

// cuBLAS 基准测试
float benchmark_cublas(cublasHandle_t handle, const half* d_A, const half* d_B, float* d_C,
                       int M, int N, int K, int warmup = 5, int repeat = 20) {
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_16F, N,
                                  d_A, CUDA_R_16F, K,
                                  &beta,
                                  d_C, CUDA_R_32F, N,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_16F, N,
                                  d_A, CUDA_R_16F, K,
                                  &beta,
                                  d_C, CUDA_R_32F, N,
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count() / repeat;
    return elapsed_ms;
}

void run_benchmark(int M, int N, int K) {
    std::cout << "\n============================================\n";
    std::cout << "Matrix size: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "============================================\n";

    // 分配主机内存
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    half* h_A = new half[size_A];
    half* h_B = new half[size_B];
    float* h_C = new float[size_C];
    float* h_C_ref = new float[size_C];

    // 初始化随机数据
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < size_A; i++) {
        h_A[i] = __float2half(dist(rng));
    }
    for (size_t i = 0; i < size_B; i++) {
        h_B[i] = __float2half(dist(rng));
    }

    // 分配设备内存
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(half), cudaMemcpyHostToDevice));

    // 计算 TFLOPS
    double flops = 2.0 * M * N * K;  // MAC = 2 ops

    // 创建 cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // 结果表格
    std::cout << std::setw(25) << "Kernel" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "TFLOPS" 
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    float cublas_time = 0.0f;

    // cuBLAS 基准
    {
        float time_ms = benchmark_cublas(handle, d_A, d_B, d_C, M, N, K);
        cublas_time = time_ms;
        double tflops = flops / (time_ms * 1e9);
        std::cout << std::setw(25) << "cuBLAS (Tensor Core)"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << "1.00x (baseline)" << std::endl;
        
        CHECK_CUDA(cudaMemcpy(h_C_ref, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // 版本1: Naive WMMA
    {
        CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
        float time_ms = benchmark_kernel(launch_wmma_gemm_v1, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (time_ms * 1e9);
        float speedup = cublas_time / time_ms;
        std::cout << std::setw(25) << "V1: Naive WMMA"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        // 验证
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify_result(h_C_ref, h_C, size_C)) {
            std::cout << "  [FAILED] Result mismatch!" << std::endl;
        }
    }

    // 版本2: Shared Memory
    {
        CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
        float time_ms = benchmark_kernel(launch_wmma_gemm_v2, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (time_ms * 1e9);
        float speedup = cublas_time / time_ms;
        std::cout << std::setw(25) << "V2: Shared Memory"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify_result(h_C_ref, h_C, size_C)) {
            std::cout << "  [FAILED] Result mismatch!" << std::endl;
        }
    }

    // 版本3: Double Buffer
    {
        CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
        float time_ms = benchmark_kernel(launch_wmma_gemm_v3, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (time_ms * 1e9);
        float speedup = cublas_time / time_ms;
        std::cout << std::setw(25) << "V3: Double Buffer"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify_result(h_C_ref, h_C, size_C)) {
            std::cout << "  [FAILED] Result mismatch!" << std::endl;
        }
    }

    // 版本4: Vectorized
    {
        CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
        float time_ms = benchmark_kernel(launch_wmma_gemm_v4, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (time_ms * 1e9);
        float speedup = cublas_time / time_ms;
        std::cout << std::setw(25) << "V4: Vectorized Load"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify_result(h_C_ref, h_C, size_C)) {
            std::cout << "  [FAILED] Result mismatch!" << std::endl;
        }
    }

    // 版本5: Large Tile 128x64
    {
        CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
        float time_ms = benchmark_kernel(launch_wmma_gemm_v5, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (time_ms * 1e9);
        float speedup = cublas_time / time_ms;
        std::cout << std::setw(25) << "V5: Large Tile 128x64"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify_result(h_C_ref, h_C, size_C)) {
            std::cout << "  [FAILED] Result mismatch!" << std::endl;
        }
    }

    // 版本6: cp.async Pipeline
    {
        CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
        float time_ms = benchmark_kernel(launch_wmma_gemm_v6, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (time_ms * 1e9);
        float speedup = cublas_time / time_ms;
        std::cout << std::setw(25) << "V6: cp.async Pipeline"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify_result(h_C_ref, h_C, size_C)) {
            std::cout << "  [FAILED] Result mismatch!" << std::endl;
        }
    }

    // 版本7: Combined (Large Tile + cp.async)
    {
        CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
        float time_ms = benchmark_kernel(launch_wmma_gemm_v7, d_A, d_B, d_C, M, N, K);
        double tflops = flops / (time_ms * 1e9);
        float speedup = cublas_time / time_ms;
        std::cout << std::setw(25) << "V7: Combined Best"
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify_result(h_C_ref, h_C, size_C)) {
            std::cout << "  [FAILED] Result mismatch!" << std::endl;
        }
    }

    // 清理
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
}

int main(int argc, char** argv) {
    // 检查 GPU
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    
    if (prop.major < 7) {
        std::cerr << "Error: Tensor Cores require compute capability 7.0 or higher!" << std::endl;
        return 1;
    }

    // 运行不同大小的测试
    std::vector<std::tuple<int, int, int>> sizes = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };

    for (auto& [M, N, K] : sizes) {
        run_benchmark(M, N, K);
    }

    return 0;
}

