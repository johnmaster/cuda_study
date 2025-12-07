#include "kernels.cuh"

float cpu_sum(float* host_data, size_t N) {
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += host_data[i];
    }
    return sum;
}

void printf_format(const char* name, double time_us, float cpu_ref, float gpu_result)
{
    std::cout << std::left                           // 名字左对齐
              << std::setw(32) << name               // 方法名占 32 字符
              << std::right                          // 后面的数字右对齐
              << std::fixed << std::setprecision(3)
              << std::setw(12) << time_us << " us"   // 时间，带单位
              << std::setw(18) << std::scientific << std::setprecision(2)
              << (cpu_ref - gpu_result)               // 绝对误差
              << "\n";
}

float execute_cublas_Sdot_func(float* device_data, float* result, const size_t N, const size_t epoches) {
    float *device_ones;
    std::vector<float> host_ones(N, 1.0f);
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    CHECK_CUDA( cudaMalloc(&device_ones, N * sizeof(float)) );
    CHECK_CUDA( cudaMemcpy(device_ones, host_ones.data(), N * sizeof(float), cudaMemcpyHostToDevice) );

    double time_us = benchmark([&]() {
        CHECK_CUBLAS( cublasSdot(handle, static_cast<int>(N), device_ones, 1, device_data, 1, result) );
    }, epoches);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(device_ones));

    return time_us;
}

template<typename KernelFunc>
double benchmark_reduction(KernelFunc kernel, float* device_in, float* device_tmp, float* device_backup,
                           size_t N, int epoches, float& result,
                           int elements_per_thread = 1) {
    size_t elements_per_block = BLOCK_SIZE * elements_per_thread;
    size_t numBlocks = (N + elements_per_block - 1) / elements_per_block;

    double time_us = benchmark([&]() {
        CHECK_CUDA(cudaMemcpy(device_in, device_backup, N * sizeof(float), cudaMemcpyDeviceToDevice));

        kernel<<<numBlocks, BLOCK_SIZE>>>(device_tmp, device_in, N);

        size_t remaining = numBlocks;
        while (remaining > 1) {
            size_t newBlocks = (remaining + elements_per_block - 1) / elements_per_block;
            
            kernel<<<newBlocks, BLOCK_SIZE>>>(device_tmp, device_tmp, remaining);
            remaining = newBlocks;
        }
    }, epoches);

    CHECK_CUDA(cudaMemcpy(&result, device_tmp, sizeof(float), cudaMemcpyDeviceToHost));
    return time_us;
}

int main() {
    CHECK_CUDA(cudaDeviceReset());

    const size_t N = 1 << 15;
    const int epoches = 100;
    float* host_data = alloc_host(N);

    rand_init(host_data, N);
    float cpu_ref = cpu_sum(host_data, N);
    float result = 0.0f;
    
    std::cout << "Array size: " << N << " elements (" 
              << N*sizeof(float)/1024.0/1024 << " MB)\n";
    std::cout << "CPU reference sum = " << std::fixed << std::setprecision(6) << cpu_ref << "\n\n";

    float* device_data = alloc_device(N);
    float* device_out = alloc_device(N);
    float* device_backup = alloc_device(N);
    CHECK_CUDA(cudaMemcpy(device_data, host_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_backup, host_data, N * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "=== CUDA Reduction Performance Comparison ===\n";
    std::cout << std::left << std::setw(30) << "Kernel" << " | Avg Time (100 runs) | Relative Error\n";
    std::cout << std::string(65, '-') << "\n";

    double time_us = execute_cublas_Sdot_func(device_data, &result, N, epoches);
    printf_format("cublas_sdot", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_naive, device_data, device_out, device_backup,
                                  N, epoches, result, 1);
    printf_format("reduce_naive", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_no_divergence, device_data, device_out, device_backup,
                                  N, epoches, result, 1);
    printf_format("reduce_no_divergence", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_sequential, device_data, device_out, device_backup,
                                  N, epoches, result, 1);
    printf_format("reduce_sequential", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_first_add, device_data, device_out, device_backup,
                                  N, epoches, result, 2);
    printf_format("reduce_first_add", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_unroll_last_warp, device_data, device_out, device_backup,
                                  N, epoches, result, 2);
    printf_format("reduce_unroll_last_warp", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_warp_shuffle, device_data, device_out, device_backup,
                                  N, epoches, result, 2);
    printf_format("reduce_warp_shuffle", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_vectorized, device_data, device_out, device_backup,
                                  N, epoches, result, 4);
    printf_format("reduce_vectorized", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_float4, device_data, device_out, device_backup,
                                  N, epoches, result, 4);
    printf_format("reduce_float4", time_us, cpu_ref, result);

    time_us = benchmark_reduction(reduce_float4x2, device_data, device_out, device_backup,
                                  N, epoches, result, 8);
    printf_format("reduce_float4x2", time_us, cpu_ref, result);
    
    free_device(device_data);
    free_device(device_out);
    free_device(device_backup);
    free_host(host_data);
}