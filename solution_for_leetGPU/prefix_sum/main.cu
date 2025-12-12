#include "kernels.cuh"
#include <cmath>

// CPU 参考实现: inclusive prefix sum
void cpu_prefix_sum(float* input, float* output, size_t n) {
    if (n == 0) return;
    output[0] = input[0];
    for (size_t i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// 验证结果 (浮点数需要容差)
bool verify(float* cpu_result, float* gpu_result, size_t n, float tolerance = 1e-3f) {
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        float max_val = std::max(std::abs(cpu_result[i]), std::abs(gpu_result[i]));
        float rel_error = (max_val > 1e-6f) ? diff / max_val : diff;
        
        if (rel_error > tolerance && diff > tolerance) {
            std::cerr << "Mismatch at position " << i << ": "
                      << "CPU = " << cpu_result[i] << ", GPU = "
                      << gpu_result[i] << ", diff = " << diff << std::endl;
            return false;
        }
    }
    return true;
}

void print_array(const char* name, float* arr, size_t n, size_t max_print = 10) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(n, max_print); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << arr[i];
    }
    if (n > max_print) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

void test_prefix_sum(size_t n, int epochs = 100) {
    std::cout << "\n=====================" << std::endl;
    std::cout << "Testing Prefix Sum (N = " << n << ")" << std::endl;
    std::cout << "=====================\n" << std::endl;

    float* h_input = alloc_host(n);
    float* h_output_cpu = alloc_host(n);
    float* h_output_gpu = alloc_host(n);
    
    float* d_input = alloc_device(n);
    float* d_output = alloc_device(n);

    rand_init(h_input, n);
    print_array("Input", h_input, n);

    // CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_prefix_sum(h_input, h_output_cpu, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    std::cout << "CPU time: " << std::fixed << std::setprecision(2) << cpu_time << " us" << std::endl;
    print_array("CPU Output", h_output_cpu, n);

    // GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    
    double gpu_time = benchmark(
        [&]() {
            solve(d_input, d_output, n);
        }, epochs
    );
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "GPU time: " << std::fixed << std::setprecision(2) << gpu_time << " us" << std::endl;
    print_array("GPU Output", h_output_gpu, n);
    
    bool match = verify(h_output_cpu, h_output_gpu, n);
    std::cout << "Result match: " << (match ? "YES" : "NO") << std::endl;

    free_host(h_input);
    free_host(h_output_cpu);
    free_host(h_output_gpu);
    free_device(d_input);
    free_device(d_output);

    if (!match) {
        std::cerr << "Test FAILED" << std::endl;
        exit(1);
    }
    std::cout << "TEST PASSED" << std::endl;
}

int main() {
    CHECK_CUDA(cudaDeviceReset());

    test_prefix_sum(8, 100);           // 小数组
    test_prefix_sum(512, 100);         // 单 block
    test_prefix_sum(1024, 100);        // 多 block
    test_prefix_sum(1 << 16, 100);     // 64K
    test_prefix_sum(1 << 20, 100);     // 1M
    
    std::cout << "\n============ All tests passed ===========\n" << std::endl;
    return 0;
}
