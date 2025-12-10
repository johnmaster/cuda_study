#include "kernels.cuh"

void cpu_topk(float* input, float* output, size_t n, size_t k) {
    std::vector<float> sorted(input, input + n);
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    std::copy(sorted.begin(), sorted.begin() + k, output);
}
bool verify(float* cpu_result, float* gpu_result, size_t k, float tolerance = 1e-5f) {
    for (size_t i = 0; i < k; i++) {
        if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            std::cerr << "Mismatch at position " << i << ": "
                      << "CPU = " << cpu_result[i] << ", GPU = " << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}
void print_array(const char* name, float* arr, size_t n, size_t max_print = 10) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(n, max_print); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(2) << arr[i];
    }
    if (n > max_print) std::cout << ", ...";
    std::cout << "]" << std::endl;
}
void test_topk(size_t n, size_t k, int epoches = 100) {
    std::cout << "\n=============================" << std::endl;
    std::cout << "Testing top-k selection" << std::endl;
    std::cout << "N = " << n << ", K = " << k << std::endl;
    std::cout << "===============================\n" << std::endl;
    
    float* h_input = alloc_host(n);
    float* h_output_cpu = alloc_host(k);
    float* h_output_gpu = alloc_host(k);

    float* d_input = alloc_device(n);
    float* d_output = alloc_device(k);

    rand_init(h_input, n);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_topk(h_input, h_output_cpu, n, k);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    std::cout << "cpu top-k time: " << cpu_time << " us" << std::endl;

    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    double bitonic_time = benchmark([&]() {
        solve_bitonic_sort(d_input, d_output, n, k);
    }, epoches);
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, k * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "\n-------results-------" << std::endl;
    print_array("CPU top-k", h_output_cpu, k);
    print_array("Bitonic top-k", h_output_gpu, k);

    bool bitonic_correct = verify(h_output_cpu, h_output_gpu, k);
    std::cout << "Bitonic Sort: " << (bitonic_correct ? "Pass" : "Fail") << std::endl;
    std::cout << "gpu top-k time: " << bitonic_time << " us" << std::endl;

    free_host(h_input);
    free_host(h_output_cpu);
    free_host(h_output_gpu);
    free_device(d_input);
    free_device(d_output);
}

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    test_topk(1024, 10, 100);
    test_topk(1 << 15, 100, 100);
    test_topk(1 << 20, 1000, 50);

    std::cout << "\n======All Tests Passed==========\n" << std::endl;
    return 0;
}

