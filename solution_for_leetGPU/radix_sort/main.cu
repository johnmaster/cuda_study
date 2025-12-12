#include "kernels.cuh"

void cpu_radix_sort(uint32_t* input, uint32_t* output, size_t n) {
    std::vector<uint32_t> data(input, input + n);
    
    std::vector<uint32_t> temp(n);
    std::vector<uint32_t> count(RADIX);
    
    uint32_t* src = data.data();
    uint32_t* dst = temp.data();
    
    for (int bit = 0; bit < 32; bit += RADIX_BITS) {
        std::fill(count.begin(), count.end(), 0);
        
        //count the number of elements for each digit
        for (size_t i = 0; i < n; i++) {
            uint32_t digit = (src[i] >> bit) & (RADIX - 1);
            count[digit]++;
        }
        
        //prefix sum (exclusive scan)
        uint32_t total = 0;
        for (int i = 0; i < RADIX; i++) {
            uint32_t c = count[i];
            count[i] = total;
            total += c;
        }

        //write to output based on the count
        for (size_t i = 0; i < n; i++) {
            uint32_t digit = (src[i] >> bit) & (RADIX - 1);
            dst[count[digit]++] = src[i];
        }
        
        std::swap(src, dst);
    }

    if (src == data.data()) {
        std::copy(data.begin(), data.end(), output);
    } else {
        std::copy(temp.begin(), temp.end(), output);
    }
}

bool verify(uint32_t* cpu_result, uint32_t* gpu_result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            std::cerr << "Mismatch at position " << i << ": "
                      << "CPU = " << cpu_result[i] << ", GPU = "
                      << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}
bool is_sorted(uint32_t* arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i] < arr[i-1]) {
            return false;
        }
    }
    return true;
}
void print_array(const char* name, uint32_t* arr, size_t n, size_t max_print = 10) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(n, max_print); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << arr[i];
    }
    if (n > max_print) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

void test_radix_sort(size_t n, int epoches = 100) {
    std::cout << "\n=====================" << std::endl;
    std::cout << "Testing Radix Sort" << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "=======================\n" << std::endl;

    uint32_t* h_input = alloc_host(n);
    uint32_t* h_output_cpu = alloc_host(n);
    uint32_t* h_output_gpu = alloc_host(n);
    
    uint32_t* d_input = alloc_device(n);
    uint32_t* d_output = alloc_device(n);

    rand_init(h_input, n);
    print_array("Input", h_input, n);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_radix_sort(h_input, h_output_cpu, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();
    std::cout << "CPU time sort time: " << std::fixed << std::setprecision(2) << cpu_time
              << " us" << std::endl;

    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    double gpu_time = benchmark(
        [&]() {
            radix_sort_gpu(d_input, d_output, n);
        }, epoches
    );
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::cout << "GPU time sort time: " << std::fixed << std::setprecision(2)
              << gpu_time << " us" << std::endl;
    
    bool cpu_sorted = is_sorted(h_output_cpu, n);
    bool gpu_sorted = is_sorted(h_output_gpu, n);
    bool match = verify(h_output_cpu, h_output_gpu, n);

    std::cout << "CPU result sorted: " << (cpu_sorted ? "YES" : "NO") << std::endl;
    std::cout << "GPU result sorted: " << (gpu_sorted ? "YES" : "NO") << std::endl;
    std::cout << "Result match: " << (match ? "YES" : "NO") << std::endl;

    free_host(h_input);
    free_host(h_output_cpu);
    free_host(h_output_gpu);
    free_device(d_input);
    free_device(d_output);

    if (!match || !gpu_sorted) {
        std::cerr << "Test Failed" << std::endl;
        exit(1);
    }
    std::cout << "TEST PASSED" << std::endl;
}

int main() {
    CHECK_CUDA(cudaDeviceReset());

    test_radix_sort(1 << 20, 100);
    
    std::cout << "\n============ All test passed ===========\n" << std::endl;
    return 0;
}
