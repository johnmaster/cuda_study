#include "kernels.cuh"

void test_row_reduction() {
    printf("\n========== 测试行规约 (Row Reduction) ==========\n");
    
    const size_t rows = 1024;
    const size_t cols = 2048;
    const size_t input_size = rows * cols;
    const size_t output_size = rows;
    
    printf("矩阵大小: %zu x %zu\n", rows, cols);
    
    // 分配主机内存
    float* h_input = alloc_host(input_size);
    float* h_output = alloc_host(output_size);
    
    // 初始化输入数据
    rand_init(h_input, rows, cols);
    
    // 分配设备内存
    float* d_input = alloc_device(input_size);
    float* d_output = alloc_device(output_size);
    
    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 测试朴素版本
    printf("\n--- 朴素版本 ---\n");
    {
        dim3 block(256);
        dim3 grid(rows);
        size_t shared_mem = block.x * sizeof(float);
        
        reduce_rows_naive<<<grid, block, shared_mem>>>(d_output, d_input, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        verify_row_reduction(h_input, h_output, rows, cols);
        
        double time = benchmark([&]() {
            reduce_rows_naive<<<grid, block, shared_mem>>>(d_output, d_input, rows, cols);
            cudaDeviceSynchronize();
        });
        
        printf("时间: %.2f μs\n", time);
        printf("带宽: %.2f GB/s\n", (input_size * sizeof(float)) / time / 1000.0);
    }
    
    // 测试优化版本
    printf("\n--- 优化版本 (Warp Shuffle) ---\n");
    {
        dim3 block(256);
        dim3 grid(rows);
        
        reduce_rows_optimized<<<grid, block>>>(d_output, d_input, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        verify_row_reduction(h_input, h_output, rows, cols);
        
        double time = benchmark([&]() {
            reduce_rows_optimized<<<grid, block>>>(d_output, d_input, rows, cols);
            cudaDeviceSynchronize();
        });
        
        printf("时间: %.2f μs\n", time);
        printf("带宽: %.2f GB/s\n", (input_size * sizeof(float)) / time / 1000.0);
    }
    
    // 清理
    free_host(h_input);
    free_host(h_output);
    free_device(d_input);
    free_device(d_output);
}

void test_col_reduction() {
    printf("\n========== 测试列规约 (Column Reduction) ==========\n");
    
    const size_t rows = 2048;
    const size_t cols = 1024;
    const size_t input_size = rows * cols;
    const size_t output_size = cols;
    
    printf("矩阵大小: %zu x %zu\n", rows, cols);
    
    // 分配主机内存
    float* h_input = alloc_host(input_size);
    float* h_output = alloc_host(output_size);
    
    // 初始化输入数据
    rand_init(h_input, rows, cols);
    
    // 分配设备内存
    float* d_input = alloc_device(input_size);
    float* d_output = alloc_device(output_size);
    
    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 测试朴素版本
    printf("\n--- 朴素版本 ---\n");
    {
        dim3 block(256);
        dim3 grid(cols);
        size_t shared_mem = block.x * sizeof(float);
        
        reduce_cols_naive<<<grid, block, shared_mem>>>(d_output, d_input, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        verify_col_reduction(h_input, h_output, rows, cols);
        
        double time = benchmark([&]() {
            reduce_cols_naive<<<grid, block, shared_mem>>>(d_output, d_input, rows, cols);
            cudaDeviceSynchronize();
        });
        
        printf("时间: %.2f μs\n", time);
        printf("带宽: %.2f GB/s\n", (input_size * sizeof(float)) / time / 1000.0);
    }
    
    // 测试优化版本
    printf("\n--- 优化版本 (Warp Shuffle) ---\n");
    {
        dim3 block(256);
        dim3 grid(cols);
        
        reduce_cols_optimized<<<grid, block>>>(d_output, d_input, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        verify_col_reduction(h_input, h_output, rows, cols);
        
        double time = benchmark([&]() {
            reduce_cols_optimized<<<grid, block>>>(d_output, d_input, rows, cols);
            cudaDeviceSynchronize();
        });
        
        printf("时间: %.2f μs\n", time);
        printf("带宽: %.2f GB/s\n", (input_size * sizeof(float)) / time / 1000.0);
    }
    
    // 清理
    free_host(h_input);
    free_host(h_output);
    free_device(d_input);
    free_device(d_output);
}

void test_global_reduction() {
    printf("\n========== 测试全局规约 (Global Reduction) ==========\n");
    
    const size_t rows = 2048;
    const size_t cols = 2048;
    const size_t input_size = rows * cols;
    
    printf("矩阵大小: %zu x %zu (总元素: %zu)\n", rows, cols, input_size);
    
    // 分配主机内存
    float* h_input = alloc_host(input_size);
    float h_output;
    
    // 初始化输入数据
    rand_init(h_input, rows, cols);
    
    // 分配设备内存
    float* d_input = alloc_device(input_size);
    float* d_output;
    float* d_temp;
    
    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("\n--- 两阶段全局规约 ---\n");
    {
        const int block_size = 256;
        const int num_blocks = (input_size + block_size * 2 - 1) / (block_size * 2);
        
        d_temp = alloc_device(num_blocks);
        d_output = alloc_device(1);
        
        // 第一阶段
        reduce_global_stage1<<<num_blocks, block_size>>>(d_temp, d_input, input_size);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 第二阶段
        reduce_global_stage2<<<1, block_size>>>(d_output, d_temp, num_blocks);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        verify_global_reduction(h_input, h_output, input_size);
        
        double time = benchmark([&]() {
            reduce_global_stage1<<<num_blocks, block_size>>>(d_temp, d_input, input_size);
            reduce_global_stage2<<<1, block_size>>>(d_output, d_temp, num_blocks);
            cudaDeviceSynchronize();
        });
        
        printf("时间: %.2f μs\n", time);
        printf("带宽: %.2f GB/s\n", (input_size * sizeof(float)) / time / 1000.0);
        
        free_device(d_temp);
        free_device(d_output);
    }
    
    // 清理
    free_host(h_input);
    free_device(d_input);
}

int main() {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║      2D 矩阵规约 (2D Reduction) 测试          ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");
    
    // 获取GPU信息
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("\nGPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("最大线程/块: %d\n", prop.maxThreadsPerBlock);
    
    // 运行测试
    test_row_reduction();
    test_col_reduction();
    test_global_reduction();
    
    printf("\n所有测试完成!\n");
    
    return 0;
}

