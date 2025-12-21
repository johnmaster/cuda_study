#include "kernels.cuh"

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    // Print GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory Bandwidth: " << prop.memoryBusWidth * prop.memoryClockRate * 2 / 1e6 << " GB/s (theoretical)\n";
    
    // Test different matrix/vector sizes
    // GEMV: y = A * x, where A is MxK, x is K, y is M
    std::vector<std::tuple<int, int>> shapes = {
        // Small sizes
        {1024, 1024},
        {2048, 2048},
        
        // LLM-style: large K (embedding dimension), various M (sequence length)
        {1, 4096},          // Single token, 4K embedding
        {32, 4096},         // Batch of 32 tokens
        {128, 4096},        // Batch of 128 tokens
        {512, 4096},        // Batch of 512 tokens
        
        // Large matrices
        {4096, 4096},
        {8192, 8192},
        
        // Wide matrices (M >> K)
        {8192, 1024},
        
        // Tall matrices (K >> M)
        {1024, 8192},
    };

    for (auto [M, K] : shapes) {
        benchMark(M, K);
    }
    
    std::cout << "\n========================================\n";
    std::cout << "All GEMV benchmarks completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}

