#include "kernels.cuh"

int main() {
    CHECK_CUDA(cudaDeviceReset());
    
    // Print GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory Bandwidth: " << prop.memoryBusWidth * prop.memoryClockRate * 2 / 1e6 
              << " GB/s (theoretical)\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    
    // Test different batch sizes and hidden dimensions
    // Typical configurations in deep learning
    std::vector<std::pair<int, int>> configs = {
        // (batch_size, hidden_dim)
        
        // Small models / inference
        {1, 768},           // BERT-base, single sequence
        {8, 768},           // BERT-base, small batch
        {32, 768},          // BERT-base, medium batch
        
        // Large language models
        {1, 4096},          // GPT-style, single token
        {16, 4096},         // GPT-style, small batch
        {64, 4096},         // GPT-style, medium batch
        
        // Very large models
        {1, 12288},         // GPT-3 style, single token
        {8, 12288},         // GPT-3 style, small batch
        
        // Different aspect ratios
        {128, 512},         // Smaller hidden dim, larger batch
        {256, 1024},        // Medium size
        {512, 2048},        // Larger configurations
    };
    
    std::cout << "\n========================================\n";
    std::cout << "LayerNorm Benchmark Suite\n";
    std::cout << "========================================\n";
    std::cout << "\nTesting " << configs.size() << " configurations...\n";
    
    for (const auto& [batch_size, hidden_dim] : configs) {
        benchMark(batch_size, hidden_dim);
    }
    
    std::cout << "\n========================================\n";
    std::cout << "All LayerNorm benchmarks completed!\n";
    std::cout << "========================================\n";
    
    std::cout << "\nKey Optimizations Demonstrated:\n";
    std::cout << "1. Naive (3-pass): Baseline with separate mean, variance, normalize passes\n";
    std::cout << "2. Block Reduction: Single-pass computation with shared memory\n";
    std::cout << "3. Warp Reduction: Using warp shuffle for faster reduction\n";
    std::cout << "4. Vectorized: float4 loads/stores for better memory coalescing\n";
    std::cout << "5. Welford's Algorithm: Numerically stable online variance computation\n";
    
    return 0;
}

