#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA device found.\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("========== GPU %d: %s ==========\n", dev, prop.name);
        printf("  Compute Capability     : %d.%d\n", prop.major, prop.minor);
        printf("  SM 数量 (multiprocessorCount) : %d\n", prop.multiProcessorCount);
        printf("  每 SM 最大线程数 (maxThreadsPerMultiProcessor) : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp 大小 (warpSize)   : %d 线程\n", prop.warpSize);
        printf("  每 Block 最大线程数 (maxThreadsPerBlock) : %d\n", prop.maxThreadsPerBlock);
        printf("  每 SM 最大 Block 数 (maxBlocksPerMultiProcessor): %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  每 Block 最大寄存器数 (regsPerBlock) : %d\n", prop.regsPerBlock);
        printf("  每 SM 寄存器总数 (regsPerMultiprocessor) : %d\n", prop.regsPerMultiprocessor);
        printf("  共享内存每 Block (sharedMemPerBlock) : %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  共享内存每 SM (sharedMemPerMultiprocessor) : %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  全局内存 (totalGlobalMem) : %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  每 SM 最大 warp 数 (理论) : %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
        printf("========================================\n");
    }
    return 0;
}
