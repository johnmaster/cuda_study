/**
 * GPU 设备信息查询工具
 * 编译: nvcc -o device_query device_query.cu
 * 运行: ./device_query
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found." << std::endl;
        return 1;
    }

    for (int d = 0; d < deviceCount; ++d) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, d));

        std::cout << "\n============== GPU " << d << " ==============\n";
        std::cout << "  名称                    : " << prop.name << "\n";
        std::cout << "  Compute Capability      : " << prop.major << "." << prop.minor << "\n";
        std::cout << "  SM 数量 (多处理器)      : " << prop.multiProcessorCount << "\n";
        std::cout << "  每 SM 最大线程数        : " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  全卡最大线程数 (理论)   : " << (prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor) << "\n";
        std::cout << "  Warp 大小               : " << prop.warpSize << "\n";
        std::cout << "  每 SM 最大 warp 数      : " << (prop.maxThreadsPerMultiProcessor / prop.warpSize) << "\n";
        std::cout << "  每 block 最大线程数     : " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  每 SM 最大 block 数     : " << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  全局内存 (MB)           : " << (prop.totalGlobalMem / (1024 * 1024)) << "\n";
        std::cout << "  每 SM 共享内存 (KB)     : " << (prop.sharedMemPerMultiprocessor / 1024) << "\n";
        std::cout << "  每 block 共享内存 (KB)  : " << (prop.sharedMemPerBlock / 1024) << "\n";
        std::cout << "  每 SM 寄存器数          : " << prop.regsPerMultiprocessor << "\n";
        std::cout << "  每 block 寄存器数       : " << prop.regsPerBlock << "\n";
        std::cout << "  L2 缓存大小 (KB)        : " << (prop.l2CacheSize / 1024) << "\n";

        // WMMA (Warp Matrix Multiply-Accumulate) / Tensor Cores
        bool wmma_support = (prop.major > 7) || (prop.major == 7 && prop.minor >= 0);
        std::cout << "\n  --- Tensor Core / WMMA ---\n";
        std::cout << "  支持 WMMA (Tensor Core) : " << (wmma_support ? "是" : "否") << " (需 Compute 7.0+)\n";
        if (wmma_support) {
            std::cout << "  FP16 矩阵运算          : 支持\n";
            if (prop.major >= 8) std::cout << "  BF16 / TF32 (Ampere+)   : 支持\n";
        }

        std::cout << "==============================\n" << std::endl;
    }

    return 0;
}
